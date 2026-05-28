#!/usr/bin/env python
"""Train the standalone three-codebook VQ fragment tokenizer."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqfrag.data import FragmentUnitDataset, collate_fragment_units
from vqfrag.metrics import code_usage_stats
from vqfrag.model import VQFragmentTokenizer, entropy_regularizer, initialize_codebooks_from_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/canopus_hplus_peak_units.jsonl")
    parser.add_argument("--out-dir", default="runs/canopus_vq_small")
    parser.add_argument("--max-records", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--init-codebooks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-batches", type=int, default=4)
    parser.add_argument("--refresh-codebooks-each-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _epoch(
    model: VQFragmentTokenizer,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    entropy_weight: float,
) -> dict:
    train = optimizer is not None
    model.train(train)
    totals = {
        "loss": 0.0,
        "raw_loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "n": 0,
    }
    all_peak_codes = []
    all_fragment_codes = []
    all_event_codes = []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch)
            entropy_bonus = entropy_regularizer(outputs, model.codebook_size)
            loss = outputs["loss"] - entropy_weight * entropy_bonus
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_n = int(batch["peak_x"].shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_n
        totals["raw_loss"] += float(outputs["loss"].detach().cpu()) * batch_n
        totals["recon_loss"] += float(outputs["recon_loss"].detach().cpu()) * batch_n
        totals["vq_loss"] += float(outputs["vq_loss"].detach().cpu()) * batch_n
        totals["n"] += batch_n
        all_peak_codes.append(outputs["peak_codes"].detach().cpu().numpy())
        all_fragment_codes.append(outputs["fragment_codes"].detach().cpu().numpy())
        all_event_codes.append(outputs["event_codes"].detach().cpu().numpy())

    n = max(totals.pop("n"), 1)
    metrics = {k: v / n for k, v in totals.items()}
    for name, chunks in [
        ("peak", all_peak_codes),
        ("fragment", all_fragment_codes),
        ("event", all_event_codes),
    ]:
        stats = code_usage_stats(np.concatenate(chunks), model.codebook_size) if chunks else {}
        metrics[f"{name}_active_fraction"] = stats.get("active_fraction", 0.0)
        metrics[f"{name}_perplexity_fraction"] = stats.get("perplexity_fraction", 0.0)
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FragmentUnitDataset(args.index, max_records=args.max_records)
    if len(dataset) == 0:
        raise RuntimeError(f"No records found in {args.index}")
    indices = list(range(len(dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_fraction)) if len(indices) > 10 else max(1, len(indices) // 5)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:] or indices

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fragment_units,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fragment_units,
    )

    device = _device(args.device)
    model = VQFragmentTokenizer(
        hidden_dim=args.hidden_dim,
        code_dim=args.code_dim,
        codebook_size=args.codebook_size,
    ).to(device)
    if args.init_codebooks:
        initialize_codebooks_from_loader(
            model,
            train_loader,
            device=device,
            num_batches=args.init_batches,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            entropy_weight=args.entropy_weight,
        )
        if args.refresh_codebooks_each_epoch:
            initialize_codebooks_from_loader(
                model,
                train_loader,
                device=device,
                num_batches=args.init_batches,
            )
        with torch.no_grad():
            val_metrics = _epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                entropy_weight=args.entropy_weight,
            )
        entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(entry)
        print(json.dumps(entry, sort_keys=True))

        if val_metrics["raw_loss"] < best_val:
            best_val = val_metrics["raw_loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config_dict(),
                    "args": vars(args),
                    "best_val_raw_loss": best_val,
                },
                out_dir / "best_model.pt",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
