#!/usr/bin/env python
"""Train the observation-only SetVQ spectrum tokenizer."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqfrag.data import SpectrumFeatureConfig, SpectrumSetDataset, collate_spectrum_sets
from vqfrag.metrics import code_usage_stats
from vqfrag.model import SetVQSpectrumTokenizer, initialize_setvq_codebook_from_loader, setvq_entropy_regularizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/canopus_hplus_peak_units.jsonl")
    parser.add_argument("--out-dir", default="runs/canopus_setvq_obs")
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--max-peaks", type=int, default=256)
    parser.add_argument("--max-mz", type=float, default=1500.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--init-codebook", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-batches", type=int, default=4)
    parser.add_argument("--refresh-codebook-each-epoch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def _epoch(
    model: SetVQSpectrumTokenizer,
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
        "bce_loss": 0.0,
        "intensity_loss": 0.0,
        "spectral_cosine_loss": 0.0,
        "n": 0,
    }
    all_codes = []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch)
            entropy_bonus = setvq_entropy_regularizer(outputs, model.codebook_size)
            loss = outputs["loss"] - entropy_weight * entropy_bonus
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_n = int(batch["peak_x"].shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_n
        totals["raw_loss"] += float(outputs["loss"].detach().cpu()) * batch_n
        for key in ["recon_loss", "vq_loss", "bce_loss", "intensity_loss", "spectral_cosine_loss"]:
            totals[key] += float(outputs[key].detach().cpu()) * batch_n
        totals["n"] += batch_n
        all_codes.append(outputs["slot_codes"].detach().cpu().numpy().reshape(-1))

    n = max(totals.pop("n"), 1)
    metrics = {key: value / n for key, value in totals.items()}
    usage = code_usage_stats(np.concatenate(all_codes), model.codebook_size) if all_codes else {}
    metrics["code_active_fraction"] = usage.get("active_fraction", 0.0)
    metrics["code_perplexity_fraction"] = usage.get("perplexity_fraction", 0.0)
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_config = SpectrumFeatureConfig(
        max_peaks=args.max_peaks,
        max_mz=args.max_mz,
        bin_width=args.bin_width,
        mz_scale=args.max_mz,
    )
    dataset = SpectrumSetDataset(
        args.index,
        max_spectra=args.max_spectra,
        max_units=args.max_units,
        feature_config=feature_config,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No spectra found in {args.index}")

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
        collate_fn=collate_spectrum_sets,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_spectrum_sets,
    )

    device = _device(args.device)
    model = SetVQSpectrumTokenizer(
        hidden_dim=args.hidden_dim,
        code_dim=args.code_dim,
        codebook_size=args.codebook_size,
        num_slots=args.num_slots,
        num_bins=feature_config.num_bins,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    if args.init_codebook:
        initialize_setvq_codebook_from_loader(model, train_loader, device=device, num_batches=args.init_batches)
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
        if args.refresh_codebook_each_epoch:
            initialize_setvq_codebook_from_loader(model, train_loader, device=device, num_batches=args.init_batches)
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
                    "feature_config": asdict(feature_config),
                    "args": vars(args),
                    "best_val_raw_loss": best_val,
                },
                out_dir / "best_model.pt",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
