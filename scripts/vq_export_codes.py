#!/usr/bin/env python
"""Export per-spectrum VQ code histograms for downstream FRIGID conditioning."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqfrag.data import FragmentUnitDataset, collate_fragment_units
from vqfrag.model import VQFragmentTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/canopus_hplus_peak_units.jsonl")
    parser.add_argument("--checkpoint", default="runs/canopus_vq_small/best_model.pt")
    parser.add_argument("--out", default="runs/canopus_vq_small/vq_code_histograms.npz")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = VQFragmentTokenizer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = FragmentUnitDataset(args.index, max_records=args.max_records)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fragment_units)

    hist = defaultdict(lambda: np.zeros(model.codebook_size * 3, dtype=np.float32))
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(tensor_batch)
            peak = outputs["peak_codes"].detach().cpu().numpy()
            frag = outputs["fragment_codes"].detach().cpu().numpy()
            event = outputs["event_codes"].detach().cpu().numpy()
            for spec_id, p, f, e in zip(batch["spectrum_id"], peak, frag, event):
                hist[spec_id][int(p)] += 1.0
                hist[spec_id][model.codebook_size + int(f)] += 1.0
                hist[spec_id][2 * model.codebook_size + int(e)] += 1.0

    names = np.array(sorted(hist.keys()), dtype=object)
    features = np.stack([hist[name] / max(hist[name].sum(), 1.0) for name in names], axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spectrum_id=names, vq_histogram=features)
    meta = {
        "checkpoint": str(args.checkpoint),
        "index": str(args.index),
        "num_spectra": int(len(names)),
        "feature_dim": int(features.shape[1]),
        "layout": "peak_hist | fragment_hist | event_hist",
        "codebook_size": int(model.codebook_size),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
