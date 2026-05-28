#!/usr/bin/env python
"""Export per-spectrum SetVQ slot-code sequences and histograms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqfrag.data import SpectrumFeatureConfig, SpectrumSetDataset, collate_spectrum_sets
from vqfrag.model import SetVQSpectrumTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/canopus_hplus_peak_units.jsonl")
    parser.add_argument("--checkpoint", default="runs/canopus_setvq_obs/best_model.pt")
    parser.add_argument("--out", default="runs/canopus_setvq_obs/setvq_codes.npz")
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def code_histogram(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    counts = np.bincount(codes.reshape(-1).astype(np.int64), minlength=codebook_size).astype(np.float32)
    return counts / max(float(counts.sum()), 1.0)


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = dict(checkpoint["model_config"])
    feature_config_raw = dict(checkpoint["feature_config"])
    feature_config_raw.setdefault("formula_conditioned", bool(model_config.get("formula_conditioned", model_config.get("peak_dim", 3) > 3)))
    feature_config = SpectrumFeatureConfig(**feature_config_raw)
    model = SetVQSpectrumTokenizer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = SpectrumSetDataset(
        args.index,
        max_spectra=args.max_spectra,
        max_units=args.max_units,
        feature_config=feature_config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_spectrum_sets)

    names = []
    features = []
    sequences = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(tensor_batch)
            codes = outputs["slot_codes"].detach().cpu().numpy()
            for spec_id, row_codes in zip(batch["spectrum_id"], codes):
                names.append(spec_id)
                sequences.append(row_codes.astype(np.int64, copy=False))
                features.append(code_histogram(row_codes, model.codebook_size))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feature_arr = np.stack(features, axis=0) if features else np.zeros((0, model.codebook_size), dtype=np.float32)
    sequence_arr = np.stack(sequences, axis=0) if sequences else np.zeros((0, model.num_slots), dtype=np.int64)
    name_arr = np.asarray(names, dtype=object)
    np.savez_compressed(
        out,
        spectrum_id=name_arr,
        vq_histogram=feature_arr,
        vq_sequence=sequence_arr,
        slot_codes=sequence_arr,
    )
    meta = {
        "checkpoint": str(args.checkpoint),
        "index": str(args.index),
        "num_spectra": int(len(names)),
        "feature_dim": int(feature_arr.shape[1]),
        "sequence_shape": list(sequence_arr.shape),
        "histogram_shape": list(feature_arr.shape),
        "layout": "vq_sequence [num_spectra,num_slots] | vq_histogram [num_spectra,codebook_size]",
        "codebook_size": int(model.codebook_size),
        "num_slots": int(model.num_slots),
        "formula_conditioned": bool(model.formula_conditioned),
        "feature_config": feature_config_raw,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
