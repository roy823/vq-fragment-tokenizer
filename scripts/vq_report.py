#!/usr/bin/env python
"""Generate a tokenizer report from a trained VQ fragment model."""

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

from vqfrag.data import FragmentUnitDataset, collate_fragment_units
from vqfrag.metrics import (
    code_usage_stats,
    decoded_peak_features_to_peaks,
    grouped_spectral_reconstruction,
    summarize_codes,
)
from vqfrag.model import VQFragmentTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/canopus_hplus_peak_units.jsonl")
    parser.add_argument("--checkpoint", default="runs/canopus_vq_small/best_model.pt")
    parser.add_argument("--out-dir", default="runs/canopus_vq_small/report")
    parser.add_argument("--max-records", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = VQFragmentTokenizer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = FragmentUnitDataset(args.index, max_records=args.max_records)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fragment_units)

    peak_codes, fragment_codes, event_codes = [], [], []
    peak_recs = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(tensor_batch)
            peak_codes.append(outputs["peak_codes"].detach().cpu().numpy())
            fragment_codes.append(outputs["fragment_codes"].detach().cpu().numpy())
            event_codes.append(outputs["event_codes"].detach().cpu().numpy())
            peak_recs.append(outputs["peak_rec"].detach().cpu().numpy())

    peak_codes_arr = np.concatenate(peak_codes)
    fragment_codes_arr = np.concatenate(fragment_codes)
    event_codes_arr = np.concatenate(event_codes)
    peak_rec_arr = np.concatenate(peak_recs)

    units = dataset.units
    usage = {
        "peak": code_usage_stats(peak_codes_arr, model.codebook_size),
        "fragment": code_usage_stats(fragment_codes_arr, model.codebook_size),
        "event": code_usage_stats(event_codes_arr, model.codebook_size),
    }
    summaries = {
        "peak": summarize_codes(units, peak_codes_arr, codebook_size=model.codebook_size),
        "fragment": summarize_codes(units, fragment_codes_arr, codebook_size=model.codebook_size),
        "event": summarize_codes(units, event_codes_arr, codebook_size=model.codebook_size),
    }
    recon_peaks = decoded_peak_features_to_peaks(peak_rec_arr)
    recon = grouped_spectral_reconstruction(units, recon_peaks, max_spectra=500)

    report = {
        "checkpoint": str(args.checkpoint),
        "records": len(units),
        "codebook_size": model.codebook_size,
        "usage": usage,
        "reconstruction": recon,
        "pseudo_dag": True,
        "criterion_perplexity_fraction_min": 0.20,
        "criterion_pass": {
            name: stats["perplexity_fraction"] >= 0.20 for name, stats in usage.items()
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "code_summaries.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# VQ Fragment Tokenizer Report",
        "",
        f"- Records evaluated: {len(units)}",
        f"- Codebook size: {model.codebook_size}",
        "- DAG status: pseudo-DAG root-to-fragment events from FRIGID peak-formula JSONs",
        "",
        "## Code Usage",
        "",
        "| Codebook | Active fraction | Perplexity fraction | Criterion >= 0.20 |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in usage.items():
        lines.append(
            f"| {name} | {stats['active_fraction']:.3f} | {stats['perplexity_fraction']:.3f} | {report['criterion_pass'][name]} |"
        )
    lines.extend(
        [
            "",
            "## Reconstruction",
            "",
            f"- Mean grouped spectral cosine: {recon['spectral_cosine_mean']:.4f}",
            f"- Median grouped spectral cosine: {recon['spectral_cosine_median']:.4f}",
            "",
            "## Next Integration Step",
            "",
            "Use `scripts/vq_export_codes.py` to export per-spectrum code histograms for FRIGID-side conditioning experiments.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
