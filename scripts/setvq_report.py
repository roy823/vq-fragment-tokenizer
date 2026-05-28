#!/usr/bin/env python
"""Generate a report for an observation-only SetVQ spectrum model."""

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
from vqfrag.metrics import code_usage_stats, peak_recall_ppm
from vqfrag.model import SetVQSpectrumTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/canopus_hplus_peak_units.jsonl")
    parser.add_argument("--checkpoint", default="runs/canopus_setvq_obs/best_model.pt")
    parser.add_argument("--out-dir", default="runs/canopus_setvq_obs/report")
    parser.add_argument("--max-spectra", type=int, default=5000)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--neighbor-spectra", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros(a.shape[0], dtype=np.float64)
    ok = denom > 0
    out[ok] = (a[ok] * b[ok]).sum(axis=1) / denom[ok]
    return out


def _predicted_peak_mz(row: np.ndarray, *, bin_width: float, max_mz: float, top_k: int) -> list[float]:
    if top_k <= 0:
        return []
    k = min(top_k, row.shape[0])
    bins = np.argpartition(-row, k - 1)[:k]
    bins = bins[np.argsort(-row[bins])]
    return [float(min(idx * bin_width, max_mz)) for idx in bins if row[idx] > 0]


def _predicted_peak_bins(row: np.ndarray, top_k: int) -> set[int]:
    if top_k <= 0:
        return set()
    k = min(top_k, row.shape[0])
    bins = np.argpartition(-row, k - 1)[:k]
    return {int(idx) for idx in bins if row[idx] > 0}


def _bin_recall(target_presence: np.ndarray, predicted_bins: set[int]) -> float:
    target_bins = set(np.flatnonzero(target_presence > 0))
    if not target_bins:
        return 0.0
    return len(target_bins & predicted_bins) / len(target_bins)


def _code_histogram(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    counts = np.bincount(codes.reshape(-1).astype(np.int64), minlength=codebook_size).astype(np.float32)
    return counts / max(float(counts.sum()), 1.0)


def _nearest_neighbors(names: list[str], hist: np.ndarray, max_items: int) -> list[dict]:
    n = min(len(names), max_items)
    if n <= 1:
        return []
    x = hist[:n].astype(np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = x @ x.T
    examples = []
    for row in range(min(10, n)):
        order = np.argsort(-sim[row])
        neighbors = [
            {"spectrum_id": names[col], "cosine": float(sim[row, col])}
            for col in order
            if col != row
        ][:5]
        examples.append({"query": names[row], "neighbors": neighbors})
    return examples


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    feature_config = SpectrumFeatureConfig(**checkpoint["feature_config"])
    model = SetVQSpectrumTokenizer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = SpectrumSetDataset(
        args.index,
        max_spectra=args.max_spectra,
        max_units=args.max_units,
        feature_config=feature_config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_spectrum_sets)

    all_codes = []
    cosines = []
    random_cosines = []
    recalls = []
    bin_recalls = []
    names = []
    histograms = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
            outputs = model(tensor_batch)
            codes = outputs["slot_codes"].detach().cpu().numpy()
            recon = outputs["binned_recon"].detach().cpu().numpy()
            target = batch["target_intensity"].detach().cpu().numpy()
            presence = batch["target_presence"].detach().cpu().numpy()

            all_codes.append(codes.reshape(-1))
            cosines.extend(_cosine(recon, target).tolist())
            random_recon = rng.random(recon.shape, dtype=np.float32)
            random_cosines.extend(_cosine(random_recon, target).tolist())

            for spec_id, row, pres, peaks, row_codes in zip(batch["spectrum_id"], recon, presence, batch["peaks"], codes):
                ref_mz = [float(mz) for mz, _ in peaks if 0 <= float(mz) <= feature_config.max_mz]
                top_k = int(pres.sum())
                pred_bins = _predicted_peak_bins(row, top_k)
                pred_mz = _predicted_peak_mz(
                    row,
                    bin_width=feature_config.bin_width,
                    max_mz=feature_config.max_mz,
                    top_k=top_k,
                )
                recalls.append(peak_recall_ppm(ref_mz, pred_mz, ppm=args.ppm))
                bin_recalls.append(_bin_recall(pres, pred_bins))
                names.append(spec_id)
                histograms.append(_code_histogram(row_codes, model.codebook_size))

    codes_arr = np.concatenate(all_codes) if all_codes else np.array([], dtype=np.int64)
    usage = code_usage_stats(codes_arr, model.codebook_size)
    hist_arr = np.stack(histograms, axis=0) if histograms else np.zeros((0, model.codebook_size), dtype=np.float32)
    report = {
        "checkpoint": str(args.checkpoint),
        "spectra_evaluated": int(len(names)),
        "codebook_size": int(model.codebook_size),
        "num_slots": int(model.num_slots),
        "num_bins": int(model.num_bins),
        "feature_config": checkpoint["feature_config"],
        "usage": usage,
        "reconstruction": {
            "binned_spectral_cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
            "binned_spectral_cosine_median": float(np.median(cosines)) if cosines else 0.0,
            "random_binned_spectral_cosine_mean": float(np.mean(random_cosines)) if random_cosines else 0.0,
            "peak_recall_ppm_mean": float(np.mean(recalls)) if recalls else 0.0,
            "peak_recall_ppm_median": float(np.median(recalls)) if recalls else 0.0,
            "peak_recall_bin_mean": float(np.mean(bin_recalls)) if bin_recalls else 0.0,
            "peak_recall_bin_median": float(np.median(bin_recalls)) if bin_recalls else 0.0,
            "ppm": float(args.ppm),
        },
        "nearest_neighbors": _nearest_neighbors(names, hist_arr, args.neighbor_spectra),
        "criterion_perplexity_fraction_min": 0.20,
        "criterion_pass": {
            "code_perplexity_fraction": usage["perplexity_fraction"] >= 0.20,
            "beats_random_cosine": (float(np.mean(cosines)) if cosines else 0.0)
            > (float(np.mean(random_cosines)) if random_cosines else 0.0),
        },
    }
    (out_dir / "setvq_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "nearest_neighbors.json").write_text(
        json.dumps(report["nearest_neighbors"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Observation-Only SetVQ Report",
        "",
        f"- Spectra evaluated: {len(names)}",
        f"- Codebook size: {model.codebook_size}",
        f"- Slots per spectrum: {model.num_slots}",
        f"- Bins: {model.num_bins} at width {feature_config.bin_width}",
        "- Inputs: mz, intensity, collision energy, adduct condition, instrument condition",
        "- Excluded from model input: root formula, fragment formula, neutral loss, mz error",
        "",
        "## Code Usage",
        "",
        "| Active fraction | Perplexity fraction | Criterion >= 0.20 |",
        "|---:|---:|---:|",
        f"| {usage['active_fraction']:.3f} | {usage['perplexity_fraction']:.3f} | {report['criterion_pass']['code_perplexity_fraction']} |",
        "",
        "## Reconstruction",
        "",
        f"- Mean binned spectral cosine: {report['reconstruction']['binned_spectral_cosine_mean']:.4f}",
        f"- Median binned spectral cosine: {report['reconstruction']['binned_spectral_cosine_median']:.4f}",
        f"- Random mean binned spectral cosine: {report['reconstruction']['random_binned_spectral_cosine_mean']:.4f}",
        f"- Mean peak recall@{args.ppm:g}ppm: {report['reconstruction']['peak_recall_ppm_mean']:.4f}",
        f"- Median peak recall@{args.ppm:g}ppm: {report['reconstruction']['peak_recall_ppm_median']:.4f}",
        f"- Mean peak recall@bin: {report['reconstruction']['peak_recall_bin_mean']:.4f}",
        f"- Median peak recall@bin: {report['reconstruction']['peak_recall_bin_median']:.4f}",
        "",
        "## Next Probe",
        "",
        "Run `scripts/vq_probe.py` on `setvq_code_histograms.npz` to test whether observation-only codes predict weak formula/loss labels.",
    ]
    (out_dir / "setvq_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
