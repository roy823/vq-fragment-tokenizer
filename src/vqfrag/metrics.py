"""Metrics and reporting helpers for VQ fragment tokenization."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np

from .data import FragmentSpectrumUnit


def spectral_cosine(
    peaks_a: Sequence[tuple[float, float]],
    peaks_b: Sequence[tuple[float, float]],
    *,
    bin_width: float = 0.1,
    max_mz: float = 1500.0,
) -> float:
    """Cosine similarity between two binned spectra."""
    n_bins = int(np.ceil(max_mz / bin_width)) + 1
    vec_a = np.zeros(n_bins, dtype=np.float32)
    vec_b = np.zeros(n_bins, dtype=np.float32)
    for mz, inten in peaks_a:
        if 0 <= mz <= max_mz:
            vec_a[min(int(round(mz / bin_width)), n_bins - 1)] = max(vec_a[min(int(round(mz / bin_width)), n_bins - 1)], float(inten))
    for mz, inten in peaks_b:
        if 0 <= mz <= max_mz:
            vec_b[min(int(round(mz / bin_width)), n_bins - 1)] = max(vec_b[min(int(round(mz / bin_width)), n_bins - 1)], float(inten))
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def peak_recall_ppm(reference_mz: Sequence[float], query_mz: Sequence[float], *, ppm: float = 20.0) -> float:
    """Fraction of reference peaks matched by at least one query peak."""
    if len(reference_mz) == 0:
        return 0.0
    query = np.asarray(query_mz, dtype=np.float64)
    if query.size == 0:
        return 0.0
    matched = 0
    for mz in reference_mz:
        mz = float(mz)
        tol = abs(mz) * ppm * 1e-6
        if np.any(np.abs(query - mz) <= tol):
            matched += 1
    return matched / len(reference_mz)


def code_usage_stats(codes: Sequence[int] | np.ndarray, codebook_size: int) -> dict:
    arr = np.asarray(codes, dtype=np.int64).reshape(-1)
    counts = np.bincount(arr, minlength=codebook_size).astype(np.float64)
    total = counts.sum()
    probs = counts / total if total else counts
    active = probs > 0
    entropy = float(-(probs[active] * np.log(probs[active])).sum()) if total else 0.0
    perplexity = float(np.exp(entropy))
    return {
        "total": int(total),
        "active_codes": int(active.sum()),
        "active_fraction": float(active.sum() / codebook_size),
        "entropy": entropy,
        "perplexity": perplexity,
        "perplexity_fraction": float(perplexity / codebook_size),
        "counts": counts.astype(int).tolist(),
    }


def formula_subset_rate(units: Iterable[FragmentSpectrumUnit]) -> float:
    from .chem import formula_subset

    total = 0
    ok = 0
    for unit in units:
        total += 1
        ok += int(formula_subset(unit.fragment_formula, unit.root_formula))
    return ok / total if total else 0.0


def summarize_codes(
    units: Sequence[FragmentSpectrumUnit],
    codes: Sequence[int] | np.ndarray,
    *,
    codebook_size: int,
    top_k: int = 8,
) -> dict[int, dict]:
    """Summarize code semantics by frequent fragment/loss formulae."""
    by_code: dict[int, dict[str, Counter]] = {
        idx: {"fragment_formulae": Counter(), "neutral_losses": Counter(), "spectra": Counter()}
        for idx in range(codebook_size)
    }
    for unit, code in zip(units, np.asarray(codes, dtype=np.int64).reshape(-1)):
        if not 0 <= int(code) < codebook_size:
            continue
        entry = by_code[int(code)]
        entry["fragment_formulae"][unit.fragment_formula] += 1
        entry["neutral_losses"][unit.neutral_loss_formula] += 1
        entry["spectra"][unit.spectrum_id] += 1

    out = {}
    for code, entry in by_code.items():
        count = sum(entry["fragment_formulae"].values())
        if count == 0:
            continue
        out[code] = {
            "count": count,
            "top_fragment_formulae": entry["fragment_formulae"].most_common(top_k),
            "top_neutral_losses": entry["neutral_losses"].most_common(top_k),
            "unique_spectra": len(entry["spectra"]),
        }
    return out


def grouped_spectral_reconstruction(
    units: Sequence[FragmentSpectrumUnit],
    reconstructed_peaks: Sequence[tuple[float, float]],
    *,
    max_spectra: int | None = None,
    bin_width: float = 0.1,
    max_mz: float = 1500.0,
) -> dict:
    """Compute grouped spectrum reconstruction cosine from per-unit peaks."""
    original_by_spec: dict[str, list[tuple[float, float]]] = defaultdict(list)
    recon_by_spec: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for unit, peak in zip(units, reconstructed_peaks):
        original_by_spec[unit.spectrum_id].append((unit.mz, unit.intensity))
        recon_by_spec[unit.spectrum_id].append(peak)

    values = []
    for idx, spec_id in enumerate(original_by_spec):
        if max_spectra is not None and idx >= max_spectra:
            break
        values.append(
            spectral_cosine(
                original_by_spec[spec_id],
                recon_by_spec.get(spec_id, []),
                bin_width=bin_width,
                max_mz=max_mz,
            )
        )

    arr = np.asarray(values, dtype=np.float64)
    return {
        "spectra_evaluated": int(arr.size),
        "spectral_cosine_mean": float(arr.mean()) if arr.size else 0.0,
        "spectral_cosine_median": float(np.median(arr)) if arr.size else 0.0,
        "spectral_cosine_p10": float(np.percentile(arr, 10)) if arr.size else 0.0,
    }


def decoded_peak_features_to_peaks(peak_features: np.ndarray, *, mz_scale: float = 1500.0) -> list[tuple[float, float]]:
    """Convert decoded normalized peak features into ``(mz, intensity)`` pairs."""
    arr = np.asarray(peak_features, dtype=np.float32)
    mz = np.clip(arr[:, 0] * mz_scale, 0.0, mz_scale)
    inten = np.clip(arr[:, 1], 0.0, 1.0)
    return list(zip(mz.tolist(), inten.tolist()))
