#!/usr/bin/env python
"""Probe whether exported VQ histograms preserve chemically useful signals.

This script intentionally evaluates representation quality, not reconstruction
quality. It trains small linear probes from per-spectrum features to spectrum-
level weak labels derived from the JSONL index:

- root formula element presence
- neutral loss formula presence among frequent losses
- fragment formula presence among frequent fragments
- coarse m/z-bin presence
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqfrag.chem import NORM_VEC, VALID_ELEMENTS, formula_mass, formula_to_vector, ion_mass_shift, normalize_ion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Exported .npz from vq_export_codes.py")
    parser.add_argument("--index", required=True, help="FragmentSpectrumUnit JSONL index")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--top-losses", type=int, default=128)
    parser.add_argument("--top-fragments", type=int, default=128)
    parser.add_argument("--mz-bin-width", type=float, default=10.0)
    parser.add_argument("--max-mz", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--retrieval-max-queries", type=int, default=5000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_features(path: str | Path, max_spectra: int | None) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    names = np.asarray(data["spectrum_id"], dtype=str)
    x = np.asarray(data["vq_histogram"], dtype=np.float32)
    if max_spectra is not None:
        names = names[:max_spectra]
        x = x[:max_spectra]
    return names, x


def _safe_formula_vec(formula: str | None) -> np.ndarray | None:
    try:
        return formula_to_vector(formula)
    except ValueError:
        return None


def collect_targets(
    *,
    index: str | Path,
    spectrum_names: np.ndarray,
    top_losses: int,
    top_fragments: int,
    mz_bin_width: float,
    max_mz: float,
) -> dict:
    """Stream the JSONL index and aggregate spectrum-level weak labels."""
    spec_to_row = {str(spec): idx for idx, spec in enumerate(spectrum_names.tolist())}
    n = len(spectrum_names)
    n_mz_bins = int(math.ceil(max_mz / mz_bin_width)) + 1

    root_elements = np.zeros((n, len(VALID_ELEMENTS)), dtype=np.float32)
    formula_features = np.zeros((n, len(VALID_ELEMENTS) + 3), dtype=np.float32)
    mz_bins = np.zeros((n, n_mz_bins), dtype=np.float32)
    peak_stats_raw = np.zeros((n, 10), dtype=np.float64)

    losses_by_spec: list[Counter[str]] = [Counter() for _ in range(n)]
    fragments_by_spec: list[Counter[str]] = [Counter() for _ in range(n)]
    global_losses: Counter[str] = Counter()
    global_fragments: Counter[str] = Counter()

    seen_root = np.zeros(n, dtype=bool)
    root_formula_names = [""] * n
    parent_mass_by_row = np.zeros(n, dtype=np.float64)
    precursor_mz_by_row = np.zeros(n, dtype=np.float64)
    adduct_shift_by_row = np.zeros(n, dtype=np.float64)
    rows_seen = 0
    with Path(index).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            row = spec_to_row.get(str(rec.get("spectrum_id", "")))
            if row is None:
                continue
            rows_seen += 1

            if not seen_root[row]:
                root_formula_names[row] = str(rec.get("root_formula") or "")
                vec = _safe_formula_vec(rec.get("root_formula"))
                if vec is not None:
                    root_elements[row] = (vec > 0).astype(np.float32)
                    parent_mass = formula_mass(vec)
                    adduct = normalize_ion(rec.get("adduct"))
                    try:
                        shift = ion_mass_shift(adduct)
                    except ValueError:
                        shift = 0.0
                    precursor_mz = parent_mass + shift
                    parent_mass_by_row[row] = parent_mass
                    precursor_mz_by_row[row] = precursor_mz
                    adduct_shift_by_row[row] = shift
                    formula_features[row, : len(VALID_ELEMENTS)] = (vec / NORM_VEC).astype(np.float32)
                    formula_features[row, len(VALID_ELEMENTS)] = parent_mass / max_mz
                    formula_features[row, len(VALID_ELEMENTS) + 1] = precursor_mz / max_mz
                    formula_features[row, len(VALID_ELEMENTS) + 2] = vec.sum() / float(NORM_VEC.sum())
                seen_root[row] = True

            frag = str(rec.get("fragment_formula") or "")
            if frag:
                fragments_by_spec[row][frag] += 1
                global_fragments[frag] += 1

            loss = str(rec.get("neutral_loss_formula") or "")
            if loss:
                losses_by_spec[row][loss] += 1
                global_losses[loss] += 1

            mz = float(rec.get("mz", 0.0))
            intensity = float(rec.get("intensity", 0.0))
            neutral_peak_mass = max(mz - adduct_shift_by_row[row], 0.0)
            loss_mass = max(parent_mass_by_row[row] - neutral_peak_mass, 0.0) if parent_mass_by_row[row] > 0 else 0.0
            if 0 <= mz <= max_mz:
                mz_bins[row, min(int(round(mz / mz_bin_width)), n_mz_bins - 1)] = 1.0

            # count, sum_mz, sum_mz2, max_mz, sum_intensity, max_intensity,
            # weighted_mz_sum, sum_loss_mass, max_loss_mass, valid_loss_count
            peak_stats_raw[row, 0] += 1.0
            peak_stats_raw[row, 1] += mz
            peak_stats_raw[row, 2] += mz * mz
            peak_stats_raw[row, 3] = max(peak_stats_raw[row, 3], mz)
            peak_stats_raw[row, 4] += intensity
            peak_stats_raw[row, 5] = max(peak_stats_raw[row, 5], intensity)
            peak_stats_raw[row, 6] += mz * intensity
            if loss_mass > 0:
                peak_stats_raw[row, 7] += loss_mass
                peak_stats_raw[row, 8] = max(peak_stats_raw[row, 8], loss_mass)
                peak_stats_raw[row, 9] += 1.0

    top_loss_names = [name for name, _ in global_losses.most_common(top_losses)]
    top_fragment_names = [name for name, _ in global_fragments.most_common(top_fragments)]
    loss_to_col = {name: idx for idx, name in enumerate(top_loss_names)}
    fragment_to_col = {name: idx for idx, name in enumerate(top_fragment_names)}

    loss_labels = np.zeros((n, len(top_loss_names)), dtype=np.float32)
    fragment_labels = np.zeros((n, len(top_fragment_names)), dtype=np.float32)
    for row, counter in enumerate(losses_by_spec):
        for name in counter:
            col = loss_to_col.get(name)
            if col is not None:
                loss_labels[row, col] = 1.0
    for row, counter in enumerate(fragments_by_spec):
        for name in counter:
            col = fragment_to_col.get(name)
            if col is not None:
                fragment_labels[row, col] = 1.0

    count = np.maximum(peak_stats_raw[:, 0], 1.0)
    valid_loss_count = np.maximum(peak_stats_raw[:, 9], 1.0)
    mean_mz = peak_stats_raw[:, 1] / count
    var_mz = np.maximum(peak_stats_raw[:, 2] / count - mean_mz * mean_mz, 0.0)
    weighted_mz = peak_stats_raw[:, 6] / np.maximum(peak_stats_raw[:, 4], 1e-8)
    mean_loss = peak_stats_raw[:, 7] / valid_loss_count
    peak_stats = np.stack(
        [
            np.log1p(peak_stats_raw[:, 0]),
            mean_mz,
            np.sqrt(var_mz),
            peak_stats_raw[:, 3],
            peak_stats_raw[:, 4],
            peak_stats_raw[:, 5],
            weighted_mz,
            mean_loss,
            peak_stats_raw[:, 8],
            np.log1p(peak_stats_raw[:, 9]),
        ],
        axis=1,
    ).astype(np.float32)

    missing = int((~seen_root).sum())
    return {
        "targets": {
            "root_element": {"y": root_elements, "labels": VALID_ELEMENTS},
            "neutral_loss": {"y": loss_labels, "labels": top_loss_names},
            "fragment_formula": {"y": fragment_labels, "labels": top_fragment_names},
            "mz_bin": {"y": mz_bins, "labels": [f"{i * mz_bin_width:.1f}" for i in range(n_mz_bins)]},
        },
        "formula_features": formula_features,
        "root_formula": root_formula_names,
        "peak_stats": peak_stats,
        "rows_seen": rows_seen,
        "missing_root_labels": missing,
    }


def make_split(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def standardize_features(x: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float32).reshape(-1)
    s = np.asarray(y_score, dtype=np.float32).reshape(-1)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s)
    ranked = y[order]
    cumsum = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1, dtype=np.float32)
    return float((cumsum[ranked > 0] / ranks[ranked > 0]).sum() / positives)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float32).reshape(-1)
    s = np.asarray(y_score, dtype=np.float32).reshape(-1)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y > 0].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def topk_recall(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    values = []
    for truth, score in zip(y_true, y_score):
        positives = int(truth.sum())
        if positives == 0:
            continue
        kk = min(k, len(score))
        pred = np.argpartition(-score, kk - 1)[:kk]
        values.append(float(truth[pred].sum() / positives))
    return float(np.mean(values)) if values else float("nan")


def multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    per_label_ap = []
    per_label_auc = []
    for col in range(y_true.shape[1]):
        if y_true[:, col].sum() == 0:
            continue
        per_label_ap.append(average_precision(y_true[:, col], y_score[:, col]))
        per_label_auc.append(roc_auc(y_true[:, col], y_score[:, col]))
    return {
        "micro_ap": average_precision(y_true, y_score),
        "macro_ap": float(np.nanmean(per_label_ap)) if per_label_ap else float("nan"),
        "macro_auc": float(np.nanmean(per_label_auc)) if per_label_auc else float("nan"),
        "recall_at_1": topk_recall(y_true, y_score, 1),
        "recall_at_5": topk_recall(y_true, y_score, 5),
        "recall_at_10": topk_recall(y_true, y_score, 10),
    }


def same_formula_retrieval(
    *,
    x: np.ndarray,
    root_formula: list[str],
    loss_y: np.ndarray,
    fragment_y: np.ndarray,
    max_queries: int,
    seed: int,
) -> dict:
    """Nearest-neighbor retrieval within equal-root-formula groups."""

    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for idx, formula in enumerate(root_formula):
        if formula:
            groups.setdefault(formula, []).append(idx)
    query_pool = [idx for values in groups.values() if len(values) > 1 for idx in values]
    if len(query_pool) > max_queries:
        query_pool = rng.choice(np.asarray(query_pool), size=max_queries, replace=False).tolist()
    if not query_pool:
        return {"queries": 0, "loss_jaccard": float("nan"), "fragment_jaccard": float("nan")}

    x_std = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    x_norm = x_std / np.maximum(np.linalg.norm(x_std, axis=1, keepdims=True), 1e-12)
    loss_scores = []
    fragment_scores = []
    for query in query_pool:
        candidates = [idx for idx in groups[root_formula[query]] if idx != query]
        if not candidates:
            continue
        sims = x_norm[candidates] @ x_norm[query]
        nn = candidates[int(np.argmax(sims))]
        for y, scores in [(loss_y, loss_scores), (fragment_y, fragment_scores)]:
            a = y[query] > 0
            b = y[nn] > 0
            union = np.logical_or(a, b).sum()
            if union == 0:
                continue
            scores.append(float(np.logical_and(a, b).sum() / union))
    return {
        "queries": int(len(query_pool)),
        "loss_jaccard": float(np.mean(loss_scores)) if loss_scores else float("nan"),
        "fragment_jaccard": float(np.mean(fragment_scores)) if fragment_scores else float("nan"),
    }


def train_linear_probe(
    *,
    x: np.ndarray,
    y: np.ndarray,
    split: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> dict:
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
    keep = (y[train_idx].sum(axis=0) >= 3) & ((len(train_idx) - y[train_idx].sum(axis=0)) >= 3) & (y[test_idx].sum(axis=0) > 0)
    y = y[:, keep]
    if y.shape[1] == 0:
        return {"num_labels": 0, "error": "no probe labels after train/test support filtering"}

    x_std = standardize_features(x, train_idx)
    x_train = torch.from_numpy(x_std[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x_std[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)
    x_test = torch.from_numpy(x_std[test_idx]).to(device)

    model = nn.Linear(x.shape[1], y.shape[1]).to(device)
    pos = torch.from_numpy(y[train_idx].sum(axis=0)).float().to(device)
    neg = float(len(train_idx)) - pos
    pos_weight = torch.clamp(neg / torch.clamp(pos, min=1.0), max=50.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).detach().cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_score = torch.sigmoid(model(x_test)).detach().cpu().numpy()

    metrics = multilabel_metrics(y[test_idx], test_score)
    metrics.update(
        {
            "num_labels": int(y.shape[1]),
            "best_val_loss": best_val,
            "test_positive_fraction": float(y[test_idx].mean()),
        }
    )
    return metrics


def write_markdown(results: dict, out_path: Path) -> None:
    lines = [
        "# VQ Representation Probe",
        "",
        f"- spectra: {results['num_spectra']}",
        f"- rows matched from index: {results['rows_seen']}",
        f"- missing root labels: {results['missing_root_labels']}",
        "",
        "| task | feature | labels | micro AP | macro AP | macro AUC | R@1 | R@5 | R@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, by_feature in results["probes"].items():
        for feature_name, metrics in by_feature.items():
            if metrics.get("num_labels", 0) == 0:
                lines.append(f"| {task} | {feature_name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                "| {task} | {feature} | {labels} | {micro:.4f} | {macro:.4f} | {auc:.4f} | {r1:.4f} | {r5:.4f} | {r10:.4f} |".format(
                    task=task,
                    feature=feature_name,
                    labels=metrics["num_labels"],
                    micro=metrics["micro_ap"],
                    macro=metrics["macro_ap"],
                    auc=metrics["macro_auc"],
                    r1=metrics["recall_at_1"],
                    r5=metrics["recall_at_5"],
                    r10=metrics["recall_at_10"],
                )
            )
    lines.extend(
        [
            "",
            "## Same-Formula Retrieval",
            "",
            "| feature | queries | loss Jaccard | fragment Jaccard |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for feature_name, metrics in results.get("same_formula_retrieval", {}).items():
        lines.append(
            "| {feature} | {queries} | {loss:.4f} | {fragment:.4f} |".format(
                feature=feature_name,
                queries=metrics["queries"],
                loss=metrics["loss_jaccard"],
                fragment=metrics["fragment_jaccard"],
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    names, vq_x = load_features(args.features, args.max_spectra)
    target_bundle = collect_targets(
        index=args.index,
        spectrum_names=names,
        top_losses=args.top_losses,
        top_fragments=args.top_fragments,
        mz_bin_width=args.mz_bin_width,
        max_mz=args.max_mz,
    )
    peak_stats = target_bundle["peak_stats"]
    formula_features = target_bundle["formula_features"]
    feature_sets = {
        "vq": vq_x,
        "peak_stats": peak_stats,
        "formula_only": formula_features,
        "formula_peak_stats": np.concatenate([formula_features, peak_stats], axis=1),
        "formula_vq": np.concatenate([formula_features, vq_x], axis=1),
        "vq_plus_peak_stats": np.concatenate([vq_x, peak_stats], axis=1),
        "formula_peak_stats_vq": np.concatenate([formula_features, peak_stats, vq_x], axis=1),
    }
    split = make_split(len(names), args.seed)

    probes: dict[str, dict[str, dict]] = {}
    for task, payload in target_bundle["targets"].items():
        probes[task] = {}
        y = np.asarray(payload["y"], dtype=np.float32)
        for feature_name, x in feature_sets.items():
            probes[task][feature_name] = train_linear_probe(
                x=np.asarray(x, dtype=np.float32),
                y=y,
                split=split,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )

    retrieval_features = {
        key: feature_sets[key]
        for key in ["vq", "peak_stats", "formula_peak_stats", "formula_vq", "formula_peak_stats_vq"]
        if key in feature_sets
    }
    same_formula = {
        name: same_formula_retrieval(
            x=np.asarray(x, dtype=np.float32),
            root_formula=target_bundle["root_formula"],
            loss_y=np.asarray(target_bundle["targets"]["neutral_loss"]["y"], dtype=np.float32),
            fragment_y=np.asarray(target_bundle["targets"]["fragment_formula"]["y"], dtype=np.float32),
            max_queries=args.retrieval_max_queries,
            seed=args.seed,
        )
        for name, x in retrieval_features.items()
    }

    results = {
        "features": str(args.features),
        "index": str(args.index),
        "num_spectra": int(len(names)),
        "feature_dim": int(vq_x.shape[1]),
        "device": str(device),
        "rows_seen": int(target_bundle["rows_seen"]),
        "missing_root_labels": int(target_bundle["missing_root_labels"]),
        "split_sizes": {key: int(len(value)) for key, value in split.items()},
        "target_labels": {task: list(payload["labels"]) for task, payload in target_bundle["targets"].items()},
        "probes": probes,
        "same_formula_retrieval": same_formula,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(results, out_dir / "probe_report.md")
    print(json.dumps({k: results[k] for k in ["num_spectra", "feature_dim", "device", "split_sizes"]}, indent=2))
    print(f"Wrote {out_dir / 'probe_report.md'}")


if __name__ == "__main__":
    main()
