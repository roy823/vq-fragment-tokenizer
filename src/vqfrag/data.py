"""Data loading and featurization for the standalone VQ fragment tokenizer."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .chem import (
    ION_MASS_SHIFT,
    NORM_VEC,
    formula_difference,
    formula_mass,
    formula_subset,
    formula_to_vector,
    ion_mass_shift,
    mass_error_ppm,
    normalize_ion,
    vector_to_formula,
)

ION_ORDER = list(ION_MASS_SHIFT.keys())
ION_TO_INDEX = {ion: idx for idx, ion in enumerate(ION_ORDER)}

INSTRUMENT_TYPES = ["unknown", "qtof", "orbitrap", "iontrap", "fticr"]


@dataclass(frozen=True)
class FragmentSpectrumUnit:
    """One peak-level fragment/spectrum training unit.

    FRIGID peak-formula files do not store full parent-child DAG edges. For the
    first tokenizer prototype each peak is represented as a pseudo-edge from the
    root molecule to one annotated fragment. ICEBERG fragment IDs can be slotted
    into the same object later.
    """

    spectrum_id: str
    source: str
    root_formula: str
    adduct: str
    collision_energy: float | None
    instrument: str | None
    fragment_id: str
    fragment_formula: str
    mz: float
    intensity: float
    neutral_loss_formula: str
    neutral_loss_mass: float
    mz_error_ppm: float | None
    peak_set: tuple[tuple[float, float], ...]
    parent_ids: tuple[str, ...] = ("root",)
    child_ids: tuple[str, ...] = ()
    pseudo_edge: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "FragmentSpectrumUnit":
        peak_set = tuple(tuple(float(v) for v in pair) for pair in data.get("peak_set", ()))
        parent_ids = tuple(data.get("parent_ids", ("root",)))
        child_ids = tuple(data.get("child_ids", ()))
        return cls(
            spectrum_id=str(data["spectrum_id"]),
            source=str(data["source"]),
            root_formula=str(data["root_formula"]),
            adduct=str(data["adduct"]),
            collision_energy=None if data.get("collision_energy") is None else float(data["collision_energy"]),
            instrument=data.get("instrument"),
            fragment_id=str(data["fragment_id"]),
            fragment_formula=str(data["fragment_formula"]),
            mz=float(data["mz"]),
            intensity=float(data["intensity"]),
            neutral_loss_formula=str(data.get("neutral_loss_formula", "")),
            neutral_loss_mass=float(data.get("neutral_loss_mass", 0.0)),
            mz_error_ppm=None if data.get("mz_error_ppm") is None else float(data["mz_error_ppm"]),
            peak_set=peak_set,
            parent_ids=parent_ids,
            child_ids=child_ids,
            pseudo_edge=bool(data.get("pseudo_edge", True)),
        )


def _as_list(value: object, length: int, default: object) -> list:
    if value is None:
        return [default] * length
    if isinstance(value, list):
        return value
    return [value] * length


def _instrument_bucket(instrument: str | None) -> str:
    if not instrument:
        return "unknown"
    lowered = instrument.lower()
    if "q-tof" in lowered or "qtof" in lowered or "tof" in lowered:
        return "qtof"
    if "orbitrap" in lowered:
        return "orbitrap"
    if "ion trap" in lowered or "iontrap" in lowered:
        return "iontrap"
    if "fticr" in lowered:
        return "fticr"
    return "unknown"


def read_label_map(labels_file: str | Path | None) -> dict[str, dict]:
    """Read FRIGID/CANOPUS labels.tsv into a spectrum-id metadata map."""
    if labels_file is None:
        return {}
    path = Path(labels_file)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows = {}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            spec = row.get("spec")
            if spec:
                rows[spec] = row
        return rows


def iter_units_from_peakformula_file(
    json_file: str | Path,
    *,
    source: str,
    label_map: dict[str, dict] | None = None,
    require_hplus: bool = True,
    min_intensity: float = 0.0,
    max_peaks: int | None = None,
) -> Iterator[FragmentSpectrumUnit]:
    """Yield peak-level units from one FRIGID peak-formula JSON file."""
    path = Path(json_file)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    spectrum_id = path.stem
    labels = (label_map or {}).get(spectrum_id, {})
    root_formula = data.get("cand_form") or labels.get("formula")
    adduct = normalize_ion(data.get("cand_ion") or labels.get("ionization"))
    instrument = data.get("instrument") or labels.get("instrument") or None

    if not root_formula or not adduct:
        return
    if require_hplus and adduct != "[M+H]+":
        return

    output_tbl = data.get("output_tbl") or {}
    mzs = output_tbl.get("mz") or output_tbl.get("mono_mass") or []
    intensities = output_tbl.get("ms2_inten") or output_tbl.get("rel_inten") or []
    if not mzs or not intensities:
        return
    n = min(len(mzs), len(intensities))

    formulas = _as_list(output_tbl.get("formula"), n, "")
    ions = _as_list(output_tbl.get("ions"), n, adduct)

    order = list(range(n))
    if max_peaks is not None and n > max_peaks:
        order = sorted(order, key=lambda i: float(intensities[i]), reverse=True)[:max_peaks]

    root_mass = formula_mass(root_formula)
    root_vec = formula_to_vector(root_formula)
    for local_rank, idx in enumerate(order):
        intensity = float(intensities[idx])
        if intensity < min_intensity:
            continue
        frag_formula = formulas[idx]
        if not frag_formula:
            continue
        peak_ion = normalize_ion(ions[idx]) or adduct
        try:
            expected_mz = formula_mass(frag_formula) + ion_mass_shift(peak_ion)
            ppm = mass_error_ppm(float(mzs[idx]), expected_mz)
            if formula_subset(frag_formula, root_vec):
                loss_vec = formula_difference(root_vec, frag_formula)
                loss_formula = vector_to_formula(loss_vec, allow_zero=True)
                loss_mass = root_mass - formula_mass(frag_formula)
            else:
                loss_formula = ""
                loss_mass = float("nan")
        except ValueError:
            continue

        yield FragmentSpectrumUnit(
            spectrum_id=spectrum_id,
            source=source,
            root_formula=root_formula,
            adduct=adduct,
            collision_energy=None,
            instrument=instrument,
            fragment_id=f"{spectrum_id}:{local_rank}",
            fragment_formula=frag_formula,
            mz=float(mzs[idx]),
            intensity=intensity,
            neutral_loss_formula=loss_formula,
            neutral_loss_mass=float(loss_mass) if np.isfinite(loss_mass) else 0.0,
            mz_error_ppm=ppm,
            peak_set=((float(mzs[idx]), intensity),),
        )


def discover_frigid_peakformula_files(
    frigid_root: str | Path,
    *,
    include_real: bool = True,
    include_aug: bool = True,
    max_real_files: int | None = None,
    max_aug_files: int | None = None,
) -> list[tuple[Path, str]]:
    """Discover peak-formula JSONs under a FRIGID checkout."""
    root = Path(frigid_root)
    found: list[tuple[Path, str]] = []

    if include_real:
        real_dir = root / "data" / "canopus" / "subformulae" / "subformulae_default"
        real_files = sorted(real_dir.glob("*.json"))
        if max_real_files is not None:
            real_files = real_files[:max_real_files]
        found.extend((path, "canopus") for path in real_files)

    if include_aug:
        aug_root = root / "data" / "canopus" / "aug_iceberg_canopus_train"
        aug_files = sorted(aug_root.glob("*/subforms/*.json"))
        if max_aug_files is not None:
            aug_files = aug_files[:max_aug_files]
        found.extend((path, "iceberg_aug") for path in aug_files)

    return found


class AuditAccumulator:
    """Streaming summary for JSONL index construction."""

    def __init__(self) -> None:
        self.units = 0
        self.spectra: set[str] = set()
        self.sources = Counter()
        self.adducts = Counter()
        self.instruments = Counter()
        self.invalid_formula_subset = 0
        self.large_mz_error = 0
        self.ppm_abs_values: list[float] = []

    def add(self, unit: FragmentSpectrumUnit) -> None:
        self.units += 1
        self.spectra.add(unit.spectrum_id)
        self.sources[unit.source] += 1
        self.adducts[unit.adduct] += 1
        self.instruments[_instrument_bucket(unit.instrument)] += 1
        if not formula_subset(unit.fragment_formula, unit.root_formula):
            self.invalid_formula_subset += 1
        if unit.mz_error_ppm is not None and np.isfinite(unit.mz_error_ppm):
            abs_ppm = abs(float(unit.mz_error_ppm))
            self.ppm_abs_values.append(abs_ppm)
            if abs_ppm > 20:
                self.large_mz_error += 1

    def to_dict(self) -> dict:
        ppm = np.asarray(self.ppm_abs_values, dtype=np.float64)
        return {
            "units": self.units,
            "spectra": len(self.spectra),
            "sources": dict(self.sources),
            "adducts": dict(self.adducts),
            "instruments": dict(self.instruments),
            "invalid_formula_subset": self.invalid_formula_subset,
            "large_mz_error_gt_20ppm": self.large_mz_error,
            "mz_error_abs_ppm_mean": float(ppm.mean()) if ppm.size else None,
            "mz_error_abs_ppm_p95": float(np.percentile(ppm, 95)) if ppm.size else None,
            "mz_error_abs_ppm_max": float(ppm.max()) if ppm.size else None,
        }


def build_jsonl_index(
    *,
    frigid_root: str | Path,
    out_path: str | Path,
    include_real: bool = True,
    include_aug: bool = True,
    max_real_files: int | None = None,
    max_aug_files: int | None = None,
    require_hplus: bool = True,
    min_intensity: float = 0.0,
    max_peaks: int | None = None,
) -> dict:
    """Create a standalone JSONL index from FRIGID peak-formula data."""
    root = Path(frigid_root)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    label_map = read_label_map(root / "data" / "canopus" / "labels.tsv")
    files = discover_frigid_peakformula_files(
        root,
        include_real=include_real,
        include_aug=include_aug,
        max_real_files=max_real_files,
        max_aug_files=max_aug_files,
    )
    audit = AuditAccumulator()
    with out.open("w", encoding="utf-8") as handle:
        for path, source in files:
            for unit in iter_units_from_peakformula_file(
                path,
                source=source,
                label_map=label_map,
                require_hplus=require_hplus,
                min_intensity=min_intensity,
                max_peaks=max_peaks,
            ):
                audit.add(unit)
                handle.write(unit.to_json() + "\n")
    summary = audit.to_dict()
    summary["input_files"] = len(files)
    summary["index_path"] = str(out)
    return summary


def read_jsonl_units(path: str | Path, *, max_records: int | None = None) -> list[FragmentSpectrumUnit]:
    units: list[FragmentSpectrumUnit] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            units.append(FragmentSpectrumUnit.from_dict(json.loads(line)))
            if max_records is not None and len(units) >= max_records:
                break
    return units


@dataclass(frozen=True)
class FeatureConfig:
    mz_scale: float = 1500.0
    mass_scale: float = 1500.0
    ce_scale: float = 100.0
    ppm_scale: float = 50.0


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        out[index] = 1.0
    return out


def unit_to_features(unit: FragmentSpectrumUnit, config: FeatureConfig = FeatureConfig()) -> dict[str, np.ndarray]:
    """Map one unit into model inputs.

    Adduct and instrument are condition features only. They are not fed into any
    encoder before VQ assignment, which keeps codebooks from becoming instrument
    or adduct labels.
    """
    root_vec = formula_to_vector(unit.root_formula).astype(np.float32)
    frag_vec = formula_to_vector(unit.fragment_formula).astype(np.float32)
    loss_vec = np.maximum(root_vec - frag_vec, 0).astype(np.float32)
    root_norm = root_vec / NORM_VEC
    frag_norm = frag_vec / NORM_VEC
    loss_norm = loss_vec / NORM_VEC

    ce_norm = 0.0 if unit.collision_energy is None else float(unit.collision_energy) / config.ce_scale
    ppm = 0.0 if unit.mz_error_ppm is None or not np.isfinite(unit.mz_error_ppm) else unit.mz_error_ppm
    peak_x = np.array(
        [
            unit.mz / config.mz_scale,
            unit.intensity,
            ce_norm,
            unit.neutral_loss_mass / config.mass_scale,
            np.clip(ppm / config.ppm_scale, -10.0, 10.0),
        ],
        dtype=np.float32,
    )
    fragment_x = np.concatenate([frag_norm, root_norm]).astype(np.float32)
    event_x = np.concatenate(
        [
            loss_norm,
            np.array([unit.neutral_loss_mass / config.mass_scale, ce_norm], dtype=np.float32),
        ]
    ).astype(np.float32)

    adduct_idx = ION_TO_INDEX.get(unit.adduct, -1)
    inst_idx = INSTRUMENT_TYPES.index(_instrument_bucket(unit.instrument))
    cond_x = np.concatenate(
        [
            _one_hot(adduct_idx, len(ION_ORDER)),
            _one_hot(inst_idx, len(INSTRUMENT_TYPES)),
        ]
    ).astype(np.float32)

    return {
        "peak_x": peak_x,
        "fragment_x": fragment_x,
        "event_x": event_x,
        "cond_x": cond_x,
    }


class FragmentUnitDataset(Dataset):
    """Torch dataset over JSONL-indexed fragment spectrum units."""

    def __init__(
        self,
        units_or_path: Sequence[FragmentSpectrumUnit] | str | Path,
        *,
        max_records: int | None = None,
        feature_config: FeatureConfig = FeatureConfig(),
    ) -> None:
        if isinstance(units_or_path, (str, Path)):
            self.units = read_jsonl_units(units_or_path, max_records=max_records)
        else:
            self.units = list(units_or_path)
            if max_records is not None:
                self.units = self.units[:max_records]
        self.feature_config = feature_config

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, idx: int) -> dict:
        unit = self.units[idx]
        feats = unit_to_features(unit, self.feature_config)
        return {
            "peak_x": torch.from_numpy(feats["peak_x"]),
            "fragment_x": torch.from_numpy(feats["fragment_x"]),
            "event_x": torch.from_numpy(feats["event_x"]),
            "cond_x": torch.from_numpy(feats["cond_x"]),
            "spectrum_id": unit.spectrum_id,
            "fragment_formula": unit.fragment_formula,
            "neutral_loss_formula": unit.neutral_loss_formula,
            "mz": float(unit.mz),
            "intensity": float(unit.intensity),
        }


def collate_fragment_units(batch: list[dict]) -> dict:
    tensor_keys = ["peak_x", "fragment_x", "event_x", "cond_x"]
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    for key in ["spectrum_id", "fragment_formula", "neutral_loss_formula"]:
        out[key] = [item[key] for item in batch]
    for key in ["mz", "intensity"]:
        out[key] = torch.tensor([item[key] for item in batch], dtype=torch.float32)
    return out


def group_units_by_spectrum(units: Iterable[FragmentSpectrumUnit]) -> dict[str, list[FragmentSpectrumUnit]]:
    grouped: dict[str, list[FragmentSpectrumUnit]] = defaultdict(list)
    for unit in units:
        grouped[unit.spectrum_id].append(unit)
    return grouped
