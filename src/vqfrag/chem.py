"""Small chemistry helpers for formula- and mass-consistency checks.

The element order mirrors FRIGID/MIST's ``mist.utils.chem_utils`` order so
vectors can be compared directly with peak-formula JSON files.
"""

from __future__ import annotations

import re
from typing import Mapping

import numpy as np

FORMULA_PATTERN = re.compile(r"([A-Z][a-z]*)([0-9]*)")

VALID_ELEMENTS = [
    "C",
    "H",
    "As",
    "B",
    "Br",
    "Cl",
    "Co",
    "F",
    "Fe",
    "I",
    "K",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Se",
    "Si",
]

ELEMENT_TO_INDEX = {el: idx for idx, el in enumerate(VALID_ELEMENTS)}

MONO_MASSES = np.array(
    [
        12.00000000000,  # C
        1.00782503223,  # H
        74.92159457000,  # As
        11.00930536000,  # B
        78.91833760000,  # Br
        34.96885268200,  # Cl
        58.93319429000,  # Co
        18.99840316273,  # F
        55.93493633000,  # Fe
        126.90447190000,  # I
        38.96370648640,  # K
        14.00307400443,  # N
        22.98976928200,  # Na
        15.99491461957,  # O
        30.97376199842,  # P
        31.97207117440,  # S
        79.91652180000,  # Se
        27.97692653465,  # Si
    ],
    dtype=np.float64,
)

NORM_VEC = np.array([81, 158, 2, 1, 3, 10, 1, 17, 1, 6, 1, 19, 2, 34, 6, 6, 2, 6], dtype=np.float32)

ELECTRON_MASS = 0.00054858

ION_MASS_SHIFT = {
    "[M+H]+": MONO_MASSES[ELEMENT_TO_INDEX["H"]] - ELECTRON_MASS,
    "[M+Na]+": MONO_MASSES[ELEMENT_TO_INDEX["Na"]] - ELECTRON_MASS,
    "[M+K]+": MONO_MASSES[ELEMENT_TO_INDEX["K"]] - ELECTRON_MASS,
    "[M-H2O+H]+": -MONO_MASSES[ELEMENT_TO_INDEX["O"]] - MONO_MASSES[ELEMENT_TO_INDEX["H"]] - ELECTRON_MASS,
    "[M+H3N+H]+": MONO_MASSES[ELEMENT_TO_INDEX["N"]] + MONO_MASSES[ELEMENT_TO_INDEX["H"]] * 4 - ELECTRON_MASS,
    "[M]+": -ELECTRON_MASS,
    "[M-H4O2+H]+": -MONO_MASSES[ELEMENT_TO_INDEX["O"]] * 2 - MONO_MASSES[ELEMENT_TO_INDEX["H"]] * 3 - ELECTRON_MASS,
}

ION_ALIASES = {
    "M+H": "[M+H]+",
    "M+Na": "[M+Na]+",
    "M+K": "[M+K]+",
    "M+H-H2O": "[M-H2O+H]+",
    "M-H2O+H": "[M-H2O+H]+",
    "M+NH4": "[M+H3N+H]+",
    "[M+NH4]+": "[M+H3N+H]+",
    "M-2H2O+H": "[M-H4O2+H]+",
    "[M-2H2O+H]+": "[M-H4O2+H]+",
}


def normalize_ion(ion: str | None) -> str | None:
    if ion is None:
        return None
    return ION_ALIASES.get(ion, ion)


def parse_formula(formula: str | None) -> dict[str, int]:
    """Parse a molecular formula into element counts."""
    if not formula:
        return {}
    
    formula_clean = re.sub(r"[\+\-]\d*$", "", formula)

    counts: dict[str, int] = {}
    consumed = ""
    for element, raw_count in FORMULA_PATTERN.findall(formula_clean):
        if element not in ELEMENT_TO_INDEX:
            raise ValueError(f"Unsupported element in formula {formula!r}: {element}")
        count = 1 if raw_count == "" else int(raw_count)
        counts[element] = counts.get(element, 0) + count
        consumed += f"{element}{raw_count}"
    if consumed != formula_clean:
        raise ValueError(f"Could not parse formula completely: {formula!r}")
    return counts


def formula_to_vector(formula: str | Mapping[str, int] | None) -> np.ndarray:
    """Convert a formula string or mapping into a dense element vector."""
    counts = parse_formula(formula) if isinstance(formula, str) or formula is None else dict(formula)
    vec = np.zeros(len(VALID_ELEMENTS), dtype=np.float32)
    for element, count in counts.items():
        if element not in ELEMENT_TO_INDEX:
            raise ValueError(f"Unsupported element: {element}")
        vec[ELEMENT_TO_INDEX[element]] = float(count)
    return vec


def vector_to_formula(vec: np.ndarray, *, allow_zero: bool = False) -> str:
    """Convert an integer-like dense vector back to formula notation."""
    rounded = np.rint(vec).astype(int)
    if np.any(rounded < 0):
        raise ValueError(f"Cannot render formula with negative counts: {rounded.tolist()}")
    chunks = []
    for element, count in zip(VALID_ELEMENTS, rounded):
        if count == 0:
            continue
        chunks.append(element if count == 1 else f"{element}{count}")
    if not chunks and allow_zero:
        return ""
    return "".join(chunks)


def formula_mass(formula: str | np.ndarray | Mapping[str, int] | None) -> float:
    vec = formula if isinstance(formula, np.ndarray) else formula_to_vector(formula)
    return float(np.asarray(vec, dtype=np.float64).dot(MONO_MASSES))


def ion_mass_shift(ion: str | None) -> float:
    norm = normalize_ion(ion)
    if norm not in ION_MASS_SHIFT:
        raise ValueError(f"Unsupported ion/adduct: {ion!r}")
    return float(ION_MASS_SHIFT[norm])


def formula_subset(child: str | np.ndarray, parent: str | np.ndarray) -> bool:
    child_vec = child if isinstance(child, np.ndarray) else formula_to_vector(child)
    parent_vec = parent if isinstance(parent, np.ndarray) else formula_to_vector(parent)
    return bool(np.all(parent_vec - child_vec >= -1e-6))


def formula_difference(parent: str | np.ndarray, child: str | np.ndarray) -> np.ndarray:
    parent_vec = parent if isinstance(parent, np.ndarray) else formula_to_vector(parent)
    child_vec = child if isinstance(child, np.ndarray) else formula_to_vector(child)
    return np.asarray(parent_vec, dtype=np.float32) - np.asarray(child_vec, dtype=np.float32)


def mass_error_ppm(observed_mz: float, expected_mz: float) -> float:
    if expected_mz == 0:
        return float("nan")
    return float((observed_mz - expected_mz) / expected_mz * 1e6)
