import numpy as np

from vqfrag.chem import (
    formula_difference,
    formula_mass,
    formula_subset,
    formula_to_vector,
    ion_mass_shift,
    mass_error_ppm,
    vector_to_formula,
)


def test_formula_roundtrip_and_subset():
    root = formula_to_vector("C6H12O6")
    child = formula_to_vector("C3H6O3")

    assert formula_subset(child, root)
    assert vector_to_formula(formula_difference(root, child)) == "C3H6O3"


def test_formula_mass_and_ppm_error_are_stable():
    expected = formula_mass("C2H6O") + ion_mass_shift("[M+H]+")
    assert expected > 47.0
    assert abs(mass_error_ppm(expected * (1 + 5e-6), expected) - 5.0) < 1e-6


def test_formula_vector_uses_mist_order():
    vec = formula_to_vector("CHNOPS")
    assert np.count_nonzero(vec) == 6
    assert vec[0] == 1  # C
    assert vec[1] == 1  # H
