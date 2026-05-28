import json

from vqfrag.chem import formula_mass, ion_mass_shift
from vqfrag.data import (
    FragmentUnitDataset,
    collate_fragment_units,
    iter_units_from_peakformula_file,
)


def test_iter_units_from_peakformula_file(tmp_path):
    ethanol_mz = formula_mass("C2H6O") + ion_mass_shift("[M+H]+")
    ethene_mz = formula_mass("C2H4") + ion_mass_shift("[M+H]+")
    path = tmp_path / "spec_0.json"
    path.write_text(
        json.dumps(
            {
                "cand_form": "C2H6O",
                "cand_ion": "[M+H]+",
                "output_tbl": {
                    "mz": [ethanol_mz, ethene_mz],
                    "ms2_inten": [1.0, 0.5],
                    "formula": ["C2H6O", "C2H4"],
                    "ions": ["[M+H]+", "[M+H]+"],
                },
            }
        ),
        encoding="utf-8",
    )

    units = list(iter_units_from_peakformula_file(path, source="unit_test"))

    assert len(units) == 2
    assert units[0].neutral_loss_formula == ""
    assert units[1].neutral_loss_formula == "H2O"
    assert abs(units[0].mz_error_ppm) < 1e-6


def test_fragment_unit_dataset_and_collate(tmp_path):
    mz = formula_mass("CH4") + ion_mass_shift("[M+H]+")
    path = tmp_path / "spec_0.json"
    path.write_text(
        json.dumps(
            {
                "cand_form": "CH4",
                "cand_ion": "[M+H]+",
                "output_tbl": {
                    "mz": [mz],
                    "ms2_inten": [1.0],
                    "formula": ["CH4"],
                    "ions": ["[M+H]+"],
                },
            }
        ),
        encoding="utf-8",
    )
    units = list(iter_units_from_peakformula_file(path, source="unit_test"))
    dataset = FragmentUnitDataset(units)
    batch = collate_fragment_units([dataset[0]])

    assert batch["peak_x"].shape == (1, 5)
    assert batch["fragment_x"].shape == (1, 36)
    assert batch["event_x"].shape == (1, 20)
    assert batch["cond_x"].shape == (1, 12)
