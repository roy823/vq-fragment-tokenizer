import json

from vqfrag.chem import formula_mass, ion_mass_shift
from vqfrag.data import (
    FragmentUnitDataset,
    SpectrumFeatureConfig,
    SpectrumSetDataset,
    collate_fragment_units,
    collate_spectrum_sets,
    iter_units_from_peakformula_file,
    read_spectrum_set_records,
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


def test_spectrum_set_dataset_is_observation_only(tmp_path):
    path = tmp_path / "units.jsonl"
    rows = [
        {
            "spectrum_id": "spec_a",
            "adduct": "[M+H]+",
            "collision_energy": 20.0,
            "instrument": "qtof",
            "mz": 100.2,
            "intensity": 0.4,
            "root_formula": "C2H6O",
            "fragment_formula": "CH4",
            "neutral_loss_formula": "CO",
            "mz_error_ppm": 1.0,
        },
        {
            "spectrum_id": "spec_a",
            "adduct": "[M+H]+",
            "collision_energy": 20.0,
            "instrument": "qtof",
            "mz": 55.0,
            "intensity": 1.0,
            "root_formula": "C2H6O",
            "fragment_formula": "C2H4",
            "neutral_loss_formula": "H2O",
            "mz_error_ppm": 2.0,
        },
        {
            "spectrum_id": "spec_b",
            "adduct": "[M+H]+",
            "collision_energy": None,
            "instrument": None,
            "mz": 42.0,
            "intensity": 0.8,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    records = read_spectrum_set_records(path)
    assert len(records) == 2
    assert records[0].spectrum_id == "spec_a"
    assert records[0].peaks[0] == (55.0, 1.0)

    config = SpectrumFeatureConfig(max_peaks=4, max_mz=200.0, bin_width=1.0)
    dataset = SpectrumSetDataset(path, feature_config=config)
    item = dataset[0]
    batch = collate_spectrum_sets([item])

    assert set(batch) == {
        "peak_x",
        "peak_mask",
        "cond_x",
        "target_presence",
        "target_intensity",
        "spectrum_id",
        "peaks",
    }
    assert batch["peak_x"].shape == (1, 4, 3)
    assert batch["peak_mask"].tolist() == [[True, True, False, False]]
    assert batch["target_presence"].shape == (1, 201)
    assert batch["target_presence"][0, 55] == 1.0
    assert batch["target_presence"][0, 100] == 1.0
