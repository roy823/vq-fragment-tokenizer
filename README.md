# VQ Fragment Tokenizer Prototype

Standalone prototype for learning discrete VQ tokens from FRIGID/CANOPUS peak-formula spectra. It does not modify `FRIGID-main`; it only reads FRIGID data and writes local indexes, checkpoints, reports, and exported features.

## What Is Implemented

- `FragmentSpectrumUnit` JSONL index from FRIGID peak-formula JSONs.
- Three independent random-init VQ codebooks:
  - `peak-code` for local peak features.
  - `fragment-code` for fragment/root formula context.
  - `event-code` for root-to-fragment pseudo-events and neutral losses.
- Lightweight reconstruction objective plus VQ commitment and entropy anti-collapse bonus.
- Default codebook seeding and per-epoch refresh sample random encoder states from the dataset. This is unsupervised and rule-free; pass `--no-init-codebooks --no-refresh-codebooks-each-epoch` to use raw trainable random codebooks.
- Tokenizer report with code usage, grouped spectral reconstruction cosine, and code semantics summaries.
- Per-spectrum VQ histogram export for later FRIGID conditioning experiments.

The current data source does not expose full fragment DAG edges, so event-code training starts with pseudo-DAG `root -> fragment` edges. ICEBERG/MAGMa DAG edges can be added later without changing the model interface.

## Quick Start

From this folder:

```powershell
python scripts/vq_audit.py --frigid-root ..\FRIGID\FRIGID-main --max-aug-files 5000 --out data\canopus_hplus_peak_units.jsonl --summary-out data\audit_summary.json
python scripts/vq_train.py --index data\canopus_hplus_peak_units.jsonl --out-dir runs\canopus_vq_small --epochs 5 --max-records 200000
python scripts/vq_report.py --index data\canopus_hplus_peak_units.jsonl --checkpoint runs\canopus_vq_small\best_model.pt --out-dir runs\canopus_vq_small\report
python scripts/vq_export_codes.py --index data\canopus_hplus_peak_units.jsonl --checkpoint runs\canopus_vq_small\best_model.pt --out runs\canopus_vq_small\vq_code_histograms.npz
```

For a smoke run:

```powershell
python scripts/vq_audit.py --frigid-root ..\FRIGID\FRIGID-main --max-real-files 25 --max-aug-files 25 --max-peaks 50 --out data\smoke_units.jsonl
python scripts/vq_train.py --index data\smoke_units.jsonl --out-dir runs\smoke --epochs 1 --max-records 2000 --batch-size 128 --hidden-dim 64 --code-dim 16 --codebook-size 32
python scripts/vq_report.py --index data\smoke_units.jsonl --checkpoint runs\smoke\best_model.pt --out-dir runs\smoke\report --max-records 2000
```

## Representation Probe

After exporting per-spectrum VQ histograms, test whether the discrete tokens
preserve chemically useful information:

```powershell
python scripts/vq_probe.py `
  --features runs\canopus_vq_50k_k128\vq_code_histograms.npz `
  --index data\canopus_hplus_50k_units.jsonl `
  --out-dir runs\canopus_vq_50k_k128\probe `
  --epochs 80 `
  --batch-size 2048
```

The probe trains only linear heads, comparing:

- `vq`: exported VQ histogram.
- `peak_stats`: simple peak-count/mz/intensity/loss-mass statistics.
- `vq_plus_peak_stats`: concatenation of both.

Targets are weak labels derived from the JSONL index: root formula element
presence, frequent neutral losses, frequent fragment formulae, and coarse m/z
bin presence. The main question is whether `vq` beats `peak_stats`, and whether
`vq_plus_peak_stats` improves over both. If not, the current VQ codes are mostly
compressing simple peak statistics rather than learning a useful fragmentation
vocabulary.

## Formula-Conditioned SetVQ

This is the Step-1b spectrum-level route. It uses root formula as a legal
global condition, plus observed spectrum inputs: `mz`, `intensity`, collision
energy, observed loss mass derived from formula and m/z, adduct condition, and
instrument condition. Fragment formula, neutral-loss formula, peak-formula
assignment, DAG labels, and m/z error are not fed to the model.

```powershell
python scripts/setvq_train.py `
  --index data\canopus_hplus_50k_units.jsonl `
  --out-dir runs\canopus_setvq_formula_k256 `
  --epochs 10 `
  --batch-size 128 `
  --codebook-size 256 `
  --num-slots 16

python scripts/setvq_report.py `
  --index data\canopus_hplus_50k_units.jsonl `
  --checkpoint runs\canopus_setvq_formula_k256\best_model.pt `
  --out-dir runs\canopus_setvq_formula_k256\report

python scripts/setvq_export_codes.py `
  --index data\canopus_hplus_50k_units.jsonl `
  --checkpoint runs\canopus_setvq_formula_k256\best_model.pt `
  --out runs\canopus_setvq_formula_k256\setvq_codes.npz

python scripts/vq_probe.py `
  --features runs\canopus_setvq_formula_k256\setvq_codes.npz `
  --index data\canopus_hplus_50k_units.jsonl `
  --out-dir runs\canopus_setvq_formula_k256\probe
```

Pass `--no-formula-conditioned` to `setvq_train.py` to reproduce the earlier
observation-only ablation.

For a smoke run:

```powershell
python scripts/setvq_train.py --index data\smoke_units.jsonl --out-dir runs\smoke_setvq --epochs 1 --batch-size 8 --hidden-dim 32 --code-dim 8 --codebook-size 16 --num-slots 4 --encoder-layers 1 --decoder-layers 2 --max-peaks 32 --max-mz 300 --bin-width 1
python scripts/setvq_report.py --index data\smoke_units.jsonl --checkpoint runs\smoke_setvq\best_model.pt --out-dir runs\smoke_setvq\report --max-spectra 40 --batch-size 8
python scripts/setvq_export_codes.py --index data\smoke_units.jsonl --checkpoint runs\smoke_setvq\best_model.pt --out runs\smoke_setvq\setvq_codes.npz --batch-size 8
python scripts/vq_probe.py --features runs\smoke_setvq\setvq_codes.npz --index data\smoke_units.jsonl --out-dir runs\smoke_setvq\probe --epochs 1 --batch-size 16 --top-losses 16 --top-fragments 16 --max-spectra 40 --device cpu
```

`setvq_codes.npz` contains both `vq_sequence` with shape
`[num_spectra, num_slots]` for sequence-style decoder experiments and
`vq_histogram` with shape `[num_spectra, codebook_size]` for linear probes.

## Verification

```powershell
python -m pytest
```

## Outputs

- `data/*.jsonl`: per-peak fragment spectrum units.
- `runs/*/best_model.pt`: trained VQ tokenizer checkpoint.
- `runs/*/history.json`: epoch metrics.
- `runs/*/report/report.md`: human-readable tokenizer report.
- `runs/*/vq_code_histograms.npz`: per-spectrum features laid out as `peak_hist | fragment_hist | event_hist`.
- `runs/*/probe/probe_report.md`: linear-probe representation benchmark.
- `runs/*/setvq_codes.npz`: SetVQ slot-code sequences plus histogram features.
