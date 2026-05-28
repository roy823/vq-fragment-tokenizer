#!/usr/bin/env python
"""Build a JSONL FragmentSpectrumUnit index from a FRIGID checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqfrag.data import build_jsonl_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frigid-root", default="../FRIGID/FRIGID-main", help="Path to FRIGID-main")
    parser.add_argument("--out", default="data/canopus_hplus_peak_units.jsonl", help="Output JSONL path")
    parser.add_argument("--summary-out", default=None, help="Optional summary JSON path")
    parser.add_argument("--include-real", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-real-files", type=int, default=None)
    parser.add_argument("--max-aug-files", type=int, default=5000)
    parser.add_argument("--max-peaks", type=int, default=100)
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--require-hplus", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_jsonl_index(
        frigid_root=args.frigid_root,
        out_path=args.out,
        include_real=args.include_real,
        include_aug=args.include_aug,
        max_real_files=args.max_real_files,
        max_aug_files=args.max_aug_files,
        require_hplus=args.require_hplus,
        min_intensity=args.min_intensity,
        max_peaks=args.max_peaks,
    )
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    print(summary_json)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
