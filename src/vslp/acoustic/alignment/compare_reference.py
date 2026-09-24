"""CLI for descriptive comparison with manual alignment and feature references."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .validation import compare_boundaries, compare_feature_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("word", "phone", "features"))
    parser.add_argument("mfa_csv", type=Path)
    parser.add_argument("manual_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--tolerance", type=float, action="append", default=[])
    args = parser.parse_args()
    first = pd.read_csv(args.mfa_csv)
    second = pd.read_csv(args.manual_csv)
    if args.mode == "features":
        report = compare_feature_sensitivity(first, second,
                                             tolerances=tuple(args.tolerance))
    else:
        report = compare_boundaries(first, second, token_type=args.mode,
                                    tolerances_sec=tuple(args.tolerance))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
