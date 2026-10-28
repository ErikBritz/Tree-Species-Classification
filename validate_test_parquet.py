from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Parquet test files and report missing values."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {args.data_dir}")

    for path in files:
        frame = pd.read_parquet(path)
        print(f"\n{path.name}: {len(frame):,} rows, {len(frame.columns)} columns")
        print("Columns:", ", ".join(map(str, frame.columns)))
        missing = frame.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("Missing values: none")
        else:
            print("Columns with missing values:")
            print(missing.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
