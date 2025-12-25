#!/usr/bin/env python3
"""Merge all CSV files in a directory into a single CSV."""

import csv
import sys
from pathlib import Path


def merge_csvs(input_dir, output_file):
    """Merge all CSV files from input_dir into a single output file."""
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        sys.exit(1)

    csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files to merge")

    all_rows = []
    fieldnames = None

    # Read all CSVs
    for csv_file in sorted(csv_files):
        print(f"  Reading {csv_file.name}...")
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    # Write merged CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓ Merged {len(all_rows)} rows into {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_csvs.py <input_directory> [output_file]")
        print(
            "Example: python merge_csvs.py fp4_legacy_results fp4_legacy_results/merged_results.csv"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = (
        sys.argv[2] if len(sys.argv) > 2 else f"{input_dir}/merged_results.csv"
    )

    merge_csvs(input_dir, output_file)
