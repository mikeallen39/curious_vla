#!/usr/bin/env python3
"""ADAS single-round pipeline: scorer CSVs -> token list for training.

Runs three steps:
  1. Merge scorer CSVs into a single Parquet.
  2. Compute per-scenario group-level statistics.
  3. Filter dynamic (high-variance) samples and output a ``.txt`` token list.

The output ``.txt`` file is consumed directly by the verl dataloader via
``data.token_filter_file``, eliminating the need to pre-build a filtered
Parquet dataset.

ADAS outer loop (multi-round) is driven manually:
  Round N: run inference -> run this pipeline -> train with token_filter_file
  Round N+1: use new checkpoint, re-run inference, re-run pipeline, ...
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running from scripts/adas/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merge_scorer_csv import main as merge_csv
from compute_stats import main as compute_stats
from filter_dynamic import main as filter_dynamic


def run_pipeline(
    infer_folder: str,
    p: float = 0.1,
    conf: float = 0.1,
    n_rollout: int = 8,
    group_size: int = 32,
    std_threshold: float = 0.01,
    include_glob: str | None = None,
    exclude_glob: str | None = None,
) -> str:
    """Run one round of ADAS filtering.

    Returns:
        Path to the output ``.txt`` token list file.
    """
    exp_name = os.path.basename(os.path.normpath(infer_folder))
    print("=" * 60)
    print(f"ADAS Pipeline - {exp_name}")
    print("=" * 60)

    # Step 1: Merge scorer CSVs into Parquet
    print("\n[Step 1] Merging scorer CSVs into Parquet...")
    parquet_path = merge_csv(
        infer_folder,
        output_parquet=None,
        include_glob=include_glob,
        exclude_glob=exclude_glob,
    )

    # Step 2: Compute group-level statistics
    print("\n[Step 2] Computing group-level statistics...")
    stats_csv = compute_stats(parquet_path)

    # Step 3: Filter dynamic samples
    print("\n[Step 3] Filtering dynamic samples...")
    output_csv, txt_path, final_count = filter_dynamic(
        stats_csv, None, p, n_rollout, group_size,
        std_threshold, conf,
    )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Token list: {txt_path}")
    print(f"  Sample count: {final_count}")
    print(f"\nTo train with these samples:")
    print(f"  data.token_filter_file={txt_path}")
    print("=" * 60)

    return txt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ADAS pipeline: scorer CSVs -> token list for training"
    )
    parser.add_argument("--infer_folder", type=str, required=True,
                        help="Directory containing scorer CSV output from parallel inference")
    parser.add_argument("-p", type=float, default=0.1,
                        help="Diversity threshold (lower = stricter, keeps more mixed samples)")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Confidence threshold (max allowed prediction error)")
    parser.add_argument("--n_rollout", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--std_threshold", type=float, default=0.01)
    parser.add_argument("--include_glob", type=str, default=None)
    parser.add_argument("--exclude_glob", type=str, default=None)

    args = parser.parse_args()
    run_pipeline(
        infer_folder=args.infer_folder,
        p=args.p,
        conf=args.conf,
        n_rollout=args.n_rollout,
        group_size=args.group_size,
        std_threshold=args.std_threshold,
        include_glob=args.include_glob,
        exclude_glob=args.exclude_glob,
    )
