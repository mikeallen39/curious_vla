#!/usr/bin/env python3
"""Compute per-scenario group-level statistics from the merged scorer Parquet.

Outputs a CSV with columns: token, group_size,
pdms_mean, pdms_std, pdms_range, pdms_scaled_mean, pdms_scaled_std, pdms_scaled_range.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.dataset as ds


def main(parquet_path: str, output_csv: Optional[str] = None) -> str:
    out_root = os.path.dirname(parquet_path)
    parquet_stem = Path(parquet_path).stem
    tmp_csv_path = os.path.join(out_root, f"{parquet_stem}.tmp_filtered.csv")
    if output_csv is None:
        output_csv = os.path.join(out_root, f"{parquet_stem}.csv")

    dataset = ds.dataset(parquet_path, format="parquet")

    # Streaming accumulators keyed by token
    cnt: dict[str, int] = {}
    sum_pdms: dict[str, float] = {}
    sumsq_pdms: dict[str, float] = {}
    min_pdms: dict[str, float] = {}
    max_pdms: dict[str, float] = {}
    sum_pdms_scaled: dict[str, float] = {}
    sumsq_pdms_scaled: dict[str, float] = {}
    min_pdms_scaled: dict[str, float] = {}
    max_pdms_scaled: dict[str, float] = {}

    with open(tmp_csv_path, "w") as f_out:
        header_written = False
        for batch in dataset.to_batches():
            df = batch.to_pandas()
            if "pdms_scaled" in df.columns and "pdms" in df.columns:
                df = df[["token", "pdms", "pdms_scaled"]]
            else:
                df = df[["token", "score"]]
                df["pdms"] = df["score"]
                df["pdms_scaled"] = df["score"]

            df = df[df["token"].astype(str) != "average_all_frames"].copy()

            if not header_written:
                df.to_csv(f_out, index=False, header=True)
                header_written = True
            else:
                df.to_csv(f_out, index=False, header=False)

            df["pdms"] = pd.to_numeric(df["pdms"], errors="coerce")
            df["pdms_scaled"] = pd.to_numeric(df["pdms_scaled"], errors="coerce")
            df = df.dropna(subset=["pdms", "pdms_scaled", "token"])
            if df.empty:
                continue

            df["pdms_sq"] = df["pdms"] * df["pdms"]
            df["pdms_scaled_sq"] = df["pdms_scaled"] * df["pdms_scaled"]

            g = df.groupby("token", sort=False).agg(
                n=("pdms", "size"),
                pdms_sum=("pdms", "sum"),
                pdms_sumsq=("pdms_sq", "sum"),
                pdms_min=("pdms", "min"),
                pdms_max=("pdms", "max"),
                pdms_scaled_sum=("pdms_scaled", "sum"),
                pdms_scaled_sumsq=("pdms_scaled_sq", "sum"),
                pdms_scaled_min=("pdms_scaled", "min"),
                pdms_scaled_max=("pdms_scaled", "max"),
            )

            for token, row in g.iterrows():
                token = str(token)
                n = int(row["n"])
                if n <= 0:
                    continue

                if token not in cnt:
                    cnt[token] = 0
                    sum_pdms[token] = 0.0
                    sumsq_pdms[token] = 0.0
                    min_pdms[token] = float(row["pdms_min"])
                    max_pdms[token] = float(row["pdms_max"])
                    sum_pdms_scaled[token] = 0.0
                    sumsq_pdms_scaled[token] = 0.0
                    min_pdms_scaled[token] = float(row["pdms_scaled_min"])
                    max_pdms_scaled[token] = float(row["pdms_scaled_max"])

                cnt[token] += n
                sum_pdms[token] += float(row["pdms_sum"])
                sumsq_pdms[token] += float(row["pdms_sumsq"])
                min_pdms[token] = min(min_pdms[token], float(row["pdms_min"]))
                max_pdms[token] = max(max_pdms[token], float(row["pdms_max"]))
                sum_pdms_scaled[token] += float(row["pdms_scaled_sum"])
                sumsq_pdms_scaled[token] += float(row["pdms_scaled_sumsq"])
                min_pdms_scaled[token] = min(min_pdms_scaled[token], float(row["pdms_scaled_min"]))
                max_pdms_scaled[token] = max(max_pdms_scaled[token], float(row["pdms_scaled_max"]))

    print(f"Intermediate CSV written: {tmp_csv_path}")

    def _std(sum_x: float, sumsq_x: float, n: int) -> float:
        if n <= 1:
            return float("nan")
        numerator = max(sumsq_x - (sum_x * sum_x) / n, 0.0)
        return math.sqrt(numerator / (n - 1))

    rows = []
    for token in cnt:
        n = cnt[token]
        rows.append({
            "token": token,
            "group_size": n,
            "pdms_mean": sum_pdms[token] / n,
            "pdms_std": _std(sum_pdms[token], sumsq_pdms[token], n),
            "pdms_range": max_pdms[token] - min_pdms[token],
            "pdms_scaled_mean": sum_pdms_scaled[token] / n,
            "pdms_scaled_std": _std(sum_pdms_scaled[token], sumsq_pdms_scaled[token], n),
            "pdms_scaled_range": max_pdms_scaled[token] - min_pdms_scaled[token],
        })

    grouped = pd.DataFrame(rows)
    grouped.to_csv(output_csv, index=False)
    print(f"Stats written: {output_csv}  ({len(grouped)} groups, {sum(cnt.values())} total rows)")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute group-level PDMS statistics from scorer Parquet")
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()
    main(args.parquet_path, args.output_csv)
