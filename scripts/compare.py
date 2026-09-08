from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from bacl.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison of two per-case evaluation CSV files")
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", default="results/comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proposed = pd.read_csv(args.proposed)
    baseline = pd.read_csv(args.baseline)
    paired = proposed.merge(baseline, on="id", suffixes=("_proposed", "_baseline"), validate="one_to_one")
    results = {"n": int(len(paired)), "metrics": {}}
    for metric in ("dice", "iou"):
        first = paired[f"{metric}_proposed"].to_numpy()
        second = paired[f"{metric}_baseline"].to_numpy()
        difference = first - second
        t_result = stats.ttest_rel(first, second)
        if len(difference) > 1 and difference.std(ddof=1) > 0:
            standardized = (difference - difference.mean()) / difference.std(ddof=1)
            ks_result = stats.kstest(standardized, "norm")
            ks = {"statistic": float(ks_result.statistic), "pvalue": float(ks_result.pvalue)}
        else:
            ks = {"statistic": None, "pvalue": None}
        results["metrics"][metric] = {
            "mean_difference": float(difference.mean()),
            "paired_t": {"statistic": float(t_result.statistic), "pvalue": float(t_result.pvalue)},
            "ks_normality_of_standardized_differences": ks,
        }
    save_json(results, Path(args.output))
    print(results)


if __name__ == "__main__":
    main()
