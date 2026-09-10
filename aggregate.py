"""
Aggregate the per-round rows written by run_experiment.py into the comparisons
that actually answer the ablation questions.

    python3 aggregate.py results/stage2.csv
    python3 aggregate.py results/stage1.csv --metric Spearman
"""

import argparse

import numpy as np
import pandas as pd


LABELS = {
    "C0":  "original (spectral)",
    "C1":  "reference (kmeans, unfixed)",
    "C2":  "random acquisition",
    "C3":  "diversity fixed, kmer",
    "C4":  "diversity fixed, hamming",
    "C5":  "diversity fixed, embedding",
    "C6":  "diversity stage removed",
    "C7":  "new-query weight on",
    "C8":  "no clustering",
    "C9":  "rank normalization",
    "C10": "pure uncertainty",
    "C11": "pure coverage",
    "C12": "exploitation term",
    "C13": "all repairs, kmer",
    "C14": "all repairs, hamming",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--metric", default="R2")
    p.add_argument("--reference", default="C1")
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    m = a.metric
    last = int(df["round"].max())

    print(f"file: {a.csv}   metric: {m}   rounds 0..{last}")
    print(f"conditions: {df['condition'].nunique()}   seeds: {sorted(df['seed'].unique())}\n")

    # --- determinism check: identical (condition, seed) pairs run more than once
    if "tag" in df.columns:
        z = df[df["tag"].astype(str).str.startswith("S1-Z")]
        if len(z):
            piv = z.pivot_table(index="round", columns="tag", values=m)
            spread = (piv.max(axis=1) - piv.min(axis=1)).abs()
            print("--- determinism (S1-Z) ---")
            print(piv.to_string())
            print(f"max abs difference across rounds: {spread.max():.3e}")
            print(f"bit-identical: {bool(spread.max() == 0)}\n")

    # --- endpoint table
    end = df[df["round"] == last]
    g = end.groupby("condition")[m].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))

    ref = g.loc[a.reference, "mean"] if a.reference in g.index else np.nan

    print(f"--- final round ({last}) ---")
    print(f"{'cond':6s} {'condition':30s} {'mean':>9s} {'se':>8s} {'n':>3s} {'vs '+a.reference:>10s}")
    for c in sorted(g.index, key=lambda x: (len(x), x)):
        d = g.loc[c, "mean"] - ref
        delta = "  ref" if c == a.reference else f"{d:+10.4f}"
        print(f"{c:6s} {LABELS.get(c,''):30s} {g.loc[c,'mean']:9.4f} {g.loc[c,'se']:8.4f} "
              f"{int(g.loc[c,'count']):3d} {delta:>10s}")

    # --- budget efficiency: rounds needed to match the random baseline's endpoint
    if "C2" in g.index:
        target = g.loc["C2", "mean"]
        print(f"\n--- rounds to reach the random baseline endpoint ({m} = {target:.4f}) ---")
        curves = df.groupby(["condition", "round"])[m].mean().reset_index()
        for c in sorted(curves["condition"].unique(), key=lambda x: (len(x), x)):
            sub = curves[curves["condition"] == c].sort_values("round")
            hit = sub[sub[m] >= target]
            r = int(hit["round"].iloc[0]) if len(hit) else None
            print(f"  {c:6s} {LABELS.get(c,''):30s} " + (f"round {r}" if r is not None else "never"))

    # --- the free correctness assertion
    if {"C1", "C6"} <= set(g.index):
        d = abs(g.loc["C1", "mean"] - g.loc["C6", "mean"])
        pooled = np.hypot(g.loc["C1", "se"], g.loc["C6", "se"])
        print(f"\n--- assertion: C1 == C6 (broken diversity stage is a no-op) ---")
        print(f"  |C1 - C6| = {d:.4f}   pooled se = {pooled:.4f}   "
              f"{'consistent' if d <= 2 * pooled else 'INCONSISTENT -- investigate'}")

    print(f"\n--- AL curves ({m}, mean over seeds) ---")
    piv = df.pivot_table(index="round", columns="condition", values=m, aggfunc="mean")
    cols = sorted(piv.columns, key=lambda x: (len(x), x))
    print(piv[cols].round(4).to_string())


if __name__ == "__main__":
    main()
