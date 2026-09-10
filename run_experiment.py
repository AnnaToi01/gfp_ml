"""
Single entry point for one ablation condition at one seed.

    python3 run_experiment.py --condition C1 --seed 0

Appends one row per AL round to --out (default results/runs.csv), so results
across conditions and seeds aggregate into a single tidy table.

Condition definitions live in CONDITIONS below. Every flag not named by a
condition keeps its AblationConfig default, which reproduces the original
behaviour -- so C0 is the untouched code.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import pandas as pd

from src.utls import setup, create_config, run_training, get_peaks_sources
from src.data_loading import prepare_data_protein, prepare_data_multiple_proteins


# ---------------------------------------------------------------- conditions

REFERENCE = {"cluster": "kmeans"}          # C1 and everything derived from it

CONDITIONS: dict[str, dict] = {
    # anchors
    "C0":  {"cluster": "spectral"},
    "C1":  {**REFERENCE},
    "C2":  {**REFERENCE, "acquisition": "random"},
    # the diversity fix
    "C3":  {**REFERENCE, "diversity_input": "sequence", "diversity_metric": "kmer"},
    "C4":  {**REFERENCE, "diversity_input": "sequence", "diversity_metric": "hamming"},
    "C5":  {**REFERENCE, "diversity_input": "sequence", "diversity_metric": "embedding"},
    "C6":  {**REFERENCE, "diversity_metric": "none"},
    # other dead / untested machinery
    "C7":  {**REFERENCE, "use_query_weight": True},
    "C8":  {**REFERENCE, "cluster": "none"},
    "C9":  {**REFERENCE, "normalize": "rank"},
    # acquisition composition
    "C10": {**REFERENCE, "acquisition": "uncert"},
    "C11": {**REFERENCE, "acquisition": "dist"},
    "C12": {**REFERENCE, "acquisition": "greedy"},
    # combined
    "C13": {**REFERENCE, "diversity_input": "sequence", "diversity_metric": "kmer",
            "use_query_weight": True, "normalize": "rank"},
    "C14": {**REFERENCE, "diversity_input": "sequence", "diversity_metric": "hamming",
            "use_query_weight": True, "normalize": "rank"},
}


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-peak", default="cgreGFPWT-00")
    p.add_argument("--test-peak", default="cgre1338-06",
                   help="AL pool. Empty string splits the train peak instead.")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--budget", type=int, default=96 * 10)
    p.add_argument("--train-size", type=float, default=0.01)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=0, help="Override TrainConfig.epochs (0 = default).")
    p.add_argument("--start-mult", type=int, default=0, help="Override starting_epoch_multiplier (0 = default).")
    p.add_argument("--pool-cap", type=int, default=0,
                   help="Subsample the AL pool to this many rows (0 = no cap). For smoke tests.")
    p.add_argument("--no-frozen-eval", action="store_true")
    p.add_argument("--out", default="results/runs.csv")
    p.add_argument("--tag", default="", help="Free-text label recorded in the output row.")
    return p.parse_args()


def main():
    a = build_args()
    flags = dict(CONDITIONS[a.condition])

    # Seed before anything else touches an RNG.
    setup(seed=a.seed)

    wt = prepare_data_protein(
        filepath="data/cgreGFP/raw_data/cgreGFPWT-00.csv",
        target_row_label="brightness",
        rescale="log10",
    )
    peaks = prepare_data_multiple_proteins(
        folderpath="data/mpcgreGFP/data_raw/",
        target_row_label="log_brightness",
        concat_peaks=False,
        additional_peaks=[wt],
    )

    if a.pool_cap:
        for i, df in enumerate(peaks):
            if df["gene"].iloc[0] == a.test_peak and len(df) > a.pool_cap:
                peaks[i] = df.sample(n=a.pool_cap, random_state=a.seed).reset_index(drop=True)

    peak_names, _ = get_peaks_sources(peaks_df=peaks)
    assert a.train_peak in peak_names, f"{a.train_peak} not in {peak_names}"
    test_peaks = [a.test_peak] if a.test_peak else []
    for t in test_peaks:
        assert t in peak_names, f"{t} not in {peak_names}"

    cfg = create_config(
        data_cfg={
            "train_size": a.train_size,
            "val_size": a.val_size,
            "test_size": 0.1,
            "split": "perc",
            "batch_size": a.batch_size,
            "num_workers": a.num_workers,
        },
        train_cfg={
            "use_al": True, "num_runs": 1,
            **({"epochs": a.epochs} if a.epochs else {}),
            **({"starting_epoch_multiplier": a.start_mult} if a.start_mult else {}),
        },
        al_cfg={"acquisition_budget_total": a.budget, "rounds": a.rounds},
        model_cfg={},
        abl_cfg={
            **flags,
            "seed": a.seed,
            "frozen_eval": not a.no_frozen_eval,
            "eval_frac": a.eval_frac,
            "frozen_val": True,
        },
    )

    print(f"=== condition {a.condition} seed {a.seed} :: {json.dumps(flags, sort_keys=True)}")

    t0 = time.time()
    _, _, history = run_training(
        peaks_df=peaks,
        train_peaks=[a.train_peak],
        test_peaks=test_peaks,
        cfg=cfg,
    )
    elapsed = time.time() - t0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "condition": a.condition,
            "seed": a.seed,
            "train_peak": a.train_peak,
            "test_peak": a.test_peak or "(split)",
            "pool_cap": a.pool_cap,
            "tag": a.tag,
            "elapsed_s": round(elapsed, 2),
            **{k: v for k, v in flags.items()},
            **h,
        }
        for h in history
    ]
    fields = sorted({k for r in rows for k in r})
    write_header = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)

    print(f"\n=== {a.condition} seed {a.seed} done in {elapsed:.1f}s -> {out} ({len(rows)} rows)")
    print(pd.DataFrame(history).to_string(index=False))


if __name__ == "__main__":
    main()
