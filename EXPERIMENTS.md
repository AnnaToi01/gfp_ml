# Active-learning ablations

Fork of [jaca00001/paper_shotgun_gfp](https://github.com/jaca00001/paper_shotgun_gfp)
with a flag-gated harness for testing what the active-learning machinery
actually contributes.

Every `AblationConfig` default reproduces the original behaviour, so an
unmodified run is condition **C0** and each fix is a single flag.

The full plan -- the three measurement defects, the condition matrix with
rationale, and further improvements -- is in [`docs/ablation_matrix.html`](docs/ablation_matrix.html)
(open it in a browser), also published at
<https://claude.ai/code/artifact/9ea65f33-2e6f-4853-a234-1edd0fbcefcb>.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p logs results
sbatch slurm/stage1.sh        # validate the apparatus first
python3 aggregate.py results/stage1.csv

sbatch slurm/stage2.sh        # 15 conditions x 5 seeds
python3 aggregate.py results/stage2.csv
```

A single run, outside SLURM:

```bash
python3 run_experiment.py --condition C4 --seed 0 --test-peak cgre1338-06
```

Fast smoke test (minutes, meaningless numbers, proves the path executes):

```bash
python3 run_experiment.py --condition C4 --seed 0 --test-peak cgre900x-25 \
    --pool-cap 400 --rounds 2 --budget 40 --epochs 2 --start-mult 1 \
    --num-workers 0 --out /tmp/smoke.csv
```

## Conditions

| ID | Condition | Isolates |
|----|-----------|----------|
| C0 | original, spectral clustering | provenance anchor |
| C1 | **reference** — original, still unfixed, KMeans | baseline for every row below |
| C2 | random acquisition | whether any of this beats chance |
| C3 | batch diversity repaired, k-mer | the diversity fix as designed |
| C4 | batch diversity, Hamming | better metric for aligned equal-length seqs |
| C5 | batch diversity, embedding space | model-space vs sequence-space diversity |
| C6 | diversity stage removed | asserts C1 == C6 (broken stage is a no-op) |
| C7 | `new_query_weight` honoured | upweighting fresh acquisitions |
| C8 | no clustering | what the 2-way split contributes |
| C9 | rank normalization | min-max outlier sensitivity |
| C10 | pure uncertainty (alpha=1) | is coverage earning its 0.62? |
| C11 | pure coverage (alpha=0) | is uncertainty earning its 0.38? |
| C12 | exploitation term restored | finding bright variants vs modelling the landscape |
| C13 | all repairs, k-mer | headline "fixed" number |
| C14 | all repairs, Hamming | best configuration |

C1 repairs nothing on purpose: a difference like C3 − C1 is only attributable
if the reference is the unfixed behaviour.

## Measurement apparatus

Three defects made comparisons meaningless; all three are addressed, and all
are opt-in so C0 still reproduces the original numbers.

**Seeding.** Nothing in the original was seeded — `split_data` was called
without `random=False`, and there was no `torch`/`numpy`/`random` seed anywhere.
`setup(seed=...)` now calls `src.repro.set_seed`, and the split takes an
explicit seed. Verified: two runs of the same condition and seed acquire
identical sequences and agree on metrics to ~1e-6. Not bit-identical —
`set_seed` uses `warn_only=True`, so some non-deterministic kernels are still
permitted. Set `warn_only=False` to have PyTorch name them.

**Frozen test set** (`frozen_eval`, default on in `run_experiment.py`). The
original evaluated on the *remaining* pool after each round's acquisitions were
removed, so the test set shrank by 96 per round and systematically lost the
sequences the acquisition function judged most informative — and each condition
ended up scored on a different test set. Now `eval_frac` of the pool is held out
before round 0, never acquirable, identical across rounds, conditions and seeds.

**Frozen validation** (`frozen_val`, default on in `run_experiment.py`). The
original diverted 5% of each round's acquisitions into validation, so model
selection depended on the acquisition strategy being tested.

## Data notes

- `data/cgreGFP/raw_data/cgreGFPWT-00.csv` — 24,537 rows. Three in-frame
  insertion variants (`.201H`, `.235R`, `.215D`) were removed: they are genuine
  single-codon insertions, the original encoded them as substitutions, and a
  fixed-length one-hot encoder cannot represent them. 20 deletion variants
  (`.` in the mutant slot) remain.
- Sequence reconstruction convention is **0-based** throughout —
  `aa_genotype_native` position *p* maps to `protein_seq[p]`; `aa_genotype_pseudo`
  position *p* is column *p* of `protein_seq_aligned`.
- `cgre900x`'s `nucleotide_seq` in `wt_gene_sequences.csv` is the *pre-cloning*
  cgre900 allele (codon 217 `ACA`/Thr). The library parent is cgre900x
  (`GCA`/Ala), which is what `protein_seq` and `data/protein_seqs.fa` carry, so
  the model data is correct. Patch position 648 (0-based) `A`→`G` before any
  nucleotide-level analysis.
- Fitness is strongly bimodal, and the dark mode is a **censored point mass** at
  the assay floor (`10**2.771` = 590.2 linear): 31.9% of `cgreGFPWT-00` rows hold
  that exact value. R² on this target therefore largely measures dark-vs-bright
  separation rather than brightness prediction.
- Budget 960 exceeds the pool size for 8 of 14 peaks. Only cgre1338 (8,934),
  cgre4111 (8,214), cgre132 (4,267), cgre9708 (4,180), cgre2880 (1,780) and
  cgre900x (1,276) can host the full experiment.

## Known issues not addressed here

- `evaluate_all_peaks` and the Optuna `objective` index `run_training`'s return
  value as a dict, but it returns a tuple. That path was already broken before
  these changes; `run_training` now returns a 3-tuple, which does not make it
  newly broken but will need updating if hyperopt is revived.
- `run_hyperopt`'s objective passes `val_percentage` to `ActiveLearningConfig`
  and `loss_weight` to `TrainConfig`; neither field exists.
- `save_metrics` calls `torch.save(unlabeled_dataset, ...)` inside the
  per-source loop, rewriting the same tensor once per source per round.
- `ModelConfig.patience = 20` cannot fire during AL rounds, which train for 11
  epochs, so "best model on validation" is "last epoch" for 10 of 11 rounds.

## Tests

```bash
python3 tests/check_select_distance_based.py
```

Prints a step-by-step audit showing that the original `select_distance_based`
receives one-hot tensors where strings are expected, that `torch.Tensor` hashes
by identity so every k-mer intersection is empty, and that the function
therefore returns `range(k)` — i.e. plain top-k by acquisition score.
