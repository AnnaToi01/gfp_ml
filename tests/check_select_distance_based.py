"""
Step-by-step audit of select_distance_based() as it is actually called by
get_queried_samples() in src/utls.py.

Run from the repo root:  python3 tests/check_select_distance_based.py
Nothing here is modified or monkeypatched -- it imports the real functions.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import torch, numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data_loading import create_dataset
from src.utls import kmers, select_distance_based

def rule(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)

# ---------------------------------------------------------------- setup
df = pd.read_csv(f"{REPO}/data/mpcgreGFP/data_raw/cgre132-06.csv").head(40).copy()
df = df.rename(columns={"log_brightness": "fitness"})
df["source_id"] = 0
base_dataset = create_dataset(df)
strings = df["sequence"].tolist()

rule("STEP 0 -- the call site, verbatim from src/utls.py:600-605")
print("""    top_local  = np.argsort(-acquisition_cluster)[:diversity_budget]
    top_global = cluster_global[top_local].tolist()
    seqs       = [base_dataset[i][0] for i in top_global]   <-- element [0]
    diverse_local = select_distance_based(seqs, cluster_budget)""")
print("\n  select_distance_based's own signature and docstring:")
print("    def select_distance_based(seqs: List[str], k: int, kmer_size: int=3)")
print('    """seqs : list[str]  --  Amino acid sequences."""')

rule("STEP 1 -- what is base_dataset[i][0] ?")
x = base_dataset[0][0]
print(f"  type(base_dataset[0][0])  = {type(x).__name__}")
print(f"  .shape                    = {tuple(x.shape)}   (20 amino acids x 236 positions)")
print(f"  .dtype                    = {x.dtype}")
print(f"  first 8 values of row 0   = {x[0, :8].tolist()}")
print("\n  -> a one-hot TENSOR is passed where a str was expected.")
print(f"  the real sequence for the same row is a str of len {len(strings[0])}:")
print(f"    {strings[0][:60]}...")

rule("STEP 2 -- len() disagreement (drives L and num_kmers)")
print(f"  len(tensor)  = {len(x):3d}   <- iterates dim 0 = the 20 aa channels")
print(f"  len(string)  = {len(strings[0]):3d}   <- the actual 236 residues")
for label, L in (("tensor", len(x)), ("string", len(strings[0]))):
    print(f"    with {label}: L={L:3d}  num_kmers = L-3+1 = {L-3+1}")

rule("STEP 3 -- what kmers() returns for each input type")
kt, ks = kmers(x, 3), kmers(strings[0], 3)
print(f"  kmers(tensor)  -> set of {len(kt)} items, element type {type(next(iter(kt))).__name__}")
print(f"                    an element has shape {tuple(next(iter(kt)).shape)}")
print(f"  kmers(string)  -> set of {len(ks)} items, element type {type(next(iter(ks))).__name__}")
print(f"                    e.g. {sorted(ks)[:6]}")

rule("STEP 4 -- WHY tensor intersection is always empty (hashing)")
a = x[0:3]
b = x[0:3].clone()
print(f"  a = x[0:3]; b = a.clone()   (identical values, different objects)")
print(f"  torch.equal(a, b)  = {torch.equal(a, b)}")
print(f"  hash(a)            = {hash(a)}")
print(f"  hash(b)            = {hash(b)}")
print(f"  hash(a) == hash(b) = {hash(a) == hash(b)}   <- Tensor.__hash__ is identity-based")
print(f"  len({{a}} & {{b}})       = {len({a} & {b})}   <- so equal tensors never match in a set")
print("\n  contrast with strings:")
print(f"  hash('MTA') == hash('MTA')  = {hash('MTA') == hash('MTA')}")
print(f"  len({{'MTA'}} & {{'MTA'}})       = {len({'MTA'} & {'MTA'})}")

rule("STEP 5 -- the distance formula, computed both ways")
print("  formula:  d = 1.0 - inter / (2*num_kmers - inter)")
for label, seqs in (("TENSORS", [base_dataset[i][0] for i in range(40)]), ("STRINGS", strings)):
    L = len(seqs[0]); nk = L - 3 + 1
    kk = [kmers(s, 3) for s in seqs]
    print(f"\n  {label}:  num_kmers = {nk}")
    print(f"    {'pair':10s} {'inter':>6s} {'d':>10s}")
    for i in (1, 2, 3, 4, 5):
        inter = len(kk[0] & kk[i])
        d = 1.0 - inter / (2 * nk - inter)
        print(f"    {'s0 vs s'+str(i):10s} {inter:6d} {d:10.5f}")
    ds = [1.0 - len(kk[0] & kk[i]) / (2*nk - len(kk[0] & kk[i])) for i in range(1, 40)]
    print(f"    distinct distance values among all 39 pairs: {sorted(set(round(v,6) for v in ds))[:8]}")

rule("STEP 6 -- ground truth: how similar ARE these sequences?")
print("  cgre132 rows are point mutants of one parent, so as strings they should")
print("  share nearly all 3-mers. Hamming distances from row 0:")
for i in (1, 2, 3, 4, 5):
    ham = sum(c1 != c2 for c1, c2 in zip(strings[0], strings[i]))
    inter = len(kmers(strings[0],3) & kmers(strings[i],3))
    print(f"    row {i}: hamming={ham:3d}  shared 3-mers={inter:3d} / 234")
print("\n  -> real overlap is ~215/234. The tensor path reports 0/18 for every pair.")

rule("STEP 7 -- the tie-break that makes the result trivial")
print("  when every d == 1.0, min_dist is uniformly 1.0, so:")
mask = np.zeros(8, dtype=bool); mask[0] = True
md = np.full(8, 1.0)
arr = np.where(mask, -1.0, md)
print(f"    min_dist                                  = {md.tolist()}")
print(f"    np.where(selected_mask, -1.0, min_dist)   = {arr.tolist()}")
print(f"    np.argmax(...)                            = {np.argmax(arr)}  <- always the first unselected index")

rule("STEP 8 -- end result, real function, no modifications")
tensors = [base_dataset[i][0] for i in range(40)]
for k in (5, 8, 12):
    got_t = [int(i) for i in select_distance_based(tensors, k)]
    got_s = [int(i) for i in select_distance_based(strings, k)]
    print(f"  k={k:2d}")
    print(f"    tensors (current) -> {got_t}    == range({k})? {got_t == list(range(k))}")
    print(f"    strings (intended)-> {got_s}")

rule("STEP 9 -- consequence at the call site")
print("  top_global is sorted DESCENDING by acquisition score (np.argsort(-acq)),")
print("  and select_distance_based returns range(k), so:")
print("      selected = [top_global[i] for i in range(k)] = top-k by acquisition score")
print("  i.e. the farthest-point re-ranking, and the 5x shortlist it ranks over,")
print("  currently have no effect on which sequences get queried.")
print("\n  NOTE: the min_dist term inside compute_acquisition() is UNAFFECTED --")
print("  that one uses torch.cdist on embeddings and works correctly.")
