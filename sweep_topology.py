#!/usr/bin/env python3
"""Sweep n_neighbors for both DiRe and UMAP, measure topology preservation.

For each n_neighbors in the grid:
  1. Compute DiRe and cuML UMAP 2-d layouts on the full corpus.
  2. On N=sample stratified points, compute ripser Betti curves for the
     reference 384-d cloud and each layout.
  3. Record DTW(β₀) and DTW(β₁) between reference and each layout.
  4. Record "spurious components": count of significant H₀ bars in each
     cloud (persistence ≥ thresh_frac × longest finite bar). Reference
     should have ~1; spurious_components_layout - spurious_components_ref
     measures over-clustering in the layout.

Outputs:
  data/sweep_topology.csv   flat table, one row per (n_neighbors, method)
  data/sweep_topology.png   DTW β₀ / DTW β₁ / spurious-β₀ vs n_neighbors
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def stratified_sample(labels, n, rng):
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    idx = []
    for lab, c in zip(unique, counts):
        take = max(1, int(round(n * c / total)))
        pool = np.where(labels == lab)[0]
        sel = rng.choice(pool, size=min(take, len(pool)), replace=False)
        idx.extend(sel.tolist())
    return np.sort(np.asarray(idx, dtype=np.int64))


def rescale_unit_diameter(points):
    from scipy.spatial.distance import pdist
    d = float(pdist(points).max())
    return (points / d).astype(np.float32) if d > 0 else points.astype(np.float32)


def run_betti(points, n_steps=50):
    from dire_rapids.betti_curve import compute_betti_curve_ripser
    return compute_betti_curve_ripser(rescale_unit_diameter(points), n_steps=n_steps, maxdim=1)


def get_persistence(points, maxdim=1):
    """Raw ripser persistence diagrams on the unit-diameter-rescaled cloud."""
    from ripser import ripser
    P = rescale_unit_diameter(points)
    return ripser(P, maxdim=maxdim)["dgms"]


def significant_bars(dgm, thresh_frac=0.3):
    """Count bars whose persistence is at least thresh_frac of the longest finite bar."""
    if len(dgm) == 0:
        return 0
    # H0's longest bar is [0, inf) — drop it; ripser replaces that with max_filt.
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return 0
    pers = finite[:, 1] - finite[:, 0]
    if pers.max() <= 0:
        return 0
    return int((pers >= thresh_frac * pers.max()).sum())


def dtw_curve(a, b):
    from fastdtw import fastdtw
    d, _ = fastdtw(a.astype(float), b.astype(float))
    return float(d)


def ensure_layout(method, n_neighbors, data_dir):
    """Run run_reducer.py if the layout doesn't exist yet."""
    path = data_dir / f"{method}_layout_n{n_neighbors}_d2.npy"
    if path.exists():
        return path
    cmd = [
        sys.executable, str(HERE / "run_reducer.py"),
        "--method", method, "--n-neighbors", str(n_neighbors),
        "--n-components", "2",
    ]
    print(f"  >> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=HERE / "data", type=Path)
    ap.add_argument("--n-grid", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128])
    ap.add_argument("--sample", type=int, default=4000,
                    help="stratified sample size for ripser")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--thresh-frac", type=float, default=0.3)
    args = ap.parse_args()

    X = np.load(args.data / "embeddings.npy")
    meta = pd.read_parquet(args.data / "meta.parquet")
    labels = meta["primary_category"].fillna("").values

    # L2-normalize (same as run_reducer / evaluate)
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    X = (X / nrm).astype(np.float32)
    print(f"Loaded {len(X):,} papers, dim {X.shape[1]}")

    # Ensure all layouts exist (triggers run_reducer.py as needed)
    for n in args.n_grid:
        for m in ("dire", "umap"):
            ensure_layout(m, n, args.data)

    # Prepare stratified samples (one per seed)
    sample_indices = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        sample_indices.append(stratified_sample(labels, args.sample, rng))

    # Reference Betti + significant-bars per seed (doesn't depend on n_neighbors)
    ref_cache = {}
    for seed, idx in enumerate(sample_indices):
        t0 = time.time()
        ref_curve = run_betti(X[idx], n_steps=args.n_steps)
        ref_dgms = get_persistence(X[idx], maxdim=1)
        ref_sig_b0 = significant_bars(ref_dgms[0], args.thresh_frac)
        ref_cache[seed] = (ref_curve, ref_sig_b0)
        print(f"[ref seed={seed}] sig_H0_bars={ref_sig_b0}  "
              f"β1.max={int(ref_curve['beta_1'].max())}  ({time.time()-t0:.1f}s)")

    rows = []
    for n in args.n_grid:
        for method in ("dire", "umap"):
            Y = np.load(args.data / f"{method}_layout_n{n}_d2.npy")
            for seed, idx in enumerate(sample_indices):
                ref_curve, ref_sig_b0 = ref_cache[seed]
                t0 = time.time()
                emb_curve = run_betti(Y[idx], n_steps=args.n_steps)
                emb_dgms = get_persistence(Y[idx], maxdim=1)
                emb_sig_b0 = significant_bars(emb_dgms[0], args.thresh_frac)
                dtw_b0 = dtw_curve(ref_curve["beta_0"], emb_curve["beta_0"])
                dtw_b1 = dtw_curve(ref_curve["beta_1"], emb_curve["beta_1"])
                rows.append({
                    "n_neighbors": n, "method": method, "seed": seed,
                    "dtw_b0": dtw_b0, "dtw_b1": dtw_b1,
                    "ref_sig_b0": ref_sig_b0, "emb_sig_b0": emb_sig_b0,
                    "spurious_b0": emb_sig_b0 - ref_sig_b0,
                })
                print(f"  n={n:3d} {method:4s} seed={seed}  "
                      f"DTW(β0)={dtw_b0:7.1f}  DTW(β1)={dtw_b1:6.1f}  "
                      f"sig_H0: ref={ref_sig_b0} emb={emb_sig_b0}  "
                      f"({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    out_csv = args.data / "sweep_topology.csv"
    df.to_csv(out_csv, index=False)

    agg = df.groupby(["n_neighbors", "method"]).agg(
        dtw_b0=("dtw_b0", "mean"), dtw_b0_std=("dtw_b0", "std"),
        dtw_b1=("dtw_b1", "mean"), dtw_b1_std=("dtw_b1", "std"),
        spurious_b0=("spurious_b0", "mean"), spurious_b0_std=("spurious_b0", "std"),
        ref_sig_b0=("ref_sig_b0", "mean"),
    ).reset_index()
    print("\n=== Mean ± std across seeds ===")
    print(agg.to_string(index=False))
    agg.to_csv(args.data / "sweep_topology_agg.csv", index=False)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"dire": "#d62728", "umap": "#1f77b4"}
    for method in ("dire", "umap"):
        sub = agg[agg.method == method].sort_values("n_neighbors")
        axes[0].errorbar(sub.n_neighbors, sub.dtw_b0, yerr=sub.dtw_b0_std,
                         marker="o", label=method, color=colors[method], capsize=3)
        axes[1].errorbar(sub.n_neighbors, sub.dtw_b1, yerr=sub.dtw_b1_std,
                         marker="o", label=method, color=colors[method], capsize=3)
        axes[2].errorbar(sub.n_neighbors, sub.emb_sig_b0 if "emb_sig_b0" in sub
                         else sub.spurious_b0 + sub.ref_sig_b0,
                         yerr=None,
                         marker="o", label=method, color=colors[method])
    # Plot reference sig_b0 as a horizontal dashed line on panel 3
    ref_line = agg["ref_sig_b0"].mean()
    axes[2].axhline(ref_line, color="black", linestyle="--", alpha=0.6,
                    label=f"reference (384-d) ≈ {ref_line:.1f}")

    axes[0].set_title("DTW(β₀) vs n_neighbors  (lower = better)")
    axes[1].set_title("DTW(β₁) vs n_neighbors  (lower = better)")
    axes[2].set_title(f"# significant H₀ bars in layout  (persistence ≥ {args.thresh_frac}·max)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n_neighbors")
        ax.set_xticks(args.n_grid, [str(x) for x in args.n_grid])
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[2].set_ylabel("# components")

    fig.suptitle(f"Topology-preservation sweep — N={args.sample:,} stratified, "
                 f"{args.n_seeds} seeds", fontsize=13)
    fig.tight_layout()
    out_png = args.data / "sweep_topology.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=140)
    print(f"\nSaved → {out_csv}, {out_png}")


if __name__ == "__main__":
    main()
