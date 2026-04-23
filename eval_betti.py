#!/usr/bin/env python3
"""Topology-preservation comparison via ripser-based Betti curves + DTW.

For each random stratified sample of N points:
  * Compute Betti curve (β₀, β₁) of the original 384-d cloud (reference)
  * Compute Betti curve of each 2-d layout restricted to the same indices
  * Compute DTW(reference, layout) for β₀ and β₁

Ripser's Vietoris-Rips is ~O(N²) for maxdim=1 (fine at N≈3-5K, brutal at 723K),
so we repeat over several seeds and average DTW — this also gives a stddev.

Outputs:
  data/eval_betti_n<S>.csv       per-seed raw DTW numbers
  data/eval_betti_n<S>_curves.npz  the actual β curves (for plotting)
  data/eval_betti_n<S>.png         overlay of β₀, β₁ curves for each method
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastdtw import fastdtw


def stratified_sample(labels: np.ndarray, n: int, rng: np.random.Generator):
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


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def curve_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """fastdtw on the two Betti curves (1-D sequences); Euclidean on scalars."""
    dist, _ = fastdtw(a.astype(float), b.astype(float))
    return float(dist)


def _rescale_to_unit_diameter(points: np.ndarray) -> np.ndarray:
    """Rescale so the max pairwise distance = 1.

    Without this, UMAP's natural coordinate range is ~10× the reference's,
    shifting its Betti curve against the filtration axis and inflating DTW
    for reasons that have nothing to do with topology.
    """
    from scipy.spatial.distance import pdist
    d_max = float(pdist(points).max())
    if d_max == 0:
        return points.astype(np.float32)
    return (points / d_max).astype(np.float32)


def run_betti(points, n_steps: int, rescale: bool = True):
    """Ripser-based Betti curve, β₀ and β₁.

    If ``rescale`` is True, each cloud is rescaled to unit diameter first so
    the Betti curves share a common filtration axis (essential for DTW to
    compare topology rather than coordinate scale).
    """
    from dire_rapids.betti_curve import compute_betti_curve_ripser
    if rescale:
        points = _rescale_to_unit_diameter(points)
    return compute_betti_curve_ripser(points, n_steps=n_steps, maxdim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path(__file__).parent / "data", type=Path)
    ap.add_argument("--methods", nargs="+", default=["dire", "umap"])
    ap.add_argument("--n-neighbors", type=int, default=16)
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    X = np.load(args.data / "embeddings.npy")
    meta = pd.read_parquet(args.data / "meta.parquet")
    labels = meta["primary_category"].fillna("").values
    print(f"Loaded {len(X):,} papers, dim {X.shape[1]}")

    if not args.no_normalize:
        X = normalize_rows(X)

    layouts = {}
    for method in args.methods:
        p = args.data / f"{method}_layout_n{args.n_neighbors}.npy"
        if not p.exists():
            print(f"missing {p} — skipping {method}", file=sys.stderr)
            continue
        layouts[method] = np.load(p)
    print(f"Loaded layouts: {list(layouts)}")

    rows = []
    # keep last seed's curves for plotting
    last_curves = None

    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        idx = stratified_sample(labels, args.sample, rng)
        print(f"\n=== seed {seed}: sample N={len(idx):,} ===")

        t0 = time.time()
        ref = run_betti(X[idx], n_steps=args.n_steps)
        print(f"  reference (384-d): β₀[0]={ref['beta_0'][0]}, "
              f"β₀[-1]={ref['beta_0'][-1]}, "
              f"β₁.max={int(ref['beta_1'].max())}  ({time.time()-t0:.1f}s)")

        curves = {"reference": ref}
        for method, Y in layouts.items():
            t0 = time.time()
            emb = run_betti(Y[idx], n_steps=args.n_steps)
            curves[method] = emb
            dtw_b0 = curve_dtw(ref["beta_0"], emb["beta_0"])
            dtw_b1 = curve_dtw(ref["beta_1"], emb["beta_1"])
            print(f"  {method:>5s}: β₁.max={int(emb['beta_1'].max()):3d}  "
                  f"DTW β₀={dtw_b0:8.1f}  DTW β₁={dtw_b1:8.1f}  "
                  f"({time.time()-t0:.1f}s)")
            rows.append({
                "seed": seed, "N": len(idx), "method": method,
                "dtw_b0": dtw_b0, "dtw_b1": dtw_b1,
                "ref_b1_max": int(ref["beta_1"].max()),
                "emb_b1_max": int(emb["beta_1"].max()),
            })

        last_curves = (idx, curves)

    df = pd.DataFrame(rows)
    out_csv = args.data / f"eval_betti_n{args.sample}.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Per-seed ===")
    print(df.to_string(index=False))
    print("\n=== Mean ± std across seeds ===")
    agg = df.groupby("method").agg(
        dtw_b0_mean=("dtw_b0", "mean"), dtw_b0_std=("dtw_b0", "std"),
        dtw_b1_mean=("dtw_b1", "mean"), dtw_b1_std=("dtw_b1", "std"),
    )
    print(agg.to_string())
    print(f"\nSaved raw → {out_csv}")

    # Save curves from the last seed + a summary plot
    if last_curves is not None:
        idx, curves = last_curves
        out_npz = args.data / f"eval_betti_n{args.sample}_curves.npz"
        to_save = {}
        for tag, c in curves.items():
            to_save[f"{tag}_filt"] = c["filtration_values"]
            to_save[f"{tag}_b0"] = c["beta_0"]
            to_save[f"{tag}_b1"] = c["beta_1"]
        np.savez(out_npz, sample_idx=idx, **to_save)
        print(f"Saved curves → {out_npz}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for tag, c in curves.items():
            f = c["filtration_values"]
            axes[0].plot(f, c["beta_0"], label=tag, linewidth=2)
            axes[1].plot(f, c["beta_1"], label=tag, linewidth=2)
        axes[0].set_title(f"β₀ (connected components) — N={len(idx):,}")
        axes[1].set_title(f"β₁ (1-cycles) — N={len(idx):,}")
        for ax in axes:
            ax.set_xlabel("filtration")
            ax.set_ylabel("Betti number")
            ax.legend()
            ax.grid(True, alpha=0.3)
        out_png = args.data / f"eval_betti_n{args.sample}.png"
        fig.savefig(out_png, bbox_inches="tight", dpi=140)
        print(f"Saved plot → {out_png}")


if __name__ == "__main__":
    main()
