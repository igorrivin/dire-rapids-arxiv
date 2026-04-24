#!/usr/bin/env python3
"""Quantify how much each layout exaggerates cluster boundaries.

"Island-ness" metrics, on rescaled (unit-diameter) point clouds — comparing
the reference 384-d cloud against each layout at each n_neighbors.

Three complementary views of the H₀ persistence diagram:

  longest_bar    absolute length of the longest H₀ bar (= birth-time of the
                 last merge). High = a very isolated cluster survives deep
                 into the filtration.

  top5_ratio     (top-5 longest H₀ bars) / (50th-percentile bar length).
                 High = a handful of bars dominate; equivalent to "a few
                 hard islands on a smooth background".

  gini_h0        Gini coefficient of H₀ bar lengths. 0 = uniform bar
                 lengths, 1 = all persistence mass in one bar.

If UMAP islands math.AG while DiRe keeps it as a tendril, UMAP should score
higher on all three — that is the quantitative version of the complaint
that UMAP "invents clusters where there aren't any".
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def stratified_sample(labels, n, rng):
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    idx = []
    for lab, c in zip(unique, counts):
        take = max(1, int(round(n * c / counts.sum())))
        pool = np.where(labels == lab)[0]
        idx.extend(rng.choice(pool, size=min(take, len(pool)), replace=False).tolist())
    return np.sort(np.asarray(idx, dtype=np.int64))


def rescale_unit_diameter(points):
    from scipy.spatial.distance import pdist
    d = float(pdist(points).max())
    return (points / d).astype(np.float32) if d > 0 else points.astype(np.float32)


def h0_bars(points, maxdim=0):
    """Return finite H₀ bar lengths on a unit-diameter-rescaled cloud.

    The infinite bar (final connected component that never dies) is dropped.
    """
    from ripser import ripser
    P = rescale_unit_diameter(points)
    dgm = ripser(P, maxdim=maxdim)["dgms"][0]
    finite = dgm[np.isfinite(dgm[:, 1])]
    return finite[:, 1] - finite[:, 0]


def gini(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float(1.0 - 2.0 * (cum.sum() - cum[-1] / 2.0) / (n * cum[-1]))


def island_metrics(bars):
    bars = np.asarray(bars, dtype=float)
    if len(bars) == 0:
        return {"longest_bar": 0.0, "top5_ratio": 0.0, "gini_h0": 0.0,
                "n_bars": 0}
    bars_sorted = np.sort(bars)[::-1]
    top5 = bars_sorted[:5].mean() if len(bars_sorted) >= 5 else bars_sorted.mean()
    median = float(np.median(bars_sorted))
    return {
        "longest_bar": float(bars_sorted[0]),
        "top5_ratio": float(top5 / median) if median > 0 else 0.0,
        "gini_h0": gini(bars_sorted),
        "n_bars": int(len(bars)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=HERE / "data", type=Path)
    ap.add_argument("--n-grid", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128])
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()

    X = np.load(args.data / "embeddings.npy")
    meta = pd.read_parquet(args.data / "meta.parquet")
    labels = meta["primary_category"].fillna("").values
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    X = (X / nrm).astype(np.float32)

    # Same stratified samples as sweep_topology for comparability
    sample_indices = [stratified_sample(labels, args.sample, np.random.default_rng(s))
                      for s in range(args.n_seeds)]

    rows = []
    for seed, idx in enumerate(sample_indices):
        t0 = time.time()
        bars = h0_bars(X[idx])
        m = island_metrics(bars)
        m.update({"method": "reference", "n_neighbors": -1, "seed": seed})
        rows.append(m)
        print(f"[ref seed={seed}] longest={m['longest_bar']:.4f}  "
              f"top5_ratio={m['top5_ratio']:.2f}  gini={m['gini_h0']:.3f}  "
              f"({time.time()-t0:.1f}s)")

    for n in args.n_grid:
        for method in ("dire", "umap"):
            Y = np.load(args.data / f"{method}_layout_n{n}_d2.npy")
            for seed, idx in enumerate(sample_indices):
                bars = h0_bars(Y[idx])
                m = island_metrics(bars)
                m.update({"method": method, "n_neighbors": n, "seed": seed})
                rows.append(m)
                print(f"  n={n:3d} {method:4s} seed={seed}  "
                      f"longest={m['longest_bar']:.4f}  "
                      f"top5_ratio={m['top5_ratio']:.2f}  "
                      f"gini={m['gini_h0']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.data / "island_ness.csv", index=False)

    agg = df.groupby(["method", "n_neighbors"]).agg(
        longest_bar=("longest_bar", "mean"),
        longest_bar_std=("longest_bar", "std"),
        top5_ratio=("top5_ratio", "mean"),
        top5_ratio_std=("top5_ratio", "std"),
        gini_h0=("gini_h0", "mean"),
        gini_h0_std=("gini_h0", "std"),
    ).reset_index()
    print("\n=== Mean ± std across seeds ===")
    print(agg.to_string(index=False))
    agg.to_csv(args.data / "island_ness_agg.csv", index=False)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"dire": "#d62728", "umap": "#1f77b4", "reference": "black"}

    ref_agg = agg[agg.method == "reference"].iloc[0]
    metrics = [
        ("longest_bar", "longest H₀ bar (rescaled)",
         "Longest H₀ bar length  (higher = 1 cluster dominates)"),
        ("top5_ratio", "top-5 / median H₀ bar",
         "Top-5 / median bar ratio  (higher = a few bars dominate)"),
        ("gini_h0", "Gini of H₀ bar lengths",
         "Gini coefficient of H₀ bar lengths  (higher = concentrated)"),
    ]
    for ax, (key, _, title) in zip(axes, metrics):
        for method in ("dire", "umap"):
            sub = agg[agg.method == method].sort_values("n_neighbors")
            ax.errorbar(sub.n_neighbors, sub[key], yerr=sub[f"{key}_std"],
                        marker="o", label=method, color=colors[method], capsize=3)
        ax.axhline(ref_agg[key], color="black", linestyle="--", alpha=0.7,
                   label=f"reference 384-d: {ref_agg[key]:.3f}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n_neighbors")
        ax.set_xticks(args.n_grid, [str(x) for x in args.n_grid])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Island-ness of layouts vs reference — N={args.sample:,} stratified, "
                 f"{args.n_seeds} seeds", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.data / "island_ness.png", bbox_inches="tight", dpi=140)
    print(f"\nSaved → {args.data / 'island_ness.csv'}, {args.data / 'island_ness.png'}")


if __name__ == "__main__":
    main()
