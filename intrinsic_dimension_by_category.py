#!/usr/bin/env python3
"""Per-category intrinsic dimension: is the arXiv embedding manifold stratified?

Tests the heterogeneity hypothesis -- do different arXiv primary categories
live on sheets of different intrinsic dimension, or is local d ~ 22 uniform?
For each sufficiently large primary_category we restrict to its documents,
dedup, and report:

  TwoNN, MLE  : density-based global intrinsic dimension (local geometry)
  PR          : PCA participation ratio (soft linear dimension)
  PCA95       : #components for 95% variance (linear ceiling)
  fluffiness  : PCA95 / TwoNN  -- a coiling/curvature proxy

Assumes data/meta.parquet is row-aligned with data/embeddings.npy (both built
by build_doc_embeddings.py in the same pass; row counts are checked).
"""
import numpy as np
import pandas as pd
import skdim

EMB = "data/embeddings.npy"
META = "data/meta.parquet"
SEED = 0
MIN_N = 3_000      # skip categories smaller than this (ID estimate too noisy)
CAP_N = 20_000     # cap per-category sample for speed
TOP_K = 18         # number of largest categories to report


def participation_ratio(eigs):
    eigs = np.asarray(eigs, dtype=np.float64)
    return eigs.sum() ** 2 / np.square(eigs).sum()


def pca_stats(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    eigs = s ** 2 / (len(X) - 1)
    cum = np.cumsum(eigs) / eigs.sum()
    pca95 = int(np.searchsorted(cum, 0.95) + 1)
    return participation_ratio(eigs), pca95


def main():
    rng = np.random.default_rng(SEED)
    X = np.load(EMB, mmap_mode="r")
    meta = pd.read_parquet(META, columns=["primary_category"])
    assert len(meta) == len(X), f"meta {len(meta)} != emb {len(X)} (not aligned)"
    cats = meta["primary_category"].to_numpy()

    counts = pd.Series(cats).value_counts()
    top = counts[counts >= MIN_N].head(TOP_K)
    print(f"{len(counts)} primary categories; "
          f"{len(top)} with >= {MIN_N:,} docs (showing top {TOP_K})\n")

    hdr = f"{'category':<12} {'n':>7} {'TwoNN':>6} {'MLE':>6} {'PR':>6} {'PCA95':>6} {'fluff':>6}"
    print(hdr); print("-" * len(hdr))

    rows = []
    for cat, cnt in top.items():
        pos = np.flatnonzero(cats == cat)
        if len(pos) > CAP_N:
            pos = np.sort(rng.choice(pos, CAP_N, replace=False))
        Xs = np.ascontiguousarray(X[pos], dtype=np.float64)
        _, uniq = np.unique(np.round(Xs, 6), axis=0, return_index=True)
        Xs = Xs[np.sort(uniq)]
        two = skdim.id.TwoNN().fit(Xs).dimension_
        mle = skdim.id.MLE().fit(Xs).dimension_
        pr, pca95 = pca_stats(Xs)
        fluff = pca95 / two if two else float("nan")
        rows.append((cat, len(Xs), two, mle, pr, pca95, fluff))
        print(f"{cat:<12} {len(Xs):>7,} {two:>6.1f} {mle:>6.1f} {pr:>6.1f} "
              f"{pca95:>6d} {fluff:>6.1f}")

    arr = np.array([(r[2], r[3], r[4]) for r in rows])
    print("\nacross categories:")
    for j, name in enumerate(["TwoNN", "MLE", "PR"]):
        col = arr[:, j]
        print(f"  {name:<6} min={col.min():5.1f}  max={col.max():5.1f}  "
              f"mean={col.mean():5.1f}  std={col.std():4.1f}  "
              f"spread={col.max()-col.min():4.1f}")


if __name__ == "__main__":
    main()
