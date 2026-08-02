#!/usr/bin/env python3
"""Density-matched within/between intrinsic dimension (follow-up #1).

The per-category finding (within-field d ~12-18 < whole-corpus d ~22) was
confounded: per-category samples are near-complete (dense) while the full-corpus
sample is ~7% (sparse), and kNN ID estimators read higher at coarser scales.
This script removes the confound by estimating ID *as a function of physical
neighborhood radius* and comparing the full corpus to single categories AT EQUAL
RADIUS.

Method: Levina-Bickel MLE (NIPS 2004), swept over neighbor count k, with the
MacKay-Ghahramani reciprocal-average correction. For point i with sorted
neighbor distances T_1<=...<=T_k (self excluded),
    1/m_k(i) = (1/(k-1)) * sum_{j=1}^{k-1} log(T_k(i)/T_j(i)),
    m_k      = 1 / mean_i( 1/m_k(i) ).
Larger k => larger neighborhood radius => coarser scale. One kNN query per
dataset yields the whole d-vs-radius curve. We then compare each category's
(radius, d) curve to the full corpus's, interpolating the full-corpus d to the
category's radii: a persistent negative gap (category lower at matched radius)
means the within<between split is real, not a density artifact.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

EMB = "data/embeddings.npy"
META = "data/meta.parquet"
SEED = 0
FULL_N = 50_000
CAT_CAP = 25_000
CATS = ["math.CO", "math.AP", "math.FA"]   # highest-d, middle, lowest-d
KS = [4, 8, 16, 32, 64, 128, 256]
KMAX = max(KS)


def dedup(X):
    _, uniq = np.unique(np.round(X, 6), axis=0, return_index=True)
    return X[np.sort(uniq)]


def knn_dists(X):
    """Sorted distances to the KMAX nearest neighbors (self excluded), all > 0."""
    nn = NearestNeighbors(n_neighbors=KMAX + 1, algorithm="brute", n_jobs=-1)
    nn.fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, 1:]                      # drop self (column 0, distance 0)


def mle_vs_scale(d):
    """Return (radii, dims): median k-NN radius and MG-corrected LB MLE per k."""
    logd = np.log(d)
    radii, dims = [], []
    for k in KS:
        Tk = logd[:, k - 1][:, None]
        inv_m = (Tk - logd[:, :k - 1]).sum(1) / (k - 1)   # 1/m_k per point
        dims.append(1.0 / inv_m.mean())                    # MacKay-Ghahramani
        radii.append(np.median(d[:, k - 1]))               # physical scale
    return np.array(radii), np.array(dims)


def main():
    rng = np.random.default_rng(SEED)
    X = np.load(EMB, mmap_mode="r")
    cats = pd.read_parquet(META, columns=["primary_category"])["primary_category"].to_numpy()
    assert len(cats) == len(X)

    datasets = {}
    full_idx = np.sort(rng.choice(len(X), FULL_N, replace=False))
    datasets["FULL"] = dedup(np.ascontiguousarray(X[full_idx], dtype=np.float64))
    for c in CATS:
        pos = np.flatnonzero(cats == c)
        if len(pos) > CAT_CAP:
            pos = np.sort(rng.choice(pos, CAT_CAP, replace=False))
        datasets[c] = dedup(np.ascontiguousarray(X[pos], dtype=np.float64))

    curves = {}
    for name, D in datasets.items():
        curves[name] = mle_vs_scale(knn_dists(D))
        r, d = curves[name]
        print(f"\n{name}  (n={len(D):,})")
        print("    k :  " + "  ".join(f"{k:>5d}" for k in KS))
        print("  radius:" + " ".join(f"{x:6.3f}" for x in r))
        print("   d_MLE:" + " ".join(f"{x:6.1f}" for x in d))

    # ---- matched-radius comparison: category d vs full-corpus d at same radius
    fr, fd = curves["FULL"]
    order = np.argsort(fr)
    fr_s, fd_s = fr[order], fd[order]
    print("\n== category d vs FULL d, interpolated to matched radius ==")
    print(f"{'cat':<9} {'radius':>7} {'d_cat':>6} {'d_full@r':>9} {'gap':>6}")
    for c in CATS:
        r, d = curves[c]
        in_range = (r >= fr_s[0]) & (r <= fr_s[-1])     # only where curves overlap
        gaps = []
        for ri, di, ok in zip(r, d, in_range):
            if not ok:
                continue
            df = np.interp(ri, fr_s, fd_s)
            gaps.append(di - df)
            print(f"{c:<9} {ri:>7.3f} {di:>6.1f} {df:>9.1f} {di - df:>6.1f}")
        if gaps:
            print(f"  -> {c} mean gap over overlap: {np.mean(gaps):+.1f}  "
                  f"({len(gaps)} matched scales)\n")

    # ---- plot ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    for name, (r, d) in curves.items():
        lw = 2.5 if name == "FULL" else 1.5
        plt.plot(r, d, "o-", lw=lw, label=name)
    plt.xlabel("median k-NN radius (neighborhood scale)")
    plt.ylabel("Levina-Bickel MLE intrinsic dimension")
    plt.title("ID vs scale: full corpus vs single categories\n"
              "(compare curves at equal radius)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("data/intrinsic_dim_scale.png", dpi=130)
    print("wrote data/intrinsic_dim_scale.png")


if __name__ == "__main__":
    main()
