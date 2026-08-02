#!/usr/bin/env python3
"""Intrinsic-dimension estimation for the arXiv doc embeddings.

The question "what is the *true* dimension of the embeddings" is exactly the
intrinsic-dimension (ID) problem. Rather than train a bottleneck autoencoder
(whose answer is entangled with architecture/training), we run a battery of
dedicated ID estimators from `scikit-dimension` (imports as `skdim`) plus a
linear PCA baseline, and check that the estimates are stable across subsample
sizes (a known small-sample bias of the kNN-based estimators).

Estimators
----------
PCA variance / participation ratio : linear upper bound on ID.
lPCA (skdim, local)                : per-point local-PCA dimension; reports a
                                     distribution, robust to global curvature.
TwoNN (Facco et al., Sci.Rep. 2017): primary, near-assumption-free global ID.
MLE   (Levina-Bickel, NIPS 2004)   : MLE on kNN distances; second opinion.
CorrInt (correlation dimension)    : fractal/box-counting sanity check.

The embeddings are L2-normalized, i.e. they live on (a patch of) the unit
sphere S^383 in R^384, so the ambient/linear ceiling is 383, not 384.
"""
import time
import numpy as np
import skdim

EMB = "data/embeddings.npy"
SEED = 0
SAMPLE_SIZES = [10_000, 20_000, 50_000]   # stability sweep for kNN estimators
CORRINT_N = 5_000                          # CorrInt is O(n^2); cap it
LPCA_NN = 100                              # neighborhood size for local PCA


def draw(Xfull, n, ns, rng):
    """Draw a size-ns subsample and drop exact-duplicate embeddings.

    Exact duplicates (identical abstracts, cross-listings, boilerplate) give
    zero nearest-neighbor distances, which is repeated sampling rather than
    manifold structure: it breaks the MLE log-ratio and mildly biases the
    other kNN estimators. We remove them before estimating dimension.
    """
    sub = np.sort(rng.choice(n, size=ns, replace=False))
    Xs = np.ascontiguousarray(Xfull[sub], dtype=np.float64)
    _, uniq = np.unique(np.round(Xs, 6), axis=0, return_index=True)
    return Xs[np.sort(uniq)]


def participation_ratio(eigs):
    """Effective linear dimension: (sum lambda)^2 / sum(lambda^2).

    For k equal eigenvalues this returns exactly k; it is the standard
    "soft" count of significant principal directions.
    """
    eigs = np.asarray(eigs, dtype=np.float64)
    return eigs.sum() ** 2 / np.square(eigs).sum()


def pca_report(X):
    """Linear baseline: variance-explained thresholds + participation ratio."""
    Xc = X - X.mean(0, keepdims=True)
    # eigenvalues of the covariance = singular values^2 / (n-1)
    s = np.linalg.svd(Xc, compute_uv=False)
    eigs = s ** 2 / (len(X) - 1)
    cum = np.cumsum(eigs) / eigs.sum()
    thr = {f"{int(p*100)}%": int(np.searchsorted(cum, p) + 1)
           for p in (0.90, 0.95, 0.99)}
    return thr, participation_ratio(eigs), eigs


def run_global(name, estimator, X):
    t0 = time.time()
    d = estimator.fit(X).dimension_
    return name, float(d), time.time() - t0


def main():
    rng = np.random.default_rng(SEED)
    Xfull = np.load(EMB, mmap_mode="r")
    n, amb = Xfull.shape
    print(f"embeddings: {n:,} x {amb}  (L2-normalized -> on S^{amb-1})\n")

    # ---- linear baseline on the largest sample ----------------------------
    Xbig = draw(Xfull, n, max(SAMPLE_SIZES), rng)
    thr, pr, _ = pca_report(Xbig)
    print("== Linear PCA baseline (n=%d) ==" % len(Xbig))
    print("  components for variance:", thr)
    print(f"  participation ratio (effective linear dim): {pr:.1f}\n")

    # ---- local PCA: per-point dimension distribution ----------------------
    print("== lPCA (local, n=%d, %d-NN) ==" % (len(Xbig), LPCA_NN))
    t0 = time.time()
    lpca = skdim.id.lPCA().fit_pw(Xbig, n_neighbors=LPCA_NN, n_jobs=-1)
    dpw = np.asarray(lpca.dimension_pw_)
    q = np.percentile(dpw, [25, 50, 75])
    print(f"  per-point local dim  median={q[1]:.0f}  IQR=[{q[0]:.0f}, {q[2]:.0f}]"
          f"  mean={dpw.mean():.1f}   ({time.time()-t0:.1f}s)\n")

    # ---- global kNN estimators, swept over sample size --------------------
    print("== Global ID estimators (stability over n) ==")
    header = f"  {'n':>7} | {'TwoNN':>7} {'MLE':>7} {'CorrInt':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ns in SAMPLE_SIZES:
        Xs = draw(Xfull, n, ns, rng)
        _, two, _ = run_global("TwoNN", skdim.id.TwoNN(), Xs)
        _, mle, _ = run_global("MLE", skdim.id.MLE(), Xs)
        # CorrInt is O(n^2): run on a fixed-size subset of this sample
        Xc = Xs[:CORRINT_N]
        _, ci, _ = run_global("CorrInt", skdim.id.CorrInt(), Xc)
        print(f"  {ns:>7,} | {two:>7.1f} {mle:>7.1f} {ci:>8.1f}")
    print(f"\n  (CorrInt computed on the first {CORRINT_N:,} pts of each sample)")


if __name__ == "__main__":
    main()
