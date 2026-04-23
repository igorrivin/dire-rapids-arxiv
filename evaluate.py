#!/usr/bin/env python3
"""Quantitative comparison of 2-d layouts against the original 384-d embeddings.

Loads:
    data/embeddings.npy              (N, 384) float32 — mean-pooled BGE
    data/meta.parquet                has primary_category
    data/dire_layout_n<N>.npy        (N, 2)
    data/umap_layout_n<N>.npy        (N, 2)

Computes two groups of metrics:

1. Faithfulness to high-D structure (on a stratified sample for tractability):
   - trustworthiness   (sklearn, penalizes false neighbors in 2-d)
   - continuity        (penalizes lost HD neighbors)
   - kNN-preservation  (|kNN_HD ∩ kNN_2D| / k, averaged)

2. Category separation (on full corpus):
   - silhouette_score(2D, primary_category)  — sampled
   - NMI(primary_category, KMeans(2D))       — cuML KMeans, K = #cats

Betti-curve DTW (topology preservation via Hodge Laplacian or ripser) is a
separate step — see eval_betti.py when ready.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_sample(labels: np.ndarray, n: int, rng: np.random.Generator):
    """Return indices for a stratified sample of size ~n."""
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


def knn_preservation(X_hd: np.ndarray, Y_2d: np.ndarray, k: int = 15) -> float:
    """Mean |kNN_HD(i) ∩ kNN_2D(i)| / k over all points (self excluded)."""
    from sklearn.neighbors import NearestNeighbors
    nn_hd = NearestNeighbors(n_neighbors=k + 1).fit(X_hd)
    nn_2d = NearestNeighbors(n_neighbors=k + 1).fit(Y_2d)
    _, idx_hd = nn_hd.kneighbors(X_hd)
    _, idx_2d = nn_2d.kneighbors(Y_2d)
    # drop self (first column)
    idx_hd = idx_hd[:, 1:]
    idx_2d = idx_2d[:, 1:]
    overlap = np.array([
        len(set(a) & set(b)) for a, b in zip(idx_hd, idx_2d)
    ])
    return float(overlap.mean() / k)


def faithfulness(X_hd: np.ndarray, Y_2d: np.ndarray, k: int):
    from sklearn.manifold import trustworthiness
    t = float(trustworthiness(X_hd, Y_2d, n_neighbors=k))
    # Continuity: trustworthiness with args flipped
    c = float(trustworthiness(Y_2d, X_hd, n_neighbors=k))
    knn = knn_preservation(X_hd, Y_2d, k=k)
    return {"trustworthiness": t, "continuity": c, f"knn_pres@{k}": knn}


def silhouette(Y_2d: np.ndarray, labels: np.ndarray, sample_size: int = 30000,
               rng_seed: int = 42) -> float:
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(Y_2d, labels, sample_size=sample_size,
                                   random_state=rng_seed))


def nmi_via_kmeans(Y_2d: np.ndarray, labels: np.ndarray, k_clusters: int) -> float:
    try:
        from cuml import KMeans as cuKMeans
        km = cuKMeans(n_clusters=k_clusters, n_init=10, random_state=42)
        preds = np.asarray(km.fit_predict(Y_2d.astype(np.float32)))
    except Exception as e:
        print(f"  cuML KMeans failed ({e}), using sklearn", file=sys.stderr)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
        preds = km.fit_predict(Y_2d)
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(labels, preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path(__file__).parent / "data", type=Path)
    ap.add_argument("--methods", nargs="+", default=["dire", "umap"])
    ap.add_argument("--n-neighbors", type=int, default=16)
    ap.add_argument("--sample", type=int, default=30000,
                    help="sample size for faithfulness metrics")
    ap.add_argument("--silhouette-sample", type=int, default=30000)
    ap.add_argument("--k", type=int, default=15,
                    help="k for kNN-based faithfulness metrics")
    ap.add_argument("--k-clusters", type=int, default=25,
                    help="#clusters for NMI via KMeans")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X = np.load(args.data / "embeddings.npy")
    meta = pd.read_parquet(args.data / "meta.parquet")
    labels = meta["primary_category"].fillna("").values
    print(f"Loaded {len(X):,} papers, dim {X.shape[1]}")

    if not args.no_normalize:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        X = (X / n).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    sample_idx = stratified_sample(labels, args.sample, rng)
    X_s = X[sample_idx]
    labels_s = labels[sample_idx]
    print(f"Faithfulness sample: {len(sample_idx):,} (stratified by primary_category)")

    rows = []
    for method in args.methods:
        tag = f"{method}_layout_n{args.n_neighbors}"
        Y_path = args.data / f"{tag}.npy"
        if not Y_path.exists():
            print(f"missing {Y_path} — skipping {method}", file=sys.stderr)
            continue
        Y = np.load(Y_path)
        print(f"\n=== {method} · {tag} ===")

        print(f"[faithfulness on {len(sample_idx):,} sample]")
        t0 = time.time()
        faith = faithfulness(X_s, Y[sample_idx], k=args.k)
        print(f"  {faith}  ({time.time()-t0:.1f}s)")

        print(f"[silhouette on {min(args.silhouette_sample, len(Y)):,} sample]")
        t0 = time.time()
        sil = silhouette(Y, labels, sample_size=args.silhouette_sample, rng_seed=args.seed)
        print(f"  silhouette={sil:.4f}  ({time.time()-t0:.1f}s)")

        print(f"[NMI via KMeans K={args.k_clusters} on full {len(Y):,}]")
        t0 = time.time()
        nmi = nmi_via_kmeans(Y, labels, args.k_clusters)
        print(f"  NMI={nmi:.4f}  ({time.time()-t0:.1f}s)")

        rows.append({
            "method": method,
            "n_neighbors": args.n_neighbors,
            **faith,
            "silhouette": sil,
            "nmi_kmeans": nmi,
        })

    if rows:
        df = pd.DataFrame(rows)
        out = args.data / f"eval_n{args.n_neighbors}.csv"
        df.to_csv(out, index=False)
        print(f"\n=== Summary ===")
        print(df.to_string(index=False))
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
