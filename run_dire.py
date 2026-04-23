#!/usr/bin/env python3
"""Run DiRe-Rapids on mean-pooled arXiv doc embeddings.

Loads artifacts produced by `build_doc_embeddings.py`:
    data/embeddings.npy   (N, 384) float32
    data/meta.parquet     has columns primary_category, arxiv_id, title, ...

Saves:
    data/dire_layout.npy           (N, n_components) float32
    data/dire_layout.png           scatter colored by top-K primary categories
    data/dire_layout.html          interactive plotly scatter (if --interactive)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path(__file__).parent / "data", type=Path)
    ap.add_argument("--n-neighbors", type=int, default=32)
    ap.add_argument("--n-components", type=int, default=2)
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip L2-normalization before running DiRe")
    ap.add_argument("--sample", type=int, default=0,
                    help="random subsample to N points (0 = all)")
    ap.add_argument("--top-k-cats", type=int, default=15,
                    help="number of primary categories to color distinctly")
    ap.add_argument("--interactive", action="store_true",
                    help="also write an interactive Plotly HTML")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load
    emb_path = args.data / "embeddings.npy"
    meta_path = args.data / "meta.parquet"
    if not emb_path.exists() or not meta_path.exists():
        print(f"Missing {emb_path} or {meta_path}. Run build_doc_embeddings.py first.",
              file=sys.stderr)
        sys.exit(1)

    X = np.load(emb_path)
    meta = pd.read_parquet(meta_path)
    assert len(X) == len(meta), f"embedding/meta length mismatch: {len(X)} vs {len(meta)}"
    print(f"Loaded {len(X):,} papers, dim {X.shape[1]}")

    # Optional subsample
    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(X):
        idx = rng.choice(len(X), size=args.sample, replace=False)
        idx.sort()
        X = X[idx]
        meta = meta.iloc[idx].reset_index(drop=True)
        print(f"Subsampled to {len(X):,} papers")

    # Normalize (mean-pooled BGE unit vectors are ~0.85-1.0 norm; cosine = euclidean on unit sphere)
    if not args.no_normalize:
        X = normalize_rows(X)
        print("L2-normalized embeddings")

    # Run DiRe-Rapids
    from dire_rapids import create_dire
    reducer = create_dire(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        verbose=True,
    )
    print(f"Reducer: {type(reducer).__name__}")
    t0 = time.time()
    Y = reducer.fit_transform(X)
    elapsed = time.time() - t0
    print(f"DiRe fit_transform: {elapsed:.1f}s for {len(X):,} points")

    out_layout = args.data / "dire_layout.npy"
    np.save(out_layout, Y.astype(np.float32))
    print(f"Saved layout → {out_layout}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = meta["primary_category"].fillna("").values
    counts = pd.Series(cats).value_counts()
    # Drop empty-string from top-K
    top_cats = [c for c in counts.index if c][:args.top_k_cats]
    color_map = {c: i for i, c in enumerate(top_cats)}
    color_idx = np.array([color_map.get(c, -1) for c in cats])  # -1 = "other"

    fig, ax = plt.subplots(figsize=(14, 12), dpi=120)
    cmap = plt.get_cmap("tab20", len(top_cats))

    # "Other" (gray) first, so top categories draw on top
    other_mask = color_idx == -1
    if other_mask.any():
        ax.scatter(Y[other_mask, 0], Y[other_mask, 1], s=1.5, c="lightgray",
                   alpha=0.25, linewidths=0, label=f"other ({other_mask.sum():,})")

    for i, cat in enumerate(top_cats):
        m = color_idx == i
        if not m.any():
            continue
        ax.scatter(Y[m, 0], Y[m, 1], s=2.5, color=cmap(i), alpha=0.7, linewidths=0,
                   label=f"{cat} ({m.sum():,})")

    ax.set_title(f"DiRe-Rapids layout: {len(X):,} arXiv papers, "
                 f"mean-pooled BGE-small-384\n"
                 f"n_neighbors={args.n_neighbors}, n_components={args.n_components}, "
                 f"time={elapsed:.1f}s")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, markerscale=3, framealpha=0.9)
    out_png = args.data / "dire_layout.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved plot → {out_png}")

    # Optional interactive Plotly
    if args.interactive:
        try:
            import plotly.express as px
            plot_df = meta.copy()
            plot_df["x"] = Y[:, 0]
            plot_df["y"] = Y[:, 1]
            plot_df["color_cat"] = [c if c in color_map else "other" for c in cats]
            fig = px.scatter(
                plot_df, x="x", y="y", color="color_cat",
                hover_data=["arxiv_id", "title", "primary_category", "n_chunks"],
                title=f"DiRe-Rapids · {len(X):,} arXiv papers",
                render_mode="webgl",
            )
            fig.update_traces(marker=dict(size=3, opacity=0.7))
            fig.update_layout(height=900)
            out_html = args.data / "dire_layout.html"
            fig.write_html(str(out_html))
            print(f"Saved interactive HTML → {out_html}")
        except ImportError as e:
            print(f"Plotly not available ({e}); skipping interactive plot.")


if __name__ == "__main__":
    main()
