#!/usr/bin/env python3
"""Interactive 3-D Plotly viewer for pre-computed layouts.

Consumes `{method}_layout_n{N}_d3.npy` + `meta.parquet`, writes a rotatable
scrollable HTML the user can open in a browser.

Full 723K points in a single HTML is ~200 MB and choky in-browser — default
is a 120K stratified sample with WebGL rendering, which is smooth on modern
hardware.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path(__file__).parent / "data", type=Path)
    ap.add_argument("--method", choices=["dire", "umap"], default="dire")
    ap.add_argument("--n-neighbors", type=int, default=16)
    ap.add_argument("--sample", type=int, default=120_000,
                    help="stratified subsample for the HTML")
    ap.add_argument("--top-k-cats", type=int, default=15)
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    layout_path = args.data / f"{args.method}_layout_n{args.n_neighbors}_d3.npy"
    meta_path = args.data / "meta.parquet"
    if not layout_path.exists() or not meta_path.exists():
        print(f"Missing {layout_path} or {meta_path}.", file=sys.stderr)
        sys.exit(1)

    Y = np.load(layout_path)
    meta = pd.read_parquet(meta_path)
    assert Y.shape[1] == 3, f"expected 3-d layout, got shape {Y.shape}"
    assert len(Y) == len(meta)
    print(f"Loaded {len(Y):,} papers, {Y.shape[1]}-d")

    labels = meta["primary_category"].fillna("").values
    counts = pd.Series(labels).value_counts()
    top_cats = [c for c in counts.index if c][:args.top_k_cats]

    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(Y):
        idx = stratified_sample(labels, args.sample, rng)
        Y = Y[idx]
        meta = meta.iloc[idx].reset_index(drop=True)
        labels = labels[idx]
        print(f"Subsampled to {len(Y):,}")

    color_cat = np.array([c if c in top_cats else "other" for c in labels])

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Install plotly: python3 -m pip install plotly", file=sys.stderr)
        sys.exit(1)

    # One trace per category so legend toggles work.
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
               "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]
    traces = []
    # 'other' first so it's drawn underneath
    other_mask = color_cat == "other"
    if other_mask.any():
        traces.append(go.Scatter3d(
            x=Y[other_mask, 0], y=Y[other_mask, 1], z=Y[other_mask, 2],
            mode="markers",
            marker=dict(size=args.point_size * 0.7, color="lightgray",
                        opacity=0.25),
            name=f"other ({other_mask.sum():,})",
            hovertext=[f"{aid}<br>{ttl[:80]}" for aid, ttl in
                       zip(meta.loc[other_mask, "arxiv_id"],
                           meta.loc[other_mask, "title"])],
            hoverinfo="text",
        ))

    for i, cat in enumerate(top_cats):
        m = color_cat == cat
        if not m.any():
            continue
        color = palette[i % len(palette)]
        traces.append(go.Scatter3d(
            x=Y[m, 0], y=Y[m, 1], z=Y[m, 2],
            mode="markers",
            marker=dict(size=args.point_size, color=color, opacity=0.8),
            name=f"{cat} ({m.sum():,})",
            hovertext=[f"{aid}<br>{ttl[:80]}" for aid, ttl in
                       zip(meta.loc[m, "arxiv_id"],
                           meta.loc[m, "title"])],
            hoverinfo="text",
        ))

    method_label = {"dire": "DiRe-Rapids", "umap": "cuML UMAP"}[args.method]
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{method_label} · {len(Y):,} arXiv papers (3-d, n_neighbors={args.n_neighbors})",
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
            aspectmode="data",
        ),
        width=1200, height=900,
        legend=dict(itemsizing="constant"),
    )
    out = args.data / f"{args.method}_layout_n{args.n_neighbors}_d3.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
