#!/usr/bin/env python3
"""Build per-paper document embeddings by mean-pooling chunk embeddings.

Source: local PostgreSQL `chunks` + `papers` tables.
Output:
    data/paper_ids.npy        int32, shape (N,) — DB paper_id
    data/arxiv_ids.npy        object (unicode), shape (N,) — arXiv ID
    data/embeddings.npy       float32, shape (N, 384) — mean-pooled doc vectors
    data/chunk_counts.npy     int32,  shape (N,) — how many chunks were pooled
    data/meta.parquet         title, categories, primary_category keyed by paper_id

Aggregation uses pgvector's native AVG(vector), streamed in keyset-paginated
batches over paper_id so we don't buffer the full result server-side. Paper
metadata is JOINed from the local `papers` table in the same query.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, "/home/igor/devel/heretic")
from shared.db_config import LOCAL_DB_CONFIG  # noqa: E402


def fetch_batch(conn, after_paper_id: int, batch_size: int):
    """Fetch one batch of rows, joining chunks to papers for metadata.

    Returns tuples: (paper_id, arxiv_id, title, categories_json, avg_embedding_text, n_chunks).
    Keyset pagination on paper_id for stable streaming.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.paper_id,
                   MAX(c.arxiv_id)              AS arxiv_id,
                   MAX(p.title)                 AS title,
                   MAX(p.categories::text)      AS categories,
                   AVG(c.embedding)::text       AS avg_embedding,
                   COUNT(*)                     AS n_chunks
            FROM chunks c
            LEFT JOIN papers p ON p.id = c.paper_id
            WHERE c.paper_id > %s
            GROUP BY c.paper_id
            ORDER BY c.paper_id
            LIMIT %s
            """,
            (after_paper_id, batch_size),
        )
        return cur.fetchall()


def _primary_category(cats_json: str | None) -> str:
    """Extract the first (primary) category from a JSONB text array."""
    if not cats_json:
        return ""
    try:
        cats = json.loads(cats_json)
    except json.JSONDecodeError:
        return ""
    if isinstance(cats, list) and cats:
        return str(cats[0])
    if isinstance(cats, str):
        # Sometimes stored as a space-separated string
        return cats.split()[0] if cats else ""
    return ""


def parse_pgvector(text: str) -> np.ndarray:
    # pgvector text format: "[0.123,-0.456,...]"
    return np.fromstring(text.strip("[]"), sep=",", dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path(__file__).parent / "data", type=Path)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="stop after N papers (0=all)")
    ap.add_argument("--min-chunks", type=int, default=1,
                    help="skip papers with fewer than N chunks")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(**LOCAL_DB_CONFIG)
    conn.set_session(readonly=True)
    with conn.cursor() as cur:
        # Bigger work_mem keeps hash aggregation in-process instead of
        # spilling into parallel-worker shmem segments (Docker default 64MB
        # /dev/shm is too small). Small batch sizes also help.
        cur.execute("SET work_mem = '512MB';")

    paper_ids: list[int] = []
    arxiv_ids: list[str] = []
    titles: list[str] = []
    primary_cats: list[str] = []
    categories_raw: list[str] = []
    chunk_counts: list[int] = []
    embeddings: list[np.ndarray] = []

    after = -1
    total = 0
    t0 = time.time()

    while True:
        rows = fetch_batch(conn, after, args.batch_size)
        if not rows:
            break
        for paper_id, arxiv_id, title, cats_json, avg_text, n_chunks in rows:
            if n_chunks < args.min_chunks:
                after = paper_id
                continue
            paper_ids.append(paper_id)
            arxiv_ids.append(arxiv_id or "")
            titles.append(title or "")
            categories_raw.append(cats_json or "")
            primary_cats.append(_primary_category(cats_json))
            chunk_counts.append(n_chunks)
            embeddings.append(parse_pgvector(avg_text))
            after = paper_id
            total += 1
        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f"[{elapsed:7.1f}s] {total:>7d} papers  ({rate:6.0f}/s)  "
              f"last paper_id={after}",
              flush=True)
        if args.limit and total >= args.limit:
            break

    conn.close()

    if not paper_ids:
        print("No rows fetched. Exiting.", file=sys.stderr)
        sys.exit(1)

    emb = np.stack(embeddings)
    np.save(args.out / "paper_ids.npy", np.asarray(paper_ids, dtype=np.int32))
    np.save(args.out / "arxiv_ids.npy", np.asarray(arxiv_ids, dtype=object))
    np.save(args.out / "embeddings.npy", emb)
    np.save(args.out / "chunk_counts.npy", np.asarray(chunk_counts, dtype=np.int32))

    meta = pd.DataFrame({
        "paper_id": paper_ids,
        "arxiv_id": arxiv_ids,
        "title": titles,
        "primary_category": primary_cats,
        "categories": categories_raw,
        "n_chunks": chunk_counts,
    })
    meta.to_parquet(args.out / "meta.parquet", index=False)

    elapsed = time.time() - t0
    print(f"\nDone. {total} papers in {elapsed:.1f}s ({total/elapsed:.0f}/s)")
    print(f"Embeddings shape: {emb.shape}, dtype {emb.dtype}")
    print(f"Mean chunks/paper: {float(np.mean(chunk_counts)):.1f}")
    top_cats = meta["primary_category"].value_counts().head(10)
    print(f"Top 10 primary categories:\n{top_cats.to_string()}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
