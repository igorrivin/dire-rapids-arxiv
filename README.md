# DiRe-Rapids on the arXiv corpus

Quick experiment: mean-pool BGE-small chunk embeddings to get one vector per
paper, then run [DiRe-Rapids](https://github.com/sashakolpakov/dire-rapids)
for a 2-D layout.

## Pipeline

```
local Postgres chunks (130M rows)
        │ pgvector AVG() grouped by paper_id
        ▼
data/embeddings.npy  (N=723,457, D=384, float32)
data/meta.parquet    (arxiv_id, title, primary_category, n_chunks, ...)
        │ DiReCuVS (cuVS k-NN + cuML PCA init + force-directed layout on GPU)
        ▼
data/dire_layout.npy (N, 2)
data/dire_layout.png colored by top-15 primary categories
```

## Environment

Everything runs in the `rapids-26.04` conda env. `dire-rapids` is installed
editable from `/home/igor/devel/dire-rapids/`. Only extras needed:

```bash
conda run -n rapids-26.04 python3 -m pip install psycopg2-binary python-dotenv
```

## Scripts

- `build_doc_embeddings.py` — stream chunks → mean-pool → save per-paper
  vectors + metadata. Keyset-paginated SQL with `work_mem = 512 MB` (the
  Docker postgres has a 64 MB `/dev/shm`, so we keep batches small and avoid
  big hash-agg spills).
- `run_dire.py` — load artifacts, optionally subsample, L2-normalize,
  `create_dire()` auto-selects `DiReCuVS`, plot colored by primary category.

## Performance

GH200 96 GB · WD SSD · arxiv-postgres Docker container.

| Step | N | Time |
|---|---:|---:|
| Build doc embeddings | 723,457 | ~17 min (720 papers/s) |
| DiRe fit_transform | 50,000 | 6.3 s |
| DiRe fit_transform | 723,457 | 20.3 s |

## Usage

```bash
# Build (one-off)
conda run -n rapids-26.04 python3 build_doc_embeddings.py --batch-size 1000

# Full corpus
conda run -n rapids-26.04 python3 run_dire.py --n-neighbors 32

# Sample for fast iteration
conda run -n rapids-26.04 python3 run_dire.py --sample 50000 --n-neighbors 32

# Interactive Plotly HTML (can be large; prefer --sample)
conda run -n rapids-26.04 python3 run_dire.py --sample 100000 --interactive
```

## Notes

- Source data: `chunks` table in `arxiv-postgres` (130 M rows, BGE-small-en-v1.5,
  384-d). `papers` table (local) provides `title` + `categories` metadata.
- Each chunk embedding is unit-normalized from BGE; mean-pooling gives
  norms in ~[0.83, 1.0]. We re-normalize before DiRe so cosine similarity
  between docs maps cleanly to Euclidean distance on the unit sphere.
- ~723 K of the 933 K papers on Supabase have chunks (the rest lack LaTeX
  or are mid-OCR).
