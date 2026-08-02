# Public release: paper-level arXiv embeddings (v1)

## Goal

Release the 723,457 × 384 paper-level pooled embedding corpus + reproducibility
artifacts for the PNAS paper, satisfying PNAS's data-availability requirement.

- **Canonical citation:** Zenodo DOI (reserve before submission, publish on accept)
- **Convenience mirror:** Hugging Face Datasets
- **License:** CC0 for arXiv-derived metadata, CC-BY-4.0 for our numerical artifacts
- **Out of scope for v1:** chunk-level embeddings, raw text/PDFs, full sweep of pooling strategies

## Decisions (2026-04-27)

- (a) Ship the metadata we already have: `arxiv_id, title, primary_category, categories, n_chunks`. No DB enrichment for v1.
- HF namespace: TBD (kolpakov-rivin org doesn't exist on igriv account; pending user choice between igriv/, new org, or other)
- Layout sweep: include all 12 layouts (n=8,16,32,64,128 × {DiRe, UMAP} for d=2; plus d=3 at n=16)
- Indices for the Betti / island-ness samples: regenerable from seeds; not preserved as separate artifacts in v1
- Embedding dtype: float32, fixed_size_list[float32, 384]

## Future work (TODO, not v1)

- Test whether chunk-level embeddings cluster around their paper-level pool (i.e. is the per-paper mean a good summary?)
- Test whether random per-document chunks give the same 2-d visualization as the pooled embedding (sensitivity check)

## Release tree

```
release/
  README.md                              # HF dataset card with YAML frontmatter
  LICENSE                                # CC-BY-4.0
  CITATION.cff
  load_example.py
  metadata.parquet                       # arxiv_id + descriptive fields
  embeddings/
    part-00000-of-00004.parquet          # arxiv_id + fixed_size_list<float32>[384]
    part-00001-of-00004.parquet
    part-00002-of-00004.parquet
    part-00003-of-00004.parquet
  layouts/
    layouts_2d.parquet                   # long: arxiv_id, method, n_neighbors, x, y
    layouts_3d.parquet                   # long: arxiv_id, method, n_neighbors, x, y, z
  provenance/
    embedding_config.json
    software_versions.json
    file_hashes.json
```

## Steps

1. [x] Build `release/build_release.py` script — deterministic regeneration from `data/`.
2. [x] Write metadata parquet (drop `paper_id` internal id).
3. [x] Write sharded embedding parquet (8 shards, fixed_size_list[float32, 384]).
4. [x] Write layouts parquets (long format, all sweep entries).
5. [x] Write provenance JSONs (model + versions; no arxiv harvest snapshot for v1).
6. [x] Compute SHA256 file hashes, write `file_hashes.json`.
7. [x] Write README.md (HF dataset card with frontmatter).
8. [x] Write CITATION.cff (with Rivin ORCID 0000-0001-9302-2169; Kolpakov ORCID still TODO).
9. [x] Write LICENSE (CC-BY-4.0 with note on metadata source / arXiv terms).
10. [x] Write `load_example.py`.
11. [x] Smoke-test: `load_dataset(...)` from local path, verify shape + dtype + arxiv_id roundtrip.
12. [x] Push to HF: https://huggingface.co/datasets/igriv/dire-arxiv-bge-small-embeddings (2026-04-27, commit 5e24908).
13. [x] Remote smoke test: `load_dataset(..., streaming=True)` returns first row correctly.
14. [x] Zenodo deposit created via REST API from this machine (igor's PAT). 19 files uploaded as flat keys (Zenodo doesn't support subdirs). Draft at https://zenodo.org/deposit/19837857.
   - Reserved version DOI: `10.5281/zenodo.19837857`
   - Reserved concept DOI: `10.5281/zenodo.19837856` (use this in publications — always points to latest)
15. [x] DOI baked into `release/README.md` and `release/CITATION.cff`; re-pushed to both HF and Zenodo.
16. [ ] **(User cleanup) Revoke the Zenodo PAT** that was shared in chat — Settings → Applications → Personal access tokens.
17. [ ] Get Sasha's ORCID, add to CITATION.cff (under both authors entries) and README, re-push to HF + Zenodo.
18. [ ] After PNAS accepts: click Publish on the Zenodo draft (files become immutable, concept+version DOIs go live).
19. [ ] (Optional) Link the GitHub repo to Zenodo for separate code DOI (https://zenodo.org/account/settings/github/).
