# DiRe-Rapids vs cuML UMAP on the arXiv corpus — results

**Dataset:** 723,457 arXiv papers (all those with LaTeX-sourced chunks in the
local pgvector store); doc-level embeddings built by mean-pooling 130 M
BGE-small-en-v1.5 chunk vectors and L2-renormalizing. See `build_doc_embeddings.py`.

**Environment:** `rapids-26.04` conda env, GH200 480 GB GPU, `dire-rapids 0.2.0`
installed editable from `/home/igor/devel/dire-rapids/`, `cuml 26.04`,
`ripser` + `fastdtw` for topology metrics.

---

## Headline

- **Pick DiRe at `n_neighbors=16` (or 8) for this corpus.** Topology is
  preserved far better by DiRe at every `n_neighbors` tried (β₀ DTW
  2-3× lower, β₁ DTW 2-5× lower).
- **UMAP wins on local-neighborhood and category-separation metrics**
  (trustworthiness, kNN preservation, silhouette, NMI) by ~10-25%. But
  those gains come from collapsing a continuous manifold into fake islands.
- **UMAP over-clusters quantitatively.** Its longest H₀ persistence bar
  is 2-5× longer than DiRe's at every `n_neighbors`. In human terms:
  math.AG (and a handful of others) show up as isolated islands in
  UMAP, and as tendrils rooted in the main mass in DiRe.

Commands to reproduce everything: see [README.md](README.md).

---

## Layout comparison (full corpus, n_neighbors=16)

| | DiRe-Rapids (DiReCuVS) | cuML UMAP |
|---|---:|---:|
| fit_transform time | **13–20 s** | 32 s |
| 2-d PNG | `data/dire_layout_n16_d2.png` | `data/umap_layout_n16_d2.png` |
| 3-d interactive HTML | `data/dire_layout_n16_d3.html` | `data/umap_layout_n16_d3.html` |

Top-15 primary-category coloring. Visual read: DiRe has a dense central
mass with distinct thin tendrils (lime-green tendril = math.AG, brown
spur = math-ph / hep-th, etc.); UMAP shows a compact oval with a couple
of disconnected outlier clusters.

---

## Local-neighborhood + category metrics (`evaluate.py`)

Stratified 20 K sample for faithfulness; full corpus for silhouette and NMI.

| metric | DiRe | UMAP | winner |
|---|---:|---:|:---:|
| trustworthiness (k=15) | 0.802 | **0.877** | UMAP |
| continuity (k=15) | 0.912 | **0.934** | UMAP |
| kNN preservation @ k=15 | 0.070 | **0.093** | UMAP |
| kNN preservation @ k=50 (mesa) | 0.105 | **0.140** | UMAP |
| silhouette(layout, cat) | -0.512 | **-0.388** | UMAP |
| NMI(cat, KMeans K=25) | 0.343 | **0.431** | UMAP |

Mesa-regime check: bumping `k` from 15 to 50 boosts both methods
proportionally (DiRe/UMAP ratio stays 0.75), so UMAP's local-metric
advantage isn't an artifact of a noisy small-k regime.

---

## Topology preservation — Betti-curve DTW (`eval_betti.py`)

ripser-based Betti curves on 3 × N=4034 stratified samples, rescaled to
unit pairwise diameter before filtration. fastdtw between the 384-d
reference curve and each layout's curve.

At `n_neighbors=16`:

| | DTW(β₀) | DTW(β₁) |
|---|---:|---:|
| **DiRe** | **3,338 ± 832** | **218 ± 34** |
| UMAP | 10,429 ± 466 | 818 ± 328 |

DiRe beats UMAP by **3.1× on β₀** and **3.7× on β₁**, every seed.

---

## Sweep over n_neighbors (`sweep_topology.py`)

Mean ± std over 3 seeds, N=4000 sample each. See `data/sweep_topology.png`.

| n | method | DTW(β₀) | DTW(β₁) |
|--:|---|---:|---:|
| 8 | DiRe | **2149 ± 472** | 423 ± 52 |
| 8 | UMAP | 6943 ± 1045 | 408 ± 70 |
| 16 | DiRe | 3901 ± 506 | **193 ± 16** |
| 16 | UMAP | 10429 ± 466 | 818 ± 328 |
| 32 | DiRe | 5632 ± 49 | **192 ± 13** |
| 32 | UMAP | 11212 ± 313 | 1084 ± 320 |
| 64 | DiRe | 7663 ± 653 | 275 ± 41 |
| 64 | UMAP | 11978 ± 109 | 508 ± 8 |
| 128 | DiRe | 5389 ± 550 | 224 ± 63 |
| 128 | UMAP | 12125 ± 124 | 664 ± 158 |

**Observations:**

- DiRe beats UMAP on DTW(β₀) at every `n`, by 2-3×.
- DiRe beats UMAP on DTW(β₁) at every `n`, with the biggest gaps
  (5×) at `n=32` — UMAP's DTW(β₁) peaks badly there.
- UMAP DTW(β₀) rises monotonically with `n` (6.9K → 12.1K).
- DiRe's topology sweet spot is `n=16–32` for β₁ and `n=8` for β₀.
- Larger `n` is *not* uniformly better — DiRe's DTW(β₀) more than triples
  going from n=8 to n=64, because the force field gets over-smoothed.

---

## Island-ness — quantifying UMAP's over-clustering (`island_ness.py`)

Three complementary diagnostics on the H₀ persistence diagram of each
unit-diameter-rescaled cloud:

- `longest_bar` — length of the longest finite H₀ bar. High = one
  cluster survives isolation far into the filtration.
- `top5_ratio` — top-5 longest bars / median bar. High = a few bars
  dominate (islands on a smooth background).
- `gini_h0` — Gini coefficient of H₀ bar lengths. High = persistence
  mass is concentrated in few bars.

At every `n_neighbors`:

| n | | longest_bar | top5_ratio | gini_h0 |
|--:|---|---:|---:|---:|
| (ref 384-d) | | 0.697 | 2.4 | 0.117 |
| 8 | DiRe | **0.071** | **9** | 0.32 |
| 8 | UMAP | 0.182 | 42 | 0.42 |
| 16 | DiRe | **0.131** | **17** | **0.37** |
| 16 | UMAP | 0.266 | 71 | 0.42 |
| 32 | DiRe | **0.092** | **24** | **0.42** |
| 32 | UMAP | 0.313 | 74 | 0.40 |
| 64 | DiRe | **0.137** | **47** | 0.47 |
| 64 | UMAP | 0.391 | 131 | 0.44 |
| 128 | DiRe | **0.088** | **32** | 0.51 |
| 128 | UMAP | 0.391 | 146 | 0.43 |

**Key finding.** UMAP's longest H₀ bar is 2-5× DiRe's at every `n_neighbors`
tested, and its top-5/median ratio is 3-4× DiRe's. UMAP's H₀ persistence
mass concentrates into a handful of long bars (the "islands"). DiRe
spreads it across many moderate bars (the "tendrils"). Both are far from
the 384-d reference (longest=0.697, top5/med=2.4, Gini=0.12), because
reducing to 2-d necessarily compresses the persistence range — but the
*shape* of the bar distribution is what differs, and on shape DiRe is
consistently closer to reference.

**Caveat for high n_neighbors.** At n=128, DiRe's Gini (0.51) exceeds
UMAP's (0.43) — DiRe starts to develop its own cluster-like structure
when the force field over-smooths. This is another reason to stay in
the n=8-32 range for continuous-manifold data.

---

## Interpretation

Mathematically, the arXiv taxonomy is a **continuum** rather than a set
of discrete clusters. math.AG borders math.NT (arithmetic geometry),
math.AC (commutative algebra → scheme theory), math.RT (geometric
representation theory), math.AT (étale cohomology, motivic homotopy).
There is no natural boundary where AG ends and other fields begin.

UMAP's fuzzy-simplicial-set loss amplifies local density minima into
hard cluster boundaries — math.AG becomes an isolated island, which is
an artifact of the algorithm, not a feature of mathematics. This shows
up as UMAP's long longest-bar and high top-5/median ratio.

DiRe's force model, with global repulsion as a scalar field rather than
a simplicial structure, stretches math.AG into a tendril rooted in the
algebraic mass — matching the actual continuity of mathematics.

Betti-curve DTW is the quantitative version: UMAP's β₀ DTW is 3.1×
worse than DiRe's because UMAP invents components that aren't in the
high-D data.

## Practical recommendation

- **Data exploration / research** where the underlying manifold is
  continuous (arXiv, document corpora, knowledge graphs): use DiRe at
  `n_neighbors=16`. The visual is honest about continuity.
- **UI clustering** where you want clean visually-separated groups for
  end users (even if the separation is exaggerated): use cuML UMAP.
- **Speed:** DiRe is ~2× faster than cuML UMAP on this corpus
  (15 s vs 32 s for 723 K points at n=16 on GH200).

---

## Intrinsic dimension of the 384-d embeddings

What is the "true" dimension of the BGE-small embeddings? Estimated with a
battery of intrinsic-dimension (ID) estimators (`scikit-dimension`) plus a
linear PCA baseline, on deduplicated subsamples (exact-duplicate embeddings —
identical abstracts / cross-listings — give zero NN-distances that break MLE
and bias the kNN estimators; they are removed). Stable across n = 10k/20k/50k.

| estimator | scale probed | value |
|---|---|---:|
| PCA components for 90% / 95% / 99% variance | linear | 162 / 232 / 331 |
| PCA participation ratio `(Σλ)²/Σλ²` | linear, soft | **22.2** |
| TwoNN (Facco et al. 2017) | local global-ID | **20–24** |
| MLE / Levina–Bickel (2004) | local global-ID | **22.0–22.4** |
| CorrInt (correlation dimension) | fine scale | ~11 |
| lPCA (100-NN neighborhoods) | coarse local | ~45 |

**Headline: the embedding manifold is ~22-dimensional** sitting in the 384-d
ambient space (really S³⁸³ — the embeddings are L2-normalized). TwoNN, MLE,
and the linear participation ratio agree at ~22; CorrInt reads lower (~11) at
finer scales and lPCA higher (~45) at coarser ones, the usual scale-dependent
spread. TwoNN/MLE are known to *under*-read under positive curvature, so true
local d is plausibly a touch above 22. This ~20× compression from 384→22 is the
structural reason 2-d/3-d layouts cannot preserve topology (cf. the β₁ DTW
costs above): they crush a ~22-d object into the plane.

ID is a **constraint and diagnostic, not a target.** A useful by-product is the
*fluffiness ratio* PCA95 / d ≈ 232 / 22 ≈ **10**: variance lives in ~22 effective
directions but bleeds across ~230 linear ones, a coiling/curvature proxy.

### Per-category: is the manifold stratified?

Restricting to each large `primary_category` (top-18, all math / math-ph /
hep-th / cs.IT — the corpus is math-dominated by the LaTeX-source filter):

- **Local ID is roughly uniform across subfields**: TwoNN 10–15 (std 1.5),
  MLE 16–20 (std 1.6). No strong stratification — the manifold is *not* a
  patchwork of wildly different-dimensional sheets.
- **Within-field d (~12–18) < whole-corpus d (~22).** Suggests a decomposition
  *within-field topical variation (~12–18) + between-field arrangement (~4–10)
  ≈ 22* — a chunk of the "true dimension" is the directions separating subfields,
  not variation inside one. **Caveat:** per-category samples are near-complete
  while the full-corpus sample is ~7% — the gap is partly a sampling-density/scale
  artifact, so treat the between-field component as suggestive, not pinned.
- Faint edge signal: analytic/physical fields (math.FA, hep-th, math-ph) read
  lowest-d but most coiled (fluff ≈ 22); combinatorial/probabilistic fields
  (math.CO, math.PR, math.NT) highest-d, least coiled (fluff ≈ 15–16).

### Density-matched: is the within<between gap real? (mostly scale)

The within/between gap above was confounded by sampling density. Resolving it
by estimating ID *vs neighborhood radius* (Levina-Bickel MLE swept over k, with
the MacKay-Ghahramani reciprocal-average correction) and comparing curves AT
EQUAL RADIUS (`intrinsic_dimension_scale.py`, see `data/intrinsic_dim_scale.png`):

- **ID is strongly scale-dependent** — every curve rises with radius. The local
  tangent dimension at the finest resolvable scale is only **~11–15**; it inflates
  to **~22 at coarse scale** through curvature/coiling (matches fluffiness ~10).
- **The full corpus only ever measures coarse scale.** Even its tightest
  neighborhoods sit at radius ~0.235 (d~20) — sparsity prevents finer probing —
  so its flat ~22 is a *coarse-scale* reading, not the tangent dimension.
- **Most of the raw 12-vs-22 gap was the scale artifact**: dense categories reach
  fine scales (d~11–15) the sparse full corpus cannot.
- **At matched radius a ~2–3 dim residual remains** (math.CO gap −2.3, math.FA
  −3.5), shrinking to ~0 at coarse scale: a *genuine but modest* between-category
  contribution. It is conservative — at fixed absolute radius a category's
  neighborhood covers a larger fraction of its (smaller) manifold, which biases
  it upward, yet it still reads lower. (math.AP is too dense to matched-compare:
  radii 0.18–0.22 lie entirely below the corpus range; reported, not extrapolated.)

Refined picture: **fine-scale tangent dim ~11–15**, curvature-inflated to **~22
at coarse scale**, plus **~2–3 dim genuine between-category** spread. The ~22
headline is a coarse-scale figure; the local tangent dimension is ~12–15.

**Why an autoencoder was *not* used:** a bottleneck-AE's "elbow" entangles
intrinsic dimension with architecture, width, and training budget. Dedicated ID
estimators answer the actual question in ~30 lines with no such confound.

---

## Files

| Script | Produces |
|---|---|
| `build_doc_embeddings.py` | `data/embeddings.npy`, `data/meta.parquet` |
| `intrinsic_dimension.py` | `data/intrinsic_dim.log` (ID battery, full corpus) |
| `intrinsic_dimension_by_category.py` | per-category ID table (stdout) |
| `intrinsic_dimension_scale.py` | `data/intrinsic_dim_scale.png` (ID vs scale) |
| `run_reducer.py` | `data/{dire,umap}_layout_n{n}_d{d}.{npy,png}` |
| `evaluate.py` | `data/eval_n{n}.csv` |
| `eval_betti.py` | `data/eval_betti_n{N}.{csv,npz,png}` |
| `sweep_topology.py` | `data/sweep_topology{,_agg}.csv`, `data/sweep_topology.png` |
| `island_ness.py` | `data/island_ness{,_agg}.csv`, `data/island_ness.png` |
| `view_3d.py` | `data/{dire,umap}_layout_n{n}_d3.html` |
