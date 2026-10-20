# TP4 — Unsupervised Learning

Sistemas de Inteligencia Artificial (72.27), ITBA. Assignment statement: `docs/SIATP42026.pdf`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp -R docs/datos/. data/   # data/ is gitignored
pytest
```

## Layout

```
src/tp4/
  data.py            shared: load creadores.csv, feature selection, scaling
  linear/            Area A: eda.py, pca.py, oja.py
  kohonen/           Area B: som.py, metrics.py
  hopfield/          Area C: network.py, patterns.py
notebooks/           exploration only, never imported by src/
data/                raw data (gitignored)
results/             generated figures/outputs (gitignored)
docs/                statement, report source, slides
tests/               one test file per area + shared data
```

## Proposed split

The statement has two exercises and five deliverable blocks. Exercise 1 parts 1–3 share the same
data and preprocessing; part 4 (campaign decisions) consumes all of them. Exercise 2 (Hopfield) is
fully independent. That gives three areas with a single, small shared interface.

| Area | Statement sections | Packages | Depends on |
|------|--------------------|----------|------------|
| **A — Exploration & linear methods** | 1.1 EDA + PCA, 1.3 Oja index | `tp4.data`, `tp4.linear` | — |
| **B — Kohonen map** | 1.2 SOM, U-matrix, hyperparameter search | `tp4.kohonen` | `tp4.data` from A |
| **C — Associative memory** | 2.2 letters, 2.1 QR codes | `tp4.hopfield` | — |

Section 1.4 (campaign decisions) is **shared**: it needs A's components and index and B's grid
regions side by side. Suggested: whoever owns C coordinates 1.4, since C has no stake in either
method's results and has the lightest dependency load.

### Area A — Exploration & linear methods

Owns `tp4.data`, which is the contract B builds on, so it should land first.

- EDA: distributions, outliers, scale differences, correlations.
- Decide the feature set and the scaling, and justify both in the report.
- PCA: explained variance, loadings, PC1/PC2 biplot, whether to reduce dimensionality.
- Oja's rule: converge to the first component, compare the resulting index with the PC1
  projection, discuss what the index means and whether "pick the highest score" is valid.

Questions to settle early:
- Which columns are deterministic functions of other columns? What does keeping them do to PCA?
- Which columns are compositional (percentages summing to 100)?
- Is `creator_id` / `name` a feature?

### Area B — Kohonen map

- SOM over the feature matrix from `tp4.data`.
- Hyperparameter sweep (grid size, initial radius, learning rate, epochs, initialisation) with a
  stated evaluation metric for choosing the final model.
- Hit map (creators per neuron), U-matrix, per-feature component planes.
- Interpret grid regions and compare them with the PCA biplot from A.

B does not have to wait for A: the SOM can be developed and tested on a toy dataset with known
clusters, then switched to `tp4.data` once the feature set is fixed.

### Area C — Associative memory

- Hopfield network: storage rule, asynchronous update, energy per step, stopping criterion.
- Letters (5×5, ±1): choose 4 letters, orthogonality analysis, capacity, a spurious state from a
  heavily noisy input. The statement recommends doing this first.
- QR codes (37×37 from `data/hopfield/`): binarise, store all five, add noise at several levels,
  report recovery rate, iterations and energy; explain limitations of the use case.

## Shared interface

`tp4.data` is the only module two areas both depend on. Agree on, in words, before writing it:

1. What it returns: a feature matrix, the matching column names, and the creator identifiers so
   results can be mapped back to names.
2. Which preprocessing is applied inside it (column drop, scaling) versus left to the caller.
3. That scaling parameters are kept, so a new month of data or a new creator can be projected
   onto the same PCA/SOM (needed for the maintenance question in 1.4).

Changing this contract after B starts using it should be a PR both A and B review.

## Workflow

- Branches: `feat/<area>-<topic>`, e.g. `feat/kohonen-umatrix`. Rebase onto `main` before merging.
- Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`; imperative, one change each.
- Figures for the report are produced by committed code writing to `results/`; PNGs are not committed.
