# TP4 — Unsupervised Learning

Sistemas de Inteligencia Artificial (72.27), ITBA.

## Setup

Requires Python 3.11+. Run every command from the project root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Or with [uv](https://docs.astral.sh/uv/): `uv sync --extra dev`, then prefix the commands below with `uv run`.

## Exercise 1 — Creators (PCA, Kohonen, Oja)

Train and tune the Kohonen map first; it writes `results/`, which the notebooks read:

```bash
python -m experiments.run_kohonen_tuning
```

Then open the notebooks with `jupyter lab notebooks/`:

| Notebook | Section |
|---|---|
| `exercise_1_1_eda_pca.ipynb` | 1.1 EDA and PCA |
| `kohonen_tuning.ipynb` | 1.2 Kohonen map |
| `exercise_1_3_oja.ipynb` | 1.3 Oja index |
| `exercise_1_4_decisions.ipynb` | 1.4 Campaign decisions |

## Exercise 2 — Hopfield

```bash
python -m tp4.hopfield.run --part all      # or --part letters / --part qr
```

Figures are saved to `results/hopfield/`. Run `python -m tp4.hopfield.run --help` for the other options.
