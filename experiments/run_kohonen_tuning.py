"""Sequential (one-factor-at-a-time) hyperparameter tuning for the Kohonen SOM.

Each stage sweeps ONE hyperparameter with all others frozen at the current best
values, then the winner is frozen and carried into the next stage.

Every candidate is scored with k-fold cross-validation: the SOM is trained on
(k-1)/k of the creators and evaluated on the held-out creators. Held-out
quantization error (QE) penalises memorisation, so larger maps no longer win
automatically. Held-out topographic error (TE) is recorded alongside it.

Run from the project root:
    python -m experiments.run_kohonen_tuning
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.tp4.data import load_creators, standardise
from src.tp4.kohonen.kohonen import Kohonen
from src.tp4.kohonen.metrics import (
    bmu_assignments,
    calculate_u_matrix,
    quantization_error,
    topographic_error,
)

# --------------------------------------------------
# Setup
# --------------------------------------------------

DATA_PATH = "data/creadores.csv"
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "kohonen_models"

SEEDS = [0, 1, 2]
K_FOLDS = 5

BASELINE = {
    "map_size": 6,
    "grid_type": "rectangular",
    "learning_rate": 0.5,
    "n_iterations": 1000,
    "initial_radius": 6,
}

# Parameters where "smaller is simpler": the winner is the smallest value within
# one std of the best mean held-out QE (see pick_winner).
PARSIMONY = {"map_size"}

# Parameters with no natural "simpler" value, where candidates whose held-out QE
# is within one std of the best are treated as tied and the lowest held-out
# topographic error wins. Caveat: TE is not strictly comparable across grid types
# (a hexagon has 6 neighbours, a rectangle 4).
TE_TIEBREAK = {"grid_type"}

# (hyperparameter, candidate values) in the order they are tuned.
# Map size goes first because every later stage depends on it.
# initial_radius must be >= 2 because time_constant = n_iterations / ln(radius).
STAGES = [
    ("map_size", [3, 4, 5, 6, 7, 8, 10, 12, 14, 15]),
    ("learning_rate", [0.1, 0.3, 0.5, 0.7, 0.9]),
    ("n_iterations", [500, 1000, 2000, 3000, 5000]),
    ("initial_radius", [2, 3, 4, 6, 8]),
    ("grid_type", ["rectangular", "hexagonal"]),
]


# --------------------------------------------------
# Core
# --------------------------------------------------

def build_and_train(X_train, config, seed):
    size = config["map_size"]
    time_constant = config["n_iterations"] / np.log(config["initial_radius"])

    som = Kohonen(
        X_train, size, size,
        config["n_iterations"], config["learning_rate"],
        config["initial_radius"], time_constant,
        np.random.default_rng(seed),
        grid_type=config["grid_type"],
    )
    som.create_map()
    return som


def evaluate_cv(df, config, seed):
    """K-fold CV on the RAW feature DataFrame. Returns one dict per fold with
    train and held-out metrics.

    Scaling is fitted on each fold's training creators only and then applied to
    the held-out ones, so held-out creators never influence the scaler."""
    folds = np.array_split(np.random.default_rng(seed).permutation(len(df)), K_FOLDS)
    rows = []
    for fold, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
        X_train, scaler = standardise(df.iloc[train_idx])
        X_test = scaler.transform(df.iloc[test_idx])
        som = build_and_train(X_train, config, seed)
        rows.append({
            "fold": fold,
            "train_qe": quantization_error(som, X_train),
            "heldout_qe": quantization_error(som, X_test),
            "heldout_te": topographic_error(som, X_test),
            "used_neurons": int(np.count_nonzero(bmu_assignments(som, X_train)[1])),
            "total_neurons": config["map_size"] ** 2,
        })
    return rows


def run_stage(df, current_best, param, values, stage_idx):
    rows = []
    for value in values:
        config = {**current_best, param: value}
        for seed in SEEDS:
            for result in evaluate_cv(df, config, seed):
                rows.append({"stage": stage_idx, "param": param, "value": value,
                             "seed": seed, **config, **result})
        part = pd.DataFrame(rows[-len(SEEDS) * K_FOLDS:])
        print(f"  {param}={value}: held-out QE={part['heldout_qe'].mean():.4f} "
              f"TE={part['heldout_te'].mean():.3f} "
              f"train QE={part['train_qe'].mean():.4f}")
    return pd.DataFrame(rows)


def pick_winner(stage_df, param):
    """Pick a candidate by held-out QE. Held-out TE is reported next to it, not used to rank.

    Default: lowest mean held-out QE. For parameters in PARSIMONY the winner is
    instead the SMALLEST value whose mean QE is within one std (of the best
    candidate's runs) of the best mean, so noise-level gains from a bigger map
    do not win. For parameters in TE_TIEBREAK, candidates within the same cutoff
    are tied on QE and the one with the lowest held-out TE wins."""
    summary = (
        stage_df.groupby("value")
        .agg(heldout_qe=("heldout_qe", "mean"), heldout_qe_std=("heldout_qe", "std"),
             heldout_te=("heldout_te", "mean"), train_qe=("train_qe", "mean"))
        .reset_index()
        .sort_values(["heldout_qe", "heldout_qe_std"])
    )
    best = summary.iloc[0]
    if param in PARSIMONY or param in TE_TIEBREAK:
        cutoff = best["heldout_qe"] + best["heldout_qe_std"]
        within = summary[summary["heldout_qe"] <= cutoff]
        print(f"  cutoff = {best['heldout_qe']:.4f} + {best['heldout_qe_std']:.4f} = {cutoff:.4f}; "
              f"within cutoff: {sorted(within['value'].tolist())}")
        if param in PARSIMONY:
            return within["value"].min(), summary
        return within.sort_values("heldout_te")["value"].iloc[0], summary
    return summary["value"].iloc[0], summary


def save_final_model(df, config):
    """Train the final config on ALL creators (best of the seeds by training QE)
    and save the arrays used by the notebook, plus the scaler parameters needed
    to project a new creator onto the same map."""
    X, scaler = standardise(df)
    soms = [(build_and_train(X, config, s), s) for s in SEEDS]
    som, seed = min(soms, key=lambda r: quantization_error(r[0], X))
    bmu_indices, bmu_counts = bmu_assignments(som, X)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODELS_DIR / "som_tuned.npz",
        weights=som.weights,
        bmu_counts=bmu_counts,
        bmu_indices=bmu_indices,
        u_matrix=calculate_u_matrix(som),
        feature_names=np.asarray(df.columns, dtype=str),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    metadata = {**config, "seed": seed,
                "train_quantization_error": quantization_error(som, X),
                "train_topographic_error": topographic_error(som, X),
                "used_neurons": int(np.count_nonzero(bmu_counts)),
                "total_neurons": config["map_size"] ** 2}
    with open(MODELS_DIR / "som_tuned.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def main():
    df = load_creators(DATA_PATH)
    print(f"{df.shape[0]} samples, {df.shape[1]} features, {K_FOLDS}-fold CV x {len(SEEDS)} seeds")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    current_best = dict(BASELINE)
    all_stages = []
    winners = []

    for stage_idx, (param, values) in enumerate(STAGES):
        print(f"\nStage {stage_idx}: {param} over {values} | frozen: "
              f"{ {k: v for k, v in current_best.items() if k != param} }")
        stage_df = run_stage(df, current_best, param, values, stage_idx)
        winner, summary = pick_winner(stage_df, param)
        # numpy scalars -> plain Python so the config stays JSON-serialisable
        winner = winner.item() if hasattr(winner, "item") else winner
        current_best[param] = winner
        all_stages.append(stage_df)
        winners.append({"stage": stage_idx, "param": param, "winner": winner})
        print(summary.round(4).to_string(index=False))
        print(f"-> {param} = {winner}")

    pd.concat(all_stages, ignore_index=True).to_csv(
        RESULTS_DIR / "kohonen_tuning.csv", index=False)
    with open(RESULTS_DIR / "kohonen_best_config.json", "w", encoding="utf-8") as f:
        json.dump({"config": current_best, "winners": winners}, f, indent=4)

    save_final_model(df, current_best)
    print("\nFinal config:", current_best)


if __name__ == "__main__":
    main()
