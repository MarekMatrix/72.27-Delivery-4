"""Analysis helpers for Exercise 2: outcome classification, overlap, plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .patterns import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results" / "hopfield"


# ---------- outcome ----------

def hamming(a, b):
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def classify(state, patterns, target_idx, tol=0.0):
    """Classify a final state against the stored patterns.

    Exact matches: 'correct', 'wrong' (another stored pattern) or 'inverse'.
    With tol > 0, a state within tol * N pixels of the nearest stored pattern
    (or its inverse) also counts: 'near' if that is the target, otherwise
    'wrong' or 'inverse'. Anything else is 'spurious'.
    Returns (outcome, index of the matched pattern or None).
    """
    state = np.asarray(state)
    n = state.size
    dist = [(hamming(state, p), "pattern", i) for i, p in enumerate(patterns)]
    dist += [(hamming(state, -p), "inverse", i) for i, p in enumerate(patterns)]
    d, kind, i = min(dist)
    if d > tol * n:
        return "spurious", None
    if kind == "inverse":
        return "inverse", i
    if i != target_idx:
        return "wrong", i
    return ("correct" if d == 0 else "near"), i


def overlap_matrix(patterns):
    """Normalized overlap m_ab = (xi_a . xi_b) / N, values in [-1, 1]."""
    X = np.asarray(patterns, dtype=float)
    return X @ X.T / X.shape[1]


# ---------- plotting ----------

def save(fig, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(REPO_ROOT)}")


def show_grid(ax, vec, side, title=None):
    ax.imshow(np.asarray(vec).reshape(side, side), cmap="gray_r", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def plot_patterns(patterns, labels, side, name, cols=None):
    n = len(patterns)
    cols = cols or min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p, lab in zip(axes, patterns, labels):
        ax.axis("on")
        show_grid(ax, p, side, lab)
    save(fig, name)


def plot_overlap(patterns, labels, name, title="Normalized overlap (xi_a . xi_b) / N"):
    M = overlap_matrix(patterns)
    fig, ax = plt.subplots(figsize=(0.5 * len(labels) + 2, 0.5 * len(labels) + 1.5))
    im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90 if len(labels) > 8 else 0)
    ax.set_yticks(range(len(labels)), labels)
    if len(labels) <= 10:
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, name)
    return M


def plot_recall(net, original, noisy, result, side, name, title=""):
    """Original | noisy input | final state | energy curve."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
    show_grid(axes[0], original, side, "Stored pattern")
    show_grid(axes[1], noisy, side, f"Input ({hamming(noisy, original)} px wrong)")
    show_grid(axes[2], result["state"], side,
              f"Output after {result['sweeps']} sweeps\n({hamming(result['state'], original)} px wrong)")
    axes[3].plot(result["energy_update"])
    axes[3].set_xlabel("update step")
    axes[3].set_ylabel("energy")
    axes[3].grid(alpha=0.3)
    fig.suptitle(title)
    save(fig, name)