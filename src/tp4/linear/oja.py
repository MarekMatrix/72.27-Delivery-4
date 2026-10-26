"""Oja's learning rule for the first principal component (creator index)."""
import numpy as np


def fit_oja(X, learning_rate=0.001, epochs=2000, seed=42):
    """Learn weights from standardised rows; return weights and training history."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.size or not np.isfinite(X).all():
        raise ValueError('X must be a non-empty finite 2D array')
    if learning_rate <= 0 or epochs < 1:
        raise ValueError('learning_rate and epochs must be positive')
    rng = np.random.default_rng(seed)
    w = rng.normal(size=X.shape[1])
    w /= np.linalg.norm(w)
    history = []
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            x = X[i]
            y = x @ w
            w += learning_rate * y * (x - y * w)
        if not np.isfinite(w).all() or np.linalg.norm(w) == 0:
            raise ValueError('Training diverged; try a smaller learning rate')
        history.append(np.mean((X @ w) ** 2))
    return w / np.linalg.norm(w), np.asarray(history)


def creator_index(X, weights):
    return np.asarray(X) @ np.asarray(weights)
