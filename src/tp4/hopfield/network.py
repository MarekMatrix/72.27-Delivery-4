"""Hopfield network: storage, asynchronous update, energy."""

import numpy as np

# Define standard 5x5 alphabet letters using 1 (active) and -1 (inactive)
ALPHABET_5X5 = {
    "J": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [ 1, -1, -1,  1, -1],
        [-1,  1,  1,  1, -1]
    ]),
    "X": np.array([
        [ 1, -1, -1, -1,  1],
        [-1,  1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
        [-1,  1, -1,  1, -1],
        [ 1, -1, -1, -1,  1]
    ]),
    "T": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1]
    ]),
    "L": np.array([
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1]
    ])
}

def get_pattern_matrix(letter: str) -> np.ndarray:
    """Returns the 5x5 numpy array for a given letter."""
    if letter not in ALPHABET_5X5:
        raise ValueError(f"Letter '{letter}' not defined in ALPHABET_5X5.")
    return ALPHABET_5X5[letter].copy()

def flatten_pattern(matrix: np.ndarray) -> np.ndarray:
    """Flattens a 2D binary matrix (e.g., 5x5) into a 1D vector of length N=25."""
    return matrix.flatten()

def reshape_pattern(vector: np.ndarray, shape=(5, 5)) -> np.ndarray:
    """Reshapes a 1D vector back into a 2D matrix shape."""
    return vector.reshape(shape)
def add_noise(pattern: np.ndarray, noise_level: float = None, rng=None,
              num_flips: int = None) -> np.ndarray:

    rng = np.random.default_rng(rng)  # returns the same Generator if one is passed
    flat = np.asarray(pattern).flatten()  # flatten always returns a copy
    n = flat.size

    if (noise_level is None) == (num_flips is None):
        raise ValueError("Give exactly one of noise_level or num_flips.")
    if num_flips is None:
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be between 0.0 and 1.0.")
        num_flips = int(round(noise_level * n))
    if not 0 <= num_flips <= n:
        raise ValueError(f"num_flips must be between 0 and {n}.")

    indices_to_flip = rng.choice(n, size=num_flips, replace=False)
    flat[indices_to_flip] *= -1  # flip selected bits between 1 and -1

    return flat.reshape(np.shape(pattern))