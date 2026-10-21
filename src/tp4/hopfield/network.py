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
    "C": np.array([
        [-1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [-1,  1,  1,  1,  1]
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

def add_noise(pattern: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    """
    Randomly flips a percentage of bits in a pattern to simulate damage or noise.
    
    Parameters:
    - pattern: 1D or 2D binary array containing values in {-1, 1}.
    - noise_level: Float between 0.0 and 1.0 representing the fraction of bits to flip.
    - seed: Optional random seed for reproducibility.
    
    Returns:
    - Noisy copy of the pattern with the same shape.
    """
    rng = np.random.default_rng(seed)
    noisy = pattern.copy()
    flat = noisy.ravel()
    
    num_to_flip = int(np.ceil(noise_level * len(flat)))
    indices_to_flip = rng.choice(len(flat), size=num_to_flip, replace=False)
    
    # Multiply selected bits by -1 to flip them between 1 and -1
    flat[indices_to_flip] *= -1
    
    return flat.reshape(pattern.shape)