"""SOM evaluation: U-matrix, quantisation and topographic error.

Every metric takes a trained `Kohonen` instance (and the data `X` when needed).
`X` can be any data, so the same metrics work on training and held-out samples.
"""
import numpy as np

from src.tp4.kohonen.distances import euclidean_distance
from src.tp4.kohonen.topology import are_adjacent


def bmu_assignments(som, X):
    """Return (bmu_indices, bmu_counts): the BMU (i, j) of every sample, shape
    (n_samples, 2), and how many samples each neuron wins, shape (width, height)."""
    bmu_indices = np.array([som.find_bmu(x) for x in X], dtype=int)
    bmu_counts = np.zeros((som.map_width, som.map_height), dtype=int)
    for i, j in bmu_indices:
        bmu_counts[i, j] += 1
    return bmu_indices, bmu_counts


def quantization_error(som, X):
    """Mean euclidean distance between each sample and the weights of its BMU."""
    bmu_indices, _ = bmu_assignments(som, X)
    return float(np.mean([
        euclidean_distance(x, som.weights[i, j]) for x, (i, j) in zip(X, bmu_indices)
    ]))


def used_neurons(som, X):
    """Number of neurons that are the BMU of at least one sample."""
    _, bmu_counts = bmu_assignments(som, X)
    return int(np.count_nonzero(bmu_counts))


def topographic_error(som, X):
    """Fraction of samples whose first and second BMU are not grid neighbours."""
    errors = 0
    for x in X:
        distances = np.linalg.norm(som.weights - x, axis=2)
        # stable sort -> ties resolved in row-major order, like `find_bmu`
        order = np.argsort(distances, axis=None, kind="stable")[:2]
        first = np.unravel_index(order[0], distances.shape)
        second = np.unravel_index(order[1], distances.shape)
        if not are_adjacent(first, second, som.grid_type):
            errors += 1
    return errors / len(X)


def calculate_u_matrix(som):
    """Mean weight distance from each neuron to its immediate grid neighbours."""
    u_matrix = np.zeros((som.map_width, som.map_height), dtype=float)

    for i in range(som.map_width):
        for j in range(som.map_height):
            neighbor_distances = []

            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue

                    ni, nj = i + di, j + dj

                    # Ignore coordinates outside the map
                    if not (0 <= ni < som.map_width and 0 <= nj < som.map_height):
                        continue

                    if not are_adjacent((i, j), (ni, nj), som.grid_type):
                        continue

                    neighbor_distances.append(
                        euclidean_distance(som.weights[i, j], som.weights[ni, nj])
                    )

            if neighbor_distances:
                u_matrix[i, j] = np.mean(neighbor_distances)

    return u_matrix
