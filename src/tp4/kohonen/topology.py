import numpy as np

def hex_coordinates(i, j):
    x = i + 0.5 * (j % 2)
    y = j * np.sqrt(3) / 2
    return np.array([x, y])


def get_radius(initial_radius, time_constant, iteration):
    return max(
        1.0,
        initial_radius * np.exp(-iteration / time_constant)
    )

def are_adjacent(pos_a, pos_b, grid_type):
    """True if two grid cells (i, j) are immediate neighbours.

    Rectangular: 4-neighbourhood (orthogonal only). Hexagonal: centres at
    distance 1 in the offset layout of `hex_coordinates`.
    """
    if grid_type == "rectangular":
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]) == 1
    if grid_type == "hexagonal":
        return bool(np.isclose(
            np.linalg.norm(hex_coordinates(*pos_a) - hex_coordinates(*pos_b)), 1.0
        ))
    raise ValueError("Unknown grid type")
