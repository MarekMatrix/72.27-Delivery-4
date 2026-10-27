"""Distance functions between vectors and between grid cells."""
import numpy as np

from src.tp4.kohonen.topology import hex_coordinates

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1-v2)

def grid_distance(i, j, bmu_index, grid_type):
    if grid_type == "rectangular":
        pos = np.array([i, j])
        bmu_pos = np.array(bmu_index)
    else: 
        pos = hex_coordinates(i, j)
        bmu_pos = hex_coordinates(*bmu_index)
    return euclidean_distance(pos, bmu_pos)
