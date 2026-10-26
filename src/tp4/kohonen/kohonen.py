"""Kohonen network: grid, BMU search, neighbourhood and learning-rate schedules."""
import numpy as np
import matplotlib.pyplot as plt
from src.tp4.kohonen.metrics import euclidean_distance, exponential_distance
from src.tp4.kohonen.topology import get_radius, hex_coordinates

class Kohonen:
    def __init__(self, X: np.ndarray, map_width: int, map_height: int, n_iterations: int, learning_rate: float, initial_radius: float, time_constant: float, rng: np.random.Generator, grid_type="rectangular", distance_metric="euclidean") -> None: 
        self.X = X
        self.n_features = X.shape[1]
        self.map_width = map_width
        self.map_height = map_height
        self.n_iterations = n_iterations
        self.initial_learning_rate = learning_rate
        self.initial_radius = initial_radius
        self.time_constant = time_constant 
        self.rng = rng
    
        if grid_type not in ("rectangular", "hexagonal"):
            raise ValueError("Unknown grid type")

        if distance_metric not in ("euclidean", "exponential"):
            raise ValueError("Unknown distance metric")

        self.grid_type = grid_type
        self.distance_metric = distance_metric
    
        # Initialization of weights
        indices = rng.choice(X.shape[0], size=(map_width, map_height), replace=True)
        self.weights = X[indices]
        #self.weights = rng.random((self.map_width, self.map_height, self.n_features))

    def feature_distance(self, v1, v2):
        if self.distance_metric == "euclidean":
            return euclidean_distance(v1, v2)

        return exponential_distance(v1, v2)
    
    def grid_distance(self, i, j, bmu_index):
        if self.grid_type == "rectangular":
            pos = np.array([i, j])
            bmu_pos = np.array(bmu_index)
        else: 
            pos = hex_coordinates(i, j)
            bmu_pos = hex_coordinates(*bmu_index)
        return euclidean_distance(pos, bmu_pos)
    
    def find_bmu(self, input_vector):
        bmu_index = None
        if self.distance_metric == "euclidean":
            best_score = np.inf
            better = lambda score, best: score < best
        else:
            best_score = -np.inf
            better = lambda score, best: score > best
        
        # Compare with each neuron
        for i in range(self.map_width):
            for j in range(self.map_height):
                score = self.feature_distance(input_vector, self.weights[i, j])
                if better(score, best_score):
                    best_score = score
                    bmu_index = (i, j)
        
        return bmu_index
    
    def update_weights(self, input_vector, bmu_index, radius, learning_rate):    
        # Update weights of neurons within the BMU's neighborhood
        for i in range(self.map_width):
            for j in range(self.map_height):
                dist_to_bmu = self.grid_distance(i, j, bmu_index)
                
                # If the neuron is within the neighborhood
                if dist_to_bmu < radius: 
                    self.weights[i, j] += learning_rate * (input_vector - self.weights[i, j])
    
    def create_map(self):
        for iteration in range(self.n_iterations):
            # Select a random input vector
            input_vector = self.X[self.rng.integers(0, self.X.shape[0])]
            # Find the Best Matching Unit (BMU)
            bmu_index = self.find_bmu(input_vector)
            # Radius update
            radius = get_radius(self.initial_radius, self.time_constant, iteration)
            # Gradually reduce the learning rate over iterations
            learning_rate = self.initial_learning_rate * np.exp(- iteration / self.n_iterations) # Or we can do 1 / (iteration + 1)
            # Update the weights of the BMU and neighboring neurons
            self.update_weights(input_vector, bmu_index, radius, learning_rate)
        return self.weights

    def calculate_u_matrix(self):
        u_matrix = np.zeros((self.map_width, self.map_height), dtype=float)
        
        for i in range(self.map_width):
            for j in range(self.map_height):
                neighbor_distances = []
                
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj ==0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        
                        # Ignore coordinates outside the map
                        if not (0 <= ni < self.map_width and 0 <= nj < self.map_height):
                            continue
                        
                        if self.grid_type == "rectangular":
                            # Four orthogonal neighbors only
                            if abs(di) + abs(dj) != 1:
                                continue
                        elif self.grid_type == "hexagonal":
                            p1 = hex_coordinates(i, j)
                            p2 = hex_coordinates(ni, nj)
                            
                            # Only immediate hexagonal neighbors
                            if not np.isclose(euclidean_distance(p1, p2), 1.0):
                                continue
                        
                        distance = euclidean_distance(self.weights[i, j], self.weights[ni, nj])
                        neighbor_distances.append(distance)
                        
                if neighbor_distances:
                    u_matrix[i, j] = np.mean(neighbor_distances)
                
        return u_matrix


