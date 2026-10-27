"""Kohonen network: grid, BMU search, neighbourhood and learning-rate schedules."""
import numpy as np
import matplotlib.pyplot as plt
from src.tp4.kohonen.distances import euclidean_distance, grid_distance
from src.tp4.kohonen.topology import get_radius

class Kohonen:
    def __init__(self, X: np.ndarray, map_width: int, map_height: int, n_iterations: int, learning_rate: float, initial_radius: float, time_constant: float, rng: np.random.Generator, grid_type="rectangular") -> None: 
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

        self.grid_type = grid_type
    
        # Initialization of weights
        indices = rng.choice(X.shape[0], size=(map_width, map_height), replace=True)
        self.weights = X[indices]
        #self.weights = rng.random((self.map_width, self.map_height, self.n_features))

    def find_bmu(self, input_vector):
        bmu_index = None
        best_score = np.inf

        # Compare with each neuron
        for i in range(self.map_width):
            for j in range(self.map_height):
                score = euclidean_distance(input_vector, self.weights[i, j])
                if score < best_score:
                    best_score = score
                    bmu_index = (i, j)
        
        return bmu_index
    
    def update_weights(self, input_vector, bmu_index, radius, learning_rate):    
        # Update weights of neurons within the BMU's neighborhood
        for i in range(self.map_width):
            for j in range(self.map_height):
                dist_to_bmu = grid_distance(i, j, bmu_index, self.grid_type)
                
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
