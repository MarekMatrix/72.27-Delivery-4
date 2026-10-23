"""Kohonen network: grid, BMU search, neighbourhood and learning-rate schedules."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def hex_coordinates(i, j):
    x = i + 0.5 * (j % 2)
    y = j * np.sqrt(3) / 2
    return np.array([x, y])

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1-v2)

def exponential_distance(v1, v2):
    return np.exp(- euclidean_distance(v1, v2)**2)

def get_radius(initial_radius, time_constant, iteration):
    return max(
        1.0,
        initial_radius * np.exp(-iteration / time_constant)
    )
    
def visualize_som_weights(weights, feature):
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights[:, :, feature], cmap="viridis", annot=False, cbar=True)
    plt.title("Kohonen Map (SOM) - Projection of Weights")
    plt.show()


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


"""
rng = np.random.default_rng(seed=42)

map_width = 10
map_height = 10
n_iterations = 500 * X.shape[1]

learning_rate = 0.5
initial_radius = max(map_width, map_height)
time_constant = n_iterations / np.log(initial_radius)

kohonen = Kohonen(
    X,
    map_width,
    map_height,
    n_iterations,
    learning_rate,
    initial_radius,
    time_constant,
    rng,
    grid_type="rectangular",
    distance_metric="euclidean"
)

weights = kohonen.create_map()
visualize_som_weights(weights, feature=1)


"""