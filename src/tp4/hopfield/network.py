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

class Hopfield:
    def __init__(self, n_units: int):
        self.n = n_units
        self.W = np.zeros((n_units, n_units))
        self.patterns = None
 
    def train(self, patterns: np.ndarray, rule: str = "hebbian") -> None:
        """Compute the weight matrix W from the stored patterns, zero diagonal.

        patterns: shape (P, N), values in {-1, +1}.
        rule:
          "hebbian"        W = (1/N) * X^T X. Simple, but similar patterns
                           interfere with each other (crosstalk).
          "pseudoinverse"  W = X^T (X X^T)^-1 X. Corrects for the overlap
                           between patterns, so each stored pattern only
                           votes for itself. Needs linearly independent patterns.
        """
        patterns = np.asarray(patterns, dtype=float)
        assert patterns.shape[1] == self.n, "pattern length must equal n_units"
        self.patterns = patterns.copy()
        if rule == "hebbian":
            self.W = patterns.T @ patterns / self.n
        elif rule == "pseudoinverse":
            overlap = patterns @ patterns.T  # P x P overlap between stored patterns
            self.W = patterns.T @ np.linalg.inv(overlap) @ patterns
        else:
            raise ValueError("rule must be 'hebbian' or 'pseudoinverse'")
        self.rule = rule
        np.fill_diagonal(self.W, 0.0)

    def energy(self, s: np.ndarray) -> float:
        return float(-0.5 * s @ self.W @ s)
 
    def recall(self, s0, max_sweeps: int = 100, mode: str = "async",
               rng=None, keep_states: bool = False) -> dict:
        """Run the network from s0 until a fixed point or max_sweeps.
 
        async: one unit at a time in random order (energy never increases).
        sync:  all units at once (can end in a 2-cycle).
 
        Returns a dict with:
          state           final state
          sweeps          number of full passes performed
          energy_sweep    energy at start and after every sweep
          energy_update   energy after every single update (async only)
          converged       True if a fixed point was reached
          states          states per sweep (if keep_states)
        """
        rng = np.random.default_rng() if rng is None else rng
        s = np.asarray(s0, dtype=float).copy()
        E = self.energy(s)
        energy_sweep, energy_update = [E], [E]
        states = [s.copy()] if keep_states else None
        converged = False
        sweeps = 0
        prev = None  # used to detect 2-cycles in sync mode
 
        for sweeps in range(1, max_sweeps + 1):
            changed = False
            if mode == "async":
                for i in rng.permutation(self.n):
                    h = self.W[i] @ s
                    if h == 0:
                        continue
                    new = 1.0 if h > 0 else -1.0
                    if new != s[i]:
                        # Delta E when flipping unit i (zero diagonal): 2 * s_old * h
                        E += 2.0 * s[i] * h
                        s[i] = new
                        changed = True
                    energy_update.append(E)
            elif mode == "sync":
                h = self.W @ s
                new = np.where(h > 0, 1.0, np.where(h < 0, -1.0, s))
                changed = not np.array_equal(new, s)
                cycle = prev is not None and np.array_equal(new, prev)
                prev, s = s, new
                E = self.energy(s)
                energy_update.append(E)
                if changed and cycle:
                    energy_sweep.append(E)
                    break  # 2-cycle: oscillates between two states
            else:
                raise ValueError("mode must be 'async' or 'sync'")
 
            energy_sweep.append(E)
            if keep_states:
                states.append(s.copy())
            if not changed:
                converged = True
                break
 
        return {
            "state": s,
            "sweeps": sweeps,
            "energy_sweep": energy_sweep,
            "energy_update": energy_update,
            "converged": converged,
            "states": states,
        }