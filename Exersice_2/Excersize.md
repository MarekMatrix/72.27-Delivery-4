Context: TP4 Exercise 2, Hopfield associative memory
Course: ITBA 72.27 Sistemas de Inteligencia Artificial, TP4 (Métodos de Aprendizaje No Supervisado).
This file covers only Exercise 2 (Hopfield network). Exercise 1 (Kohonen, Oja, PCA) is handled separately.
Language: Python + NumPy. The Hopfield network must be implemented from scratch (no Hopfield libraries).
Allowed helpers: numpy, matplotlib, Pillow/opencv for image loading, optionally pyzbar or opencv for QR decoding checks.
---
1. Assignment specification
1.1 Main case: recovering damaged QR codes
Story: a restaurant ("Trattoria Aurora") has five digital menus, each with its own QR code. The printed codes on the tables get torn, scratched, stained and worn, so they can no longer be scanned. Idea: build an associative memory with a Hopfield network that restores a damaged QR code so it can be sent to a QR reader.
Data: `datos/hopfield/` contains the five original QR codes:
Menu	File (expected)
Food (main menu)	`food_menu.png`
Wine	`wine_menu.png`
Beverages (cocktails and other drinks)	`beverages_menu.png`
Dessert	`dessert_menu.png`
Specials of the day	`specials_menu.png`
Required tasks:
Represent each QR code as a binary 37 x 37 matrix (small size because of computational cost).
Store the five vectors in the network and build query patterns.
Analyze degradation by adding random noise at different noise levels, simulating wear.
Analyze the number of iterations and the energy at each step of the network.
Explain the limitations of the use case.
1.2 Recommended warm-up: letters
Test the network on a simpler case first:
Build alphabet letter patterns as 5 x 5 matrices using 1 and -1. Store 4 letter patterns. Implement the Hopfield model to associate 5 x 5 input matrices with the stored letters.
Perform an orthogonality analysis and a storage capacity analysis of the network.
Feed in a very noisy pattern and identify a spurious state.
Example from the assignment, letter J:
```
 1  1  1  1  1
-1 -1 -1  1 -1
-1 -1 -1  1 -1
 1 -1 -1  1 -1
 1  1  1 -1 -1
```
---
2. Theory
2.1 What a Hopfield network is
A Hopfield network (John Hopfield, 1982) is an associative or content-addressable memory. It stores a small set of patterns and, given a partial or noisy version of one of them, converges to the closest stored pattern. It does not classify or predict. It is recurrent and fully connected: every unit is connected to every other unit, and the same units act as input and output. There is no gradient-based training; the weights are computed in one step.
2.2 Representation
N units (neurons), each with state s_i in {-1, +1}.
A pattern is a vector xi of length N. For a 5 x 5 letter, N = 25. For a 37 x 37 QR code, N = 1369.
Convention for images: black = +1, white = -1. Flatten row by row.
2.3 Storage (Hebbian rule)
For P stored patterns xi^1 ... xi^P:
```
W = (1/N) * sum_mu ( xi^mu  outer  xi^mu )
W_ii = 0
```
W is symmetric (W_ij = W_ji). Positive W_ij means units i and j tend to be in the same state across stored patterns; negative means they tend to be opposite. This is Hebb's principle ("neurons that fire together, wire together").
2.4 Recall (update rule)
Start with the query pattern s(0). Update unit i with:
```
h_i = sum_j W_ij * s_j
s_i = sign(h_i)        (if h_i == 0, keep the current s_i)
```
Asynchronous update (recommended): update one unit at a time, in random order. One full pass over all N units is one sweep (one "iteration"). This guarantees energy never increases and convergence to a fixed point.
Synchronous update: update all units at once with s = sign(W s). Faster, but can end in a 2-cycle (oscillating between two states). If used, detect cycles.
Stop when a full sweep produces no change (fixed point), or after a maximum number of sweeps.
2.5 Energy
```
E(s) = -1/2 * s^T W s
```
With symmetric W, zero diagonal and asynchronous updates, every update keeps E equal or lowers it. The network therefore always converges. Stored patterns are local minima (attractors) of E. Recall is "rolling downhill" from the query to the nearest minimum.
2.6 Crosstalk and stability
For a stored pattern xi^nu, the local field is:
```
h_i = xi_i^nu + (1/N) * sum_{mu != nu} xi_i^mu * (xi^mu . xi^nu)
```
The first term is the signal. The second is crosstalk from other patterns. If the crosstalk is large enough to flip the sign, the stored pattern is not stable. Crosstalk is small when patterns are close to orthogonal.
2.7 Orthogonality
Two patterns are orthogonal when their dot product is 0, which for +-1 vectors means they agree on exactly half their entries. Useful measures:
Dot product matrix: `X @ X.T` for pattern matrix X (P x N).
Normalized overlap: `m = (xi^a . xi^b) / N`, in [-1, 1]. Values near 0 are good.
For choosing letters: compute |dot products| for all candidate letters and pick the set of 4 with the smallest maximum (or mean) absolute overlap.
2.8 Capacity
For random, uncorrelated patterns, the network can reliably store about:
```
P_max ≈ 0.138 * N
```
Letters: N = 25, so P_max ≈ 3.5. Storing 4 letters is at or above the limit, so the choice of letters matters a lot.
QR codes: N = 1369, so P_max ≈ 189. Five patterns is far below the theoretical limit, but QR codes are strongly correlated (see 4), so effective capacity is lower.
Stricter criterion (error-free recall of all bits): P_max ≈ N / (2 ln N).
2.9 Spurious states
Minima of E that are not stored patterns:
Inverse states: if xi is stored, -xi is also a minimum (E is symmetric under s -> -s).
Mixture states: combinations of an odd number of stored patterns, e.g. sign(xi^1 + xi^2 + xi^3).
Spin-glass states: unrelated minima that appear when too many patterns are stored.
High noise inputs often converge to spurious states.
2.10 Uses and relevance
Pattern completion and denoising (images, characters, signals).
Optimization (Hopfield and Tank, 1985): encode a problem such as TSP in the energy function and let the network descend.
Theoretical neuroscience: model of associative memory in the brain.
Foundation for Boltzmann machines and modern Hopfield networks; the modern Hopfield update rule is equivalent to transformer attention.
Hopfield shared the 2024 Nobel Prize in Physics with Geoffrey Hinton.
---
3. Implementation plan
3.1 Suggested structure
```
ej2_hopfield/
  hopfield.py          # Hopfield class
  patterns.py          # letter definitions, QR loading and preprocessing
  noise.py             # random noise and structured damage
  exp_letters.py       # warm-up experiments
  exp_qr.py            # QR experiments
  plots.py             # plotting helpers
  results/             # saved figures and CSVs
```
3.2 Hopfield class (minimum API)
```python
class Hopfield:
    def __init__(self, n_units: int): ...
    def train(self, patterns: np.ndarray) -> None:
        # patterns: shape (P, N), values in {-1, +1}
        # W = (1/N) * patterns.T @ patterns ; np.fill_diagonal(W, 0)
    def energy(self, s: np.ndarray) -> float:
        # -0.5 * s @ W @ s
    def recall(self, s0: np.ndarray, max_sweeps: int = 100,
               mode: str = "async", rng=None) -> dict:
        # returns {"state": final_state,
        #          "sweeps": n_sweeps,
        #          "energies": list of energy per update or per sweep,
        #          "states": optional list of states per sweep (for animation),
        #          "converged": bool}
```
Implementation notes:
Keep everything as `np.int8` or `float` vectors of +-1.
Asynchronous: `for i in rng.permutation(N): h = W[i] @ s; if h != 0: s[i] = 1 if h > 0 else -1`.
Track whether any unit changed during a sweep; stop when none changed.
Log energy after every sweep (and optionally after every single update for detailed curves).
Seed the RNG for reproducibility.
3.3 Identifying the result
After recall, compare the final state with each stored pattern and its inverse:
Hamming distance to each stored pattern.
Classify the outcome as: correct pattern, wrong stored pattern, inverse of a stored pattern, or spurious state (none of the above).
3.4 Noise models
Random flip: flip a fraction p of the units, chosen at random. Sweep p from 0.0 to 0.5 (e.g. steps of 0.05).
Structured damage (realistic for the use case): blank out a corner or a rectangular region, add a scratch line, add a stain blob. Set damaged pixels to -1 (white) or to random values.
---
4. QR preprocessing
A 37 x 37 QR code is QR version 5 (4 * 5 + 17 = 37 modules per side).
Steps: load PNG, convert to grayscale, threshold to binary, crop the white quiet zone around the code, then resample to exactly 37 x 37 so that each pixel equals one module (sample at module centers or use nearest-neighbor resizing).
Map black = +1, white = -1, flatten to length 1369.
Verify visually that each 37 x 37 matrix still looks like the original QR, and optionally decode it with opencv/pyzbar to confirm it scans.
QR codes share fixed structure: three finder squares (corners), timing lines, alignment pattern, format information. This shared structure makes the five codes strongly correlated, so compute and report the overlap matrix.
---
5. Experiments and outputs
5.1 Letters
Define a set of candidate letters (5 x 5, +-1). Plot them.
Orthogonality: dot product matrix for all candidates; select the 4 most orthogonal; also select a deliberately bad set (e.g. similar letters like E and F) for comparison.
Stability check: feed each stored letter unchanged and verify it is a fixed point.
Noise: for increasing noise levels, run many trials and plot recall success rate per letter set.
Capacity: store 1, 2, 3, 4, 5, ... letters and plot how recall quality drops.
Spurious state: feed a very noisy pattern (around 40 to 50% flipped), show a case that converges to an inverse or mixture state, with its energy curve.
5.2 QR codes
Show the five preprocessed 37 x 37 QR matrices.
Overlap matrix between the five codes.
Recall examples: original, noisy input, recovered output, side by side, for several noise levels.
Success rate vs noise level (many trials per level, error bars).
Iterations: mean number of sweeps to converge vs noise level.
Energy: energy vs step for several runs (should be monotonically non-increasing).
Structured damage: torn corner, scratch, stain; does the network recover the right menu?
Optional: decode recovered codes with a QR reader to check they scan.
5.3 Plots checklist
Pattern grids (letters, QR codes).
Heatmap of overlap / dot product matrix.
Success rate vs noise level.
Sweeps to convergence vs noise level.
Energy vs step.
Before / noisy / after image triplets.
Example of a spurious state.
---
6. Limitations to discuss
The network can only return one of the stored codes; a new menu requires recomputing W, and an unknown or heavily damaged code returns the wrong menu.
Correlated patterns (shared QR structure) increase crosstalk and reduce effective capacity.
Above about 50% noise, information is lost; outputs become inverse or mixture states.
Random noise is not the same as real damage, which is spatially concentrated (torn corner, stain).
QR codes already contain Reed-Solomon error correction (7% to 30% depending on level), so the Hopfield network only adds value beyond that.
A simpler baseline for 5 known codes: pick the stored code with the smallest Hamming distance to the input. Worth comparing.
Memory and compute scale as N^2 (W has 1369^2 ≈ 1.87 million entries), so larger images get expensive quickly.