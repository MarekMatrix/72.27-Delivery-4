"""Exercise 2.2 (warm-up): Hopfield network on 5x5 letters.
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from .analysis import classify, overlap_matrix, plot_overlap, plot_patterns, plot_recall, save
from .network import Hopfield
from .patterns import ALL_LETTERS, ALPHABET_5X5, LETTER_SIDE, add_noise, letters_to_patterns

N = LETTER_SIDE * LETTER_SIDE
OUTCOMES = ["correct", "wrong", "inverse", "spurious"]
STORED = list(ALPHABET_5X5)       # the letters the team chose: J, X, T, L
BAD_SET = ["J", "C", "T", "L"]    # the first choice, kept as a non-orthogonal example


def score_set(names):
    """Off-diagonal |overlap| of a set of letters: (max, mean). Lower = more orthogonal."""
    M = np.abs(overlap_matrix(letters_to_patterns(names)))
    off = M[~np.eye(len(names), dtype=bool)]
    return off.max(), off.mean()


def best_set(size=4):
    scored = [(score_set(c), c) for c in combinations(ALL_LETTERS, size)]
    return min(scored, key=lambda x: (x[0][0], x[0][1]))


def build_net(names):
    net = Hopfield(N)
    net.train(letters_to_patterns(names))
    return net


def stability(net, names):
    """Number of pixels that change when each stored letter is fed in clean (0 = stable)."""
    return {n: int(np.sum(np.sign(net.W @ p) != p)) for n, p in zip(names, net.patterns)}


def noise_curve(names, ks, trials, rng):
    """Share of each outcome per number of flipped pixels, and mean sweeps."""
    net = build_net(names)
    counts = {o: np.zeros(len(ks)) for o in OUTCOMES}
    sweeps = np.zeros(len(ks))
    for j, k in enumerate(ks):
        for t in range(trials):
            idx = t % len(names)
            r = net.recall(add_noise(net.patterns[idx], num_flips=k, rng=rng), rng=rng)
            counts[classify(r["state"], net.patterns, idx)[0]][j] += 1
            sweeps[j] += r["sweeps"]
    return {o: c / trials for o, c in counts.items()}, sweeps / trials


def capacity_curve(max_p, subsets, k_noise, rng):
    """For P = 1..max_p random letter sets: share of stored letters that are
    fixed points, and recall rate with k_noise flipped pixels."""
    names_all = list(ALL_LETTERS)
    stable, recall = [], []
    for P in range(1, max_p + 1):
        st, rc = [], []
        for _ in range(subsets):
            names = list(rng.choice(names_all, size=P, replace=False))
            net = build_net(names)
            st.append(np.mean([v == 0 for v in stability(net, names).values()]))
            ok = 0
            for idx in range(P):
                r = net.recall(add_noise(net.patterns[idx], num_flips=k_noise, rng=rng), rng=rng)
                ok += classify(r["state"], net.patterns, idx)[0] == "correct"
            rc.append(ok / P)
        stable.append(np.mean(st))
        recall.append(np.mean(rc))
    return np.array(stable), np.array(recall)


def find_example(net, names, outcome, k, rng, tries=2000):
    for _ in range(tries):
        idx = int(rng.integers(len(names)))
        noisy = add_noise(net.patterns[idx], num_flips=k, rng=rng)
        r = net.recall(noisy, rng=rng)
        if classify(r["state"], net.patterns, idx)[0] == outcome:
            return idx, noisy, r
    return None

def evaluate_capacity_multi_run(max_p=10, n_runs=100, subsets=50, k_noise=2):
    """
    Evaluates Hopfield capacity over multiple independent runs to 
    account for random seed / letter subset variance.
    """
    P_values = np.arange(1, max_p + 1)
    stable_results = np.zeros((n_runs, max_p))
    recall_results = np.zeros((n_runs, max_p))
    
    for run in range(n_runs):
        # Use a different seed for each run as recommended for stochastic evaluations
        rng = np.random.default_rng(seed=100 + run)
        
        # Calls your existing capacity_curve function
        stable, recall = capacity_curve(max_p, subsets=subsets, k_noise=k_noise, rng=rng)
        
        stable_results[run, :] = stable
        recall_results[run, :] = recall

    # Compute average across all runs
    avg_stable = np.mean(stable_results, axis=0)
    avg_recall = np.mean(recall_results, axis=0)
    
    # Print formatted table for your presentation/report
    print(f"{'P':<5} | {'Stable (Fixed Point %)':<25} | {'Recall (2 Flipped Pixels %)'}")
    print("-" * 60)
    for i, p in enumerate(P_values):
        print(f"{p:<5} | {avg_stable[i]*100:6.1f}%                    | {avg_recall[i]*100:6.1f}%")
        
    return avg_stable, avg_recall


def run(seed=42, trials=200):
    rng = np.random.default_rng(seed)
    print("\n=== Exercise 2.2: letters ===")

    # 1. Orthogonality: all letters and the candidate sets
    names_all = list(ALL_LETTERS)
    plot_patterns(letters_to_patterns(names_all), names_all, LETTER_SIDE, "letters_alphabet.png", cols=9)
    plot_overlap(letters_to_patterns(names_all), names_all, "letters_overlap_all.png",
                 title="Overlap between all letters")

    (best_score, best) = best_set(4)
    best = list(best)
    sets = {"stored": STORED, "bad": BAD_SET, "best": best}
    for label, names in sets.items():
        mx, mean = score_set(names)
        print(f"{label:>6} set {''.join(names)}: max|m| = {mx:.2f}, mean|m| = {mean:.2f}, "
              f"unstable pixels per letter = {stability(build_net(names), names)}")
        plot_patterns(letters_to_patterns(names), names, LETTER_SIDE, f"letters_set_{label}.png")
        plot_overlap(letters_to_patterns(names), names, f"letters_overlap_{label}.png",
                     f"Overlap, {label} set {''.join(names)}")

    # 2. Noise robustness per set
    ks = np.arange(0, 13)
    curves = {label: noise_curve(names, ks, trials, rng) for label, names in sets.items()}

    fig, ax = plt.subplots(figsize=(7, 4))
    for (label, names), marker in zip(sets.items(), "osd"):
        ax.plot(ks, curves[label][0]["correct"], marker + "-", label=f"{label} {''.join(names)}")
    ax.set_xlabel("flipped pixels (out of 25)")
    ax.set_ylabel("correct recall rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "letters_recall_vs_noise.png")

    freq, sweeps = curves["stored"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(ks))
    for o in OUTCOMES:
        ax.bar(ks, freq[o], bottom=bottom, label=o)
        bottom += freq[o]
    ax.set_xlabel("flipped pixels (out of 25)")
    ax.set_ylabel("share of runs")
    ax.set_title(f"Final state per noise level, set {''.join(STORED)}")
    ax.legend()
    save(fig, "letters_outcomes_vs_noise.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, sweeps, "o-")
    ax.set_xlabel("flipped pixels (out of 25)")
    ax.set_ylabel("mean sweeps to converge")
    ax.grid(alpha=0.3)
    save(fig, "letters_sweeps_vs_noise.png")

    # 3. Capacity
    max_p = 10
    stable, recall = capacity_curve(max_p, subsets=100, k_noise=2, rng=rng)
    P = np.arange(1, max_p + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(P, stable, "o-", label="stored letters that are fixed points")
    ax.plot(P, recall, "s-", label="correct recall with 2 flipped pixels")
    ax.axvline(0.15 * N, color="gray", ls="--", label=f"0.15 N = {0.15 * N:.1f}")
    ax.set_xlabel("number of stored letters P (random sets)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "letters_capacity.png")

    avg_s, avg_r = evaluate_capacity_multi_run()

    # 4. Recall examples and spurious states with the stored set
    net = build_net(STORED)
    for k in (3, 6):
        ex = find_example(net, STORED, "correct", k, rng)
        if ex:
            idx, noisy, r = ex
            plot_recall(net, net.patterns[idx], noisy, r, LETTER_SIDE, f"letters_recall_k{k}.png",
                        f"Letter {STORED[idx]}, {k} flipped pixels")

    for outcome in ("spurious", "inverse"):
        ex = find_example(net, STORED, outcome, 11, rng)
        if ex:
            idx, noisy, r = ex
            plot_recall(net, net.patterns[idx], noisy, r, LETTER_SIDE, f"letters_{outcome}_state.png",
                        f"{outcome.capitalize()} state from very noisy {STORED[idx]} (11 px flipped)")
            print(f"Found {outcome} state: final energy {r['energy_sweep'][-1]:.3f}, "
                  f"stored letter energy {net.energy(net.patterns[idx]):.3f}")
        else:
            print(f"No {outcome} state found")


if __name__ == "__main__":
    run()