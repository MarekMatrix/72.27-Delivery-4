"""Exercise 2.1: recovering damaged QR codes with a Hopfield network.
"""
import matplotlib.pyplot as plt
import numpy as np

from .analysis import classify, hamming, plot_overlap, plot_patterns, plot_recall, save
from .network import Hopfield
from .patterns import QR_SIDE as SIDE
from .patterns import add_noise, decode_qr as decode, load_qr_codes

N = SIDE * SIDE
OUTCOMES = ["correct", "near", "wrong", "inverse", "spurious"]
TOL = 0.10  # 'near' = closest to the right code, with at most 10% of pixels wrong
RULES = ["hebbian", "pseudoinverse"]


# ---------- realistic damage ----------

def torn_corner(vec, size=12):
    g = vec.reshape(SIDE, SIDE).copy()
    g[-size:, -size:] = -1  # paper missing -> white
    return g.ravel()


def scratch(vec, width=2):
    g = vec.reshape(SIDE, SIDE).copy()
    for i in range(SIDE):
        g[i, max(0, i - width):i + width] *= -1  # diagonal scratch inverts pixels
    return g.ravel()


def stain(vec, radius=8, center=(18, 12)):
    g = vec.reshape(SIDE, SIDE).copy()
    y, x = np.ogrid[:SIDE, :SIDE]
    g[(y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius ** 2] = 1  # dark stain
    return g.ravel()


def finder_destroyed(vec, size=10):
    g = vec.reshape(SIDE, SIDE).copy()
    g[:size, :size] = -1  # top-left finder square wiped out
    return g.ravel()


DAMAGE = {
    "torn corner": torn_corner,
    "scratch": scratch,
    "stain": stain,
    "finder square wiped": finder_destroyed,
}


# ---------- experiments ----------

def nearest_pattern(state, patterns):
    """Baseline: pick the stored code with the smallest Hamming distance."""
    return int(np.argmin([hamming(state, p) for p in patterns]))


def noise_sweep(net, levels, trials, rng, check_decode=True):
    pats = net.patterns
    P = len(pats)
    out = {o: np.zeros(len(levels)) for o in OUTCOMES}
    sweeps = [[] for _ in levels]
    bit_err = np.zeros(len(levels))
    baseline = np.zeros(len(levels))
    input_scans = np.zeros(len(levels))
    output_scans = np.zeros(len(levels))
    runs = trials * P

    for j, p in enumerate(levels):
        for t in range(trials):
            for idx in range(P):
                noisy = add_noise(pats[idx], p, rng)
                r = net.recall(noisy, rng=rng)
                out[classify(r["state"], pats, idx, TOL)[0]][j] += 1
                sweeps[j].append(r["sweeps"])
                bit_err[j] += hamming(r["state"], pats[idx]) / N
                baseline[j] += nearest_pattern(noisy, pats) == idx
                if check_decode:
                    input_scans[j] += decode(noisy) is not None
                    output_scans[j] += decode(r["state"]) is not None
        print(f"  noise {p:.2f}: correct {out['correct'][j] / runs:.2f}, "
              f"near {out['near'][j] / runs:.2f}, "
              f"mean sweeps {np.mean(sweeps[j]):.1f}")

    return {
        "outcomes": {o: v / runs for o, v in out.items()},
        "sweeps_mean": np.array([np.mean(s) for s in sweeps]),
        "sweeps_std": np.array([np.std(s) for s in sweeps]),
        "bit_error": bit_err / runs,
        "baseline": baseline / runs,
        "input_scans": input_scans / runs,
        "output_scans": output_scans / runs,
    }


def run_rule(rule, patterns, labels, levels, trials, rng, check_decode):
    """All experiments for one learning rule. Figures are saved as qr_<rule>_*.png."""
    print(f"\n--- learning rule: {rule} ---")
    pre = f"qr_{rule}_"

    net = Hopfield(N)
    net.train(patterns, rule=rule)
    for lab, p in zip(labels, patterns):
        stable = np.all(np.sign(net.W @ p) == p)
        drift = hamming(net.recall(p, rng=rng)["state"], p)
        print(f"  {lab} is a fixed point: {stable} (drifts {drift} px when fed the clean code)")

    # Noise sweep
    res = noise_sweep(net, levels, trials, rng, check_decode)

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(levels))
    for o in OUTCOMES:
        ax.bar(levels, res["outcomes"][o], width=0.04, bottom=bottom, label=o)
        bottom += res["outcomes"][o]
    ax.set_xlabel("noise level (fraction of flipped pixels)")
    ax.set_ylabel("share of runs")
    ax.set_title(f"Final state per noise level ({rule})")
    ax.legend()
    save(fig, pre + "outcomes_vs_noise.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(levels, res["outcomes"]["correct"], "o-", label="Hopfield returns the exact code")
    ax.plot(levels, res["outcomes"]["correct"] + res["outcomes"]["near"], "o--",
            label=f"Hopfield ends closest to the right code (<{TOL:.0%} px off)")
    ax.plot(levels, res["baseline"], "s--", label="baseline: nearest code by Hamming distance")
    if check_decode:
        ax.plot(levels, res["input_scans"], "^:", label="damaged input still scans (no network)")
        ax.plot(levels, res["output_scans"], "v-", label="network output scans")
    ax.set_xlabel("noise level (fraction of flipped pixels)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Recall vs noise ({rule})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, pre + "recall_vs_noise.png")

    # Iterations and energy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(levels, res["sweeps_mean"], yerr=res["sweeps_std"], fmt="o-", capsize=3)
    ax.set_xlabel("noise level (fraction of flipped pixels)")
    ax.set_ylabel("sweeps to converge")
    ax.set_title(f"Iterations ({rule})")
    ax.grid(alpha=0.3)
    save(fig, pre + "sweeps_vs_noise.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for p in (0.1, 0.2, 0.3, 0.4, 0.5):
        r = net.recall(add_noise(patterns[0], p, rng), rng=rng)
        ax.plot(r["energy_update"], label=f"noise {p:.1f} ({r['sweeps']} sweeps)")
    ax.axhline(net.energy(patterns[0]), color="gray", ls="--", label=f"energy of stored {labels[0]}")
    ax.set_xlabel("update step (one unit at a time)")
    ax.set_ylabel("energy")
    ax.set_title(f"Energy per step, code {labels[0]} ({rule})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, pre + "energy_curves.png")

    # Recall examples
    for p in (0.1, 0.3, 0.45):
        idx = int(rng.integers(len(labels)))
        noisy = add_noise(patterns[idx], p, rng)
        r = net.recall(noisy, rng=rng)
        outcome = classify(r["state"], patterns, idx, TOL)[0]
        plot_recall(net, patterns[idx], noisy, r, SIDE, pre + f"recall_noise{int(p * 100)}.png",
                    f"{labels[idx]}, noise {p:.2f}: {outcome} ({rule})")

    # Realistic damage
    print("  Realistic damage:")
    for name, fn in DAMAGE.items():
        idx = 0
        damaged = fn(patterns[idx])
        r = net.recall(damaged, rng=rng)
        outcome, got = classify(r["state"], patterns, idx, TOL)
        print(f"    {name}: {hamming(damaged, patterns[idx])} px damaged -> {outcome}"
              f"{'' if got is None else ' (' + labels[got] + ')'}, "
              f"input scans={decode(damaged) is not None}, output scans={decode(r['state']) is not None}")
        plot_recall(net, patterns[idx], damaged, r, SIDE,
                    pre + f"damage_{name.replace(' ', '_')}.png",
                    f"{labels[idx]}, {name}: {outcome} ({rule})")
    return res


def run(qr_dir=None, seed=42, trials=10, check_decode=True):
    rng = np.random.default_rng(seed)
    print("\n=== Exercise 2.1: QR codes ===")

    # 1. Load and represent (shared by both rules)
    patterns, labels, files = load_qr_codes(qr_dir)
    print(f"Loaded {len(labels)} codes: {labels}, each {SIDE}x{SIDE} = {N} units")
    plot_patterns(patterns, labels, SIDE, "qr_patterns.png")
    if check_decode:
        for lab, p in zip(labels, patterns):
            print(f"  {lab}: scans after preprocessing = {decode(p) is not None}")
    else:
        print("  (scan checks skipped: CHECK_DECODE = False)")

    M = plot_overlap(patterns, labels, "qr_overlap.png")
    off = np.abs(M[~np.eye(len(labels), dtype=bool)])
    shared = int(np.all(patterns == patterns[0], axis=0).sum())
    print(f"Overlap between codes: mean |m| = {off.mean():.3f}, max |m| = {off.max():.3f}")
    print(f"Pixels identical in all {len(labels)} codes: {shared} of {N} ({shared / N:.0%})")
    print(f"Theoretical capacity 0.138 N = {0.138 * N:.0f} patterns (storing {len(labels)}), "
          f"valid for random patterns only")

    # 2. Every experiment, once per learning rule
    levels = np.round(np.arange(0.0, 0.51, 0.05), 2)
    results = {rule: run_rule(rule, patterns, labels, levels, trials, rng, check_decode)
               for rule in RULES}

    # 3. Comparison: both rules and the lookup baseline in one figure
    fig, ax = plt.subplots(figsize=(7, 4))
    for rule, marker in zip(RULES, "os"):
        ax.plot(levels, results[rule]["outcomes"]["correct"], marker + "-",
                label=f"Hopfield, {rule} rule")
    ax.plot(levels, results[RULES[0]]["baseline"], "d--", color="gray",
            label="lookup: nearest code by Hamming distance")
    ax.set_xlabel("noise level (fraction of flipped pixels)")
    ax.set_ylabel("exact code recovered")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("QR recovery: Hebbian vs pseudo-inverse vs lookup")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "qr_rules_comparison.png")

    print("\nSummary, exact code recovered:")
    print("  noise  " + "  ".join(f"{r:>13}" for r in RULES) + "         lookup")
    for j, p in enumerate(levels):
        row = "  ".join(f"{results[r]['outcomes']['correct'][j]:>13.2f}" for r in RULES)
        print(f"  {p:.2f}   {row}  {results[RULES[0]]['baseline'][j]:>13.2f}")


if __name__ == "__main__":
    run()