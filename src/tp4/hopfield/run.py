"""Run Exercise 2 (Hopfield). Figures are saved in results/hopfield/.
"""
import argparse
import sys
from pathlib import Path

# ---------------- settings for the play button ----------------
PART = "letters"     # "letters" (2.2), "qr" (2.1) or "all"
TRIALS = 3          # QR runs per code per noise level (3 = quick test)
CHECK_DECODE = False  # False = skip the QR scanning check (faster)
SEED = 42
QR_DIR = None        # None = data/hopfield, or e.g. "docs/datos/hopfield"
# ---------------------------------------------------------------

# Make the imports work when this file is started directly (play button).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the src folder
    from tp4.hopfield import exp_letters, exp_qr
else:
    from . import exp_letters, exp_qr


def main():
    ap = argparse.ArgumentParser(description="TP4 Exercise 2: Hopfield network")
    ap.add_argument("--part", choices=["letters", "qr", "all"], default=PART)
    ap.add_argument("--qr-dir", default=QR_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--no-decode", action="store_true", default=not CHECK_DECODE)
    args = ap.parse_args()

    if args.part in ("letters", "all"):
        exp_letters.run(seed=args.seed)
    if args.part in ("qr", "all"):
        exp_qr.run(args.qr_dir, seed=args.seed, trials=args.trials,
                   check_decode=not args.no_decode)


if __name__ == "__main__":
    main()