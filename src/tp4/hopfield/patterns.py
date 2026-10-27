"""Patterns for Exercise 2: 5x5 letters, QR codes and noise."""
from pathlib import Path

import numpy as np
from PIL import Image

from .network import (ALPHABET_5X5, add_noise, flatten_pattern,  # noqa: F401
                      get_pattern_matrix, reshape_pattern)

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/tp4/hopfield/patterns.py -> repo root
LETTER_SIDE = 5
QR_SIDE = 37

# ---------------------------------------------------------------------------
# Letters
# ---------------------------------------------------------------------------

# The 4 stored letters, the helpers and add_noise live in network.py.

# Full alphabet, only used for the orthogonality analysis ('#' = 1, '.' = -1).
_ALL_LETTERS_RAW = {
    "A": [".###.", "#...#", "#####", "#...#", "#...#"],
    "B": ["####.", "#...#", "####.", "#...#", "####."],
    "C": [".####", "#....", "#....", "#....", ".####"],
    "D": ["####.", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "####.", "#....", "#####"],
    "F": ["#####", "#....", "####.", "#....", "#...."],
    "G": [".####", "#....", "#..##", "#...#", ".###."],
    "H": ["#...#", "#...#", "#####", "#...#", "#...#"],
    "I": ["#####", "..#..", "..#..", "..#..", "#####"],
    "J": ["#####", "...#.", "...#.", "#..#.", "###.."],
    "K": ["#...#", "#..#.", "###..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "####."],
    "M": ["#...#", "##.##", "#.#.#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "####.", "#....", "#...."],
    "Q": [".###.", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "####.", "#..#.", "#...#"],
    "S": [".####", "#....", ".###.", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#.#.#", "##.##", "#...#"],
    "X": ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
    "Y": ["#...#", ".#.#.", "..#..", "..#..", "..#.."],
    "Z": ["#####", "...#.", "..#..", ".#...", "#####"],
}
ALL_LETTERS = {
    k: np.array([[1 if c == "#" else -1 for c in row] for row in rows])
    for k, rows in _ALL_LETTERS_RAW.items()
}
ALL_LETTERS.update(ALPHABET_5X5)  # the team's own versions of J, X, T, L take precedence


def letters_to_patterns(names, alphabet: dict = ALL_LETTERS) -> np.ndarray:
    """Stack letters as flattened vectors, shape (P, 25)."""
    return np.array([alphabet[n].flatten() for n in names], dtype=float)


# ---------------------------------------------------------------------------
# QR codes
# ---------------------------------------------------------------------------

def find_qr_dir() -> Path:
    """Default QR folder: data/hopfield, falling back to docs/datos/hopfield."""
    for rel in ("data/hopfield", "docs/datos/hopfield"):
        if (REPO_ROOT / rel).is_dir():
            return REPO_ROOT / rel
    raise FileNotFoundError("No data/hopfield or docs/datos/hopfield folder found.")


def _dark_mask(path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("L"), dtype=float) < 128


def qr_to_matrix(path, side: int = QR_SIDE) -> np.ndarray:
    """Load a QR image as a side x side matrix of 1 (black) / -1 (white).

    Crops the white border, estimates the module size from the top-left finder
    square (7 modules wide) and samples one pixel per module.
    """
    dark = _dark_mask(path)
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    crop = dark[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    h, w = crop.shape
    if h > w * 1.05:  # caption below the code: keep the square top part
        crop = crop[:w, :]
    h, w = crop.shape

    top = crop[0]
    run = np.argmax(~top) if not top.all() else w
    n_mod = int(round(w / (run / 7.0)))

    cy = ((np.arange(n_mod) + 0.5) * h / n_mod).astype(int)
    cx = ((np.arange(n_mod) + 0.5) * w / n_mod).astype(int)
    grid = crop[np.ix_(cy, cx)]

    if n_mod != side:
        print(f"  note: {Path(path).name} has {n_mod}x{n_mod} modules, resampling to {side}x{side}")
        idx = (np.arange(side) * n_mod / side).astype(int)
        grid = grid[np.ix_(idx, idx)]

    return np.where(grid, 1.0, -1.0)


def _qr_label(path: Path) -> str:
    name = path.stem
    for suffix in ("_qr", "_menu"):
        name = name.removesuffix(suffix)
    return name.capitalize()


def load_qr_codes(folder=None, side: int = QR_SIDE):
    """Returns (patterns of shape (P, side*side), labels, file paths)."""
    folder = Path(folder) if folder else find_qr_dir()
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    patterns = np.array([qr_to_matrix(f, side).ravel() for f in files])
    return patterns, [_qr_label(f) for f in files], files


def decode_qr(vec, side: int = QR_SIDE):
    """Try to read a +-1 vector as a QR code with OpenCV. Returns the text or None."""
    try:
        import cv2
    except ImportError:
        return None
    grid = np.asarray(vec).reshape(side, side)
    img = np.pad(np.where(grid > 0, 0, 255).astype(np.uint8), 4, constant_values=255)
    detectors = [cv2.QRCodeDetector()]
    if hasattr(cv2, "QRCodeDetectorAruco"):
        detectors.insert(0, cv2.QRCodeDetectorAruco())
    for det in detectors:
        for scale in (8, 5, 4, 10):
            big = np.kron(img, np.ones((scale, scale), dtype=np.uint8))
            text, _, _ = det.detectAndDecode(big)
            if text:
                return text
    return None