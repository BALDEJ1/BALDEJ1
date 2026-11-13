"""Choosing the character for each cell.

Three ways:
  ramp_chars  - classic: brightness picks a character from a dark-to-dense ramp like " .:-=+*#%@".
  pick_by_coverage - the same idea, but the ramp is ordered and spaced by the ink each glyph really
                puts on screen (measured from the font), so equal steps in brightness look equal.
  shape_chars - shape matching: a character is a small shape, not a pixel. Each glyph is described by
                how much ink falls in a 3 x 2 grid of regions (ascii/glyph_table.json, measured once
                from the Menlo font). A cell supplies the brightness of the same 6 regions and gets
                the glyph whose ink layout matches best, so edges come out as '/', '(', '_', '`'
                instead of a blocky staircase.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np

from .util import to_blocks

TABLE_PATH = Path(__file__).with_name("glyph_table.json")
GRID = (3, 2)

# Glyph sets for shape matching. Symbols read as "drawing"; letters add finer density steps but can
# start to look like text, so scenes choose.
SYMBOLS = " .,:;'`\"-_~^=+*!|/\\()<>[]{}?%#&$@"
SYMBOLS_AND_LETTERS = SYMBOLS + "ilt1fjrcvxzunoeasJLTY7CUOQ08BDEGHKMNPRSWXZ"
DEFAULT_RAMP = " .:-=+*#%@"


@cache
def _table() -> dict:
    return json.loads(TABLE_PATH.read_text())


@cache
def _vectors(charset: str) -> tuple[np.ndarray, np.ndarray]:
    glyphs = _table()["glyphs"]
    unique = "".join(dict.fromkeys(charset))
    missing = [c for c in unique if c not in glyphs]
    if missing:
        raise ValueError(f"no glyph data for {missing!r}")
    vecs = np.array([glyphs[c] for c in unique], dtype=np.float64)
    codes = np.frombuffer(unique.encode("ascii"), dtype=np.uint8).copy()
    return vecs, codes


def glyph_vector(ch: str) -> np.ndarray:
    return np.array(_table()["glyphs"][ch], dtype=np.float64)


def pool_regions(ink: np.ndarray, sy: int, sx: int) -> np.ndarray:
    """Turn a supersampled ink image (rows*sy, cols*sx) into (rows, cols, 6) region means.

    sy must be a multiple of 3 and sx a multiple of 2 (e.g. sy=6, sx=4 gives 4 samples per region).
    """
    gy, gx = GRID
    if sy % gy or sx % gx:
        raise ValueError(f"sy must be a multiple of {gy} and sx a multiple of {gx}")
    rows, cols = ink.shape[0] // sy, ink.shape[1] // sx
    fine = ink.reshape(rows * gy, sy // gy, cols * gx, sx // gx).mean(axis=(1, 3))
    return to_blocks(fine, gy, gx)


def shape_chars(regions: np.ndarray, charset: str = SYMBOLS, contrast: float = 1.0) -> np.ndarray:
    """Pick the best-matching glyph for every cell.

    regions: (rows, cols, 6) ink per region in [0, 1] (1 = as dense as the densest glyph region).
    contrast > 1 exaggerates the difference between a cell's regions, which sharpens edges.
    Returns (rows, cols) uint8 ASCII codes.
    """
    v = np.clip(np.asarray(regions, dtype=np.float64), 0.0, 1.0)
    if contrast != 1.0:
        peak = v.max(axis=-1, keepdims=True)
        safe = np.where(peak > 1e-9, peak, 1.0)
        v = np.where(peak > 1e-9, (v / safe) ** contrast * peak, 0.0)
    vecs, codes = _vectors(charset)
    # squared distance via |v|^2 - 2 v.g + |g|^2, avoiding a (rows, cols, glyphs, 6) temporary
    d = (v * v).sum(-1)[..., None] - 2.0 * (v @ vecs.T) + (vecs * vecs).sum(-1)
    return codes[d.argmin(-1)]


def ramp_chars(level: np.ndarray, ramp: str = DEFAULT_RAMP) -> np.ndarray:
    """Classic luminance ramp: level in [0, 1] -> character, 0 maps to the first (usually a space)."""
    codes = np.frombuffer(ramp.encode("ascii"), dtype=np.uint8)
    idx = np.clip(np.round(np.asarray(level) * (len(codes) - 1)), 0, len(codes) - 1).astype(np.int64)
    return codes[idx]


@cache
def coverage_ramp(chars: str) -> tuple[np.ndarray, np.ndarray]:
    """`chars` sorted by the ink each one really puts on screen (measured, not guessed).

    Returns (codes, ink) with ink scaled so the densest character is 1 and a space is 0.
    """
    ink = np.array([glyph_vector(c).mean() for c in chars])
    if ink.max() <= 0.0:
        raise ValueError("a coverage ramp needs at least one visible character")
    order = np.argsort(ink, kind="stable")
    codes = np.frombuffer("".join(chars[i] for i in order).encode("ascii"), dtype=np.uint8).copy()
    scaled = ink[order] / ink[order][-1]
    codes.setflags(write=False)
    scaled.setflags(write=False)
    return codes, scaled


def pick_by_coverage(level: np.ndarray, chars: str) -> np.ndarray:
    """For every level in [0, 1], the character of `chars` whose ink is closest (0 = blank, 1 = densest).

    Unlike shape matching this never picks a glyph for its shape, so flat areas get an even texture.
    """
    codes, ink = coverage_ramp(chars)
    level = np.clip(np.asarray(level, dtype=np.float64), 0.0, 1.0)
    return codes[np.abs(level[..., None] - ink).argmin(-1)]


def hybrid_chars(
    regions: np.ndarray,
    edge_chars: str,
    fill_chars: str,
    edge_threshold: float = 0.6,
    contrast: float = 1.3,
) -> np.ndarray:
    """Shape matching where a cell has a clear edge, an even ink ramp everywhere else.

    Shape matching alone fills flat areas with whatever glyph happens to sit closest (';' or '$' walls);
    a ramp alone turns every edge into a staircase. A cell counts as an edge when its six regions differ
    by more than `edge_threshold` times their mean.
    """
    regions = np.clip(np.asarray(regions, dtype=np.float64), 0.0, 1.0)
    mean = regions.mean(-1)
    densest = max(glyph_vector(c).mean() for c in fill_chars)
    fill = pick_by_coverage(mean / densest, fill_chars)
    spread = regions.std(-1) / (mean + 0.05)
    edge = shape_chars(regions, edge_chars, contrast)
    return np.where(spread > edge_threshold, edge, fill)
