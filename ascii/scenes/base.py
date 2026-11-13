"""Defaults and small building blocks shared by scenes."""
from __future__ import annotations

import numpy as np

from ..engine import SPACE, Frame, Style, Twinkle


class SceneBase:
    """Sensible defaults; a scene overrides what it needs and implements render()."""

    title: str = "ASCII animation"
    style: Style = Style()
    rows: int = 48
    cols: int = 100
    frames: int = 90
    fps: float = 10.0
    max_colors: int = 16
    mono_levels: int = 6
    hero: int = 0

    def render(self, k: int, n: int) -> Frame:
        raise NotImplementedError

    def static(self) -> Frame | None:
        return None

    def twinkles(self) -> list[Twinkle]:
        return []


def starfield(
    rows: int,
    cols: int,
    *,
    density: float = 0.012,
    twinkle_share: float = 0.3,
    seed: int = 7,
    keep_out: np.ndarray | None = None,
    period: tuple[float, float] = (2.5, 7.0),
) -> tuple[Frame, list[Twinkle]]:
    """Sparse background stars: a static layer plus a share of them that slowly twinkle.

    keep_out: optional (rows, cols) bool mask of cells where no star may sit (e.g. the planet).
    """
    rng = np.random.default_rng(seed)
    mask = rng.random((rows, cols)) < density
    if keep_out is not None:
        mask &= ~keep_out
    chars = np.full((rows, cols), SPACE, np.uint8)
    rgb = np.zeros((rows, cols, 3))
    twinkles: list[Twinkle] = []
    for r, c in zip(*np.nonzero(mask), strict=True):
        b = rng.random() ** 2.0
        glyph = "*" if b > 0.9 else "+" if b > 0.7 else "." if b > 0.25 else "'"
        warm = rng.random()
        tint = np.array([0.85 + 0.15 * warm, 0.88, 1.0 - 0.15 * warm])
        colour = tint * (0.35 + 0.65 * b)
        if rng.random() < twinkle_share:
            twinkles.append(Twinkle(int(r), int(c), glyph, tuple(float(v) for v in colour),
                                    float(rng.uniform(*period)), float(rng.uniform(0.0, period[1]))))
        else:
            chars[r, c] = ord(glyph)
            rgb[r, c] = colour
    return Frame(chars, rgb), twinkles
