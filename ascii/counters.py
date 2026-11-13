"""Profile views and followers, drawn with keyboard characters in the Earth's colours.

Two looks:
  big   Each number in large digits. The digits are set in the font that ships with Pillow, made bold
        with a stroke, rasterised and turned into characters by shape matching, so their edges come
        out as / \\ ( ) _ like the Earth's coasts. Haze blue at the top fades into deep sea blue at
        the bottom; a spaced-out label sits above each number.
  line  One terminal-style line: a label, a dotted leader and the number in city-light amber.

Both use the Earth's character size and width (about 749 px), and a few stars twinkle around.
The picture depends only on the two numbers, so redrawing unchanged numbers gives the same file.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engine import SPACE, Frame, Style, Twinkle, encode_svg, fit_color_map
from .glyphs import SYMBOLS, pool_regions, shape_chars
from .util import cell_weighted_mean, hex_to_rgb

LOOKS = ("big", "line")
COLS = 140                                    # the Earth's grid width, so the characters match in size
STYLE = Style().as_wide_as(COLS)
SY, SX = 6, 4                                 # sub-samples per cell for shape matching
PX = 2                                        # raster pixels per SVG unit: one cell is 12 x 24 px
CELL_W, CELL_H = int(STYLE.cell_w * PX), int(STYLE.cell_h * PX)

DIGIT_ROWS = 7                                # height of a big digit, in character rows
DIGIT_GAP = 0.10                              # space between digits, as a share of the digit height
STROKE = 0.045                                # extra boldness, as a share of the font size
HALF = COLS // 2                              # each number gets one half of the picture
MARGIN = 4                                    # empty columns at the outer edges and around the middle

HAZE = hex_to_rgb("#8fd3ff")                  # the Earth's atmosphere
SEA_LIGHT = hex_to_rgb("#5b9be8")
SEA_DEEP = hex_to_rgb("#2a5aa5")
CITY_LIGHT = hex_to_rgb("#ffd76e")
LABEL = hex_to_rgb("#9fb3c8")
LEADER = hex_to_rgb("#3d4b5c")
STAR = hex_to_rgb("#cfd8e3")

STAR_COUNT = 14
STAR_CHARS = ".'`*+"
STAR_SEED = 5
STAR_CLEARANCE = 2                            # stars keep this many cells away from any text
MAX_COLORS = 12


@dataclass(frozen=True)
class Counts:
    views: int
    followers: int

    def __post_init__(self) -> None:
        if self.views < 0 or self.followers < 0:
            raise ValueError("counts cannot be negative")


def picture(counts: Counts, look: str = "big") -> str:
    """The finished SVG for these numbers."""
    if look not in LOOKS:
        raise ValueError(f"unknown look {look!r}; choose from {', '.join(LOOKS)}")
    frame = _big(counts) if look == "big" else _line(counts)
    stars = _stars(frame)
    return encode_svg(
        [frame], fit_color_map([frame], MAX_COLORS), fps=1.0, style=STYLE,
        title=f"Profile views: {counts.views}. Followers: {counts.followers}.",
        description=describe(counts), twinkles=stars,
    )


def describe(counts: Counts) -> str:
    """Machine-readable note kept inside the SVG, so the next run knows the last numbers."""
    return f"profile views {counts.views}, followers {counts.followers}"


# ---------------------------------------------------------------- the big look

def _big(counts: Counts) -> Frame:
    rows = DIGIT_ROWS + 4                     # margin, label, gap, digits, margin
    chars = np.full((rows, COLS), SPACE, np.uint8)
    rgb = np.zeros((rows, COLS, 3))
    for half, (label, value) in enumerate((("profile views", counts.views), ("followers", counts.followers))):
        left = half * HALF
        _put_text(chars, rgb, 1, left + HALF // 2, _spaced(label.upper()), LABEL)
        ink = _digit_ink(str(value), DIGIT_ROWS, HALF - 2 * MARGIN)
        glyphs = shape_chars(pool_regions(ink, SY, SX), SYMBOLS, contrast=1.25)
        colour = cell_weighted_mean(_gradient(ink.shape), ink, SY, SX)
        width = glyphs.shape[1]
        col0 = left + (HALF - width) // 2
        chars[3:3 + DIGIT_ROWS, col0:col0 + width] = glyphs
        rgb[3:3 + DIGIT_ROWS, col0:col0 + width] = colour
    return Frame(chars, rgb)


def _spaced(text: str) -> str:
    return " ".join(text).replace("   ", "     ")


def _gradient(shape: tuple[int, int]) -> np.ndarray:
    """Haze blue at the top of the digits, through sea blue, to deep sea at the bottom."""
    t = np.linspace(0.0, 1.0, shape[0])[:, None, None]
    top = HAZE + (SEA_LIGHT - HAZE) * np.clip(t * 2.0, 0.0, 1.0)
    colour = top + (SEA_DEEP - SEA_LIGHT) * np.clip(t * 2.0 - 1.0, 0.0, 1.0)
    return np.broadcast_to(colour, (*shape, 3))


@cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _digit_ink(text: str, rows: int, max_cols: int) -> np.ndarray:
    """Ink in [0, 1] of `text` as bold digits `rows` character rows tall and at most `max_cols` wide.

    Returns sub-samples (rows * SY, cols * SX), with cols as narrow as the digits allow.
    """
    height_px = rows * CELL_H
    size = _font_size_for(height_px)
    image = _raster(text, size)
    while image.width > max_cols * CELL_W:    # too many digits: shrink them to fit, centred vertically
        size = min(size - 1, int(size * max_cols * CELL_W / image.width))  # rounding may need a second pass
        image = _raster(text, size)
    cols = -(-image.width // CELL_W)
    canvas = Image.new("L", (cols * CELL_W, height_px))
    canvas.paste(image, ((cols * CELL_W - image.width) // 2, (height_px - image.height) // 2))
    ink = np.asarray(canvas, dtype=np.float64) / 255.0
    return ink.reshape(rows * SY, CELL_H // SY, cols * SX, CELL_W // SX).mean(axis=(1, 3))


def _font_size_for(height_px: int) -> int:
    """Font size whose digits, stroke included, are `height_px` tall."""
    probe = 200
    top, bottom = _bounds("0", probe)         # the stroke grows with the size, so this scales linearly
    return int(probe * height_px / (bottom - top))


def _bounds(text: str, size: int) -> tuple[int, int]:
    _, top, _, bottom = _font(size).getbbox(text, stroke_width=round(STROKE * size))
    return top, bottom


def _raster(text: str, size: int) -> Image.Image:
    """The digits drawn white on black, cropped to their ink, with a gap between digits."""
    font = _font(size)
    stroke = round(STROKE * size)
    gap = round(DIGIT_GAP * size)
    widths = [font.getbbox(ch, stroke_width=stroke) for ch in text]
    total = sum(b[2] - b[0] for b in widths) + gap * (len(text) - 1)
    top, bottom = _bounds("0", size)
    image = Image.new("L", (total, bottom - top))
    draw = ImageDraw.Draw(image)
    x = 0
    for ch, (left, _, right, _) in zip(text, widths, strict=True):
        draw.text((x - left, -top), ch, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
        x += right - left + gap
    return image


# ---------------------------------------------------------------- the line look

def _line(counts: Counts) -> Frame:
    rows = 3
    chars = np.full((rows, COLS), SPACE, np.uint8)
    rgb = np.zeros((rows, COLS, 3))
    for half, (label, value) in enumerate((("profile views", counts.views), ("followers", counts.followers))):
        left, right = half * HALF + MARGIN, (half + 1) * HALF - MARGIN
        number = str(value)
        dots = right - left - len(label) - len(number) - 2
        _put_text(chars, rgb, 1, left, label, HAZE, centred=False)
        _put_text(chars, rgb, 1, left + len(label) + 1, "." * max(dots, 1), LEADER, centred=False)
        _put_text(chars, rgb, 1, right - len(number), number, CITY_LIGHT, centred=False)
    return Frame(chars, rgb)


# ---------------------------------------------------------------- shared pieces

def _put_text(chars: np.ndarray, rgb: np.ndarray, row: int, col: int, text: str, colour: np.ndarray,
              centred: bool = True) -> None:
    """Write plain text into the grid (in place; the grid is local to the frame being built)."""
    start = col - len(text) // 2 if centred else col
    codes = np.frombuffer(text.encode("ascii"), np.uint8)
    chars[row, start:start + len(codes)] = codes
    rgb[row, start:start + len(codes)] = colour


def _stars(frame: Frame) -> list[Twinkle]:
    """A few twinkling stars in the empty space, always at the same places."""
    busy = frame.chars != SPACE
    near = np.zeros_like(busy)
    for dy in range(-STAR_CLEARANCE, STAR_CLEARANCE + 1):
        for dx in range(-2 * STAR_CLEARANCE, 2 * STAR_CLEARANCE + 1):
            near |= np.roll(np.roll(busy, dy, axis=0), dx, axis=1)
    free = np.argwhere(~near)
    rng = np.random.default_rng(STAR_SEED)
    picks = free[rng.choice(len(free), size=min(STAR_COUNT, len(free)), replace=False)]
    return [
        Twinkle(int(r), int(c), STAR_CHARS[i % len(STAR_CHARS)], tuple(float(v) for v in STAR),
                period=float(rng.uniform(2.5, 5.0)), delay=float(rng.uniform(0.0, 4.0)))
        for i, (r, c) in enumerate(sorted(map(tuple, picks)))
    ]
