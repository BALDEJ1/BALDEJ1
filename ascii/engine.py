"""Turn ASCII animation frames into one self-contained, animated SVG for a GitHub README.

The characters stay real vector text, sharp at any zoom, and the file is plain text that compresses well.

GitHub constraints (checked 2026-11-13): README images are served from raw.githubusercontent.com with
    Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'; sandbox
so the SVG may use inline <style> and CSS animations, but no scripts and no external or data: fonts.
Text therefore uses the viewer's own monospace font, and every character must land on its column
whatever that font is (Menlo, SF Mono, Consolas, DejaVu Sans Mono...). Two layouts do that in Chromium,
Firefox and WebKit alike, and each run of one colour takes whichever is shorter in bytes:
  * a long run is its own <text> with an absolute x and an explicit textLength (stretched to the grid);
  * short runs of one colour share one <text> that gives every character its own x, plus y for the
    first one and a dy list that only changes where the row changes (dy moves the baseline, so it does
    not depend on the width of the font, unlike dx).
Every frame groups its runs by colour under one <g class>, so no run repeats its colour.
Tested 2026-11-13: a single <text> with coloured <tspan>s and one textLength is NOT stretched by WebKit,
and Firefox ignores textLength on <tspan>, which is why runs are not joined that way.

Coordinates are integers: a cell is 6 x 12 user units with a 10-unit font, and the whole picture is
scaled up for display (Style.scale), which keeps the file small.

Animation: every frame is a <g> that is hidden by default and shown for exactly 1/N of the loop by a
CSS keyframe animation; frame k starts its animation k/fps seconds late. With prefers-reduced-motion
the loop stops on one "hero" frame. Structural CSS classes have two or more letters; colour classes are
single letters, so they can never collide.
"""
from __future__ import annotations

import gzip
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .util import hex_to_rgb, oklab_to_srgb, oklch_to_hex, rgb_to_hex, srgb_to_oklab

SPACE = 32
FONT_STACK = ('ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",'
              '"DejaVu Sans Mono","Courier New",monospace')
CLASS_NAMES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
MONO_HUES = {"green": 145.0, "amber": 70.0}  # OKLCH hue angles of classic phosphor monitors
SIZE_BUDGET_BYTES = 2_000_000                 # raw size above which render.py warns
SEGMENT_GAP = 40                              # a run of more spaces than this starts a new <text>
MIN_STRETCH = 4                               # runs shorter than this skip textLength (drift < 1/3 char)
FRAME_OVERLAP_S = 1.0e-4                      # a frame hides 0.1 ms after the next appears: never a blank
LOOSE_CHAR_BYTES = 4                          # a loose character also costs a separator, itself and its dy


# ---------------------------------------------------------------- data

@dataclass(frozen=True)
class Frame:
    """One picture: a grid of characters plus the colour each character is drawn in."""

    chars: np.ndarray  # (rows, cols) uint8 printable ASCII codes; 32 (space) means empty
    rgb: np.ndarray    # (rows, cols, 3) sRGB colour in [0, 1]; ignored where the char is a space

    def __post_init__(self):
        if self.chars.ndim != 2 or self.rgb.shape != self.chars.shape + (3,):
            raise ValueError(f"chars {self.chars.shape} and rgb {self.rgb.shape} do not match")
        if self.chars.dtype != np.uint8:
            raise TypeError("chars must be a uint8 array of ASCII codes")
        if ((self.chars < 32) | (self.chars > 126)).any():
            raise ValueError("chars must be printable ASCII (codes 32..126)")

    @property
    def shape(self) -> tuple[int, int]:
        return self.chars.shape

    @staticmethod
    def empty(rows: int, cols: int) -> Frame:
        return Frame(np.full((rows, cols), SPACE, np.uint8), np.zeros((rows, cols, 3)))

    def lines(self) -> list[str]:
        return [row.tobytes().decode("ascii") for row in self.chars]


@dataclass(frozen=True)
class Twinkle:
    """A single character that fades in and out on its own clock, independent of the frame loop."""

    row: int
    col: int
    char: str
    rgb: tuple[float, float, float]
    period: float  # seconds for one fade cycle
    delay: float   # seconds; spreads twinkles so they do not pulse together


@dataclass(frozen=True)
class Style:
    font_size: float = 10.0  # in user units; see scale
    cell_w: float = 6.0      # horizontal pitch of one character: 0.6 em, the usual monospace advance
    cell_h: float = 12.0     # vertical pitch of one text line
    pad: float = 12.0
    scale: float = 1.2       # display size = user units x scale (a 10-unit font shows as 12 px)
    background: str = "#0b0e14"
    corner_radius: float = 10.0
    glow: float = 0.0        # blur radius of an optional phosphor glow; 0 disables it

    @property
    def cell_aspect(self) -> float:
        """How many times a character cell is taller than it is wide (needed to draw round circles)."""
        return self.cell_h / self.cell_w

    def size(self, rows: int, cols: int) -> tuple[float, float]:
        return 2 * self.pad + cols * self.cell_w, 2 * self.pad + rows * self.cell_h

    def as_wide_as(self, cols: int, reference_cols: int = 100) -> Style:
        """Smaller characters, so a `cols`-wide grid shows exactly as wide as a `reference_cols`-wide one."""
        return replace(self, scale=self.scale * self.size(1, reference_cols)[0] / self.size(1, cols)[0])

    def baseline(self, row: int) -> float:
        return self.pad + row * self.cell_h + 0.5 * self.cell_h + 0.3 * self.font_size


# ---------------------------------------------------------------- colour maps

class NearestColors:
    """Colour mode: every character takes the nearest of a small fitted palette (nearest in OKLab)."""

    SIMILAR = 0.06  # OKLab distance under which two palette colours count as interchangeable

    def __init__(self, lab_centroids: np.ndarray):
        order = np.argsort(lab_centroids[:, 0])
        self._lab = lab_centroids[order]
        self.colors = tuple(rgb_to_hex(oklab_to_srgb(c)) for c in self._lab)
        d = np.sqrt(((self._lab[:, None, :] - self._lab[None]) ** 2).sum(-1))
        self.similar = d < self.SIMILAR  # (k, k) bool: may a lone character be recoloured i -> j?

    def classify(self, rgb: np.ndarray) -> np.ndarray:
        lab = srgb_to_oklab(rgb)
        d = ((lab[..., None, :] - self._lab) ** 2).sum(-1)
        return d.argmin(-1)


class MonoLevels:
    """Monochrome mode: brightness is split into a few levels of one phosphor hue."""

    def __init__(self, lo: float, hi: float, hue: float, levels: int):
        self._lo, self._hi, self._levels = lo, hi, levels
        self.colors = tuple(
            oklch_to_hex(0.40 + 0.55 * i / (levels - 1), 0.16 - 0.06 * i / (levels - 1), hue)
            for i in range(levels)
        )
        idx = np.arange(levels)
        self.similar = np.abs(idx[:, None] - idx[None]) <= 1

    def classify(self, rgb: np.ndarray) -> np.ndarray:
        lightness = srgb_to_oklab(rgb)[..., 0]
        t = np.clip((lightness - self._lo) / max(self._hi - self._lo, 1e-6), 0.0, 1.0)
        return np.minimum((t * self._levels).astype(np.int64), self._levels - 1)


def _visible_rgb(frames: Sequence[Frame]) -> np.ndarray:
    picked = [f.rgb[f.chars != SPACE] for f in frames if f is not None]
    picked = [p for p in picked if len(p)]
    return np.concatenate(picked) if picked else np.zeros((1, 3))


def fit_color_map(frames: Sequence[Frame], max_colors: int = 16, seed: int = 0) -> NearestColors:
    """Fit a palette of at most `max_colors` to all visible characters (k-means in OKLab)."""
    if not 1 <= max_colors <= len(CLASS_NAMES):
        raise ValueError(f"max_colors must be within 1..{len(CLASS_NAMES)}")
    rgb = _visible_rgb(frames)
    unique = np.unique(np.round(rgb * 255).astype(np.int64), axis=0) / 255.0
    if len(unique) <= max_colors:
        return NearestColors(srgb_to_oklab(unique))
    lab = srgb_to_oklab(rgb)
    rng = np.random.default_rng(seed)
    if len(lab) > 60_000:
        lab = lab[rng.choice(len(lab), 60_000, replace=False)]
    return NearestColors(_kmeans(lab, max_colors, rng))


def _kmeans(points: np.ndarray, k: int, rng: np.random.Generator, iterations: int = 25) -> np.ndarray:
    centroids = [points[rng.integers(len(points))]]
    for _ in range(1, k):  # k-means++ seeding
        d = np.min(((points[:, None, :] - np.array(centroids)[None]) ** 2).sum(-1), axis=1)
        pick = rng.choice(len(points), p=d / d.sum()) if d.sum() > 0 else rng.integers(len(points))
        centroids.append(points[pick])
    c = np.array(centroids)
    for _ in range(iterations):
        labels = ((points[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            members = points[labels == j]
            c[j] = members.mean(0) if len(members) else points[rng.integers(len(points))]
    return c


def fit_mono_map(frames: Sequence[Frame], hue: float, levels: int = 6) -> MonoLevels:
    """Map brightness to `levels` shades of one hue, stretched over the frames' brightness range."""
    lightness = srgb_to_oklab(_visible_rgb(frames))[..., 0]
    lo, hi = np.percentile(lightness, 1.0), np.percentile(lightness, 99.5)
    return MonoLevels(float(lo), float(hi), hue, levels)


def mono_background(hue: float) -> str:
    return oklch_to_hex(0.16, 0.025, hue)


# ---------------------------------------------------------------- SVG encoding

def _num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def row_segments(codes: np.ndarray, gap: int = SEGMENT_GAP) -> list[tuple[int, int]]:
    """Inclusive (start, end) column ranges of non-space text, split where more than `gap` spaces sit."""
    nz = np.flatnonzero(codes != SPACE)
    if nz.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(nz) > gap + 1)
    starts = np.r_[nz[0], nz[breaks + 1]]
    ends = np.r_[nz[breaks], nz[-1]]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def class_runs(codes: np.ndarray, classes: np.ndarray) -> list[tuple[int, int, int]]:
    """Split one segment into (start, end_exclusive, class) runs.

    Spaces take the class of the character before them, so a gap never breaks a run in two.
    The segment must start with a non-space character.
    """
    n = len(codes)
    idx = np.where(codes != SPACE, np.arange(n), 0)
    filled = classes[np.maximum.accumulate(idx)]
    change = np.flatnonzero(np.diff(filled)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    return [(int(s), int(e), int(filled[s])) for s, e in zip(starts, ends, strict=True)]


def absorb_lone_characters(codes: np.ndarray, classes: np.ndarray, similar: np.ndarray) -> np.ndarray:
    """Recolour single characters sandwiched between two characters of one similar colour.

    A run boundary costs a whole <text> element, and a lone off-colour character inside a run is
    rarely visible, so this keeps files small at almost no visual cost. Returns a new array.
    """
    out = classes.copy()
    visible = np.flatnonzero(codes != SPACE)
    for i in range(1, len(visible) - 1):
        left, mid, right = out[visible[i - 1]], out[visible[i]], out[visible[i + 1]]
        if left == right != mid and similar[mid, left]:
            out[visible[i]] = left
    return out


class _Run(NamedTuple):
    row: int
    col: int
    cls: int
    text: str  # without trailing spaces


class _LooseRow(NamedTuple):
    row: int
    xs: list[str]
    chars: list[str]


def _frame_runs(frame: Frame, classes: np.ndarray, similar: np.ndarray | None) -> list[_Run]:
    """Every same-colour run of a frame, row by row."""
    runs = []
    for r in range(frame.chars.shape[0]):
        row = frame.chars[r]
        row_classes = classes[r] if similar is None else absorb_lone_characters(row, classes[r], similar)
        for a, b in row_segments(row):
            seg = row[a:b + 1]
            for s, e, c in class_runs(seg, row_classes[a:b + 1]):
                runs.append(_Run(r, a + s, c, seg[s:e].tobytes().decode("ascii").rstrip(" ")))
    return runs


def _own_text(run: _Run, style: Style) -> str:
    stretch = f' textLength="{_num(len(run.text) * style.cell_w)}"' if len(run.text) >= MIN_STRETCH else ""
    x, y = _num(style.pad + run.col * style.cell_w), _num(style.baseline(run.row))
    return f'<text x="{x}" y="{y}"{stretch}>{_escape(run.text)}</text>'


def _joint_text(rows: list[_LooseRow], style: Style) -> str:
    """One <text> for loose characters of several rows: x per character, dy only where the row changes."""
    xs, dys, chars = [], [], []
    previous = rows[0].row
    for loose in rows:
        xs.extend(loose.xs)
        chars.extend(loose.chars)
        dys.append(_num((loose.row - previous) * style.cell_h))
        dys.extend(["0"] * (len(loose.xs) - 1))
        previous = loose.row
    while dys and dys[-1] == "0":  # trailing zero shifts are implied
        dys.pop()
    dy = f' dy="{" ".join(dys)}"' if dys else ""
    return f'<text x="{" ".join(xs)}" y="{_num(style.baseline(rows[0].row))}"{dy}>{"".join(chars)}</text>'


def _layer(frame: Frame, classes: np.ndarray, style: Style, similar: np.ndarray | None = None) -> str:
    """All runs of a frame, one <g class> per colour; each run in whichever layout is shorter."""
    own: dict[int, list[str]] = {}
    loose: dict[int, list[_LooseRow]] = {}
    for run in _frame_runs(frame, classes, similar):
        markup = _own_text(run, style)
        glyphs = [(i, ch) for i, ch in enumerate(run.text) if ch != " "]
        xs = [_num(style.pad + (run.col + i) * style.cell_w) for i, _ in glyphs]
        if sum(len(x) + LOOSE_CHAR_BYTES for x in xs) >= len(markup):
            own.setdefault(run.cls, []).append(markup)
            continue
        rows = loose.setdefault(run.cls, [])
        if not rows or rows[-1].row != run.row:
            rows.append(_LooseRow(run.row, [], []))
        rows[-1].xs.extend(xs)
        rows[-1].chars.extend(_escape(ch) for _, ch in glyphs)
    out = []
    for c in sorted(own.keys() | loose.keys()):
        body = "".join(own.get(c, ())) + (_joint_text(loose[c], style) if c in loose else "")
        out.append(f'<g class="{CLASS_NAMES[c]}">{body}</g>')
    return "".join(out)


def encode_svg(
    frames: Sequence[Frame],
    cmap,
    *,
    fps: float,
    style: Style,
    title: str,
    background: str | None = None,
    static: Frame | None = None,
    twinkles: Sequence[Twinkle] = (),
    hero: int = 0,
    description: str = "",
) -> str:
    """Encode frames into an animated SVG string. `cmap` is a NearestColors or MonoLevels."""
    if not frames:
        raise ValueError("need at least one frame")
    shape = frames[0].shape
    if any(f.shape != shape for f in frames) or (static is not None and static.shape != shape):
        raise ValueError("all frames must have the same size")
    if len(cmap.colors) > len(CLASS_NAMES):
        raise ValueError("too many colours")
    rows, cols = shape
    n = len(frames)
    width, height = style.size(rows, cols)
    loop = n / fps
    hero = hero % n

    # Each frame stays visible FRAME_OVERLAP_S past the next one's start. Without it, rounding in the
    # written times leaves microsecond gaps, and Firefox sometimes paints in one: a blank flash.
    shown = min(100.0, 100.0 * (1.0 / n + FRAME_OVERLAP_S / loop))  # a lone frame simply stays
    css = [
        f"text{{font-family:{FONT_STACK};font-size:{_num(style.font_size)}px;white-space:pre}}",
        f".fr{{visibility:hidden;animation:fr {loop:.6f}s step-end infinite}}",
        f"@keyframes fr{{0%{{visibility:visible}}{shown:.6f}%{{visibility:hidden}}100%{{visibility:hidden}}}}",
        "@keyframes tw{0%,100%{opacity:.15}50%{opacity:1}}",
        "@media (prefers-reduced-motion:reduce){.fr,.tw{animation:none}.hero{visibility:visible}}",
    ]
    css += [f".{CLASS_NAMES[i]}{{fill:{c}}}" for i, c in enumerate(cmap.colors)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_num(width)} {_num(height)}" '
        f'width="{_num(width * style.scale)}" height="{_num(height * style.scale)}" role="img" '
        f'aria-labelledby="title" xml:space="preserve" data-frames="{n}" data-fps="{_num(fps)}">',
        f"<title id=\"title\">{_escape(title)}</title>",
    ]
    if description:
        parts.append(f"<desc>{_escape(description)}</desc>")
    if style.glow > 0:
        parts.append(
            f'<defs><filter id="glow" x="-5%" y="-5%" width="110%" height="110%">'
            f'<feGaussianBlur stdDeviation="{_num(style.glow)}" result="b"/>'
            f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
        )
    parts.append(f"<style>{''.join(css)}</style>")
    parts.append(
        f'<rect width="{_num(width)}" height="{_num(height)}" rx="{_num(style.corner_radius)}" '
        f'fill="{background or style.background}"/>'
    )
    parts.append('<g filter="url(#glow)">' if style.glow > 0 else "<g>")
    if static is not None:
        parts.append(f"<g>{_layer(static, cmap.classify(static.rgb), style)}</g>")
    for tw in twinkles:
        x = style.pad + tw.col * style.cell_w
        cls = CLASS_NAMES[int(cmap.classify(np.asarray(tw.rgb, dtype=np.float64)))]
        parts.append(
            f'<text class="tw {cls}" x="{_num(x)}" y="{_num(style.baseline(tw.row))}" '
            f'style="animation:tw {tw.period:.2f}s ease-in-out {tw.delay:.2f}s infinite">{_escape(tw.char)}</text>'
        )
    dt = 1.0 / fps
    for k, frame in enumerate(frames):
        cls = "fr hero" if k == hero else "fr"
        delay = f' style="animation-delay:{k * dt:.6f}s"' if k else ""
        body = _layer(frame, cmap.classify(frame.rgb), style, cmap.similar)
        parts.append(f'<g class="{cls}"{delay}>{body}</g>')
    parts.append("</g></svg>")
    return "".join(parts)


# ---------------------------------------------------------------- helpers for tooling

def size_report(svg: str) -> tuple[int, int]:
    """(raw bytes, gzip bytes) — GitHub serves the file gzip-compressed."""
    raw = svg.encode("utf-8")
    return len(raw), len(gzip.compress(raw, 9))


def write_text_dump(frames: Sequence[Frame], path: Path) -> None:
    """Plain-text dump of every frame, handy for reading an animation in a terminal or a diff."""
    blocks = [f"--- frame {k} ---\n" + "\n".join(f.lines()) for k, f in enumerate(frames)]
    Path(path).write_text("\n".join(blocks) + "\n", encoding="ascii")


def class_colors_rgb(cmap) -> np.ndarray:
    return np.array([hex_to_rgb(c) for c in cmap.colors])
