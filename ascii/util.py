"""Numeric helpers shared by scenes: shaping functions, rotations, supersampling, textures, noise, colour."""
from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi


# ---------------------------------------------------------------- shaping

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(edge0, edge1, x):
    t = clamp01((np.asarray(x, dtype=np.float64) - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------- rotations (column vectors: v' = R @ v)

def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------- supersampling

def subcell_centres(rows: int, cols: int, sy: int, sx: int):
    """Centres of a (rows*sy) x (cols*sx) supersampling grid, measured in character cells.

    Returns (y, x): y grows downwards from 0 to rows, x rightwards from 0 to cols.
    Remember a cell is Style.cell_aspect times taller than wide when turning these into geometry.
    """
    y = (np.arange(rows * sy) + 0.5) / sy
    x = (np.arange(cols * sx) + 0.5) / sx
    return np.meshgrid(y, x, indexing="ij")


def to_blocks(a, sy: int, sx: int) -> np.ndarray:
    """Regroup a supersampled image (rows*sy, cols*sx[, ch]) into (rows, cols, sy*sx[, ch]).

    Samples inside a cell are ordered row-major: top-left, top-right, ..., bottom-right.
    """
    a = np.asarray(a)
    rows, cols = a.shape[0] // sy, a.shape[1] // sx
    rest = a.shape[2:]
    b = a.reshape(rows, sy, cols, sx, *rest)
    b = np.moveaxis(b, 2, 1)  # rows, cols, sy, sx, ...
    return b.reshape(rows, cols, sy * sx, *rest)


def cell_mean(a, sy: int, sx: int) -> np.ndarray:
    """Average a supersampled image back to one value (or colour) per character cell."""
    return to_blocks(a, sy, sx).mean(axis=2)


def cell_weighted_mean(values, weights, sy: int, sx: int) -> np.ndarray:
    """Per-cell average of `values` (rows*sy, cols*sx[, ch]) weighted by `weights` (rows*sy, cols*sx).

    Typical use: the colour of a character is the brightness-weighted mean of its sub-samples,
    so a cell that is half bright land and half black space takes the land's colour.
    """
    v = to_blocks(values, sy, sx)
    w = to_blocks(weights, sy, sx)
    if v.ndim == w.ndim + 1:
        w = w[..., None]
    total = w.sum(axis=2)
    return (v * w).sum(axis=2) / np.where(total > 1e-9, total, 1.0)


# ---------------------------------------------------------------- textures

def sample_equirect(tex: np.ndarray, lon, lat) -> np.ndarray:
    """Bilinear lookup in an equirectangular texture; longitude wraps around.

    tex: (h, w) or (h, w, ch) float array. lon in radians, 0 at the texture centre, east positive.
    lat in radians, north positive.
    """
    h, w = tex.shape[:2]
    u = (np.asarray(lon, dtype=np.float64) / TAU + 0.5) * w - 0.5
    v = (0.5 - np.asarray(lat, dtype=np.float64) / np.pi) * h - 0.5
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    fx = u - x0
    fy = v - y0
    x1 = (x0 + 1) % w
    x0 = x0 % w
    y1 = np.clip(y0 + 1, 0, h - 1)
    y0 = np.clip(y0, 0, h - 1)
    if tex.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = tex[y0, x0] * (1.0 - fx) + tex[y0, x1] * fx
    bottom = tex[y1, x0] * (1.0 - fx) + tex[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


# ---------------------------------------------------------------- noise

_M32 = np.uint64(0xFFFFFFFF)


def _hash3(ix, iy, iz, seed: int) -> np.ndarray:
    h = (
        ix.astype(np.uint64) * np.uint64(0x8DA6B343)
        ^ iy.astype(np.uint64) * np.uint64(0xD8163841)
        ^ iz.astype(np.uint64) * np.uint64(0xCB1AB31F)
        ^ np.uint64(seed & 0xFFFFFFFF) * np.uint64(0x165667B1)
    ) & _M32
    h = ((h ^ (h >> np.uint64(13))) * np.uint64(0x5BD1E995)) & _M32
    h ^= h >> np.uint64(15)
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise3(x, y, z, seed: int = 0) -> np.ndarray:
    """Smooth 3D value noise in [0, 1], vectorised over arrays of any shape."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    xf, yf, zf = np.floor(x), np.floor(y), np.floor(z)
    ux, uy, uz = _fade(x - xf), _fade(y - yf), _fade(z - zf)
    xi, yi, zi = xf.astype(np.int64), yf.astype(np.int64), zf.astype(np.int64)

    def corner(dx, dy, dz):
        return _hash3(xi + dx, yi + dy, zi + dz, seed)

    x00 = mix(corner(0, 0, 0), corner(1, 0, 0), ux)
    x10 = mix(corner(0, 1, 0), corner(1, 1, 0), ux)
    x01 = mix(corner(0, 0, 1), corner(1, 0, 1), ux)
    x11 = mix(corner(0, 1, 1), corner(1, 1, 1), ux)
    return mix(mix(x00, x10, uy), mix(x01, x11, uy), uz)


def fbm3(x, y, z, octaves: int = 5, lacunarity: float = 2.0, gain: float = 0.5, seed: int = 0) -> np.ndarray:
    """Fractal sum of value noise, normalised to [0, 1]."""
    total = np.zeros(np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape)
    amp, freq, norm = 1.0, 1.0, 0.0
    for octave in range(octaves):
        x_, y_, z_ = np.asarray(x) * freq, np.asarray(y) * freq, np.asarray(z) * freq
        total = total + amp * value_noise3(x_, y_, z_, seed + 101 * octave)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


# ---------------------------------------------------------------- colour (sRGB <-> linear <-> OKLab)

def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


_RGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_LMS_TO_LAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_LAB_TO_LMS = np.array([
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
])
_LMS_TO_RGB = np.array([
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010],
])


def linear_to_oklab(rgb) -> np.ndarray:
    lms = np.asarray(rgb, dtype=np.float64) @ _RGB_TO_LMS.T
    return np.cbrt(lms) @ _LMS_TO_LAB.T


def oklab_to_linear(lab) -> np.ndarray:
    lms = (np.asarray(lab, dtype=np.float64) @ _LAB_TO_LMS.T) ** 3
    return lms @ _LMS_TO_RGB.T


def srgb_to_oklab(rgb) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab) -> np.ndarray:
    return linear_to_srgb(oklab_to_linear(lab))


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(float(v) * 255)) for v in np.clip(rgb, 0.0, 1.0))
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def oklch_to_hex(lightness: float, chroma: float, hue_deg: float) -> str:
    h = np.deg2rad(hue_deg)
    lab = np.array([lightness, chroma * np.cos(h), chroma * np.sin(h)])
    return rgb_to_hex(oklab_to_srgb(lab))


# ---------------------------------------------------------------- colour runs

def merge_short_runs(visible: np.ndarray, idx: np.ndarray, kind: np.ndarray, min_len: int) -> np.ndarray:
    """Recolour colour runs shorter than `min_len` into their longer neighbour of the same kind.

    visible: (rows, cols) bool, the cells that show a character (runs are counted over these only);
    idx: (rows, cols) palette indices; kind[i]: the group of palette entry i (e.g. sea, land, lights),
    so a run is only ever absorbed by a neighbour of its own group. Every colour change along a row
    costs bytes in the SVG, so this keeps colour fields coherent. Returns a new array.
    """
    out = idx.copy()
    for r in range(idx.shape[0]):
        cells = np.flatnonzero(visible[r])
        if cells.size >= 2:
            out[r, cells] = _merge_row(out[r, cells], kind, min_len)
    return out


def _merge_row(seq: np.ndarray, kind: np.ndarray, min_len: int) -> np.ndarray:
    seq = seq.copy()
    for _ in range(seq.size):  # every merge removes a run, so this always ends
        edges = np.flatnonzero(np.diff(seq)) + 1
        starts, ends = np.r_[0, edges], np.r_[edges, seq.size]
        target = _merge_target(seq, starts, ends - starts, kind, min_len)
        if target is None:
            break
        run, neighbour = target
        seq[starts[run]:ends[run]] = seq[starts[neighbour]]
    return seq


def _merge_target(
    seq: np.ndarray, starts: np.ndarray, lens: np.ndarray, kind: np.ndarray, min_len: int,
) -> tuple[int, int] | None:
    """(shortest mergeable run, its longer same-kind neighbour), or None when nothing is left to merge."""
    for i in np.argsort(lens, kind="stable"):
        if lens[i] >= min_len:
            return None
        own = kind[seq[starts[i]]]
        same = [j for j in (i - 1, i + 1) if 0 <= j < starts.size and kind[seq[starts[j]]] == own]
        if same:
            return int(i), max(same, key=lambda j: lens[j])
    return None
