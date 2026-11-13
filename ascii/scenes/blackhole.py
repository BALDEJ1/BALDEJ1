"""A Gargantua-style black hole, computed by ray tracing light through curved space.

Method, in plain words
----------------------
1. Camera. Every sub-sample of the character grid is one ray sent backwards from a distant camera that
   looks at the hole from 81 degrees above its spin axis, so the accretion disk is seen almost edge-on.
2. Light bending. A ray with impact parameter b travels in one plane through the hole. In that plane
   its inverse radius u(phi) obeys the Schwarzschild photon equation (Binet form, M = 1)
       u'' + u = 3 u^2
   which is integrated with Runge-Kutta steps. Rays with b below 3*sqrt(3) fall in (the black shadow).
3. Where the ray meets the disk plane. The plane crosses the ray's orbital plane at a known angle,
   and again every half turn after that. Crossing number 0 is the direct image of the disk, crossing 1
   is the light that went round the back of the hole (the arc over the top and under the bottom),
   crossing 2 and 3 are the thin, faint echoes. At each crossing the integration gives the disk radius
   hit and the azimuth. All of this is computed once, in the constructor.
4. Disk light. Temperature falls with radius like r^(-3/4) (1 - sqrt(r_in/r))^(1/4). The gas orbits at
   the Kepler speed, so the side moving towards us is Doppler-boosted and blue-shifted and the
   far-moving side is dim and red; gravity also reddens the inner edge. Colour comes from a
   blackbody-like ramp (white-hot, yellow, orange, deep red).
5. Motion. A turbulent streaky texture is stored once in polar coordinates. Each frame it is looked up
   with the azimuth shifted by 2 pi N(r) k / n: the ring at radius r turns N(r) times per loop, and N
   falls like r^(-3/2) (Kepler). N is blended between its two neighbouring whole numbers, so the
   shear is smooth and the loop closes exactly.
6. Background. Each escaping ray also gives the direction it came from; stars are looked up in a fixed
   star map along that direction, so they are bent into arcs around the hole (the Einstein ring).
7. Characters. The brightness image is supersampled 6 x 4 per cell and turned into glyphs by
   shape matching, so edges become / \\ ( ) _ ' instead of staircases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine import SPACE, Frame, Style, Twinkle
from ..glyphs import SYMBOLS, pool_regions, shape_chars
from ..util import TAU, cell_weighted_mean, sample_equirect, smoothstep, subcell_centres, value_noise3
from .base import SceneBase

# ---------------------------------------------------------------- constants
SY, SX = 6, 4                       # sub-samples per cell
INCLINATION = np.deg2rad(81.0)      # camera angle from the disk's spin axis
R_IN, R_OUT = 4.2, 11.5             # disk edges in units of M (artistic inner edge, spinning-hole style)
SHADOW_B = 3.0 * np.sqrt(3.0)       # critical impact parameter of a Schwarzschild hole
N_ORDERS = 4                        # disk images per ray: direct, round the back, two echoes
PHI_STEP = 0.02                     # integration step in orbital-plane angle
HORIZON_U = 0.5                     # u = 1 / r reaches 1/2 at the horizon
KEPLER_TURNS = 4.0                  # whole turns per loop at the inner edge
FRAME_SKEW = 1.6                   # log-spiral twist of the texture
TEX_R, TEX_T = 480, 1440            # polar texture resolution
SKY_SIZE = (1536, 3072)             # star map (rows, cols)
SKY_SCALE = 45.0                    # sky angle of a pixel = b / SKY_SCALE (radians)
STAR_COUNT = 1500
PHOTON_RING_WIDTH = 0.16
# turbulence octaves of the disk texture: (radial frequency, frequency around, amplitude);
# the last, finest one gives the thin filaments the denser grid can now show
TEXTURE_OCTAVES = ((0.5, 2.4, 1.0), (1.1, 4.0, 0.9), (2.4, 6.5, 0.7), (5.0, 11.0, 0.45))
ROWS, COLS = 62, 146                # 1.4 times the old 44 x 104 grid in each direction
BASE_STYLE = Style()
LUM_GAMMA, LUM_CAP = 0.85, 0.94         # lift dim glyphs, keep @ for the core only
BLEND_LOW, BLEND_HIGH = 0.08, 1.60  # texture modulation of disk brightness
EXPOSURE = 0.7                      # lower = fewer saturated '@', so the gas streaks show

# hue ramp of the disk: observed temperature -> colour (max channel 1; brightness is applied separately)
RAMP_T = np.array([0.0, 0.20, 0.38, 0.60, 0.85, 1.15, 1.6])
RAMP_RGB = np.array([
    [0.60, 0.03, 0.02],
    [0.92, 0.10, 0.03],
    [1.00, 0.28, 0.04],
    [1.00, 0.52, 0.10],
    [1.00, 0.76, 0.28],
    [1.00, 0.92, 0.66],
    [0.92, 0.96, 1.00],
])


# ---------------------------------------------------------------- ray tracing

@dataclass(frozen=True)
class RayHits:
    """Result of tracing every sub-sample: disk crossings and where escaping rays came from."""

    radius: np.ndarray    # (orders, n) disk radius of each crossing, NaN where the ray misses the disk
    azimuth: np.ndarray   # (orders, n) azimuth of the crossing in the disk plane
    sky: np.ndarray       # (n, 3) direction (camera frame: right, up, away) a ray came from, NaN if captured


def _rk4_step(u: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def acc(x: np.ndarray) -> np.ndarray:
        return -x + 3.0 * x * x

    h = PHI_STEP
    k1u, k1w = w, acc(u)
    k2u, k2w = w + 0.5 * h * k1w, acc(u + 0.5 * h * k1u)
    k3u, k3w = w + 0.5 * h * k2w, acc(u + 0.5 * h * k2u)
    k4u, k4w = w + h * k3w, acc(u + h * k3u)
    return (u + h / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u),
            w + h / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w))


def _crossing_angles(b: np.ndarray, y_img: np.ndarray) -> np.ndarray:
    """Orbital-plane angles at which each ray crosses the disk plane: first one, then one per half turn."""
    first = np.mod(np.arctan2(-np.cos(INCLINATION), np.sin(INCLINATION) * y_img / b), np.pi)
    return first[None, :] + np.pi * np.arange(N_ORDERS)[:, None]


def _hermite(t: np.ndarray, u0: np.ndarray, w0: np.ndarray, u1: np.ndarray, w1: np.ndarray) -> np.ndarray:
    t2, t3 = t * t, t * t * t
    h = PHI_STEP
    return ((2 * t3 - 3 * t2 + 1) * u0 + (t3 - 2 * t2 + t) * h * w0
            + (-2 * t3 + 3 * t2) * u1 + (t3 - t2) * h * w1)


def integrate_rays(b: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """March every ray through the Binet equation.

    Returns u at each target angle (NaN once the ray has fallen in or left) and the angle at which
    the ray escaped to infinity (NaN if it never did).
    """
    n_rays = b.size
    u, w = np.zeros(n_rays), 1.0 / b
    phi_end = np.full(n_rays, np.inf)
    phi_escape = np.full(n_rays, np.nan)
    u_at = np.full(targets.shape, np.nan)
    step_of = np.floor(targets / PHI_STEP).astype(np.int64)
    sorter = [np.argsort(step_of[k], kind="stable") for k in range(N_ORDERS)]
    sorted_steps = [step_of[k][sorter[k]] for k in range(N_ORDERS)]
    for j in range(int(np.ceil(N_ORDERS * np.pi / PHI_STEP)) + 1):
        u1, w1 = _rk4_step(u, w)
        live = np.isinf(phi_end)
        fell = live & (u1 >= HORIZON_U)
        left = live & (u1 <= 0.0) & ~fell
        phi_end[fell] = j * PHI_STEP
        phi_escape[left] = j * PHI_STEP + PHI_STEP * u[left] / np.maximum(u[left] - u1[left], 1e-12)
        phi_end[left] = phi_escape[left]
        for k in range(N_ORDERS):
            lo = np.searchsorted(sorted_steps[k], j, "left")
            hi = np.searchsorted(sorted_steps[k], j, "right")
            if hi == lo:
                continue
            idx = sorter[k][lo:hi]
            t = (targets[k, idx] - j * PHI_STEP) / PHI_STEP
            val = _hermite(t, u[idx], w[idx], u1[idx], w1[idx])
            ok = (targets[k, idx] < phi_end[idx]) & (val > 0.0)
            u_at[k, idx[ok]] = val[ok]
        dead = ~np.isinf(phi_end)
        u = np.where(dead, u, u1)
        w = np.where(dead, w, w1)
    return u_at, phi_escape


def trace(x_img: np.ndarray, y_img: np.ndarray) -> RayHits:
    """Trace rays for image-plane positions (in units of M, x to the right, y up)."""
    x, y = x_img.ravel(), y_img.ravel()
    b = np.hypot(x, y)
    targets = _crossing_angles(b, y)
    u_at, phi_escape = integrate_rays(b, targets)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = 1.0 / u_at
    radius = np.where((radius >= R_IN) & (radius <= R_OUT), radius, np.nan)
    s_inc, c_inc = np.sin(INCLINATION), np.cos(INCLINATION)
    cos_t, sin_t = np.cos(targets), np.sin(targets)
    px = -y / b * c_inc                       # in-plane unit vector p, disk coordinates
    py = x / b
    gx = cos_t * s_inc + sin_t * px           # position direction at each crossing
    gy = sin_t * py
    azimuth = np.arctan2(gy, gx)
    # background direction in the camera frame (x right, y up, z towards the camera):
    # the camera axis is (0, 0, 1) and the ray's image direction is (x/b, y/b, 0)
    image_dir = np.stack([x / b, y / b, np.zeros_like(b)], axis=-1)
    axis = np.array([0.0, 0.0, 1.0])
    sky = np.cos(phi_escape)[:, None] * axis + (np.sin(phi_escape) + b / SKY_SCALE)[:, None] * image_dir
    return RayHits(radius, azimuth, sky)


# ---------------------------------------------------------------- disk light

def temperature_profile(r: np.ndarray) -> np.ndarray:
    """Relative disk temperature, normalised to a peak of 1."""
    grid = np.linspace(R_IN, R_OUT, 2000)

    def raw(x: np.ndarray) -> np.ndarray:
        return (x / R_IN) ** -0.75 * np.maximum(1.0 - np.sqrt(R_IN / x), 0.0) ** 0.25

    return raw(r) / raw(grid).max()


def hue_ramp(temp: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(temp, RAMP_T, RAMP_RGB[:, c]) for c in range(3)], axis=-1)


def doppler_factor(r: np.ndarray, x_img: np.ndarray) -> np.ndarray:
    """Frequency ratio of disk light seen at the camera: gravity plus Kepler motion (left side approaches)."""
    omega = r ** -1.5
    lam = -x_img * np.sin(INCLINATION)
    return np.sqrt(np.clip(1.0 - 3.0 / r, 1e-4, 1.0)) / np.maximum(1.0 - omega * lam, 0.12)


def disk_texture(seed: int = 3) -> np.ndarray:
    """Streaky turbulence in polar coordinates (radius rows, azimuth columns), periodic in azimuth."""
    r = np.linspace(R_IN, R_OUT, TEX_R)[:, None]
    th = np.linspace(0.0, TAU, TEX_T, endpoint=False)[None, :]
    twisted = th + FRAME_SKEW * np.log(r / R_IN)
    total = np.zeros((TEX_R, TEX_T))
    norm = 0.0
    for octave, (radial, around, amp) in enumerate(TEXTURE_OCTAVES):
        total += amp * value_noise3(r * radial + 7 * octave, around * np.cos(twisted), around * np.sin(twisted),
                                    seed + octave)
        norm += amp
    tex = total / norm
    return smoothstep(0.30, 0.72, tex)


def sample_polar(tex: np.ndarray, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Bilinear lookup in the polar texture; azimuth wraps."""
    fr = np.clip((r - R_IN) / (R_OUT - R_IN), 0.0, 1.0) * (TEX_R - 1)
    ft = np.mod(theta, TAU) / TAU * TEX_T
    r0 = np.minimum(fr.astype(np.int64), TEX_R - 2)
    t0 = np.floor(ft).astype(np.int64)
    ar, at = fr - r0, ft - t0
    t0 %= TEX_T
    t1 = (t0 + 1) % TEX_T
    top = tex[r0, t0] * (1 - at) + tex[r0, t1] * at
    bot = tex[r0 + 1, t0] * (1 - at) + tex[r0 + 1, t1] * at
    return top * (1 - ar) + bot * ar


@dataclass(frozen=True)
class DiskLayer:
    """One disk image (one crossing order): everything that does not change from frame to frame."""

    pixels: np.ndarray    # flat sub-sample indices
    radius: np.ndarray
    azimuth: np.ndarray
    alpha: np.ndarray     # opacity (soft outer edge)
    base: np.ndarray      # brightness before the texture
    hue: np.ndarray       # (m, 3) colour at full brightness
    turns_low: np.ndarray  # whole turns per loop of the two neighbouring rings
    blend: np.ndarray     # weight of the faster ring


def _make_layer(order: int, hits: RayHits, x_flat: np.ndarray) -> DiskLayer:
    ok = np.nonzero(np.isfinite(hits.radius[order]))[0]
    r = hits.radius[order][ok]
    g = doppler_factor(r, x_flat[ok])
    t_obs = temperature_profile(r) * g
    base = np.clip(1.25 * temperature_profile(r) ** 1.5 * g ** 1.0, 0.0, 4.0)
    alpha = smoothstep(R_IN, R_IN + 0.25, r) * (1.0 - smoothstep(R_OUT - 4.0, R_OUT, r)) * 0.96
    n_real = KEPLER_TURNS * (R_IN / r) ** 1.5
    low = np.floor(n_real)
    echo_fade = 0.92 ** order
    return DiskLayer(ok, r, hits.azimuth[order][ok], alpha, base * echo_fade, hue_ramp(t_obs * 0.95),
                     low, n_real - low)


def tone(e: np.ndarray) -> np.ndarray:
    """Map unbounded emission to [0, 1] with a soft shoulder."""
    return 1.0 - np.exp(-EXPOSURE * e)


# ---------------------------------------------------------------- sky

def make_sky(seed: int = 11) -> np.ndarray:
    """Equirectangular star map (brightness, tint) around the viewing direction."""
    rng = np.random.default_rng(seed)
    h, w = SKY_SIZE
    sky = np.zeros((h, w, 2))
    lon = rng.uniform(-1.7, 1.7, STAR_COUNT)
    lat = rng.uniform(-1.1, 1.1, STAR_COUNT)
    for lo, la in zip(lon, lat, strict=True):
        mag = rng.random() ** 1.6
        tint = rng.random()
        cx = (lo / TAU + 0.5) * w
        cy = (0.5 - la / np.pi) * h
        x0, y0 = int(cx) - 4, int(cy) - 4
        yy, xx = np.mgrid[y0:y0 + 9, x0:x0 + 9]
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.4 ** 2)) * (0.45 + 0.9 * mag)
        sl = (slice(y0, y0 + 9), slice(x0, x0 + 9))
        sky[sl + (0,)] = np.maximum(sky[sl + (0,)], blob)
        sky[sl + (1,)] = np.where(blob > 0.05, tint, sky[sl + (1,)])
    return sky


def sky_lookup(sky: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Star brightness (m, 2) for directions (m, 3) in the camera frame (x right, y up, z towards camera)."""
    lon = np.arctan2(direction[:, 0], -direction[:, 2])
    lat = np.arcsin(np.clip(direction[:, 1], -1.0, 1.0))
    return sample_equirect(sky, lon, lat)


# ---------------------------------------------------------------- scene

class BlackHole(SceneBase):
    title = "ASCII black hole"
    rows, cols = ROWS, COLS
    style = BASE_STYLE.as_wide_as(COLS)
    # (smaller characters: the picture is exactly as wide as the Earth, about 749 px)
    frames, fps = 84, 12.0
    max_colors = 24
    mono_levels = 8
    hero = 0

    def __init__(self) -> None:
        self.cells_per_m = (self.cols / 2 - 2.0) / (R_OUT * 1.02)
        y, x = subcell_centres(self.rows, self.cols, SY, SX)
        self.x_img = (x - self.cols / 2) / self.cells_per_m
        self.y_img = (self.rows / 2 - y) * self.style.cell_aspect / self.cells_per_m
        x_flat, y_flat = self.x_img.ravel(), self.y_img.ravel()
        self.b = np.hypot(x_flat, y_flat)
        hits = trace(self.x_img, self.y_img)
        self.shape = self.x_img.shape
        self.layers = [_make_layer(k, hits, x_flat) for k in range(N_ORDERS)]
        self.texture = disk_texture()
        self.shadow = self.b < SHADOW_B * 0.995
        self.ring = self._photon_ring(x_flat)
        self.stars = self._stars(hits)

    # -- fixed pieces
    def _photon_ring(self, x_flat: np.ndarray) -> np.ndarray:
        """A thin bright ring hugging the shadow, brighter on the approaching side."""
        profile = np.exp(-(((self.b - SHADOW_B * 1.012) / PHOTON_RING_WIDTH) ** 2))
        side = 0.80 + 0.20 * np.clip(-x_flat / np.maximum(self.b, 1e-9), -1.0, 1.0)
        return 1.0 * profile * side * (self.b > SHADOW_B * 0.995)

    def _stars(self, hits: RayHits) -> np.ndarray:
        """Lensed star brightness per sub-sample (zero where the ray fell in or sees the disk)."""
        look = sky_lookup(make_sky(), np.nan_to_num(hits.sky))
        bright = look[:, 0] * np.isfinite(hits.sky[:, 0])
        return bright.reshape(self.shape)

    # -- frames
    def disk_light(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Brightness and hue of the disk and photon ring for loop phase t (flat arrays)."""
        n = self.b.size
        acc = np.zeros(n)
        acc_hue = np.zeros((n, 3))
        trans = np.ones(n)
        for layer in self.layers:
            ix = layer.pixels
            shift = TAU * t
            slow = sample_polar(self.texture, layer.radius, layer.azimuth - shift * layer.turns_low)
            fast = sample_polar(self.texture, layer.radius, layer.azimuth - shift * (layer.turns_low + 1))
            tex = slow * (1 - layer.blend) + fast * layer.blend
            lum = tone(layer.base * (BLEND_LOW + BLEND_HIGH * tex))
            w = trans[ix] * layer.alpha
            acc[ix] += w * lum
            acc_hue[ix] += (w * lum)[:, None] * layer.hue
            trans[ix] *= 1.0 - layer.alpha
        ring = self.ring * trans
        acc += ring
        acc_hue += ring[:, None] * np.array([1.0, 0.97, 0.92])
        hue = acc_hue / np.maximum(acc, 1e-6)[:, None]
        return np.clip(acc, 0.0, 1.0) ** LUM_GAMMA * LUM_CAP, hue

    def render(self, k: int, n: int) -> Frame:
        lum, hue = self.disk_light(k / n)
        lum_img = lum.reshape(self.shape)
        chars = shape_chars(pool_regions(lum_img, SY, SX), SYMBOLS, contrast=1.25)
        colour = hue.reshape(*self.shape, 3) * (0.36 + 0.64 * lum_img[..., None])
        rgb = cell_weighted_mean(colour, lum_img, SY, SX)
        return Frame(chars, np.clip(rgb, 0.0, 1.0))

    # -- background
    def _occupied(self) -> np.ndarray:
        """Cells where the black hole or its disk ever draws something."""
        mask = self.shadow.copy()
        for layer in self.layers:
            mask[layer.pixels] = True
        mask |= self.ring > 0.02
        cells = mask.reshape(self.rows, SY, self.cols, SX).any(axis=(1, 3))
        padded = cells.copy()
        padded[1:] |= cells[:-1]
        padded[:-1] |= cells[1:]
        padded[:, 1:] |= cells[:, :-1]
        padded[:, :-1] |= cells[:, 1:]
        return padded

    def static(self) -> Frame:
        keep = ~self._occupied()
        ink = self.stars * (~self.shadow.reshape(self.shape))
        regions = pool_regions(np.clip(ink * 1.6, 0.0, 1.0), SY, SX)
        chars = shape_chars(regions, " .'`:+*", contrast=1.0)
        keep[:2], keep[-2:], keep[:, :3], keep[:, -3:] = False, False, False, False
        chars = np.where(keep, chars, SPACE).astype(np.uint8)
        brightness = np.clip(cell_weighted_mean(ink, ink, SY, SX) * 1.8 + 0.25, 0.0, 1.0)
        rgb = np.stack([0.80 * brightness, 0.86 * brightness, 1.0 * brightness], axis=-1)
        return Frame(chars, rgb)

    def twinkles(self) -> list[Twinkle]:
        return []


def make_scene() -> BlackHole:
    return BlackHole()
