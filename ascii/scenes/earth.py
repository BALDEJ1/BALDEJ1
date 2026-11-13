"""ASCII Earth: a turning globe computed from NASA maps.

Method, in plain words:
  1. Two NASA maps are loaded once: Blue Marble with topography and sea-floor shading (the planet by
     day) and Black Marble (city lights by night), both flat maps of longitude and latitude. From the
     day map every map pixel gets: land or sea; the continental shelf (shallow sea close to a coast);
     relief (how much brighter or darker a pixel is than the land around it: ridges and valleys);
     roughness (how strong that relief is nearby: mountains); and a biome (jungle, forest, steppe, dry
     land, sand, red desert, rock, ice), smoothed by a majority filter so biomes form coherent fields.
  2. For every tiny sample inside a character cell we ask "which spot on the globe is behind this
     point?" (orthographic sphere, axial tilt of 23.4 degrees, spun by the loop phase) and look it up.
  3. A fixed Sun lights the globe (Lambert shading). Every cell becomes sunlit sea, sunlit land, night
     with city lights, or the thin atmosphere ring on the sunny side of the limb.
  4. Characters: the sea is light dots, a little denser on the shelf; land is denser where the map is
     brighter, the relief rises and the Sun is higher; '^' marks mountains; coast cells use shaped
     characters ( / \\ ( ) ` ' , ) so coastlines read as lines; at night only the cities glow.
  5. Colour: a land cell takes the biome that covers most of it; the sea is deep or shelf blue, dimmer
     towards the limb. Colour runs shorter than MIN_RUN cells join a neighbour of the same kind, since
     every colour change costs bytes in the SVG.
  6. One full turn of the planet is one loop, so the seam is invisible.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageFilter

from ..engine import SPACE, Frame, Style, Twinkle
from ..glyphs import glyph_vector, hybrid_chars, pick_by_coverage, pool_regions
from ..util import (
    TAU,
    cell_mean,
    hex_to_rgb,
    merge_short_runs,
    normalize,
    rot_x,
    rot_y,
    rot_z,
    sample_equirect,
    smoothstep,
    srgb_to_oklab,
    subcell_centres,
    to_blocks,
)
from .base import SceneBase, starfield

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DAY_MAP = DATA_DIR / "raw" / "day" / "bmng_topo_bathy_3600x1800.jpg"
NIGHT_MAP = DATA_DIR / "raw" / "night" / "blackmarble_2016_3600x1800.jpg"
TEXTURE_SIZE = (2048, 1024)  # about two map pixels per sample, so sampling does not alias
ROWS, COLS = 70, 140
BASE_STYLE = Style()
SY, SX = 6, 4                # sub-samples per cell
RADIUS_SHARE = 0.42          # globe radius as a share of the row count
AXIAL_TILT = np.deg2rad(23.4)
VIEW_PITCH = np.deg2rad(11.0)  # north pole leans slightly towards the viewer
SUN = normalize([-0.78, 0.26, 0.56])  # fixed in space: upper left, a little in front
HERO_LONGITUDE = np.deg2rad(18.0)     # Europe and Africa face the viewer in the hero frame
STAR_CLEARANCE = 1.13
STAR_DENSITY = 0.007                  # stars per cell: half of 0.014 on twice the cells, so the count stays
STAR_COLOURS = np.array([(0.96, 0.96, 0.92), (1.00, 0.84, 0.42), (0.62, 0.86, 1.00)])  # white, pale yellow, pale blue

# map channels
R, G, B, LAND, LUMA, SHELF, RELIEF, ROUGH, LIGHTS = range(9)
EXPOSURE = 0.55              # gamma that brightens the dark photographic oceans for a text display
LUMA_WEIGHTS = np.array([0.30, 0.59, 0.11])

# map analysis, in map pixels (one character cell is about 5 x 10 of them)
RELIEF_SIGMA = 10.0          # ridges and valleys finer than this count as relief
ROUGH_SIGMA = 6.0            # how far roughness is averaged
NEAR_SIGMA, NEAR_LAND = 14.0, (0.10, 0.22)  # the shelf only counts within a few degrees of land
SHELF_EDGE = (0.20, 0.42)    # blue level of the map over which deep sea turns into shelf
BIOME_MODE_FILTER = 9        # window of the majority filter that makes biome fields coherent
ROCK_ROUGH, ROCK_CHROMA = 0.07, 0.075  # rough, grey land is rock
LIGHT_BLUR = 5.0             # city glow spreads over a few map pixels

# land biomes: (name, colour to recognise it on the map, colour to draw it with)
BIOMES = (
    ("jungle", "#2f5a12", "#33903f"),
    ("forest", "#5a7a16", "#7cc044"),
    ("steppe", "#8a8a30", "#b5b455"),
    ("dry", "#a58b52", "#cfa766"),
    ("sand", "#d3bc90", "#ecd48f"),
    ("red", "#b0643a", "#cc6a37"),
    ("rock", "#8a8076", "#a49b8f"),
    ("ice", "#eef3f8", "#eef3f8"),
)
ROCK, ICE = 6, 7
BIOME_TARGETS = srgb_to_oklab(np.array([hex_to_rgb(target) for _, target, _ in BIOMES]))

# lighting
DAY_EDGE = (-0.04, 0.22)     # sun height (cosine) over which a spot turns from night to day
NIGHT_EDGE = (-0.18, 0.10)   # city lights fade in below this sun height
SHADE_FLOOR, SHADE_GAMMA = 0.15, 0.75

# cell kinds
DISC_SHARE, DAY_SHARE = 0.5, 0.4  # a cell is on the globe / sunlit when more than this share of it is
COAST_SHARE = (0.16, 0.88)   # land share above which a cell is land, and below which it is also coast
SHELF_SHARE = 0.5            # sea share of a cell that must be shelf for the shelf colour
MIN_GLOW = 0.04              # dimmer city light is not drawn
LIGHT_HOT = 0.45             # brighter city light turns from amber to yellow
HALO_MIN = 0.30              # fainter atmosphere is not drawn (so only the sunny side shows)
SEA_BRIGHT_FROM = 0.45       # sunlight above which the sea takes its brighter colour

# drawing land
LAND_BASE_INK, LAND_LUMA_INK = 0.60, 0.16  # density of flat land: a constant plus map brightness
RELIEF_GAIN = 1.8            # how strongly relief changes land density
EDGE_THRESHOLD = 0.55        # how uneven a coast cell must be to get a shaped character
MOUNTAIN_ROUGH = 0.075       # roughness over which a sunlit land cell is drawn as '^'
MOUNTAIN_MIN_SHADE = 0.25    # no '^' in the dusk, where relief cannot be seen
SHELF_INK = 0.12             # the shelf is drawn a little denser than the deep sea
MIN_RUN = 4                  # colour runs shorter than this join a neighbour of the same kind

# characters, ordered by measured ink inside pick_by_coverage
SEA_GLYPHS = " .:"
LAND_GLYPHS = " .:+*#@"
EDGE_GLYPHS = "/\\()`',"
LIGHT_GLYPHS = " .:+*#@"
MOUNTAIN_GLYPH = ord("^")

# colours, as palette indices so runs can be merged by kind before the frame is built
SEA_COLOURS = np.array([hex_to_rgb(h) for h in ("#2a5aa5", "#5b9be8", "#3f93c9", "#7ccbee")])
#                                               deep dim   deep bright shelf dim  shelf bright
LAND_COLOURS = np.array([hex_to_rgb(colour) for _, _, colour in BIOMES])
LIGHT_COLOURS = np.array([hex_to_rgb(h) for h in ("#ff9a2e", "#ffd76e")])  # amber, hot
HAZE_COLOUR = hex_to_rgb("#8fd3ff")
PALETTE = np.vstack([SEA_COLOURS, LAND_COLOURS, LIGHT_COLOURS, HAZE_COLOUR[None]])
SEA_BASE, LAND_BASE = 0, len(SEA_COLOURS)
LIGHT_BASE = LAND_BASE + len(LAND_COLOURS)
HAZE_INDEX = LIGHT_BASE + len(LIGHT_COLOURS)
KIND = np.repeat(np.arange(4), [len(SEA_COLOURS), len(LAND_COLOURS), len(LIGHT_COLOURS), 1])


# ---------------------------------------------------------------- maps

def _gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur by FFT. It wraps around both ways: right across the date line, harmless at the poles."""
    ky = np.fft.fftfreq(a.shape[0])[:, None]
    kx = np.fft.rfftfreq(a.shape[1])[None, :]
    kernel = np.exp(-2.0 * (np.pi * sigma) ** 2 * (ky ** 2 + kx ** 2))
    return np.fft.irfft2(np.fft.rfft2(a) * kernel, s=a.shape)


def _land_blur(a: np.ndarray, land: np.ndarray, sigma: float) -> np.ndarray:
    """Blur that averages land pixels only, so a coastline is not mistaken for a mountain."""
    return _gauss(a * land, sigma) / np.maximum(_gauss(land, sigma), 1e-3)


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(TEXTURE_SIZE, Image.Resampling.BOX)
    return np.asarray(img, dtype=np.float64) / 255.0


def _day_map() -> np.ndarray:
    """(h, w, 8): display rgb, land, brightness, shelf, relief, roughness."""
    raw = _load_rgb(DAY_MAP)
    r, b = raw[..., 0], raw[..., 2]
    luma = raw @ LUMA_WEIGHTS
    polar = smoothstep(0.76, 0.90, np.abs(np.linspace(1.0, -1.0, raw.shape[0])))[:, None]
    snow = smoothstep(0.55, 0.75, raw.min(axis=-1)) * polar
    land = np.maximum(1.0 - smoothstep(0.0, 0.07, b - r), snow)
    hard = (land > 0.5).astype(np.float64)
    tone = 0.74 + 0.26 * polar  # pale deserts and steppe must not read as ice
    rgb = raw * (1.0 - land[..., None] * (1.0 - tone[..., None]))
    vivid = luma[..., None] + (rgb - luma[..., None]) * 1.45
    colour = np.clip(vivid * (1.0 + 0.3 * land[..., None]), 0.0, 1.0) ** EXPOSURE
    near_land = smoothstep(*NEAR_LAND, _gauss(hard, NEAR_SIGMA))
    shelf = smoothstep(*SHELF_EDGE, _gauss(b, 1.5)) * near_land * (1.0 - hard)
    relief = (luma - _land_blur(luma, hard, RELIEF_SIGMA)) * hard
    rough = _land_blur(np.abs(relief), hard, ROUGH_SIGMA)
    return np.concatenate([colour, *(c[..., None] for c in (land, luma, shelf, relief, rough))], -1)


def _biome_map(day: np.ndarray) -> np.ndarray:
    """Biome index per map pixel: nearest biome colour, rock where it is rough and grey, then smoothed."""
    lab = srgb_to_oklab(day[..., :3])
    nearest = ((lab[..., None, :] - BIOME_TARGETS) ** 2).sum(-1).argmin(-1)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    rocky = (day[..., ROUGH] > ROCK_ROUGH) & (chroma < ROCK_CHROMA) & (day[..., LAND] > 0.5) & (nearest != ICE)
    nearest = np.where(rocky, ROCK, nearest)
    smooth = Image.fromarray(nearest.astype(np.uint8)).filter(ImageFilter.ModeFilter(BIOME_MODE_FILTER))
    return np.asarray(smooth, dtype=np.int64)


def _lights_map() -> np.ndarray:
    """City light strength in [0, 1], softened so a city covers a few cells."""
    rgb = _load_rgb(NIGHT_MAP)
    warm = 0.5 * (rgb[..., 0] + rgb[..., 1]) - 0.5 * rgb[..., 2]
    raw = smoothstep(0.04, 0.34, warm) ** 0.8
    soft = Image.fromarray((raw * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(LIGHT_BLUR))
    glow = np.asarray(soft, dtype=np.float64) / 255.0
    return np.clip(np.maximum(raw, glow * 3.4), 0.0, 1.0)


class Maps(NamedTuple):
    values: np.ndarray  # (h, w, 9) channels R..LIGHTS, sampled with interpolation
    biome: np.ndarray   # (h, w) biome index, sampled nearest


def load_maps() -> Maps:
    day = _day_map()
    return Maps(np.concatenate([day, _lights_map()[..., None]], -1), _biome_map(day))


def sample_nearest(a: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    x = np.floor((lon / TAU + 0.5) * w).astype(np.int64) % w
    y = np.clip(np.floor((0.5 - lat / np.pi) * h).astype(np.int64), 0, h - 1)
    return a[y, x]


def star_palette(rgb: np.ndarray) -> np.ndarray:
    """Snap star colours to white, pale yellow or pale blue only."""
    lab = srgb_to_oklab(STAR_COLOURS)
    d = ((srgb_to_oklab(rgb)[..., None, :] - lab) ** 2).sum(-1)
    return STAR_COLOURS[d.argmin(-1)]


# ---------------------------------------------------------------- scene

class Cells(NamedTuple):
    """What one frame looks like, summarised per character cell."""

    disc: np.ndarray        # the cell lies on the globe
    land: np.ndarray        # the cell is drawn as land
    land_share: np.ndarray  # share of the cell that is land, 0..1
    day: np.ndarray         # the cell is mostly sunlit
    shade: np.ndarray       # sunlight on the cell, 0..1
    sea_ink: np.ndarray     # how dense a sea cell should look, 0..1
    land_ink: np.ndarray    # how dense a land cell should look, 0..1
    shelf: np.ndarray       # share of the cell's sea that is shelf
    rough: np.ndarray       # roughness of the cell's land
    biome: np.ndarray       # biome covering most of the cell's land
    regions: np.ndarray     # (rows, cols, 6) land ink per region, for shaped coast characters
    glow: np.ndarray        # city light on the night side, 0..1


class Earth(SceneBase):
    title = "ASCII Earth"
    rows, cols = ROWS, COLS
    style = replace(BASE_STYLE, scale=BASE_STYLE.scale * BASE_STYLE.size(1, 100)[0] / BASE_STYLE.size(1, COLS)[0])
    # (smaller characters: the picture keeps the on-screen width of a 100-column scene, about 749 px)
    frames, fps = 108, 12.0
    max_colors = 24
    mono_levels = 7

    def __init__(self) -> None:
        self.maps = load_maps()
        self.radius = RADIUS_SHARE * self.rows
        self.centre = (self.rows / 2, self.cols / 2)
        y, x = subcell_centres(self.rows, self.cols, SY, SX)
        self.ny = (self.centre[0] - y) / self.radius
        self.nx = (x - self.centre[1]) / (self.radius * self.style.cell_aspect)
        self.rho = np.sqrt(self.nx ** 2 + self.ny ** 2)
        self.inside = self.rho < 1.0
        self.nz = np.sqrt(np.clip(1.0 - self.rho ** 2, 0.0, 1.0))
        self.normal = np.stack([self.nx, self.ny, self.nz], axis=-1)
        self.sun_dot = self.normal @ SUN
        self.hero = int(round(self.frames * (1.0 - HERO_LONGITUDE / TAU))) % self.frames
        self._tilt = rot_z(-AXIAL_TILT) @ rot_x(VIEW_PITCH)
        self._halo = cell_mean(self._atmosphere(), SY, SX)
        self._densest_land = max(glyph_vector(ch).mean() for ch in LAND_GLYPHS)

    # ------------------------------------------------------------ geometry

    def _surface(self, phase: float) -> Maps:
        """Map values (rows*SY, cols*SX, 9) and biomes of the spot behind every sample."""
        view_to_body = (self._tilt @ rot_y(TAU * phase)).T
        body = self.normal @ view_to_body.T
        lon = np.arctan2(body[..., 0], body[..., 2])
        lat = np.arcsin(np.clip(body[..., 1], -1.0, 1.0))
        return Maps(sample_equirect(self.maps.values, lon, lat), sample_nearest(self.maps.biome, lon, lat))

    def _atmosphere(self) -> np.ndarray:
        """Thin glowing ring just outside the limb, brightest towards the Sun (ink per sample)."""
        outside = self.rho >= 1.0
        dx, dy = self.nx / np.maximum(self.rho, 1e-6), self.ny / np.maximum(self.rho, 1e-6)
        sun_side = (dx * SUN[0] + dy * SUN[1]) / np.hypot(SUN[0], SUN[1])
        side = 0.10 + 0.90 * smoothstep(-0.55, 0.55, sun_side)
        glow = np.exp(-(self.rho - 1.0) / 0.035) * outside * (self.rho < 1.12)
        return 0.90 * glow * side

    # ------------------------------------------------------------ per-cell summary

    def _cells(self, surface: Maps) -> Cells:
        s = surface.values
        land, lights = s[..., LAND], s[..., LIGHTS]
        lit = smoothstep(*DAY_EDGE, self.sun_dot)
        night = 1.0 - smoothstep(*NIGHT_EDGE, self.sun_dot)
        shade = lit * (SHADE_FLOOR + (1.0 - SHADE_FLOOR) * np.clip(self.sun_dot, 0.0, 1.0) ** SHADE_GAMMA)
        inside = self.inside.astype(np.float64)
        hard = (land > 0.5).astype(np.float64)

        def mean(a: np.ndarray) -> np.ndarray:
            return cell_mean(a * inside, SY, SX)

        def share(a: np.ndarray, of: np.ndarray) -> np.ndarray:
            return mean(a * of) / np.maximum(mean(of), 1e-6)

        cover = np.maximum(mean(np.ones_like(land)), 1e-6)
        land_share = mean(land) / cover
        flat = LAND_BASE_INK + LAND_LUMA_INK * np.clip(s[..., LUMA], 0.0, 1.0)
        ink = np.clip(flat + RELIEF_GAIN * s[..., RELIEF], 0.0, 1.0) * (0.30 + 0.70 * shade) * land
        votes = [to_blocks((surface.biome == j) * land * inside, SY, SX).sum(-1) for j in range(len(BIOMES))]
        return Cells(
            disc=cover > DISC_SHARE,
            land=land_share > COAST_SHARE[0],
            land_share=land_share,
            day=mean(lit) / cover > DAY_SHARE,
            shade=mean(shade) / cover,
            sea_ink=mean((0.30 + 0.60 * shade * (0.25 + 0.75 * s[..., SHELF])) * lit) / cover,
            land_ink=mean(ink) / np.maximum(mean(land), 1e-6),
            shelf=share(s[..., SHELF], 1.0 - hard),
            rough=share(s[..., ROUGH], hard),
            biome=np.stack(votes, -1).argmax(-1),
            regions=pool_regions(ink * inside, SY, SX),
            glow=mean(np.clip(lights * 2.2, 0.0, 1.0) * night),
        )

    # ------------------------------------------------------------ drawing

    def _sea(self, c: Cells) -> tuple[np.ndarray, np.ndarray]:
        shelf = (c.shelf > SHELF_SHARE).astype(np.int64)
        bright = (c.shade > SEA_BRIGHT_FROM).astype(np.int64)
        return pick_by_coverage(c.sea_ink + SHELF_INK * shelf, SEA_GLYPHS), SEA_BASE + 2 * shelf + bright

    def _land(self, c: Cells, sunlit_land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coast = (c.land_share >= COAST_SHARE[0]) & (c.land_share < COAST_SHARE[1])
        edge = hybrid_chars(c.regions * self._densest_land, EDGE_GLYPHS, LAND_GLYPHS, EDGE_THRESHOLD)
        chars = np.where(coast, edge, pick_by_coverage(c.land_ink, LAND_GLYPHS))
        mountain = sunlit_land & ~coast & (c.rough > MOUNTAIN_ROUGH) & (c.biome != ICE) & (c.shade > MOUNTAIN_MIN_SHADE)
        return np.where(mountain, MOUNTAIN_GLYPH, chars), LAND_BASE + c.biome

    def render(self, k: int, n: int) -> Frame:
        c = self._cells(self._surface((k % n) / n))
        sea = c.disc & ~c.land & c.day
        land = c.disc & c.land & c.day
        lights = c.disc & ~c.day & (c.glow > MIN_GLOW)
        rim = ~c.disc & (self._halo > HALO_MIN)

        sea_chars, sea_idx = self._sea(c)
        land_chars, land_idx = self._land(c, land)
        layers = (
            (sea, sea_chars, sea_idx),
            (land, land_chars, land_idx),
            (lights, pick_by_coverage(0.2 + 0.8 * c.glow, LIGHT_GLYPHS), LIGHT_BASE + (c.glow > LIGHT_HOT)),
            (rim, pick_by_coverage(0.35 + 0.8 * self._halo, SEA_GLYPHS), HAZE_INDEX),
        )
        chars = np.full(c.shade.shape, SPACE, np.uint8)
        idx = np.zeros(c.shade.shape, np.int64)
        for mask, glyphs, colours in layers:
            chars = np.where(mask, glyphs, chars)
            idx = np.where(mask, colours, idx)
        idx = merge_short_runs(chars != SPACE, idx, KIND, MIN_RUN)
        return Frame(chars.astype(np.uint8), PALETTE[idx])

    # ------------------------------------------------------------ stars

    def _keep_out(self) -> np.ndarray:
        r, c = np.mgrid[0:self.rows, 0:self.cols]
        dy = (r + 0.5 - self.centre[0]) / self.radius
        dx = (c + 0.5 - self.centre[1]) / (self.radius * self.style.cell_aspect)
        return dx ** 2 + dy ** 2 < STAR_CLEARANCE ** 2

    def static(self) -> Frame:
        frame = starfield(self.rows, self.cols, density=STAR_DENSITY, keep_out=self._keep_out())[0]
        return Frame(frame.chars, star_palette(frame.rgb))

    def twinkles(self) -> list[Twinkle]:
        stars = starfield(self.rows, self.cols, density=STAR_DENSITY, keep_out=self._keep_out())[1]
        return [replace(t, rgb=tuple(float(v) for v in star_palette(np.array(t.rgb)))) for t in stars]


def make_scene() -> Earth:
    return Earth()
