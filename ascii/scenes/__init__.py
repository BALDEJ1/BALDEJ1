"""Scenes: each module in this package draws one animation.

A scene module exposes make_scene() returning an object (usually a SceneBase subclass) with:
    title: str        accessible name of the picture (SVG <title>)
    style: Style      cell size, padding, background (see ascii.engine.Style)
    rows, cols: int   size of the character grid
    frames: int       frames in one seamless loop: frame `frames` would be identical to frame 0
    fps: float        playback speed
    max_colors: int   palette size of the colour version (at most 52)
    mono_levels: int  number of shades in the green and amber versions
    hero: int         frame shown to viewers who prefer reduced motion
    render(k, n)      -> Frame: frame k of an n-frame loop (loop phase = k / n)
    static()          -> Frame | None: layer drawn under every frame (for example stars)
    twinkles()        -> list[Twinkle]: single characters that fade in and out on their own clock

Render with:  .venv/bin/python render.py <module name>
"""
