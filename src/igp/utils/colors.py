# igp/utils/colors.py
# Color utilities for visualization:
# - Fixed categorical palettes (BASIC_COLORS, COLORBLIND_COLORS).
# - HSV boost for visibility, WCAG-like contrast for text color.
# - Consistent per-class color assignment with optional seed and custom palette.

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import colorsys

try:
    import matplotlib.colors as mcolors
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# Distinct, reproducible color palette (hex). Suitable for categorical labels.
BASIC_COLORS: List[str] = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#1f78b4",
    "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#cab2d6",
    "#6a3d9a", "#b2df8a", "#ffed6f", "#a6cee3", "#b15928",
]

# Color-blind friendly palette (Okabe–Ito)
COLORBLIND_COLORS: List[str] = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]


def _to_rgb(hex_col: str) -> Tuple[float, float, float]:
    if _HAS_MPL:
        return mcolors.to_rgb(hex_col)
    hex_col = hex_col.lstrip("#")
    if len(hex_col) == 3:
        hex_col = "".join(c*2 for c in hex_col)
    r = int(hex_col[0:2], 16) / 255.0
    g = int(hex_col[2:4], 16) / 255.0
    b = int(hex_col[4:6], 16) / 255.0
    return (r, g, b)


def _to_hex(rgb: Tuple[float, float, float]) -> str:
    if _HAS_MPL:
        return mcolors.to_hex(rgb)
    r = max(0, min(255, int(round(rgb[0] * 255))))
    g = max(0, min(255, int(round(rgb[1] * 255))))
    b = max(0, min(255, int(round(rgb[2] * 255))))
    return f"#{r:02x}{g:02x}{b:02x}"


def _boost_color(hex_col: str, sat_factor: float = 1.25, val_factor: float = 1.10) -> str:
    """
    Slightly increase saturation/value in HSV space to make colors pop while keeping hue.
    """
    r, g, b = _to_rgb(hex_col)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(1.0, s * sat_factor)
    v = min(1.0, v * val_factor)
    return _to_hex(colorsys.hsv_to_rgb(h, s, v))


def _relative_luminance(rgb: Tuple[float, float, float]) -> float:
    # WCAG 2.0 relative luminance with gamma correction
    def _linearize(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = ( _linearize(rgb[0]), _linearize(rgb[1]), _linearize(rgb[2]) )
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color_for_bg(hex_col: str) -> str:
    """
    Choose black/white text color with good contrast against the background.
    Uses WCAG-like perceived luminance.
    """
    lum = _relative_luminance(_to_rgb(hex_col))
    # threshold ~0.5 gives good separation on boosted colors
    return "#000000" if lum > 0.5 else "#ffffff"


def base_label(label: str) -> str:
    """
    Extract the base label by removing a trailing numeric suffix: 'person_1' -> 'person'.
    """
    return label.rsplit("_", 1)[0] if "_" in label and label.split("_")[-1].isdigit() else label


class ColorCycler:
    """
    Assign a consistent color per (base) class label.
    - Normalizes labels to lowercase base form ('Person_2' -> 'person').
    - Cycles through a palette and applies a small HSV boost.
    - Optional seed controls starting offset for reproducibility across runs.
    """
    def __init__(
        self,
        palette: Iterable[str] | None = None,
        *,
        sat_boost: float = 1.25,
        val_boost: float = 1.10,
        seed_offset: int = 0,
    ) -> None:
        pal = list(palette) if palette is not None else list(BASIC_COLORS)
        self._palette: List[str] = pal if pal else list(BASIC_COLORS)
        self._label2color: Dict[str, str] = {}
        self._sat = float(sat_boost)
        self._val = float(val_boost)
        self._seed = int(seed_offset) % max(1, len(self._palette))

    def color_for_label(self, label: str) -> str:
        base = base_label(label).lower()
        if base not in self._label2color:
            idx = (self._seed + len(self._label2color)) % len(self._palette)
            raw = self._palette[idx]
            self._label2color[base] = _boost_color(raw, self._sat, self._val)
        return self._label2color[base]

    def reset(self) -> None:
        self._label2color.clear()

    def set_palette(self, palette: Iterable[str]) -> None:
        self._palette = list(palette) or list(BASIC_COLORS)
        self.reset()