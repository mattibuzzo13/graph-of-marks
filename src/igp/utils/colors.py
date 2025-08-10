# igp/utils/colors.py
# Color utilities for visualization:
# - Fixed categorical palette (BASIC_COLORS).
# - Small HSV "boost" to improve saturation/value for visibility.
# - Text color selection based on background luminance.
# - Consistent per-class color assignment via ColorCycler.

from __future__ import annotations

from typing import Dict

import colorsys
import matplotlib.colors as mcolors

# Distinct, reproducible color palette (hex). Suitable for categorical labels.
BASIC_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#1f78b4",
    "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#cab2d6",
    "#6a3d9a", "#b2df8a", "#ffed6f", "#a6cee3", "#b15928"
]


def _boost_color(hex_col: str, sat_factor: float = 1.3, val_factor: float = 1.15) -> str:
    """
    Slightly increase saturation/value in HSV space to make colors pop
    while keeping the original hue.
    """
    r, g, b = mcolors.to_rgb(hex_col)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(1.0, s * sat_factor)
    v = min(1.0, v * val_factor)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return mcolors.to_hex((r, g, b))


def text_color_for_bg(hex_col: str) -> str:
    """
    Choose black/white text color that contrasts with a given background color,
    using a simple perceived luminance heuristic.
    """
    r, g, b = mcolors.to_rgb(hex_col)
    # Simple perceived luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luminance > 0.6 else "#ffffff"

def base_label(label: str) -> str:
    """
    Extract the base label by removing a trailing numeric suffix.
    E.g., 'person_1' -> 'person', 'car' -> 'car'.
    """
    return label.rsplit("_", 1)[0] if "_" in label and label.split("_")[-1].isdigit() else label

class ColorCycler:
    """
    Assign a consistent color per (base) class label.
    Cycles through BASIC_COLORS and applies a small HSV boost.
    """
    def __init__(self, sat_boost: float = 1.3, val_boost: float = 1.15) -> None:
        self._label2color: Dict[str, str] = {}
        self._sat = float(sat_boost)
        self._val = float(val_boost)

    def color_for_label(self, label: str) -> str:
        """
        Return the (cached) color for a label. Labels are normalized to the
        lowercase base form so 'person_1' and 'Person_2' share the same color.
        """
        base = label.rsplit("_", 1)[0].lower()
        if base not in self._label2color:
            raw = BASIC_COLORS[len(self._label2color) % len(BASIC_COLORS)]
            self._label2color[base] = _boost_color(raw, self._sat, self._val)
        return self._label2color[base]
