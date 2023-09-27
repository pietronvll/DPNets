from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager

font_path = [Path(__file__).parent / "fonts"]
print(font_path)
font_files = font_manager.findSystemFonts(font_path)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# Colors
color_palette = [
    "#e60049",
    "#0bb4ff",
    "#50e991",
    "#e6d800",
    "#9b19f5",
    "#ffa300",
    "#dc0ab4",
    "#b3d4ff",
    "#00bfa0",
]

plt.rcParams["axes.prop_cycle"] = cycler(color=color_palette)

# Typography
plt.rcParams["font.family"] = "Source Sans 3"
plt.rcParams["font.size"] = 8
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.weight"] = "normal"

# Line style
plt.rcParams["lines.linewidth"] = 1

# Layout
plt.rcParams["figure.constrained_layout.use"] = True


def pt_to_inch(pt):
    return pt / 72.27


def figsize(wratio=1, width_to_height=3 / 2, text_width_pt=397):
    w = pt_to_inch(text_width_pt) * wratio
    h = w / width_to_height
    return (w, h)
