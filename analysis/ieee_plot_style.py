# analysis/ieee_plot_style.py
"""
IEEE plotting style helpers.

Usage:
    from analysis.ieee_plot_style import apply_ieee_style, okabe_ito
    apply_ieee_style()
"""
from __future__ import annotations
import matplotlib as mpl
from cycler import cycler
from typing import Dict, Any

# Okabe–Ito color-blind friendly palette
okabe_ito = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

def apply_ieee_style(overrides: Dict[str, Any] | None = None) -> None:
    """Apply a compact IEEE-friendly Matplotlib style (serif fonts, small labels).
    """
    rc = {
        # Fonts
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        # Lines & markers
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        # Figure layout
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        # Color cycle
        "axes.prop_cycle": cycler(color=okabe_ito),
    }
    if overrides:
        rc.update(overrides)
    mpl.rcParams.update(rc)