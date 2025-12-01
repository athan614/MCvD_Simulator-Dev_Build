#!/usr/bin/env python3
"""Plot SER/LoD comparisons for single, dual, and tri-channel physical profiles."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional, Tuple

import os
import matplotlib as mpl
if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analysis.ieee_plot_style import apply_ieee_style
DATA_DIR = PROJECT_ROOT / "results" / "data"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS: List[Tuple[str, str, str, Optional[bool]]] = [
    ("single", "_single_physical", "Single (DA only)", False),
    ("dual", "_dual_physical", "Dual (DA+SERO)", False),
    ("tri", "", "Tri (DA+SERO+CTRL)", True),
]

MODES = ["MoSK", "CSK", "Hybrid"]


def _nm_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("pipeline_Nm_per_symbol", "pipeline.Nm_per_symbol"):
        if col in df.columns:
            return col
    return None


def _load_ser(mode: str, suffix: str, use_ctrl: Optional[bool]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    path = DATA_DIR / f"ser_vs_nm_{mode.lower()}{suffix}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    nm_col = _nm_column(df)
    if nm_col is None or "ser" not in df.columns:
        return None
    if use_ctrl is not None and "use_ctrl" in df.columns:
        df = df[df["use_ctrl"] == use_ctrl]
    df = df.dropna(subset=[nm_col, "ser"])  # type: ignore[list]
    if df.empty:
        return None
    df = df.sort_values(by=nm_col)
    x = pd.to_numeric(df[nm_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["ser"], errors="coerce").to_numpy(dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if not mask.any():
        return None
    return x[mask], y[mask]


def _load_lod(mode: str, suffix: str, use_ctrl: Optional[bool]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    path = DATA_DIR / f"lod_vs_distance_{mode.lower()}{suffix}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "distance_um" not in df.columns or "lod_nm" not in df.columns:
        return None
    if use_ctrl is not None and "use_ctrl" in df.columns:
        df = df[df["use_ctrl"] == use_ctrl]
    df = df.dropna(subset=["distance_um", "lod_nm"])  # type: ignore[list]
    if df.empty:
        return None
    df = df.sort_values(by="distance_um")
    x = pd.to_numeric(df["distance_um"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["lod_nm"], errors="coerce").to_numpy(dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if not mask.any():
        return None
    return x[mask], y[mask]


def _plot_ser_profiles() -> None:
    apply_ieee_style()
    fig, axes = plt.subplots(1, len(MODES), figsize=(3.6 * len(MODES), 3.0), sharey=True)
    if len(MODES) == 1:
        axes = [axes]  # type: ignore[list-item]

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []

    for idx, mode in enumerate(MODES):
        ax = axes[idx]
        plotted = False
        for _, suffix, label, use_ctrl in SCENARIOS:
            data = _load_ser(mode, suffix, use_ctrl)
            if data is None:
                continue
            x, y = data
            line, = ax.semilogy(x, y, marker="o", label=label)
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)
            plotted = True
        ax.set_title(mode)
        ax.set_xlabel("Molecules per symbol (Nm)")
        if idx == 0:
            ax.set_ylabel("SER")
        ax.grid(True, which="both", alpha=0.3)
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    if legend_handles:
        axes[0].legend(handles=legend_handles, labels=legend_labels)
    plt.tight_layout()
    out = FIG_DIR / "fig_channel_profiles_ser.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def _plot_lod_profiles() -> None:
    apply_ieee_style()
    fig, axes = plt.subplots(1, len(MODES), figsize=(3.6 * len(MODES), 3.0), sharey=True)
    if len(MODES) == 1:
        axes = [axes]  # type: ignore[list-item]

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []

    for idx, mode in enumerate(MODES):
        ax = axes[idx]
        plotted = False
        for _, suffix, label, use_ctrl in SCENARIOS:
            data = _load_lod(mode, suffix, use_ctrl)
            if data is None:
                continue
            x, y = data
            line, = ax.plot(x, y, marker="s", label=label)
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)
            plotted = True
        ax.set_title(mode)
        ax.set_xlabel("Distance (um)")
        if idx == 0:
            ax.set_ylabel("LoD Nm @ SER=1%")
        ax.grid(True, alpha=0.3)
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    if legend_handles:
        axes[0].legend(handles=legend_handles, labels=legend_labels)
    plt.tight_layout()
    out = FIG_DIR / "fig_channel_profiles_lod.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    _plot_ser_profiles()
    _plot_lod_profiles()


if __name__ == "__main__":
    main()
