#!/usr/bin/env python3
"""Plot delta-over-sigma device figures of merit versus g_m and C_tot."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional
import sys

import numpy as np
import pandas as pd
import yaml
import os
import matplotlib as mpl
if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

results_dir = project_root / "results"
data_dir = results_dir / "data"
fig_dir = results_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

from analysis.ieee_plot_style import apply_ieee_style

MODES = ["MoSK", "CSK", "Hybrid"]


def _load_config_defaults() -> Dict[str, float]:
    cfg_path = project_root / "config" / "default.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {"gm_S": 0.0, "C_tot_F": 0.0}
    try:
        oect = cfg.get("oect", {})
        return {
            "gm_S": float(oect.get("gm_S", 0.0)),
            "C_tot_F": float(oect.get("C_tot_F", 0.0)),
        }
    except Exception:
        return {"gm_S": 0.0, "C_tot_F": 0.0}


def _prepare_series(df: pd.DataFrame, param_type: str, value_col: str) -> Optional[pd.DataFrame]:
    if df.empty:
        return None
    subset = df[df["param_type"] == param_type].copy()
    if subset.empty:
        return None
    subset["param_value"] = pd.to_numeric(subset["param_value"], errors="coerce")
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset = subset.dropna(subset=["param_value", value_col])
    if subset.empty:
        return None
    grouped = (
        subset.groupby("param_value", as_index=False)
        .agg({value_col: "median"})
        .sort_values(by="param_value")
    )
    return grouped


def _load_mode_data(mode: str, suffix: str, use_ctrl: Optional[bool]) -> Optional[pd.DataFrame]:
    csv_path = data_dir / f"device_fom_{mode.lower()}{suffix}.csv"
    if not csv_path.exists():
        print(f"??  Missing data file: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"??  Could not read {csv_path}: {exc}")
        return None
    if use_ctrl is not None and "use_ctrl" in df.columns:
        df = df[df["use_ctrl"] == use_ctrl]
    if df.empty:
        return None
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot device FoM trends versus OECT parameters.")
    parser.add_argument("--variant", default="", help="Optional variant suffix used during analysis runs.")
    parser.add_argument(
        "--ctrl",
        choices=["on", "off", "any"],
        default="on",
        help="Select control-channel configuration to plot (default: on).",
    )
    parser.add_argument(
        "--domain",
        choices=["current", "charge"],
        default="current",
        help="Metric domain: 'current' for ΔI/σ_I, 'charge' for ΔQ/σ_Q.",
    )
    args = parser.parse_args()

    suffix = f"_{args.variant}" if args.variant else ""
    if args.ctrl == "on":
        desired_ctrl: Optional[bool] = True
    elif args.ctrl == "off":
        desired_ctrl = False
    else:
        desired_ctrl = None

    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    ax_gm, ax_c = axes

    if args.domain == "current":
        value_col = "delta_over_sigma_I"
        ylabel = r"$\Delta I_{\mathrm{diff}} / \sigma_{I,\mathrm{diff}}$"
    else:
        value_col = "delta_over_sigma_Q"
        ylabel = r"$\Delta Q_{\mathrm{diff}} / \sigma_{Q,\mathrm{diff}}$"

    defaults = _load_config_defaults()

    plotted_any = False
    for mode in MODES:
        df_mode = _load_mode_data(mode, suffix, desired_ctrl)
        if df_mode is None:
            continue
        if value_col not in df_mode.columns:
            print(f"??  Skipping {mode}: column {value_col} missing")
            continue

        gm_series = _prepare_series(df_mode, "gm_S", value_col)
        c_series = _prepare_series(df_mode, "C_tot_F", value_col)

        label = mode
        if gm_series is not None and not gm_series.empty:
            x = gm_series["param_value"].to_numpy(dtype=float) * 1e3  # S -> mS
            y = gm_series[value_col].to_numpy(dtype=float)
            ax_gm.plot(x, y, marker="o", linewidth=2, label=label)
            plotted_any = True
        if c_series is not None and not c_series.empty:
            x = c_series["param_value"].to_numpy(dtype=float) * 1e9  # F -> nF
            y = c_series[value_col].to_numpy(dtype=float)
            ax_c.plot(x, y, marker="o", linewidth=2, label=label)
            plotted_any = True

    ax_gm.set_xlabel("g_m (mS)")
    ax_gm.set_ylabel(ylabel)
    ax_gm.grid(True, alpha=0.3)
    if defaults["gm_S"] > 0:
        ax_gm.axvline(defaults["gm_S"] * 1e3, color="grey", linestyle="--", alpha=0.6)

    ax_c.set_xlabel("C_tot (nF)")
    ax_c.set_ylabel(ylabel)
    ax_c.grid(True, alpha=0.3)
    if defaults["C_tot_F"] > 0:
        ax_c.axvline(defaults["C_tot_F"] * 1e9, color="grey", linestyle="--", alpha=0.6)

    handles, labels = ax_gm.get_legend_handles_labels()
    if handles:
        ax_gm.legend(loc="best", fontsize=8)
    handles_c, labels_c = ax_c.get_legend_handles_labels()
    if handles_c and not handles:
        ax_c.legend(loc="best", fontsize=8)

    if not plotted_any:
        fig.text(0.5, 0.5, "No device FoM data available", ha="center", va="center")

    fig.suptitle("Device FoM: Delta-over-Sigma vs OECT Parameters", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    out_path = fig_dir / "fig_device_fom.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"? Saved: {out_path}")


if __name__ == "__main__":
    main()
