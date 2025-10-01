#!/usr/bin/env python3
"""Plot guard-factor frontiers maximizing ISI-robust throughput."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
results_dir = project_root / "results"
data_dir = results_dir / "data"
fig_dir = results_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

from analysis.ieee_plot_style import apply_ieee_style

MODES = ["MoSK", "CSK", "Hybrid"]


def _load_pipeline_defaults() -> Dict[str, float]:
    cfg_path = project_root / "config" / "default.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {"guard_factor": 0.0}
    try:
        return {"guard_factor": float(cfg.get('pipeline', {}).get('guard_factor', 0.0))}
    except Exception:
        return {"guard_factor": 0.0}


def _load_frontier(mode: str, suffix: str, use_ctrl: Optional[bool]) -> Optional[pd.DataFrame]:
    csv_path = data_dir / f"guard_frontier_{mode.lower()}{suffix}.csv"
    if not csv_path.exists():
        print(f"??  Missing frontier file: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"??  Could not read {csv_path}: {exc}")
        return None
    if use_ctrl is not None and 'use_ctrl' in df.columns:
        df = df[df['use_ctrl'] == use_ctrl]
    if df.empty:
        return None
    df = df.copy()
    df['distance_um'] = pd.to_numeric(df['distance_um'], errors='coerce')
    df['best_guard_factor'] = pd.to_numeric(df['best_guard_factor'], errors='coerce')
    df['max_irt_bps'] = pd.to_numeric(df['max_irt_bps'], errors='coerce')
    df = df.dropna(subset=['distance_um', 'best_guard_factor', 'max_irt_bps'])
    if df.empty:
        return None
    df = df.sort_values('distance_um')
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot guard-factor design frontiers.")
    parser.add_argument("--variant", default="", help="Optional variant suffix used for analysis outputs.")
    parser.add_argument(
        "--ctrl",
        choices=["on", "off", "any"],
        default="on",
        help="Select control-channel configuration to plot (default: on).",
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
    ax_guard, ax_rate = axes

    defaults = _load_pipeline_defaults()
    plotted_any = False

    for mode in MODES:
        df_frontier = _load_frontier(mode, suffix, desired_ctrl)
        if df_frontier is None:
            continue
        label = mode
        ax_guard.plot(
            df_frontier['distance_um'],
            df_frontier['best_guard_factor'],
            marker='o', linewidth=2, label=label
        )
        ax_rate.plot(
            df_frontier['distance_um'],
            df_frontier['max_irt_bps'],
            marker='o', linewidth=2, label=label
        )
        plotted_any = True

    ax_guard.set_xlabel("Distance (um)")
    ax_guard.set_ylabel("Guard factor (fraction of T_s)")
    ax_guard.set_ylim(0.0, 1.05)
    ax_guard.grid(True, alpha=0.3)
    if defaults['guard_factor'] > 0:
        ax_guard.axhline(defaults['guard_factor'], color='grey', linestyle='--', alpha=0.6)

    ax_rate.set_xlabel("Distance (um)")
    ax_rate.set_ylabel("Max IRT (bits/s)")
    ax_rate.grid(True, alpha=0.3)

    handles, labels = ax_guard.get_legend_handles_labels()
    if handles:
        ax_guard.legend(loc='best', fontsize=8)
    handles_rate, _ = ax_rate.get_legend_handles_labels()
    if handles_rate and not handles:
        ax_rate.legend(loc='best', fontsize=8)

    if not plotted_any:
        fig.text(0.5, 0.5, "No guard frontier data", ha='center', va='center')

    fig.suptitle("Guard-Factor Frontiers Across Distance", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    out_path = fig_dir / "fig_guard_frontier.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"? Saved: {out_path}")


if __name__ == "__main__":
    main()
