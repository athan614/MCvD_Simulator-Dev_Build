#!/usr/bin/env python3
"""
Input-output transduction gain summary.

Loads SER-vs-Nm sweep CSVs (ser_vs_nm_{mode}.csv) and plots ΔI_diff, ΔQ_diff
versus Nm for each modulation mode, annotating least-squares slopes. Designed
to back the 3-D narrative by exposing device/channel gain linearity.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures"

from analysis.ieee_plot_style import apply_ieee_style

MODE_ORDER = ["MoSK", "CSK", "Hybrid"]
NM_CANDIDATES = [
    "pipeline_Nm_per_symbol",
    "pipeline.Nm_per_symbol",
    "Nm_per_symbol",
    "Nm",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ΔI/ΔQ gain vs Nm across modes.")
    parser.add_argument(
        "--mode-filter",
        type=str,
        default="all",
        help="Comma-separated subset of modes (default: all).",
    )
    parser.add_argument(
        "--distance",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of distances (µm) to keep (exact/fuzzy match).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=fig_dir / "fig_input_output_gain.png",
        help="Destination for the gain summary figure.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional filename suffix (matches ser_vs_nm_{mode}{suffix}.csv).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum scatter points per subset to fit/plot (default: 3).",
    )
    return parser.parse_args()


def _normalize_mode_list(raw: str) -> List[str]:
    if not raw or raw.strip().lower() == "all":
        return MODE_ORDER[:]
    modes = []
    for token in raw.split(","):
        name = token.strip()
        if not name:
            continue
        if name.upper() in {"MOSK", "CSK", "HYBRID"}:
            name = name.upper().title()
        modes.append(name)
    if not modes:
        return MODE_ORDER[:]
    return modes


def _distance_matches(value: float, targets: Optional[Sequence[float]], tol: float = 1e-6) -> bool:
    if targets is None or not targets:
        return True
    for ref in targets:
        if math.isclose(value, ref, rel_tol=0.0, abs_tol=tol):
            return True
    return False


def _load_mode_frame(mode: str, suffix: str) -> Optional[pd.DataFrame]:
    mode_key = mode.lower()
    pattern = f"ser_vs_nm_{mode_key}{suffix}"
    candidates = sorted(data_dir.glob(f"{pattern}*.csv"))
    if not candidates and not suffix:
        candidates = sorted(data_dir.glob(f"ser_vs_nm_{mode_key}*.csv"))
    if not candidates:
        print(f"?? Gain plot: no SER vs Nm CSV found for mode {mode}")
        return None
    frames: List[pd.DataFrame] = []
    for csv in candidates:
        try:
            df = pd.read_csv(csv)
            df["__source"] = csv.name
            frames.append(df)
        except Exception as exc:
            print(f"?? Gain plot: failed to read {csv.name}: {exc}")
    if not frames:
        return None
    df_mode = pd.concat(frames, ignore_index=True)
    df_mode["mode"] = mode
    return df_mode


def _resolve_nm_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in NM_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _least_squares(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def _prepare_groups(
    df: pd.DataFrame,
    distance_filter: Optional[Sequence[float]],
) -> Dict[str, pd.DataFrame]:
    if "distance_um" in df.columns and df["distance_um"].notna().any():
        groups: Dict[str, pd.DataFrame] = {}
        for value in sorted(df["distance_um"].dropna().unique()):
            if not _distance_matches(float(value), distance_filter, tol=0.25):
                continue
            label = f"{value:.0f} µm"
            groups[label] = df[df["distance_um"] == value]
        if groups:
            return groups
    # fallback: single group
    label = "all distances"
    if distance_filter:
        label = ", ".join(f"{d:g} µm" for d in distance_filter)
    return {label: df}


def _plot_series(
    ax,
    nm: np.ndarray,
    values: np.ndarray,
    label: str,
    min_points: int,
    color: Optional[str] = None,
) -> Optional[float]:
    mask = np.isfinite(nm) & np.isfinite(values)
    nm_clean = nm[mask]
    values_clean = values[mask]
    if nm_clean.size < min_points:
        return None
    order = np.argsort(nm_clean)
    nm_sorted = nm_clean[order]
    val_sorted = values_clean[order]
    scatter_kwargs: Dict[str, Any] = {"s": 18, "alpha": 0.85, "label": label}
    if color:
        scatter_kwargs["color"] = color
    ax.scatter(nm_sorted, val_sorted, **scatter_kwargs)
    slope, intercept = _least_squares(nm_sorted, val_sorted)
    if math.isfinite(slope) and math.isfinite(intercept):
        x_line = np.linspace(nm_sorted.min(), nm_sorted.max(), 100)
        y_line = slope * x_line + intercept
        plot_kwargs: Dict[str, Any] = {"linewidth": 1.2}
        if color:
            plot_kwargs["color"] = color
        ax.plot(x_line, y_line, **plot_kwargs)
        return slope
    return None


def _apply_annotations(ax, annotation_rows: List[str]) -> None:
    if not annotation_rows:
        return
    text = "\n".join(annotation_rows)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )


def _plot_mode_panels(
    ax_i,
    ax_q,
    df: pd.DataFrame,
    mode: str,
    nm_col: str,
    min_points: int,
    distance_filter: Optional[Sequence[float]],
) -> None:
    groups = _prepare_groups(df, distance_filter)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    annotations_i: List[str] = []
    annotations_q: List[str] = []

    for idx, (label, subset) in enumerate(groups.items()):
        if subset.empty:
            continue
        color = colors[idx % len(colors)]
        nm_vals = subset[nm_col].to_numpy(dtype=float)
        di_vals = subset.get("delta_I_diff", pd.Series(dtype=float)).to_numpy(dtype=float)
        dq_vals = subset.get("delta_Q_diff", pd.Series(dtype=float)).to_numpy(dtype=float)

        slope_i = _plot_series(ax_i, nm_vals, di_vals, label, min_points, color=color)
        slope_q = _plot_series(ax_q, nm_vals, dq_vals, label, min_points, color=color)
        if slope_i is not None:
            annotations_i.append(f"{label}: slope={slope_i:.3e} A/mol")
        if slope_q is not None:
            annotations_q.append(f"{label}: slope={slope_q:.3e} C/mol")

    ax_i.set_title(mode)
    ax_i.set_ylabel("ΔI_diff (A)")
    ax_q.set_ylabel("ΔQ_diff (C)")
    ax_q.set_xlabel("Nm per symbol")
    ax_i.set_xscale("log")
    ax_q.set_xscale("log")
    ax_i.grid(True, alpha=0.3)
    ax_q.grid(True, alpha=0.3)
    if annotations_i:
        _apply_annotations(ax_i, annotations_i)
    if annotations_q:
        _apply_annotations(ax_q, annotations_q)
    handles, labels = ax_i.get_legend_handles_labels()
    if handles:
        ax_i.legend(fontsize=8)
    handles_q, labels_q = ax_q.get_legend_handles_labels()
    if handles_q and not handles:
        ax_q.legend(fontsize=8)


def main() -> None:
    args = _parse_args()
    modes = _normalize_mode_list(args.mode_filter)
    apply_ieee_style()
    valid_frames: List[Tuple[str, pd.DataFrame, str]] = []
    for mode in modes:
        df_mode = _load_mode_frame(mode, args.suffix)
        if df_mode is None or df_mode.empty:
            continue
        nm_col = _resolve_nm_column(df_mode)
        if nm_col is None:
            print(f"?? Gain plot: no Nm column available for {mode}, skipping.")
            continue
        if "delta_I_diff" not in df_mode.columns and "delta_Q_diff" not in df_mode.columns:
            print(f"?? Gain plot: missing ΔI/ΔQ columns for {mode}, skipping.")
            continue
        valid_frames.append((mode, df_mode, nm_col))

    if not valid_frames:
        print("?? Gain plot: no valid SER vs Nm datasets located.")
        return

    cols = len(valid_frames)
    fig, axes = plt.subplots(
        2,
        cols,
        figsize=(3.2 * cols, 5.4),
        sharex=False,
        squeeze=False,
    )
    for idx, (mode, df_mode, nm_col) in enumerate(valid_frames):
        ax_i = axes[0][idx]
        ax_q = axes[1][idx]
        _plot_mode_panels(
            ax_i,
            ax_q,
            df_mode,
            mode,
            nm_col,
            args.min_points,
            args.distance,
        )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    fig.savefig(tmp, dpi=300, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    tmp.replace(args.output)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
