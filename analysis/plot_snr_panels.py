#!/usr/bin/env python3
"""
Compose SNR summary panels combining Nm, distance, and Ts sweeps.

Builds modality-specific figures (up to three subplots):
  - SNR(dB) vs Nm using ser_vs_nm_{mode}.csv
  - SNR(dB) vs distance using hds_grid_{mode}.csv or lod_vs_distance_{mode}.csv
  - SNR(dB) vs Ts using snr_vs_ts_{mode}.csv (if present)
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

try:
    from argparse import BooleanOptionalAction  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for Python < 3.9
    class BooleanOptionalAction(argparse.Action):  # type: ignore[no-redef]
        def __init__(self, option_strings, dest, default=None, **kwargs):
            if default is None:
                default = True
            super().__init__(option_strings=option_strings, dest=dest, nargs=0, default=default, **kwargs)
            self._negative = {opt for opt in option_strings if opt.startswith("--no-")}

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string in self._negative:
                setattr(namespace, self.dest, False)
            else:
                setattr(namespace, self.dest, True)

NM_COLUMNS = [
    "pipeline_Nm_per_symbol",
    "pipeline.Nm_per_symbol",
    "Nm_per_symbol",
    "Nm",
]
SNR_COLUMNS = ["snr_db", "snr_q_db", "snr_i_db"]
SUMMARY_CSV_DEFAULT = data_dir / "snr_panels_summary.csv"
COMBINED_FIG_DEFAULT = fig_dir / "fig_snr_panels.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SNR summary panels.")
    parser.add_argument(
        "--modes",
        type=str,
        default="MoSK,CSK,Hybrid",
        help="Comma-separated modes to include (default: MoSK,CSK,Hybrid).",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Filter to a specific distance for Nm panel (if present).",
    )
    parser.add_argument(
        "--nm",
        type=float,
        default=None,
        help="Preferred Nm (used to pick distance panel rows).",
    )
    parser.add_argument(
        "--ts-suffix",
        type=str,
        default="",
        help="Suffix applied to snr_vs_ts CSV names.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="fig_snr_panels",
        help="Base prefix for generated figures.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=COMBINED_FIG_DEFAULT,
        help="Path for the unified multi-mode panel figure.",
    )
    parser.add_argument(
        "--build-combined",
        action=BooleanOptionalAction,
        default=True,
        help="Generate the unified multi-mode figure (use --no-build-combined to skip).",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=SUMMARY_CSV_DEFAULT,
        help="Destination CSV for aggregated panel statistics.",
    )
    parser.add_argument(
        "--write-summary",
        action=BooleanOptionalAction,
        default=True,
        help="Write the aggregated panel statistics CSV (use --no-write-summary to skip).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate panel even if figure exists.",
    )
    return parser.parse_args()


def _normalize_modes(raw: str) -> List[str]:
    modes = []
    for token in raw.split(","):
        clean = token.strip()
        if not clean:
            continue
        if clean.upper() in {"MOSK", "CSK", "HYBRID"}:
            clean = clean.upper().title()
        modes.append(clean)
    return modes or ["MoSK"]


def _load_csv(prefix: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in data_dir.glob(f"{prefix}*.csv"):
        try:
            df = pd.read_csv(path)
            df["__source"] = path.name
            frames.append(df)
        except Exception as exc:
            print(f"?? Failed to load {path.name}: {exc}")
    return frames


def _atomic_write_csv(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)


def _aggregate_nm_curve(df_nm: Optional[pd.DataFrame], nm_col: Optional[str]) -> Optional[pd.DataFrame]:
    if df_nm is None or nm_col is None:
        return None
    df = df_nm.dropna(subset=[nm_col, "snr_plot"])
    if df.empty:
        return None
    grouped = (
        df.groupby(nm_col)["snr_plot"]
        .median()
        .reset_index(name="snr_db")
        .sort_values(by=nm_col)
    )
    grouped["x_value"] = grouped[nm_col]
    return grouped


def _aggregate_distance_curve(df_distance: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_distance is None:
        return None
    df = df_distance.dropna(subset=["distance_um", "snr_plot"])
    if df.empty:
        return None
    grouped = (
        df.groupby("distance_um")["snr_plot"]
        .median()
        .reset_index(name="snr_db")
        .sort_values(by="distance_um")
    )
    grouped["x_value"] = grouped["distance_um"]
    return grouped


def _aggregate_ts_curve(df_ts: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_ts is None:
        return None
    df = df_ts.dropna(subset=["symbol_period_s", "snr_plot"])
    if df.empty:
        return None
    agg_map: Dict[str, str] = {"snr_plot": "median"}
    if "ser" in df.columns:
        agg_map["ser"] = "median"
    grouped = (
        df.groupby("symbol_period_s")
        .agg(agg_map)
        .reset_index()
        .sort_values(by="symbol_period_s")
    )
    if "snr_plot" in grouped.columns:
        grouped = grouped.rename(columns={"snr_plot": "snr_db"})
    if "ser" in grouped.columns:
        grouped = grouped.rename(columns={"ser": "ser"})
    grouped["x_value"] = grouped["symbol_period_s"]
    return grouped


def _extend_summary(
    summary_rows: List[Dict[str, Any]],
    mode: str,
    panel: str,
    curve: Optional[pd.DataFrame],
    x_field: str,
) -> None:
    if curve is None or curve.empty:
        return
    if x_field not in curve.columns:
        return
    for _, row in curve.iterrows():
        x_val = row.get(x_field)
        snr_val = row.get("snr_db")
        if not isinstance(x_val, (int, float)) or not math.isfinite(float(x_val)):
            continue
        if not isinstance(snr_val, (int, float)) or not math.isfinite(float(snr_val)):
            continue
        entry: Dict[str, Any] = {
            "mode": mode,
            "panel": panel,
            "x_value": float(x_val),
            "snr_db": float(snr_val),
        }
        ser_val = row.get("ser")
        if isinstance(ser_val, (int, float)) and math.isfinite(float(ser_val)):
            entry["ser"] = float(ser_val)
        summary_rows.append(entry)


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _ensure_snr(row: pd.Series) -> float:
    for col in SNR_COLUMNS:
        if col in row and math.isfinite(row[col]):
            return float(row[col])
    delta_q = row.get("delta_over_sigma_Q", float("nan"))
    delta_i = row.get("delta_over_sigma_I", float("nan"))
    candidate = None
    if math.isfinite(delta_q):
        candidate = delta_q
    elif math.isfinite(delta_i):
        candidate = delta_i
    if candidate is None or not math.isfinite(candidate) or candidate == 0:
        return float("nan")
    return 20.0 * math.log10(abs(candidate))


def _prepare_nm_panel(
    mode: str,
    preferred_distance: Optional[float],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    frames = _load_csv(f"ser_vs_nm_{mode.lower()}")
    if not frames:
        return None, None
    df = pd.concat(frames, ignore_index=True)
    nm_col = _resolve_column(df, NM_COLUMNS)
    if nm_col is None:
        return None, None
    if preferred_distance is not None and "distance_um" in df.columns:
        df = df[np.isclose(df["distance_um"], preferred_distance, atol=0.25)]
    if df.empty:
        return None, None
    snr_vals = df.apply(_ensure_snr, axis=1)
    df = df.assign(snr_plot=snr_vals)
    return df, nm_col


def _pick_nm_for_distance(df: pd.DataFrame, nm_hint: Optional[float]) -> float:
    nm_col = _resolve_column(df, NM_COLUMNS)
    if nm_col is None:
        return float("nan")
    if nm_hint is not None:
        return float(nm_hint)
    ser_col = "ser"
    if ser_col in df.columns and df[ser_col].notna().any():
        row = df.iloc[(df[ser_col] - 0.01).abs().argsort()[:1]]
        if not row.empty:
            return float(row.iloc[0][nm_col])
    return float(df[nm_col].median())


def _prepare_distance_panel(mode: str, nm_hint: Optional[float]) -> Optional[pd.DataFrame]:
    candidates = _load_csv(f"hds_grid_{mode.lower()}")
    source = "hds_grid"
    if not candidates:
        candidates = _load_csv(f"lod_vs_distance_{mode.lower()}")
        source = "lod_vs_distance"
    if not candidates:
        return None
    df = pd.concat(candidates, ignore_index=True)
    if "distance_um" not in df.columns:
        return None
    nm_target = _pick_nm_for_distance(df, nm_hint)
    if math.isfinite(nm_target) and "Nm_per_symbol" in df.columns:
        df = df[np.isclose(df["Nm_per_symbol"], nm_target, atol=1.0)]
    if df.empty:
        return None
    df = df.assign(snr_plot=df.apply(_ensure_snr, axis=1))
    df = df.dropna(subset=["snr_plot", "distance_um"])
    if df.empty:
        return None
    return df


def _prepare_ts_panel(mode: str, suffix: str) -> Optional[pd.DataFrame]:
    frames = _load_csv(f"snr_vs_ts_{mode.lower()}{suffix}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if "symbol_period_s" not in df.columns:
        return None
    df = df.assign(snr_plot=df.apply(_ensure_snr, axis=1))
    df = df.dropna(subset=["snr_plot", "symbol_period_s"])
    if df.empty:
        return None
    return df


def _render_mode_panels(
    ax_nm,
    ax_dist,
    ax_ts,
    mode: str,
    nm_curve: Optional[pd.DataFrame],
    nm_col: Optional[str],
    distance_curve: Optional[pd.DataFrame],
    ts_curve: Optional[pd.DataFrame],
) -> None:
    # Nm panel
    if nm_curve is not None and nm_col:
        ax_nm.plot(nm_curve[nm_col], nm_curve["snr_db"], marker="o")
        ax_nm.set_xscale("log")
    else:
        ax_nm.text(0.5, 0.5, "No Nm data", ha="center", va="center", transform=ax_nm.transAxes)
    ax_nm.set_xlabel("Nm per symbol")
    ax_nm.set_ylabel("SNR (dB)")
    ax_nm.set_title(f"{mode}: SNR vs Nm")
    ax_nm.grid(True, which="both", alpha=0.3)

    # Distance panel
    if distance_curve is not None:
        ax_dist.plot(distance_curve["distance_um"], distance_curve["snr_db"], marker="s")
    else:
        ax_dist.text(0.5, 0.5, "No distance data", ha="center", va="center", transform=ax_dist.transAxes)
    ax_dist.set_xlabel("Distance (µm)")
    ax_dist.set_ylabel("SNR (dB)")
    ax_dist.set_title("SNR vs distance")
    ax_dist.grid(True, alpha=0.3)

    # Ts panel
    if ts_curve is not None:
        ax_ts.plot(ts_curve["symbol_period_s"], ts_curve["snr_db"], marker="^", label="SNR")
        ax_ts.set_xscale("log")
        ser_available = "ser" in ts_curve.columns and ts_curve["ser"].notna().any()
        if ser_available:
            ax_ts2 = ax_ts.twinx()
            ax_ts2.plot(
                ts_curve["symbol_period_s"],
                ts_curve["ser"],
                color="tab:red",
                linestyle="--",
                marker="o",
                label="SER",
            )
            ax_ts2.set_ylabel("SER")
            ax_ts2.set_yscale("log")
            ax_ts.legend(loc="upper left", fontsize=8)
            ax_ts2.legend(loc="upper right", fontsize=8)
        else:
            ax_ts.legend(loc="upper left", fontsize=8)
    else:
        ax_ts.text(0.5, 0.5, "No Ts data", ha="center", va="center", transform=ax_ts.transAxes)
    ax_ts.set_xlabel("Symbol period Ts (s)")
    ax_ts.set_ylabel("SNR (dB)")
    ax_ts.set_title("SNR vs Ts")
    ax_ts.grid(True, which="both", alpha=0.3)


def _plot_mode_panels(
    mode: str,
    nm_curve: Optional[pd.DataFrame],
    nm_col: Optional[str],
    distance_curve: Optional[pd.DataFrame],
    ts_curve: Optional[pd.DataFrame],
    out_path: Path,
) -> None:
    apply_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    _render_mode_panels(axes[0], axes[1], axes[2], mode, nm_curve, nm_col, distance_curve, ts_curve)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    fig.savefig(tmp, dpi=350, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    tmp.replace(out_path)
    print(f"[saved] {out_path}")


def _plot_combined_panels(
    specs: Sequence[Dict[str, Any]],
    out_path: Path,
    force: bool = False,
) -> None:
    if not specs:
        return
    if out_path.exists() and not force:
        print(f"?? Combined SNR panel exists ({out_path.name}); use --force to rebuild.")
        return
    apply_ieee_style()
    rows = len(specs)
    fig, axes = plt.subplots(rows, 3, figsize=(10.5, 3.2 * rows), squeeze=False)
    for idx, spec in enumerate(specs):
        ax_nm, ax_dist, ax_ts = axes[idx]
        _render_mode_panels(
            ax_nm,
            ax_dist,
            ax_ts,
            spec["mode"],
            spec.get("nm_curve"),
            spec.get("nm_col"),
            spec.get("distance_curve"),
            spec.get("ts_curve"),
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    fig.savefig(tmp, dpi=350, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    tmp.replace(out_path)
    print(f"[saved] {out_path}")


def main() -> None:
    args = _parse_args()
    modes = _normalize_modes(args.modes)
    summary_rows: List[Dict[str, Any]] = []
    combined_specs: List[Dict[str, Any]] = []

    for mode in modes:
        df_nm, nm_col = _prepare_nm_panel(mode, args.distance)
        df_distance = _prepare_distance_panel(mode, args.nm)
        df_ts = _prepare_ts_panel(mode, args.ts_suffix)
        if all(data is None for data in (df_nm, df_distance, df_ts)):
            print(f"?? SNR panels: no data sources for {mode}, skipping.")
            continue

        nm_curve = _aggregate_nm_curve(df_nm, nm_col)
        distance_curve = _aggregate_distance_curve(df_distance)
        ts_curve = _aggregate_ts_curve(df_ts)
        has_curves = any(curve is not None for curve in (nm_curve, distance_curve, ts_curve))
        if not has_curves:
            print(f"?? SNR panels: insufficient finite data for {mode}, skipping.")
            continue

        if args.write_summary:
            _extend_summary(summary_rows, mode, "nm", nm_curve, "x_value")
            _extend_summary(summary_rows, mode, "distance", distance_curve, "x_value")
            _extend_summary(summary_rows, mode, "ts", ts_curve, "x_value")

        combined_specs.append(
            {
                "mode": mode,
                "nm_curve": nm_curve,
                "nm_col": nm_col,
                "distance_curve": distance_curve,
                "ts_curve": ts_curve,
            }
        )

        fig_path = fig_dir / f"{args.output_prefix}_{mode.lower()}.png"
        if fig_path.exists() and not args.force:
            print(f"?? SNR panel exists for {mode} (use --force to rebuild).")
        else:
            _plot_mode_panels(mode, nm_curve, nm_col, distance_curve, ts_curve, fig_path)

    if args.write_summary and summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(by=["mode", "panel", "x_value"])
        _atomic_write_csv(summary_df, args.summary_output)
        print(f"[saved] {args.summary_output}")
    elif args.write_summary:
        print("?? SNR panels: no summary rows to write.")

    if args.build_combined and combined_specs:
        _plot_combined_panels(combined_specs, args.combined_output, force=args.force)
    elif args.build_combined:
        print("?? SNR panels: no data available for combined figure.")


if __name__ == "__main__":
    main()
