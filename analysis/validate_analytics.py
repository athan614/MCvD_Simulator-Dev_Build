#!/usr/bin/env python3
"""Overlay analytic BER/SEP curves with simulation data for MoSK and CSK-4."""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style

try:
    from src.mc_detection.algorithms import ber_mosk_analytic, sep_csk_mary
except Exception:  # pragma: no cover - fallback for older trees
    from src.detection import ber_mosk_analytic, sep_csk_mary  # type: ignore[misc]

data_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


def _load_ser_csv(mode: str) -> pd.DataFrame | None:
    path = data_dir / f"ser_vs_nm_{mode.lower()}.csv"
    if not path.exists():
        print(f"??  Missing SER CSV for {mode}: {path}")
        return None
    return pd.read_csv(path)


def _ci_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    candidates = [
        ("ser_ci_low", "ser_ci_high"),
        ("ci_low", "ci_high"),
        ("ser_ci_lower", "ser_ci_upper"),
    ]
    for low, high in candidates:
        if low in df.columns and high in df.columns:
            return low, high
    return None


def _infer_snr_label(df: pd.DataFrame) -> str:
    if "snr_semantics" in df.columns:
        non_empty = df["snr_semantics"].dropna()
        if not non_empty.empty:
            semantics = str(non_empty.iloc[0]).lower()
            if "charge" in semantics or "q-statistic" in semantics:
                return "SNR_Q (dB)"
            if "current" in semantics or "contrast" in semantics:
                return "SNR_I (dB)"
    return "SNR (dB)"


def _plot_mode(ax: Axes, mode: str, label: str, theory_fn, overlay: str) -> None:
    df = _load_ser_csv(mode)
    if df is None or "snr_db" not in df.columns or "ser" not in df.columns:
        ax.set_visible(False)
        return

    snr_db = pd.to_numeric(df["snr_db"], errors="coerce")
    ser = pd.to_numeric(df["ser"], errors="coerce")
    mask = (~snr_db.isna()) & (~ser.isna())
    if not mask.any():
        ax.set_visible(False)
        return

    snr_db = snr_db[mask]
    ser = ser[mask]
    snr_vals = snr_db.to_numpy(dtype=np.float64)
    ser_vals = ser.to_numpy(dtype=np.float64)
    ax.semilogy(snr_vals, ser_vals, marker="o", linestyle="none", label="Simulation")

    if overlay == "theory":
        snr_lin = 10.0 ** (snr_vals / 10.0)
        theory = theory_fn(snr_lin)
        ax.semilogy(snr_vals, theory, label="Theory")

    ci_cols = _ci_columns(df)
    if ci_cols is not None:
        low = pd.to_numeric(df.loc[mask, ci_cols[0]], errors="coerce").to_numpy()
        high = pd.to_numeric(df.loc[mask, ci_cols[1]], errors="coerce").to_numpy()
        if not np.isnan(low).all() and not np.isnan(high).all():
            ax.fill_between(snr_vals, low, high, color="0.8", alpha=0.5, label="95% CI")

    ax.set_xlabel(_infer_snr_label(df))
    ax.set_ylabel(label)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate analytic SER/BER curves against simulation data.")
    parser.add_argument(
        "--overlay",
        choices=["none", "theory"],
        default="none",
        help="Include analytic overlays (default: none).",
    )
    args = parser.parse_args()

    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    _plot_mode(axes[0], "MoSK", "BER (MoSK)", lambda snr: ber_mosk_analytic(snr, snr), args.overlay)
    _plot_mode(axes[1], "CSK", "SER (CSK-4)", lambda snr: sep_csk_mary(snr, M=4), args.overlay)

    plt.tight_layout()
    out = fig_dir / "fig_theory_vs_sim.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
