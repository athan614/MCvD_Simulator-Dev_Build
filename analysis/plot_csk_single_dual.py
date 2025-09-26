#!/usr/bin/env python3
"""Plot CSK single vs dual baselines into paired comparison panels."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from analysis.ieee_plot_style import apply_ieee_style

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

SER_VARIANTS = {
    "single_DA_noctrl": "Single DA",
    "single_SERO_noctrl": "Single SERO",
    "dual_noctrl": "Dual (DA+SERO)",
}

LOD_VARIANTS = {
    "single_DA_ctrl": "Single DA (CTRL)",
    "single_SERO_ctrl": "Single SERO (CTRL)",
    "dual_ctrl": "Dual (CTRL)",
}


def _nm_column(df: pd.DataFrame) -> str | None:
    for cand in ("pipeline_Nm_per_symbol", "pipeline.Nm_per_symbol"):
        if cand in df.columns:
            return cand
    return None


def _plot_ser() -> None:
    apply_ieee_style()
    plt.figure(figsize=(3.6, 3.2))
    plotted = False
    for key, label in SER_VARIANTS.items():
        csv_path = data_dir / f"ser_vs_nm_csk_{key}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        nm_col = _nm_column(df)
        if nm_col is None or "ser" not in df.columns:
            continue
        x = pd.to_numeric(df[nm_col], errors="coerce")
        y = pd.to_numeric(df["ser"], errors="coerce")
        mask = (~x.isna()) & (~y.isna())
        if not mask.any():
            continue
        plt.semilogy(x[mask], y[mask], marker="o", label=label)
        plotted = True
    if not plotted:
        print("??  No SER baseline CSVs found; skipping SER plot")
        plt.close()
        return
    plt.axhline(1e-2, linestyle="--", linewidth=1.0, color="0.5")
    plt.xlabel("Molecules per symbol (Nm)")
    plt.ylabel("SER")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = fig_dir / "fig_csk_single_dual_ser.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def _plot_lod() -> None:
    apply_ieee_style()
    plt.figure(figsize=(3.6, 3.2))
    plotted = False
    for key, label in LOD_VARIANTS.items():
        csv_path = data_dir / f"lod_vs_distance_csk_{key}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "distance_um" not in df.columns or "lod_nm" not in df.columns:
            continue
        d = pd.to_numeric(df["distance_um"], errors="coerce")
        lod = pd.to_numeric(df["lod_nm"], errors="coerce")
        mask = (~d.isna()) & (~lod.isna())
        if not mask.any():
            continue
        plt.plot(d[mask], lod[mask], marker="s", label=label)
        plotted = True
    if not plotted:
        print("??  No LoD baseline CSVs found; skipping LoD plot")
        plt.close()
        return
    plt.xlabel("Distance (um)")
    plt.ylabel("LoD Nm @ SER=1%")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = fig_dir / "fig_csk_single_dual_lod.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    _plot_ser()
    _plot_lod()
