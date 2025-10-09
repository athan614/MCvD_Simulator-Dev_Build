# analysis/plot_hybrid_multidim_benchmarks.py
"""
Stage 10 - Hybrid Multi-Dimensional Benchmarks

Panels:
  (A) Hybrid Decision Components (HDS): total SER and decomposed MoSK/CSK error
      components vs Nm. Also shows a small dominance map (which component dominates).
  (B) Charge-domain QNSI:
        QNSI = DeltaQ_diff / sigma_Q_diff
      computed from charge-domain noise over the decision window.
  (C) ISI‑Robust Throughput (IRT): R_eff(Ts) = (bits/symbol)/Ts * (1 - SER(Ts))
      from the ISI guard‑factor sweep for Hybrid mode.

Inputs (canonical CSVs written by run_final_analysis.py):
  results/data/ser_vs_nm_hybrid.csv
  results/data/lod_vs_distance_hybrid.csv
  results/data/isi_tradeoff_hybrid.csv

This script is intentionally defensive:
  • Works if some CSVs are missing (skips that panel with a message).
  • Type‑safe for mypy/Pylance (no lists where ndarrays are expected; proper Axes type, etc.)
  • No reliance on QuadContourSet.collections (removes that fragile usage).

References to data/columns & style follow the Stage 1–9 code paths:
  - Column names and CSV locations per analysis/run_final_analysis.py and
    analysis/generate_comparative_plots.py. 
  - IEEE plotting style via analysis/ieee_plot_style.py.

"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from copy import deepcopy
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# Project root & imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style
from src.pipeline import calculate_proper_noise_sigma, _resolve_decision_window
from src.config_utils import preprocess_config
from analysis.noise_correlation import compute_qnsi

# ------------------------- Helpers (data/columns) ----------------------------

def _nm_col(df: pd.DataFrame) -> Optional[str]:
    if "pipeline_Nm_per_symbol" in df.columns:
        return "pipeline_Nm_per_symbol"
    if "pipeline.Nm_per_symbol" in df.columns:
        return "pipeline.Nm_per_symbol"
    return None

def _gf_col(df: pd.DataFrame) -> Optional[str]:
    if "guard_factor" in df.columns:
        return "guard_factor"
    if "pipeline.guard_factor" in df.columns:
        return "pipeline.guard_factor"
    return None

def _try_load_csv(p: Path) -> Optional[pd.DataFrame]:
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    # Fallbacks for legacy filenames (e.g., *_uniform.csv)
    if p.suffix == ".csv":
        for cand in (
            p.with_name(p.stem + "_uniform.csv"),
            p.with_name(p.stem + "_zero.csv"),
        ):
            if cand.exists():
                try:
                    return pd.read_csv(cand)
                except Exception:
                    continue
    return None


def _edge_indices(values: np.ndarray) -> List[int]:
    if values.size <= 1:
        return [0]
    return [0, values.size - 1]


def _add_marginal_curves(ax: Axes,
                          nm_axis: np.ndarray,
                          dist_axis: np.ndarray,
                          mosk_grid: np.ndarray,
                          csk_grid: np.ndarray) -> List[Line2D]:
    """
    Overlay MoSK/CSK marginal SER curves along the near/far distance edges.
    Returns handles for legend composition.
    """
    handles: List[Line2D] = []
    if nm_axis.size == 0 or dist_axis.size == 0:
        return handles

    nm_mask = np.isfinite(nm_axis)
    nm = nm_axis[nm_mask]
    if nm.size == 0:
        return handles

    mosk_colors = ["tab:red", "tab:orange"]
    csk_colors = ["tab:blue", "tab:cyan"]
    for idx, edge in enumerate(_edge_indices(dist_axis)):
        dist = dist_axis[edge]

        if mosk_grid.size:
            mosk_row = mosk_grid[edge]
            mosk_series = mosk_row[nm_mask]
            if np.isfinite(mosk_series).any():
                line = ax.plot(
                    nm,
                    mosk_series,
                    linestyle="--",
                    linewidth=1.4,
                    color=mosk_colors[idx % len(mosk_colors)],
                    label=f"MoSK edge {dist:.0f} μm",
                )[0]
                handles.append(line)

        if csk_grid.size:
            csk_row = csk_grid[edge]
            csk_series = csk_row[nm_mask]
            if np.isfinite(csk_series).any():
                line = ax.plot(
                    nm,
                    csk_series,
                    linestyle=":",
                    linewidth=1.4,
                    color=csk_colors[idx % len(csk_colors)],
                    label=f"CSK edge {dist:.0f} μm",
                )[0]
                handles.append(line)
    return handles

# -------------------------- Panel A: HDS components -------------------------

def _prepare_hds(df: Optional[pd.DataFrame]) -> Optional[Dict[str, np.ndarray]]:
    """
    Returns dict with 'Nm', 'ser_total', 'ser_mosk', 'ser_csk' as float64 ndarrays.
    Returns None if required columns are missing or df is empty.
    """
    if df is None or df.empty:
        return None
    nmcol = _nm_col(df)
    if nmcol is None or not all(c in df.columns for c in ["ser", "mosk_ser", "csk_ser"]):
        return None

    # Group by Nm (and optionally by CTRL state if present) and take medians for stability.
    # We ignore 'use_ctrl' stratification here to keep one clean curve; your ablation panel
    # already handles with/without CTRL explicitly.
    g = (
        df.groupby(nmcol, as_index=False)
          .agg({"ser": "median", "mosk_ser": "median", "csk_ser": "median"})
          .sort_values(by=nmcol)
    )

    Nm = pd.to_numeric(g[nmcol], errors="coerce").to_numpy(dtype=np.float64)
    ser_total = pd.to_numeric(g["ser"], errors="coerce").to_numpy(dtype=np.float64)
    ser_mosk = pd.to_numeric(g["mosk_ser"], errors="coerce").to_numpy(dtype=np.float64)
    ser_csk = pd.to_numeric(g["csk_ser"], errors="coerce").to_numpy(dtype=np.float64)

    mask = np.isfinite(Nm) & np.isfinite(ser_total) & np.isfinite(ser_mosk) & np.isfinite(ser_csk)
    Nm, ser_total, ser_mosk, ser_csk = Nm[mask], ser_total[mask], ser_mosk[mask], ser_csk[mask]
    if Nm.size == 0:
        return None

    return {
        "Nm": Nm,
        "ser_total": ser_total,
        "ser_mosk": ser_mosk,
        "ser_csk": ser_csk,
    }

def _plot_hds(ax: Axes, comp: Dict[str, np.ndarray]) -> None:
    """Lines for total/MoSK/CSK error components."""
    Nm = comp["Nm"]; st = comp["ser_total"]; sm = comp["ser_mosk"]; sc = comp["ser_csk"]
    ax.loglog(Nm, st, marker="o", linewidth=2, label="Total SER")
    ax.loglog(Nm, sm, marker="^", linewidth=2, linestyle="--", label="MoSK errors")
    ax.loglog(Nm, sc, marker="s", linewidth=2, linestyle="-.", label="CSK errors")
    ax.set_xlabel("Number of Molecules per Symbol (Nm)")
    ax.set_ylabel("Error Rate")
    ax.set_title("(A) Hybrid Decision Components")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.axhline(0.01, color="k", linestyle=":", alpha=0.6)
    ax.legend()

def _plot_component_dominance(ax: Axes, comp: Dict[str, np.ndarray]) -> None:
    """
    Small categorical map: for each Nm, which component dominates (MoSK vs CSK)?
    Encoded as 0=MoSK, 1=CSK; shown via a simple 2×N "heat" image (categorical).
    """
    Nm = comp["Nm"]; sm = comp["ser_mosk"]; sc = comp["ser_csk"]
    if Nm.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # 0 if MoSK >= CSK (dominates), 1 if CSK dominates — arbitrary but consistent
    dom_idx = (sc > sm).astype(np.int64)  # shape (N,)

    # Build a 2×N categorical strip image for visual weight
    # Row 0 (MoSK) highlights where MoSK dominates; Row 1 for CSK dominance
    N = int(Nm.size)
    img = np.zeros((2, N), dtype=np.float64)
    img[0, dom_idx == 0] = 1.0
    img[1, dom_idx == 1] = 1.0

    # extent must be a tuple of floats (xmin, xmax, ymin, ymax)
    xmin: float = float(np.min(Nm))
    xmax: float = float(np.max(Nm))
    extent: Tuple[float, float, float, float] = (xmin, xmax, -0.5, 1.5)

    im = ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xscale("log")
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["MoSK dominates", "CSK dominates"])
    ax.set_xlabel("Nm (log scale)")
    ax.set_title("(A′) Component Dominance Map")
    ax.grid(False)
    # Colorbar for reference
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dominance (1.0 = active)")

# -------------------------- Panel B: QNSI proxy -----------------------------


def _load_base_config(results_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []
    if results_dir is not None:
        candidates.append(results_dir / "config" / "run_config.yaml")
        candidates.append(results_dir / "config" / "default.yaml")
    candidates.append(project_root / "config" / "default.yaml")

    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            with candidate.open("r", encoding="utf-8") as fh:
                raw_cfg = yaml.safe_load(fh)
            if not isinstance(raw_cfg, dict):
                continue
            processed = preprocess_config(raw_cfg)
            if isinstance(processed, dict):
                return processed
        except Exception:
            continue
    return None


def _extract_rho_post(row: Optional[pd.Series], cfg: Optional[Dict[str, Any]]) -> float:
    if row is not None:
        for key in ("rho_cc_measured", "rho_cc"):
            if key in row and pd.notna(row[key]):
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    pass
    if cfg is None:
        return 0.0
    noise_cfg = cfg.get("noise", {})
    rho_val = noise_cfg.get("rho_between_channels_after_ctrl",
                            noise_cfg.get("effective_correlation",
                                           noise_cfg.get("rho_corr", 0.0)))
    try:
        rho_float = float(rho_val)
    except (TypeError, ValueError):
        rho_float = 0.0
    return max(-1.0, min(1.0, rho_float))


def _prepare_qnsi(
    df_ser_nm: Optional[pd.DataFrame],
    use_simulation_noise: bool = False,
    results_dir: Optional[Path] = None
) -> Optional[Dict[str, np.ndarray]]:
    if df_ser_nm is None or df_ser_nm.empty:
        return None

    nmcol = _nm_col(df_ser_nm)
    if nmcol is None:
        return None

    summary = (
        df_ser_nm.groupby(nmcol, as_index=False)
                .median(numeric_only=True)
                .sort_values(by=nmcol)
    )

    Nm = pd.to_numeric(summary[nmcol], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(Nm)
    if not mask.any():
        return None

    summary = summary.loc[mask]
    Nm = Nm[mask]

    base_cfg = _load_base_config(results_dir)
    qnsi_values: List[float] = []

    for _, row in summary.iterrows():
        nm_value = float(row[nmcol])
        cfg = deepcopy(base_cfg) if base_cfg is not None else None
        sigma_da_Q: Optional[float] = None
        sigma_sero_Q: Optional[float] = None
        rho_post = 0.0
        delta_q_diff = math.nan

        delta_q_diff_val: float = math.nan
        if cfg is not None:
            pipeline_cfg = cfg.setdefault("pipeline", {})
            pipeline_cfg["Nm_per_symbol"] = nm_value

            Ts_candidate = row.get("symbol_period_s")
            if pd.notna(Ts_candidate):
                try:
                    Ts_new = float(Ts_candidate)
                    if Ts_new > 0.0:
                        pipeline_cfg["symbol_period_s"] = Ts_new
                        pipeline_cfg["time_window_s"] = max(float(pipeline_cfg.get("time_window_s", Ts_new)), Ts_new)
                        sim_cfg = cfg.setdefault("sim", {})
                        sim_cfg["time_window_s"] = max(float(sim_cfg.get("time_window_s", Ts_new)), Ts_new)
                except (TypeError, ValueError):
                    pass

            Ts = float(cfg.get("pipeline", {}).get("symbol_period_s", 0.0))
            sim_cfg = cfg.setdefault("sim", {})
            dt = float(sim_cfg.get("dt_s", 0.01))
            detection_window_s = _resolve_decision_window(cfg, Ts, dt)

            measured_sigma_da = row.get("noise_sigma_da")
            measured_sigma_sero = row.get("noise_sigma_sero")
            measured_sigma_diff_charge = row.get("noise_sigma_diff_charge")

            if use_simulation_noise and pd.notna(measured_sigma_da) and pd.notna(measured_sigma_sero):
                sigma_da_Q = float(measured_sigma_da)
                sigma_sero_Q = float(measured_sigma_sero)
            else:
                calc_da, calc_sero = calculate_proper_noise_sigma(cfg, detection_window_s)
                sigma_da_Q = float(calc_da)
                sigma_sero_Q = float(calc_sero)

            rho_post = _extract_rho_post(row, cfg)
            if use_simulation_noise and pd.notna(measured_sigma_diff_charge):
                sigma_q_diff = float(measured_sigma_diff_charge)
            elif sigma_da_Q is not None and sigma_sero_Q is not None:
                sigma_q_diff = math.sqrt(max(
                    sigma_da_Q**2 + sigma_sero_Q**2 - 2.0 * rho_post * sigma_da_Q * sigma_sero_Q,
                    0.0
                ))
            else:
                sigma_q_diff = math.nan

            delta_q_diff_raw = row.get("delta_Q_diff")
            delta_over_sigma_q = row.get("delta_over_sigma_Q", row.get("delta_over_sigma"))
            if pd.notna(delta_q_diff_raw) and math.isfinite(float(delta_q_diff_raw)):
                delta_q_diff_val = float(delta_q_diff_raw)
            elif pd.notna(delta_over_sigma_q) and math.isfinite(sigma_q_diff):
                delta_q_diff_val = float(delta_over_sigma_q) * sigma_q_diff
            else:
                delta_I_diff = row.get("delta_I_diff")
                gm_S = cfg.get("oect", {}).get("gm_S")
                C_tot_F = cfg.get("oect", {}).get("C_tot_F")
                if (pd.notna(delta_I_diff) and gm_S not in (None, 0.0) and
                        C_tot_F not in (None, 0.0)):
                    try:
                        delta_q_diff_val = float(delta_I_diff) * (float(C_tot_F) / float(gm_S))
                    except (TypeError, ValueError, ZeroDivisionError):
                        delta_q_diff_val = math.nan

        else:
            delta_over_sigma_q = row.get("delta_over_sigma_Q", row.get("delta_over_sigma"))
            if pd.notna(delta_over_sigma_q):
                qnsi_values.append(float(delta_over_sigma_q))
                continue

        sigma_da_scalar = float(sigma_da_Q) if sigma_da_Q is not None else math.nan
        sigma_sero_scalar = float(sigma_sero_Q) if sigma_sero_Q is not None else math.nan
        if not math.isfinite(delta_q_diff_val):
            delta_q_diff_val = math.nan
        if (not math.isfinite(sigma_da_scalar) or not math.isfinite(sigma_sero_scalar) or
                not math.isfinite(delta_q_diff_val)):
            qnsi_value = math.nan
        else:
            qnsi_value = float(compute_qnsi(delta_q_diff_val, sigma_da_scalar, sigma_sero_scalar, rho_post))
        qnsi_values.append(qnsi_value)

    qnsi_arr = np.asarray(qnsi_values, dtype=np.float64)
    finite_mask = np.isfinite(qnsi_arr)
    Nm_filtered = Nm[finite_mask] if finite_mask.any() else Nm
    qnsi_filtered = qnsi_arr[finite_mask] if finite_mask.any() else qnsi_arr

    return {"Nm": Nm_filtered.astype(np.float64), "QNSI": qnsi_filtered}


def _plot_qnsi(ax: Axes, qnsi: Dict[str, np.ndarray], use_realistic: bool = False) -> None:
    Nm = qnsi["Nm"]
    Q = qnsi["QNSI"]
    mask = np.isfinite(Nm) & np.isfinite(Q)
    if not mask.any():
        ax.set_title("(B) QNSI (charge-domain SNR)")
        ax.text(0.5, 0.5, "QNSI data unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    Nm_clean = Nm[mask]
    Q_clean = Q[mask]
    ax.semilogx(Nm_clean, Q_clean, marker="o", linewidth=2)
    ax.set_xlabel("Number of Molecules per Symbol (Nm)")
    ax.set_ylabel("QNSI (charge-domain SNR)")
    title_suffix = "(cached sigma)" if use_realistic else "(analytic noise model)"
    ax.set_title(f"(B) QNSI (charge-domain SNR) {title_suffix}")
    ax.grid(True, which="both", ls="--", alpha=0.3)

# ------------------------ Panel C: IRT (ISI throughput) ----------------------

def _prepare_irt_from_isi(df_isi: Optional[pd.DataFrame]) -> Optional[Dict[str, np.ndarray]]:
    """
    From isi_tradeoff_hybrid.csv, compute IRT(Ts) = (2 / Ts) * (1 - SER).
    Uses 'symbol_period_s' and 'ser'. Aggregates by guard factor if necessary.
    """
    if df_isi is None or df_isi.empty:
        return None
    if "ser" not in df_isi.columns or "symbol_period_s" not in df_isi.columns:
        return None

    # Median aggregate for stability across seeds
    gcols: List[str] = []  # group by nothing unless guard factor exists
    gfcol = _gf_col(df_isi)
    if gfcol:
        gcols = [gfcol]
    g = (
        df_isi.groupby(gcols, as_index=False)
              .agg({
                  "symbol_period_s": lambda x: float(np.median(pd.to_numeric(x, errors="coerce"))),
                  "ser": "median"
              })
    )
    # Now sort by Ts
    g = g.sort_values(by="symbol_period_s", kind="stable")

    Ts = pd.to_numeric(g["symbol_period_s"], errors="coerce").to_numpy(dtype=np.float64)
    ser = pd.to_numeric(g["ser"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(Ts) & np.isfinite(ser) & (Ts > 0.0)
    Ts, ser = Ts[mask], ser[mask]
    if Ts.size == 0:
        return None

    bits_per_symbol = 2.0  # Hybrid carries 2 bits/symbol
    R_eff = (bits_per_symbol / Ts) * (1.0 - ser)
    return {"Ts": Ts.astype(np.float64), "IRT": R_eff.astype(np.float64)}

def _plot_irt(ax: Axes, irt: Dict[str, np.ndarray]) -> None:
    """Plot 1D IRT curve with enhanced clarity and detailed annotations."""
    Ts = irt["Ts"]; R = irt["IRT"]
    order = np.argsort(Ts)
    
    # Main curve with enhanced styling
    ax.plot(Ts[order], R[order], marker="o", linewidth=2.5, color='darkblue', 
            markersize=5, markerfacecolor='lightblue', markeredgecolor='darkblue',
            markeredgewidth=1, label='R_eff(Ts)')
    
    # Enhanced axis labels with units and context
    ax.set_xlabel("Symbol Period Tₛ (seconds)")
    ax.set_ylabel("Effective Throughput R_eff (bits/s)")
    ax.set_title("(C) ISI‑Robust Throughput: R_eff = (2 bits/Tₛ) × (1 − SER)")
    
    # Improved grid
    ax.grid(True, ls="--", alpha=0.4, linewidth=0.8)
    ax.grid(True, which='minor', ls=":", alpha=0.2)
    
    # Add performance annotations
    if len(R[order]) > 0:
        max_throughput = np.max(R[order])
        max_idx = np.argmax(R[order])
        optimal_Ts = Ts[order][max_idx]
        
        # Annotate peak performance
        ax.annotate(f'Peak: {max_throughput:.3f} bits/s\nat Tₛ = {optimal_Ts:.2f}s', 
                   xy=(optimal_Ts, max_throughput), 
                   xytext=(0.7, 0.85), textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
                   fontsize=9, ha='center')
        
        # Add threshold lines for reference
        if max_throughput > 1.0:
            ax.axhline(1.0, color='green', linestyle=':', alpha=0.6, 
                      label='1 bit/s threshold')
        if max_throughput > 0.1:
            ax.axhline(0.1, color='orange', linestyle=':', alpha=0.6, 
                      label='0.1 bit/s threshold')
    
    # Formula explanation box
    formula_text = ("R_eff = (bits/symbol) / Tₛ × (1 − SER)\n"
                   "• Higher Tₛ → Lower ISI, Lower SER\n"
                   "• Lower Tₛ → Higher data rate potential\n"
                   "• Optimal Tₛ balances ISI vs throughput")
    
    ax.text(0.02, 0.35, formula_text, 
            transform=ax.transAxes, fontsize=8, 
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
    
    # Add legend if we have threshold lines
    if len(ax.get_lines()) > 1:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

# ------------------------------ Main / I/O ----------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate Hybrid Multi-Dimensional Benchmarks")
    parser.add_argument("--realistic-qnsi", dest="realistic_qnsi", action="store_true", 
                        help="Use simulation-measured noise for QNSI (vs analytic model)")
    parser.add_argument("--realistic-onsi", dest="realistic_qnsi", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--hds-csk-mode",
        choices=["additive", "conditional", "effective"],
        default="effective",
        help="Which CSK contour to draw in the HDS panel"
    )
    args = parser.parse_args()
    
    apply_ieee_style()

    results_dir = project_root / "results"
    data_dir = results_dir / "data"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data sets we need
    ser_nm = _try_load_csv(data_dir / "ser_vs_nm_hybrid.csv")
    lod_d  = _try_load_csv(data_dir / "lod_vs_distance_hybrid.csv")
    isi_df = _try_load_csv(data_dir / "isi_tradeoff_hybrid.csv")
    hds_grid = _try_load_csv(data_dir / "hybrid_hds_grid.csv")
    
    # ---- Prepare panels ----
    comp = _prepare_hds(ser_nm)
    # NEW: Pass results_dir for device parameter consistency
    qnsi = _prepare_qnsi(ser_nm, use_simulation_noise=args.realistic_qnsi, results_dir=results_dir)
    irt  = _prepare_irt_from_isi(isi_df)

    # ---- Build figure ----
    fig = plt.figure(figsize=(13.5, 7.8))

    # (A) HDS lines or 2-D surface if available
    axA = fig.add_subplot(2, 2, 1)  # type: Axes
    if hds_grid is not None and not hds_grid.empty \
        and all(c in hds_grid.columns for c in ["distance_um", "ser"]) \
        and _nm_col(hds_grid) is not None:
        # Build a 2-D heatmap over (Nm, distance)
        print("📊 Using 2D HDS grid for enhanced visualization")
        nmcol = _nm_col(hds_grid)
        df = hds_grid.copy()
        df[nmcol] = pd.to_numeric(df[nmcol], errors="coerce")
        df["distance_um"] = pd.to_numeric(df["distance_um"], errors="coerce")

        # ADD THIS: Generate grid resolution and seed count caption
        n_nm = df[nmcol].nunique()
        n_dist = df["distance_um"].nunique()
        if "num_runs" in df.columns:
            median_seeds = int(df["num_runs"].median())
            caption_text = f"Grid: {n_nm}×{n_dist} points, {median_seeds} seeds/point (median)"
        else:
            caption_text = f"Grid: {n_nm}×{n_dist} points"

        # Build pivot tables for total SER and components
        g = df.pivot_table(index="distance_um", columns=nmcol, values="ser", aggfunc="median")
        g_mosk = df.pivot_table(index="distance_um", columns=nmcol, values="mosk_ser", aggfunc="median")

        # Choose CSK column based on mode
        csk_col = "csk_ser"
        if args.hds_csk_mode == "conditional" and "csk_ser_cond" in df.columns:
            csk_col = "csk_ser_cond"
        elif args.hds_csk_mode == "effective" and all(c in df.columns for c in ["csk_ser_cond", "mosk_exposure_frac"]):
            # Compute effective CSK if not precomputed
            if "csk_ser_eff" not in df.columns:
                df["csk_ser_eff"] = pd.to_numeric(df["csk_ser_cond"], errors="coerce") * \
                                    pd.to_numeric(df["mosk_exposure_frac"], errors="coerce")
            csk_col = "csk_ser_eff"

        g_csk = df.pivot_table(index="distance_um", columns=nmcol, values=csk_col, aggfunc="median")

        if g.size > 0:
            # Coordinate grids
            X = g.columns.values.astype(float)
            Y = g.index.values.astype(float)
            Xg, Yg = np.meshgrid(X, Y)
            
            # Data grids
            Z_total = np.asarray(g, dtype=float)
            Z_mosk = np.asarray(g_mosk, dtype=float)
            Z_csk = np.asarray(g_csk, dtype=float)

            # Main heatmap (total SER)
            im = axA.imshow(
                Z_total, origin="lower", aspect="auto",
                extent=(X.min(), X.max(), Y.min(), Y.max()),
                cmap="viridis", interpolation="bilinear"
            )
            axA.set_xscale("log")
            cbar = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
            cbar.set_label("Total SER")

            # Component contours
            levels = [1e-3, 1e-2, 1e-1]
            cs_mosk = axA.contour(
                Xg, Yg, Z_mosk, levels=levels, colors="white",
                linestyles="--", linewidths=1.5, alpha=0.8
            )
            cs_csk = axA.contour(
                Xg, Yg, Z_csk, levels=levels, colors="white",
                linestyles=":", linewidths=1.5, alpha=0.8
            )

            # Labels for contours
            axA.clabel(cs_mosk, inline=True, fontsize=7, fmt="MoSK: %.0e")
            label_fmt = {
                "additive": "CSK: %.0e",
                "conditional": "CSK(cond): %.0e",
                "effective": "CSK(eff): %.0e",
            }[args.hds_csk_mode]
            axA.clabel(cs_csk, inline=True, fontsize=7, fmt=label_fmt)

            # Edge marginals for MoSK/CSK along grid extremes
            edge_handles = _add_marginal_curves(axA, X, Y, Z_mosk, Z_csk)

            # Legend for contours and edge marginals
            contour_handles = [
                Line2D([0], [0], color="white", linestyle="--", label="MoSK contours"),
                Line2D([0], [0], color="white", linestyle=":", label="CSK contours"),
            ]
            legend_handles = contour_handles + edge_handles
            if legend_handles:
                axA.legend(handles=legend_handles, loc="upper right", fontsize=8)

            axA.set_xlabel("Nm (molecules/symbol)")
            axA.set_ylabel("Distance (μm)")
            axA.set_title(f"(A) Hybrid Decision Surface: Total SER + Components\n{caption_text}")
        
            # Add total SER reference contours
            cs_total = axA.contour(
                Xg, Yg, Z_total, levels=[1e-3, 1e-2, 1e-1],
                colors="yellow", alpha=0.7, linewidths=1.0
            )
            axA.clabel(cs_total, inline=True, fontsize=8, fmt="%.3f")
        else:
            axA.set_title("(A) HDS Grid (insufficient data)")
            axA.text(0.5, 0.5, "Insufficient grid data for heatmap", 
                    ha="center", va="center", transform=axA.transAxes)
    else:
        # Fallback to 1D component curves if available
        if comp is not None:
            _plot_hds(axA, comp)
        else:
            axA.set_title("(A) Hybrid Decision Components")
            axA.text(0.5, 0.5, "No HDS grid or ser_vs_nm_hybrid.csv found", 
                    ha="center", va="center", transform=axA.transAxes)
            axA.axis("off")

    # (A′) Dominance map
    axA2 = fig.add_subplot(2, 2, 2)  # type: Axes
    if comp is not None:
        _plot_component_dominance(axA2, comp)
    else:
        axA2.set_title("(A′) Component Dominance Map")
        axA2.text(0.5, 0.5, "ser_vs_nm_hybrid.csv not found or incomplete",
                  ha="center", va="center", transform=axA2.transAxes)
        axA2.axis("off")

    # (B) QNSI
    axB = fig.add_subplot(2, 2, 3)  # type: Axes
    if qnsi is not None:
        _plot_qnsi(axB, qnsi, use_realistic=args.realistic_qnsi)
    else:
        axB.set_title("(B) QNSI (charge-domain SNR)")
        axB.text(0.5, 0.5, "QNSI input missing", ha="center", va="center", transform=axB.transAxes)
        axB.axis("off")

    # (C) IRT - Enhanced with 2D grid if available
    axC = fig.add_subplot(2, 2, 4)  # type: Axes
    
    # Try to load 2D ISI grid first
    isi_grid_path = data_dir / "isi_grid_hybrid.csv"
    if isi_grid_path.exists():
        try:
            print("📊 Using 2D ISI grid for enhanced IRT visualization")
            df_grid = pd.read_csv(isi_grid_path)
            df_grid['R_eff_bps'] = 2.0 / df_grid['symbol_period_s'] * (1.0 - df_grid['ser'])
            
            # Create pivot table for heatmap
            pivot = df_grid.pivot_table(
                index='distance_um', 
                columns='guard_factor',
                values='R_eff_bps', 
                aggfunc='median'
            )
            
            if pivot.size > 0:
                # Enhanced 2D heatmap with better color scaling
                vmin, vmax = 0, np.nanmax(pivot.values)
                im = axC.imshow(
                    pivot.values, 
                    aspect='auto', 
                    origin='lower',
                    extent=(
                        float(pivot.columns.min()), float(pivot.columns.max()),
                        float(pivot.index.min()), float(pivot.index.max())
                    ),
                    cmap='viridis',
                    interpolation='bilinear',
                    vmin=vmin, vmax=vmax
                )
                
                # Enhanced axis labels with context
                axC.set_xlabel('Guard Factor (fraction of symbol period Tₛ)')
                axC.set_ylabel('Communication Distance (μm)')
                axC.set_title('(C) ISI‑Robust Throughput: R_eff vs Guard Factor & Distance')
                
                # Enhanced colorbar with detailed labeling
                cbar = plt.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
                cbar.set_label('Effective Throughput R_eff (bits/s)\n[Higher is better]', fontsize=9)
                
                # Enhanced contour lines with clearer labeling
                if pivot.values.max() > 0:
                    # Adaptive contour levels based on data range
                    max_val = np.nanmax(pivot.values)
                    if max_val > 10:
                        levels = [0.1, 1.0, 5.0, 10.0]
                    elif max_val > 1:
                        levels = [0.01, 0.1, 0.5, 1.0]
                    else:
                        levels = [0.001, 0.01, 0.05, 0.1]
                    
                    valid_levels = [l for l in levels if l <= max_val and l >= vmin]
                    
                    if valid_levels:
                        col_values = pd.to_numeric(pd.Series(pivot.columns, dtype='object'), errors='coerce').to_numpy(dtype=np.float64)
                        row_values = pd.to_numeric(pd.Series(pivot.index, dtype='object'), errors='coerce').to_numpy(dtype=np.float64)
                        if not (np.isfinite(col_values).all() and np.isfinite(row_values).all()):
                            raise ValueError('Non-numeric ISI grid axes')
                        Xg, Yg = np.meshgrid(col_values, row_values)
                        cs = axC.contour(Xg, Yg, pivot.values.astype(float), levels=valid_levels, 
                                        colors='white', alpha=0.8, linewidths=1.5)
                        
                        # Enhanced contour labels
                        axC.clabel(cs, inline=True, fontsize=8, fmt='%.3f bits/s',
                                  inline_spacing=3)
                
                # NEW: Optional overlay: CSK effective error contours (requires new columns)
                if {'csk_ser_eff'}.issubset(df_grid.columns):
                    p_eff = df_grid.pivot_table(index='distance_um', columns='guard_factor',
                                                values='csk_ser_eff', aggfunc='median')
                    if p_eff.size > 0 and np.isfinite(p_eff.values).any():
                        col_eff = pd.to_numeric(pd.Series(p_eff.columns, dtype='object'), errors='coerce').to_numpy(dtype=np.float64)
                        row_eff = pd.to_numeric(pd.Series(p_eff.index, dtype='object'), errors='coerce').to_numpy(dtype=np.float64)
                        if not (np.isfinite(col_eff).all() and np.isfinite(row_eff).all()):
                            raise ValueError('Non-numeric CSK grid axes')
                        Xg, Yg = np.meshgrid(col_eff, row_eff)
                        Z = p_eff.values.astype(float)
                        # Pick levels relative to observed range
                        nz = Z[np.isfinite(Z)]
                        if nz.size:
                            vmax = float(np.nanmax(nz))
                            # Typical error‑rate contours; adjust if your data shifts
                            levels = [1e-3, 1e-2, 1e-1]
                            levels = [l for l in levels if l <= vmax and l > 0]
                            if levels:
                                cs_eff = axC.contour(Xg, Yg, Z, levels=levels,
                                                    colors='white', linestyles=':', linewidths=1.2, alpha=0.9)
                                axC.clabel(cs_eff, inline=True, fontsize=7, fmt='CSK eff: %.0e')
                
                # Add performance insights annotation
                max_throughput = np.nanmax(pivot.values)
                max_pos = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
                optimal_distance = float(pivot.index[int(max_pos[0])])
                optimal_guard = float(pivot.columns[int(max_pos[1])])
                
                insight_text = (f"Peak Performance:\n"
                               f"• {max_throughput:.3f} bits/s\n"
                               f"• Distance: {optimal_distance:.0f} μm\n"
                               f"• Guard Factor: {optimal_guard:.2f}")
                
                axC.text(0.02, 0.98, insight_text, 
                        transform=axC.transAxes, fontsize=8, 
                        verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))
                
                # Add axis context
                axC.text(0.98, 0.02, "Lower guard factor = Higher data rate\nShorter distance = Less ISI", 
                        transform=axC.transAxes, fontsize=7, 
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
            else:
                raise ValueError("Empty pivot table")
                
        except Exception as e:
            print(f"⚠️  2D ISI grid failed, falling back to 1D: {e}")
            # Fallback to 1D plot
            if irt is not None:
                _plot_irt(axC, irt)
            else:
                axC.set_title("(C) ISI‑Robust Throughput vs Ts (Hybrid)")
                axC.text(0.5, 0.5, "No IRT data available", 
                        ha="center", va="center", transform=axC.transAxes)
                axC.axis("off")
    else:
        # Original 1D fallback
        if irt is not None:
            _plot_irt(axC, irt)
        else:
            axC.set_title("(C) ISI‑Robust Throughput vs Ts (Hybrid)")
            axC.text(0.5, 0.5, "isi_tradeoff_hybrid.csv not found or incomplete",
                     ha="center", va="center", transform=axC.transAxes)
            axC.axis("off")

    fig.suptitle("Hybrid Multi-Dimensional Benchmarks (HDS | QNSI | IRT)", fontsize=10, y=0.99)
    fig.tight_layout()
    out_path = fig_dir / "fig_hybrid_multidim_benchmarks.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"✓ Saved: {out_path}")

if __name__ == "__main__":
    main()
