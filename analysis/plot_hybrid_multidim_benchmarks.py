# analysis/plot_hybrid_multidim_benchmarks.py
"""
Stage 10 — Hybrid Multi‑Dimensional Benchmarks

Panels:
  (A) Hybrid Decision Components (HDS): total SER and decomposed MoSK/CSK error
      components vs Nm. Also shows a small dominance map (which component dominates).
  (B) OECT‑Normalized Sensitivity Index (ONSI):
        ONSI = (ΔI_diff / σ_I_diff) / (g_m / C_tot)
      computed from device parameters and Nm, using a light thermal‑noise proxy.
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

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from matplotlib.axes import Axes

# Project root & imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style
# For default OECT parameters (gm, C_tot, R_ch, etc.) used in ONSI proxy
# and to keep the notation aligned with the device model.
try:
    from src.mc_receiver import oect as oect_mod  # project layout
except Exception:  # fallback for local flat layouts
    from src.mc_receiver import oect as oect_mod  # keep same symbol

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
        for cand in (p.with_name(p.stem + "_uniform.csv"),
                     p.with_name(p.stem + "_zero.csv")):
            if cand.exists():
                try:
                    return pd.read_csv(cand)
                except Exception:
                    return None
    return None

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

# -------------------------- Panel B: ONSI proxy -----------------------------

def _compute_onsi_from_nm(
    nm: np.ndarray,
    gm_S: float,
    C_tot_F: float,
    R_ch_ohm: float,
    temperature_K: float,
    bits_per_symbol: float = 2.0,
    q_eff_abs_e: float = 1.0,
    detection_B_Hz: float = 100.0,
) -> Dict[str, np.ndarray]:
    """
    Compute ONSI(Nm) = (ΔI_diff / σ_I_diff) / (g_m / C_tot).

    Modeling notes (kept simple and explicit for a figure-level proxy):
      • ΔI_diff  ≈ g_m * (q_eff_abs * e) * (ΔN_b) / C_tot  (magnitude)
         For a two-level amplitude bit in Hybrid, ΔN_b ≈ 0.5 * Nm (low=½, high=1),
         so ΔI_diff ∝ (0.5 * Nm). We use magnitude (sign not informative here).
      • σ_I_diff ≈ sqrt(4 k_B T / R_ch * B_det)  (white/thermal proxy; one-sided)
      • (g_m / C_tot) comes from device mapping; normalizing removes first-order device scaling.

    The proportionality constants cancel in the ratio; this is intended for
    **comparative** visualization across Nm and device settings (not absolute metrology).
    """
    kB = 1.380649e-23
    e = 1.602176634e-19

    nm_arr = np.asarray(nm, dtype=np.float64)
    # ΔN_b ~ 0.5 * Nm for the amplitude bit (2-level)
    delta_N = 0.5 * nm_arr

    delta_I = gm_S * (q_eff_abs_e * e) * delta_N / C_tot_F  # magnitude
    sigma_I = np.sqrt(4.0 * kB * temperature_K / R_ch_ohm * detection_B_Hz)

    gm_over_C = gm_S / C_tot_F
    with np.errstate(divide="ignore", invalid="ignore"):
        onsi = (delta_I / sigma_I) / gm_over_C
    onsi = np.where(np.isfinite(onsi), onsi, 0.0)

    return {"Nm": nm_arr, "ONSI": onsi.astype(np.float64)}

def _value_key(v):
    """
    Canonicalize numeric values to consistent string representation.
    Must match analysis/run_final_analysis.py implementation.
    """
    try:
        vf = float(v)
        return str(int(vf)) if vf.is_integer() else f"{vf:.6g}"
    except Exception:
        return str(v)

def _load_device_params(results_dir: Path) -> Tuple[float, float, float]:
    """
    Load device parameters with priority: CSV data > config/default.yaml > module defaults.
    Returns (gm_S, C_tot_F, R_ch) tuple for complete consistency.
    """
    import yaml
    
    # Initialize variables with None to track what we've found
    gm_S: Optional[float] = None
    C_tot_F: Optional[float] = None
    R_ch: Optional[float] = None
    
    data_dir = results_dir / "data"
    
    # 1) Try to load from recent CSV data (most accurate)
    for csv_name in ["ser_vs_nm_hybrid.csv", "ser_vs_nm_mosk.csv", "ser_vs_nm_csk.csv"]:
        csv_path = data_dir / csv_name
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Look for device parameter columns
                gm_col = None
                c_col = None
                r_col = None
                
                for col in df.columns:
                    if 'gm_s' in col.lower() or (col.lower() == 'gm' and 'gm_s' not in df.columns):
                        gm_col = col
                    if 'c_tot_f' in col.lower() or 'c_tot' in col.lower():
                        c_col = col
                    if 'r_ch' in col.lower() or 'r_ch_ohm' in col.lower():
                        r_col = col
                
                if gm_col and c_col:
                    gm_val = pd.to_numeric(df[gm_col], errors='coerce').dropna()
                    c_val = pd.to_numeric(df[c_col], errors='coerce').dropna()
                    
                    if len(gm_val) > 0 and len(c_val) > 0:
                        gm_S = float(gm_val.iloc[0])
                        C_tot_F = float(c_val.iloc[0])
                        
                        # Try to get R_ch from CSV too
                        if r_col:
                            r_val = pd.to_numeric(df[r_col], errors='coerce').dropna()
                            if len(r_val) > 0:
                                R_ch = float(r_val.iloc[0])
                        
                        # If we got gm and C, break from loop
                        break
                        
            except Exception as e:
                print(f"⚠️  Could not read device params from {csv_name}: {e}")
                continue
    
    # 2) Try to load from config/default.yaml 
    try:
        config_path = project_root / "config" / "default.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract OECT parameters from config
            oect_config = config.get('oect', {})
            gm_S_config = oect_config.get('gm_S')
            C_tot_F_config = oect_config.get('C_tot_F') 
            R_ch_config = oect_config.get('R_ch_Ohm')
            
            # Use config values if we don't have them from CSV
            if gm_S is None and gm_S_config is not None:
                gm_S = float(gm_S_config)
            if C_tot_F is None and C_tot_F_config is not None:
                C_tot_F = float(C_tot_F_config)
            if R_ch is None and R_ch_config is not None:
                R_ch = float(R_ch_config)
                
    except Exception as e:
        print(f"⚠️  Could not load config/default.yaml: {e}")
    
    # 3) Fallback to module defaults for any missing values
    try:
        defaults = oect_mod.default_params()
        if gm_S is None:
            gm_S = float(defaults["gm_S"])
        if C_tot_F is None:
            C_tot_F = float(defaults["C_tot_F"])
        if R_ch is None:
            R_ch = float(defaults.get("R_ch_Ohm", 500.0))  # with fallback
    except Exception:
        # Use module defaults failed, apply hardcoded fallbacks
        if gm_S is None:
            gm_S = 2e-3
        if C_tot_F is None:
            C_tot_F = 18e-9
        if R_ch is None:
            R_ch = 500.0
    
    # 4) Final safety check - ensure all values are set
    if gm_S is None:
        gm_S = 2e-3  # Conservative hardcoded fallback
    if C_tot_F is None:
        C_tot_F = 18e-9  # Conservative hardcoded fallback  
    if R_ch is None:
        R_ch = 500.0  # Conservative hardcoded fallback
    
    # Determine source for logging
    if gm_S == 2e-3 and C_tot_F == 18e-9 and R_ch == 500.0:
        source = "hardcoded fallback"
    else:
        source = "mixed sources (CSV/config/module)"
    
    print(f"📊 ONSI: Using device params from {source}: gm={gm_S:.3e} S, C={C_tot_F:.3e} F, R={R_ch:.0f} Ω")
    return gm_S, C_tot_F, R_ch

def _compute_onsi_from_cache(
    nm_values: np.ndarray,
    cache_dir: Path,
    gm_S: float,
    C_tot_F: float
) -> Dict[str, np.ndarray]:
    """
    Read ONSI from cached simulation data for realistic noise statistics.
    
    Cache field compatibility notes:
    - Current runner writes: stats_da, stats_sero (raw statistics)
    - Future Stage 14 will write: noise_sigma, sigma_I_diff (processed noise)
    - This function tries multiple fallback strategies for robustness
    """
    import json
    e = 1.602176634e-19
    nm_arr = np.asarray(nm_values, dtype=np.float64)
    onsi_values = []
    
    # Thermal noise fallback function
    def _thermal_noise_proxy(gm_S: float, T_K: float, R_ch: float, B_Hz: float) -> float:
        kB = 1.380649e-23
        return float(np.sqrt(4.0 * kB * T_K / R_ch * B_Hz))
    
    for nm in nm_arr:
        # Look for cached simulation results at this Nm
        vk = _value_key(nm)
        
        # Stage 14+: First try to get noise_sigma_I_diff from CSV if available
        # (More efficient than crawling per-seed JSONs)
        sigma_I_diff = None
        
        # Check if we have direct noise column in any CSV
        for csv_name in ["ser_vs_nm_hybrid.csv", "ser_vs_nm_mosk.csv", "ser_vs_nm_csk.csv"]:
            csv_path = cache_dir.parent / "data" / csv_name  # results/data/
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    nm_col = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in df.columns else 'pipeline.Nm_per_symbol'
                    if nm_col in df.columns and 'noise_sigma_I_diff' in df.columns:
                        # Find matching Nm row
                        nm_matches = df[df[nm_col] == nm]
                        if not nm_matches.empty:
                            sigma_I_diff = float(nm_matches['noise_sigma_I_diff'].iloc[0])
                            break
                except Exception:
                    continue
        
        # If CSV lookup succeeded, use it directly
        if sigma_I_diff is not None:
            delta_N = 0.5 * nm
            delta_I = gm_S * e * delta_N / C_tot_F
            onsi_value = (delta_I / sigma_I_diff) / (gm_S / C_tot_F)
            onsi_values.append(onsi_value)
            print(f"📊 ONSI: Using CSV noise_sigma_I_diff for Nm={nm:.0e}: σ={sigma_I_diff:.2e}")
            continue
        
        # Fallback: existing JSON cache crawling logic starts here
        # Try both CTRL variants
        cache_paths = [
            cache_dir / "hybrid" / "ser_vs_nm_wctrl" / vk,
            cache_dir / "hybrid" / "ser_vs_nm_noctrl" / vk,
            cache_dir / "hybrid" / "ser_vs_nm_ctrl_unspecified" / vk
        ]
        
        sigma_measured = []
        delta_measured = []
        
        for cache_path in cache_paths:
            if cache_path.exists():
                for seed_file in cache_path.glob("seed_*.json"):
                    try:
                        with open(seed_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract noise and signal from cached results
                        # TODO: Stage 14 will standardize these field names
                        # For now, try multiple possible field names
                        sigma_fields = ['noise_sigma', 'sigma_I_diff', 'std_da', 'std_sero']
                        delta_fields = ['signal_delta', 'delta_I_diff', 'mean_diff']
                        
                        sigma_val = None
                        delta_val = None
                        
                        for field in sigma_fields:
                            if field in data:
                                sigma_val = float(data[field])
                                break
                        
                        for field in delta_fields:
                            if field in data:
                                delta_val = float(data[field])
                                break
                        
                        # If we don't have direct noise/signal, try to compute from stats
                        if sigma_val is None and 'stats_da' in data and 'stats_sero' in data:
                            stats_da = np.array(data['stats_da'])
                            stats_sero = np.array(data['stats_sero'])
                            if len(stats_da) > 0 and len(stats_sero) > 0:
                                # Combined standard deviation
                                diff_stats = np.array(stats_da) - np.array(stats_sero)
                                sigma_val = float(np.std(diff_stats))
                                delta_val = float(abs(np.mean(stats_da) - np.mean(stats_sero)))
                        
                        if sigma_val is not None and delta_val is not None:
                            sigma_measured.append(sigma_val)
                            delta_measured.append(delta_val)
                            
                    except Exception:
                        continue
                
                # If we found data in this cache path, break
                if sigma_measured:
                    break
        
        if sigma_measured and delta_measured:
            # Use median across seeds for stability
            sigma_I = float(np.median(sigma_measured))
            delta_I = float(np.median(delta_measured))
            print(f"📊 ONSI: Using cached noise for Nm={nm:.0e}: σ={sigma_I:.2e}, Δ={delta_I:.2e}")
        else:
            # Fallback to thermal proxy
            sigma_I = _thermal_noise_proxy(gm_S, 310.0, 1e6, 100.0)
            delta_I = gm_S * e * 0.5 * nm / C_tot_F  # Use resolved device C
            print(f"⚠️  ONSI: No cached noise for Nm={nm:.0e}, using thermal proxy")
        
        # Calculate ONSI for this Nm
        gm_over_C = gm_S / C_tot_F
        if sigma_I > 0 and gm_over_C > 0:
            onsi = (delta_I / sigma_I) / gm_over_C
        else:
            onsi = 0.0
        
        onsi_values.append(onsi)
    
    return {"Nm": nm_arr, "ONSI": np.array(onsi_values, dtype=np.float64)}

def _prepare_onsi(
    df_ser_nm: Optional[pd.DataFrame],
    use_simulation_noise: bool = False,
    cache_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Pulls Nm grid from ser_vs_nm_hybrid.csv and builds ONSI curve.
    
    Args:
        df_ser_nm: SER vs Nm data
        use_simulation_noise: If True, use cached σ from simulation; if False, use thermal proxy
        cache_dir: Directory containing cached simulation results
        results_dir: Results directory for device parameter loading
    """
    # Enhanced: Load device parameters consistently including R_ch
    if results_dir is not None:
        gm_S, C_tot_F, R_ch = _load_device_params(results_dir)
    else:
        # Fallback to original method
        try:
            defaults = oect_mod.default_params()
            gm_S = float(defaults["gm_S"])
            C_tot_F = float(defaults["C_tot_F"])
            R_ch = float(defaults.get("R_ch_Ohm", 500.0))
        except Exception:
            gm_S, C_tot_F, R_ch = 2e-3, 18e-9, 500.0

    temperature_K = 310.0
    detection_B_Hz = 100.0

    if df_ser_nm is None or df_ser_nm.empty:
        # build a reasonable Nm grid if data are absent
        Nm = np.logspace(2.0, 5.0, 20, dtype=np.float64)
    else:
        nmcol = _nm_col(df_ser_nm)
        if nmcol is None:
            return None
        Nm = pd.to_numeric(df_ser_nm[nmcol], errors="coerce").dropna().to_numpy(dtype=np.float64)
        Nm = np.unique(np.clip(Nm, 1.0, np.inf))
        if Nm.size == 0:
            return None

    if use_simulation_noise and cache_dir is not None:
        return _compute_onsi_from_cache(Nm, cache_dir, gm_S, C_tot_F)
    else:
        return _compute_onsi_from_nm(
            Nm, gm_S=gm_S, C_tot_F=C_tot_F, R_ch_ohm=R_ch, temperature_K=temperature_K, detection_B_Hz=detection_B_Hz
        )

def _plot_onsi(ax: Axes, onsi: Dict[str, np.ndarray], use_realistic: bool = False) -> None:
    Nm = onsi["Nm"]; O = onsi["ONSI"]
    ax.semilogx(Nm, O, marker="o", linewidth=2)
    ax.set_xlabel("Number of Molecules per Symbol (Nm)")
    ax.set_ylabel("ONSI (arb. unit)")
    title_suffix = "(cached σ)" if use_realistic else "(thermal proxy)"
    ax.set_title(f"(B) OECT‑Normalized Sensitivity Index {title_suffix}")
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
    parser.add_argument("--realistic-onsi", action="store_true", 
                        help="Use simulation-measured noise for ONSI (vs thermal proxy)")
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
    cache_dir = results_dir / "cache"  # NEW: add cache directory
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
    onsi = _prepare_onsi(ser_nm, use_simulation_noise=args.realistic_onsi, 
                         cache_dir=cache_dir, results_dir=results_dir)
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
            cs_mosk = axA.contour(Xg, Yg, Z_mosk, levels=levels, colors="white", 
                                 linestyles="--", linewidths=1.5, alpha=0.8)
            cs_csk = axA.contour(Xg, Yg, Z_csk, levels=levels, colors="white", 
                                linestyles=":", linewidths=1.5, alpha=0.8)
            
            # Labels for contours
            axA.clabel(cs_mosk, inline=True, fontsize=7, fmt="MoSK: %.0e")
            label_fmt = {"additive": "CSK: %.0e", "conditional": "CSK(cond): %.0e", "effective": "CSK(eff): %.0e"}[args.hds_csk_mode]
            axA.clabel(cs_csk, inline=True, fontsize=7, fmt=label_fmt)
            
            # Legend for contour types
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='white', linestyle='--', label='MoSK contours'),
                Line2D([0], [0], color='white', linestyle=':', label='CSK contours')
            ]
            axA.legend(handles=legend_elements, loc='upper right', fontsize=8)

            axA.set_xlabel("Nm (molecules/symbol)")
            axA.set_ylabel("Distance (μm)")
            axA.set_title(f"(A) Hybrid Decision Surface: Total SER + Components\n{caption_text}")
        
            # Add marginal plots as overlays if desired
            # You could add contour lines for key SER levels
            cs_total = axA.contour(Xg, Yg, Z_total, levels=[1e-3, 1e-2, 1e-1], colors='yellow', alpha=0.7, linewidths=1)
            axA.clabel(cs_total, inline=True, fontsize=8, fmt='%.3f')
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

    # (B) ONSI
    axB = fig.add_subplot(2, 2, 3)  # type: Axes
    if onsi is not None:
        _plot_onsi(axB, onsi, use_realistic=args.realistic_onsi)
    else:
        axB.set_title("(B) OECT‑Normalized Sensitivity Index (proxy)")
        axB.text(0.5, 0.5, "ONSI input missing", ha="center", va="center", transform=axB.transAxes)
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
                        X = np.linspace(pivot.columns.min(), pivot.columns.max(), pivot.shape[1])
                        Y = np.linspace(pivot.index.min(), pivot.index.max(), pivot.shape[0])
                        Xg, Yg = np.meshgrid(X, Y)
                        cs = axC.contour(Xg, Yg, pivot.values, levels=valid_levels, 
                                        colors='white', alpha=0.8, linewidths=1.5)
                        
                        # Enhanced contour labels
                        axC.clabel(cs, inline=True, fontsize=8, fmt='%.3f bits/s',
                                  inline_spacing=3)
                
                # NEW: Optional overlay: CSK effective error contours (requires new columns)
                if {'csk_ser_eff'}.issubset(df_grid.columns):
                    p_eff = df_grid.pivot_table(index='distance_um', columns='guard_factor',
                                                values='csk_ser_eff', aggfunc='median')
                    if p_eff.size > 0 and np.isfinite(p_eff.values).any():
                        X = np.linspace(float(p_eff.columns.min()), float(p_eff.columns.max()), p_eff.shape[1])
                        Y = np.linspace(float(p_eff.index.min()), float(p_eff.index.max()), p_eff.shape[0])
                        Xg, Yg = np.meshgrid(X, Y)
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

    fig.suptitle("Hybrid Multi‑Dimensional Benchmarks (HDS • ONSI • IRT)", fontsize=10, y=0.99)
    fig.tight_layout()
    out_path = fig_dir / "fig_hybrid_multidim_benchmarks.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"✓ Saved: {out_path}")

if __name__ == "__main__":
    main()
