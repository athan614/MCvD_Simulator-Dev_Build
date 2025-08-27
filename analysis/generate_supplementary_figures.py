# analysis/generate_supplementary_figures.py
"""
Generate supplementary figures S3–S6 for the TMBMC paper.

S3: Constellations (illustrative; gated to synthetic points for now)
S4: SNR types I–III (illustrative)
S5: Confusion matrices (plots + an .npz artifact with proper ArrayLike dtypes)
S6: Energy/bit (illustrative; reads temperature from config)

Type-safety notes:
- All numeric values passed to Matplotlib are cast to 'float' to avoid
  Scalar/complex complaints in static type checkers.
- Any numpy saving (np.savez) uses ArrayLike meta fields (e.g., int8 for bools).
- If you later enable a "rebuild from raw" path that calls
  src.pipeline._single_symbol_currents, declare:
      tx_history: List[Tuple[int, float]] = []
  to match the function's signature in the pipeline module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from analysis.ieee_plot_style import apply_ieee_style

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# Optional: seaborn is unused; keep import guarded for environments without stubs
try:  # pragma: no cover
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None  # type: ignore

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config_utils import preprocess_config  # noqa: E402

# ENHANCEMENT: Import canonical value key formatter for consistent CSV ↔ cache matching
from analysis.run_final_analysis import canonical_value_key  # noqa: E402


# ---------- small helpers (type-safe plotting & saving) ----------------------

def _f(x: object) -> float:
    """Best-effort float conversion that strips any complex part safely (typed)."""
    # Fast paths for numeric Python / NumPy scalars
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, complex):
        return float(x.real)

    # Numpy scalar/array inputs
    try:
        arr = np.asarray(x)
    except Exception as exc:
        raise TypeError(f"Cannot convert value of type {type(x).__name__} to float") from exc

    # If complex-typed, take the real part via Python complex
    if np.iscomplexobj(arr):
        # Ensure we have at least one element
        if arr.size == 0:
            raise TypeError("Cannot convert empty array to float")
        cval = complex(arr.ravel()[0])
        return float(cval.real)

    # Real-valued: coerce to float64 and take first element (or scalar)
    try:
        arr64 = np.asarray(arr, dtype=np.float64)
        if arr64.ndim == 0:
            return float(arr64.item())
        return float(arr64.ravel()[0])
    except Exception as exc:
        raise TypeError(f"Cannot convert value to float: {x!r}") from exc


def _as_float1d(arr: np.ndarray | List[float]) -> np.ndarray:
    """Return a contiguous 1-D float64 array."""
    return np.asarray(arr, dtype=np.float64).ravel()


def _ensure_figdir(fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)


def _save_confusions_npz(
    npz_path: Path,
    matrices: Dict[str, np.ndarray],
    meta: Optional[Dict[str, object]] = None
) -> None:
    """
    Save confusion matrices with ArrayLike metadata to avoid type errors:
    - booleans become tiny int8 arrays
    - numbers become float64 arrays
    - strings become 0-d object/string arrays (supported by numpy)
    """
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    # matrices (force float64)
    for mk, mv in matrices.items():
        payload[mk] = np.asarray(mv, dtype=np.float64)

    # metadata (ArrayLike only) — use distinct names to avoid type reuse issues
    if meta:
        for meta_key, meta_val in meta.items():
            if isinstance(meta_val, (bool, np.bool_)):  # store as 0/1 int8
                payload[meta_key] = np.asarray([int(meta_val)], dtype=np.int8)
            elif isinstance(meta_val, (int, np.integer)):
                payload[meta_key] = np.asarray([int(meta_val)], dtype=np.int64)
            elif isinstance(meta_val, (float, np.floating)):
                payload[meta_key] = np.asarray([float(meta_val)], dtype=np.float64)
            elif isinstance(meta_val, str):
                # numpy allows string/object arrays
                payload[meta_key] = np.asarray([meta_val])
            else:
                # fallback to string form for arbitrary objects
                payload[meta_key] = np.asarray([str(meta_val)])

    # Finally write to disk (cast kwargs to Any to satisfy typing of numpy stubs)
    np.savez(npz_path, **cast(Dict[str, Any], payload))

def _get_csv_ser_at_nm(mode: str, nm: float, data_dir: Path, tolerance: float = 0.1) -> Optional[float]:
    """
    Get the SER for a specific mode and Nm from CSV data.
    
    Args:
        mode: Modulation mode (MoSK, CSK, Hybrid)
        nm: Target Nm value
        data_dir: Directory containing CSV files
        tolerance: Relative tolerance for Nm matching (default 10%)
    
    Returns:
        SER value if found, None otherwise
    """
    csv_path = data_dir / f"ser_vs_nm_{mode.lower()}.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Find Nm column
        nm_col = None
        for col in ["pipeline_Nm_per_symbol", "pipeline.Nm_per_symbol"]:
            if col in df.columns:
                nm_col = col
                break
        
        if nm_col is None or "ser" not in df.columns:
            return None
        
        # Find closest Nm within tolerance
        nm_values = pd.to_numeric(df[nm_col], errors='coerce')
        valid_mask = ~nm_values.isna()
        
        if not valid_mask.any():
            return None
        
        # Work entirely with pandas to avoid numpy/pandas type issues
        valid_df = df[valid_mask].copy()
        valid_nm_series = nm_values[valid_mask]
        
        # Calculate relative differences (keep as pandas Series)
        rel_diff_series = (valid_nm_series - nm).abs() / nm
        within_tolerance_mask = rel_diff_series <= tolerance
        
        if not within_tolerance_mask.any():
            return None
        
        # Filter to only rows within tolerance
        tolerance_mask_index = within_tolerance_mask[within_tolerance_mask].index
        tolerance_rel_diff = rel_diff_series.loc[tolerance_mask_index]
        
        # Find the index of minimum relative difference (using pandas idxmin)
        min_idx = tolerance_rel_diff.idxmin()  # This works on pandas Series
        
        # Get the SER value at that index (using pandas scalar extraction)
        ser_scalar = df.loc[min_idx, "ser"]
        
        # Convert to Python float using _f helper for robust conversion
        return _f(ser_scalar)
        
    except Exception as e:
        print(f"Error reading SER from CSV for {mode} at Nm={nm}: {e}")
        return None

# ---------------------- S3: Constellations (illustrative) --------------------

def plot_figure_s3_constellation(results_dir: Path, save_path: Path) -> None:
    """
    Generate Figure S3: Constellation diagrams for all modulation schemes.
    Shows decision space (q_DA vs q_SERO) with transmitted symbols marked.
    Currently illustrative; data-driven points will replace these later.
    """
    _ensure_figdir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modes = ['MoSK', 'CSK', 'Hybrid']

    for idx, mode in enumerate(modes):
        ax = axes[idx]

        # Gate on existence of some results; else display placeholder notice
        data_file = results_dir / "data" / f"ser_vs_nm_{mode.lower()}.csv"
        if not data_file.exists():
            ax.text(0.5, 0.5, f"No data for {mode}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{mode} Constellation")
            ax.set_xlabel(r'$q_{\mathrm{DA}}$ (nC)')
            ax.set_ylabel(r'$q_{\mathrm{SERO}}$ (nC)')
            ax.grid(True, alpha=0.3)
            continue

        # For S3 we still show synthetic clusters (remove once real scatter is wired)
        rng = np.random.default_rng(42)

        if mode == 'MoSK':
            # DA vs SERO binary classes
            q_da_0 = _as_float1d(rng.normal(1.0, 0.2, 120))
            q_sero_0 = _as_float1d(rng.normal(0.0, 0.2, 120))
            q_da_1 = _as_float1d(rng.normal(0.0, 0.2, 120))
            q_sero_1 = _as_float1d(rng.normal(1.0, 0.2, 120))
            ax.scatter(q_da_0, q_sero_0, c='tab:blue', alpha=0.5, label='Symbol 0 (DA)', s=30)
            ax.scatter(q_da_1, q_sero_1, c='tab:red', alpha=0.5, label='Symbol 1 (SERO)', s=30)
            # Decision boundary (y = x)
            x = np.linspace(-0.5, 1.5, 100, dtype=float)
            ax.plot(_as_float1d(x), _as_float1d(x), 'k--', alpha=0.5, label='Decision boundary')

        elif mode == 'CSK':
            # 4 amplitude levels on DA
            for level in range(4):
                q_da = _as_float1d(rng.normal(level / 3.0, 0.15, 60))
                q_sero = _as_float1d(rng.normal(0.0, 0.1, 60))
                ax.scatter(q_da, q_sero, alpha=0.5, label=f'Level {level}', s=30)

        else:  # Hybrid (2 bits: molecule bit + amplitude bit)
            symbols = [(0, 0), (0, 1), (1, 0), (1, 1)]
            colors = ['tab:blue', 'tab:cyan', 'tab:red', 'tab:orange']
            for (mol, amp), color in zip(symbols, colors):
                if mol == 0:  # DA
                    q_da = _as_float1d(rng.normal(0.5 + 0.5 * amp, 0.15, 60))
                    q_sero = _as_float1d(rng.normal(0.0, 0.1, 60))
                else:        # SERO
                    q_da = _as_float1d(rng.normal(0.0, 0.1, 60))
                    q_sero = _as_float1d(rng.normal(0.5 + 0.5 * amp, 0.15, 60))
                ax.scatter(q_da, q_sero, c=color, alpha=0.5, label=f'Symbol {mol}{amp}', s=30)

        ax.set_xlabel(r'$q_{\mathrm{DA}}$ (nC)')
        ax.set_ylabel(r'$q_{\mathrm{SERO}}$ (nC)')
        ax.set_title(f'({chr(97 + idx)}) {mode} Constellation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Figure S3: Constellation Diagrams', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------- S4: SNR (illustrative) ---------------------------

def plot_figure_s4_snr_types(results_dir: Path, save_path: Path) -> None:
    """
    Generate Figure S4: SNR analysis showing Types I, II, and III (illustrative).
    """
    _ensure_figdir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    distances = np.asarray([25, 50, 75, 100, 150, 200, 250], dtype=float)

    # Type I: Molecular
    snr_i = 30.0 * np.exp(-distances / 100.0)  # dB, decays with distance

    # Type II: Electrical (after transduction)
    eta = 0.7  # example transduction efficiency
    snr_ii = snr_i - 10.0 * np.log10(1.0 / float(eta))

    # Type III: Digital (after detection)
    coding_gain = 3.0  # dB
    snr_iii = snr_ii + coding_gain

    # Panel (a): SNR vs distance
    ax1.plot(_as_float1d(distances), _as_float1d(snr_i), 'b-o', label='Type I (Molecular)', linewidth=2)
    ax1.plot(_as_float1d(distances), _as_float1d(snr_ii), 'g--s', label='Type II (Electrical)', linewidth=2)
    ax1.plot(_as_float1d(distances), _as_float1d(snr_iii), 'r-.^', label='Type III (Digital)', linewidth=2)

    ax1.set_xlabel('Distance (μm)')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('(a) SNR Types vs Distance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0.0, 35.0])

    # Panel (b): SNR conversion “bars”
    categories = ['Molecular\n→Electrical', 'Electrical\n→Digital']
    factors = [float(-10.0 * np.log10(1.0 / float(eta))), float(coding_gain)]
    colors = ['tab:green', 'tab:red']

    bars = ax2.bar(categories, factors, color=colors, alpha=0.7)
    ax2.set_ylabel('SNR Change (dB)')
    ax2.set_title('(b) SNR Conversion Factors')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.axhline(y=0.0, color='k', linestyle='-', linewidth=0.5)

    # Value labels on bars — correct argument separation and float casts
    for bar, val in zip(bars, factors):
        height = float(bar.get_height())
        x_pos = float(bar.get_x() + bar.get_width() / 2.0)
        y_pos = float(height + 0.5 * np.sign(height))
        ax2.text(x_pos, y_pos, f'{_f(val):.1f} dB', ha='center',
                 va=('bottom' if height > 0 else 'top'))

    plt.suptitle('Figure S4: Signal-to-Noise Ratio Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# --------------------- S5: Confusion matrices (plot + npz) -------------------

def plot_figure_s5_confusion_matrices(results_dir: Path, save_path: Path, strict_mode: bool = False) -> None:
    """
    Figure S5 (Stage 8): Data-driven confusion matrices built from per-seed caches.
    - Finds Nm with SER≈1% in ser_vs_nm_{mode}.csv
    - Loads symbols_tx / symbols_rx from results/cache/<mode>/ser_vs_nm_*/<Nm>/seed_*.json
    - Falls back to illustrative matrices only if no caches are found.
    
    Args:
        results_dir: Directory containing results
        save_path: Where to save the figure
        strict_mode: If True, fail when no real data exists instead of using illustrative fallback
    """
    import json

    _ensure_figdir(save_path)

    data_dir = results_dir / "data"
    cache_dir = results_dir / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Helper: column resolution for Nm
    def _nm_col(df: pd.DataFrame) -> Optional[str]:
        for c in ("pipeline_Nm_per_symbol", "pipeline.Nm_per_symbol"):
            if c in df.columns:
                return c
        return None

    # Helper: canonical value_key (must match analysis/run_final_analysis.py)
    def _value_key(v: object) -> str:
        # ENHANCEMENT: Use canonical formatter from run_final_analysis for consistency
        return canonical_value_key(v)

    # Helper: choose target Nm (closest SER to 1%)
    def _choose_nm_for_mode(mode: str, seg: Optional[str] = None) -> Tuple[Optional[float], Optional[bool]]:
        """
        Choose target Nm (closest SER to 1%) and return (nm_value, ctrl_state).
        Returns the CTRL state from the selected row for consistent cache lookup.
        """
        csv_path = data_dir / f"ser_vs_nm_{mode.lower()}.csv"
        if not csv_path.exists():
            return None, None
    
        try:
            df = pd.read_csv(csv_path)
            nmcol = _nm_col(df)
            if nmcol is None or 'ser' not in df.columns:
                return None, None
        
            # Stage 14: Add CTRL filtering after loading CSV
            if 'use_ctrl' in df.columns:
                # Prefer CTRL=True rows when available
                ctrl_true_mask = df['use_ctrl'] == True
                ctrl_false_mask = df['use_ctrl'] == False
                
                if ctrl_true_mask.any():
                    df = df[ctrl_true_mask].copy()
                    target_ctrl = True
                elif ctrl_false_mask.any():
                    df = df[ctrl_false_mask].copy()
                    target_ctrl = False
                else:
                    # No clear CTRL state, use all data
                    target_ctrl = None
            else:
                # No CTRL column, assume legacy CTRL=True
                target_ctrl = True
        
            # Find closest to 1% SER
            target_ser = 0.01
            df['ser_diff'] = abs(df['ser'] - target_ser)
            idx = df['ser_diff'].idxmin()
        
            chosen_nm = df.loc[idx, nmcol]
        
            # Extract CTRL state from the chosen row (override target_ctrl if available)
            chosen_ctrl = target_ctrl
            if 'use_ctrl' in df.columns:
                chosen_ctrl = bool(df.loc[idx, 'use_ctrl'])
            
            return _f(chosen_nm), chosen_ctrl
        
        except Exception:
            return None, None

    # Helper: find CTRL segment actually present
    def _ctrl_seg(mode: str) -> Optional[str]:
        base = cache_dir / mode.lower()
        for seg in ["ser_vs_nm_wctrl", "ser_vs_nm_noctrl", "ser_vs_nm_ctrl_unspecified"]:
            if (base / seg).exists():
                return seg
        return None

    # Helper: collect tx/rx across seeds for a given Nm
    def _collect_txrx(mode: str, nm: float, use_ctrl: Optional[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect tx/rx across seeds for a given Nm, using specified CTRL state.
        ENHANCEMENT: Includes fallback for Nm formatting mismatches.
        """
        # Determine correct segment based on CTRL state
        if use_ctrl is True:
            seg = "ser_vs_nm_wctrl"
        elif use_ctrl is False:
            seg = "ser_vs_nm_noctrl"
        else:
            seg = "ser_vs_nm_ctrl_unspecified"  # fallback
    
        vk = _value_key(nm)
        cache_base = cache_dir / mode.lower() / seg / vk
    
        # ENHANCEMENT: Fallback logic for Nm formatting mismatches
        if not cache_base.exists():
            # Look for nearest available Nm within ±10% tolerance
            parent_dir = cache_dir / mode.lower() / seg
            if parent_dir.exists():
                tolerance = 0.1
                found_match = None
                for candidate_dir in parent_dir.iterdir():
                    if candidate_dir.is_dir():
                        try:
                            candidate_nm = float(candidate_dir.name)
                            rel_diff = abs(candidate_nm - nm) / nm
                            if rel_diff <= tolerance:
                                found_match = candidate_dir
                                print(f"⚠️  Cache fallback: Using Nm={candidate_nm:.0f} for requested Nm={nm:.0f} "
                                      f"(±{rel_diff:.1%} difference)")
                                break
                        except (ValueError, ZeroDivisionError):
                            continue
                
                if found_match:
                    cache_base = found_match
                else:
                    return np.array([]), np.array([])
            else:
                return np.array([]), np.array([])
    
        tx_all, rx_all = [], []
        for seed_file in cache_base.glob("seed_*.json"):
            try:
                with open(seed_file, 'r') as f:
                    data = json.load(f)
                if 'symbols_tx' in data and 'symbols_rx' in data:
                    tx_all.extend(data['symbols_tx'])
                    rx_all.extend(data['symbols_rx'])
            except Exception:
                continue
    
        return np.array(tx_all, dtype=int), np.array(rx_all, dtype=int)

    # Helper: build normalized confusion & accuracy
    def _confusion(tx: np.ndarray, rx: np.ndarray, M: Optional[int] = None) -> Tuple[np.ndarray, float]:
        if tx.size == 0 or rx.size == 0 or tx.size != rx.size:
            return np.zeros((2, 2), dtype=float), 0.0
        if M is None:
            M = int(max(tx.max(initial=0), rx.max(initial=0)) + 1)
        cm = np.zeros((M, M), dtype=int)
        for t, r in zip(tx, rx):
            if 0 <= t < M and 0 <= r < M:
                cm[t, r] += 1
        total = cm.sum()
        acc = float(np.trace(cm) / total) if total > 0 else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cmn)
        return cmn, acc

    modes = ["MoSK", "CSK", "Hybrid"]
    matrices: Dict[str, np.ndarray] = {}
    accuracies: Dict[str, float] = {}
    real_count = 0
    
    # NEW: Track CTRL state used for each mode
    ctrl_states: Dict[str, bool] = {}

    for mode in modes:
        # NEW: Get both Nm and CTRL state from CSV (Fix 2)
        chosen_nm, chosen_ctrl = _choose_nm_for_mode(mode)
    
        if chosen_nm is not None:
            tx, rx = _collect_txrx(mode, chosen_nm, chosen_ctrl)
            if len(tx) > 0 and len(rx) > 0:
                matrices[mode], accuracies[mode] = _confusion(tx, rx)
                real_count += 1
                ctrl_states[mode] = chosen_ctrl is True  # for meta tracking
            
                # SER Verification (enhanced)
                csv_ser = _get_csv_ser_at_nm(mode, chosen_nm, data_dir, tolerance=0.15)
                if csv_ser is not None:
                    # Calculate SER from confusion matrix using raw TX/RX data
                    cache_ser = float(np.mean(tx != rx))  # Direct calculation, handles imbalanced classes
    
                    # Compare with tolerance
                    ser_diff = abs(cache_ser - csv_ser)
                    ser_rel_diff = ser_diff / csv_ser if csv_ser > 0 else float('inf')
                
                    if ser_rel_diff <= 0.2:  # 20% tolerance
                        verification_status = "✓ PASS"
                    else:
                        verification_status = "✗ FAIL"
                
                    ctrl_label = 'CTRL' if chosen_ctrl else 'no-CTRL'
                    print(f"✓ {mode}: loaded {len(tx)} symbols from {ctrl_label} data at Nm≈{chosen_nm:.0f} (Acc={accuracies[mode]:.2%})")
                    print(f"   SER verification: Cache={cache_ser:.4f}, CSV={csv_ser:.4f}, Diff={ser_rel_diff:.1%} {verification_status}")
                else:
                    ctrl_label = 'CTRL' if chosen_ctrl else 'no-CTRL'
                    print(f"✓ {mode}: loaded {len(tx)} symbols from {ctrl_label} data at Nm≈{chosen_nm:.0f} (Acc={accuracies[mode]:.2%})")
                    print(f"   SER verification: CSV data not available")
            else:
                print(f"✗ {mode}: no cached symbols found for Nm={chosen_nm}")
        else:
            print(f"✗ {mode}: no valid CSV data found")

    # Fill missing with illustrative (or fail in strict mode)
    if len(matrices) < len(modes):
        if strict_mode:
            missing = [m for m in modes if m not in matrices]
            raise RuntimeError(
                f"Missing confusion matrix data for: {', '.join(missing)}. "
                f"Run SER vs Nm sweeps first with sufficient seeds (≥20) to persist "
                f"TX/RX caches: python analysis/run_final_analysis.py --mode ALL --num-seeds 20"
            )
        
        # Continue with illustrative fallback as before...
        if "MoSK" not in matrices:
            matrices["MoSK"] = np.array([[0.95, 0.05], [0.03, 0.97]], dtype=float)
            accuracies["MoSK"] = float(np.trace(matrices["MoSK"]) / matrices["MoSK"].sum())
            ctrl_states["MoSK"] = True  # Default for illustrative
        if "CSK" not in matrices:
            matrices["CSK"] = np.array([[0.90, 0.08, 0.02, 0.00],
                                        [0.07, 0.85, 0.07, 0.01],
                                        [0.02, 0.06, 0.87, 0.05],
                                        [0.00, 0.01, 0.04, 0.95]], dtype=float)
            accuracies["CSK"] = float(np.trace(matrices["CSK"]) / matrices["CSK"].sum())
            ctrl_states["CSK"] = True  # Default for illustrative
        if "Hybrid" not in matrices:
            matrices["Hybrid"] = np.array([[0.92, 0.05, 0.02, 0.01],
                                           [0.04, 0.91, 0.01, 0.04],
                                           [0.03, 0.01, 0.93, 0.03],
                                           [0.01, 0.03, 0.04, 0.92]], dtype=float)
            accuracies["Hybrid"] = float(np.trace(matrices["Hybrid"]) / matrices["Hybrid"].sum())
            ctrl_states["Hybrid"] = True  # Default for illustrative

    # NEW: Determine overall CTRL state (majority vote or most restrictive)
    if ctrl_states:
        # Use the most common CTRL state, with preference for actual data over illustrative
        real_ctrl_states = [ctrl_states[mode] for mode in modes if mode in matrices and real_count > 0]
        if real_ctrl_states:
            # Use CTRL state from real data
            overall_ctrl = all(real_ctrl_states)  # True only if all real data used CTRL
        else:
            # All illustrative - use default
            overall_ctrl = True
    else:
        overall_ctrl = True  # Default fallback

    meta = {
        "source": "data_driven" if real_count == len(modes) else ("mixed" if real_count > 0 else "illustrative"),
        "use_ctrl": overall_ctrl,  # NEW: Derived from actual usage
        "note": "confusion matrices assembled from cached TX/RX (per-seed)",
    }
    _save_confusions_npz(data_dir / "confusion_matrices.npz", matrices, meta=cast(Dict[str, object], meta))

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["MoSK", "CSK-4", "Hybrid"]

    for idx, title in enumerate(titles):
        ax = axes[idx]
        cm = matrices[title if title != "CSK-4" else "CSK"]
        im = ax.imshow(np.asarray(cm, dtype=float), cmap='Blues', vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', rotation=270, labelpad=15)
        n_classes = int(cm.shape[0])
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        labels = ['DA', 'SERO'] if n_classes == 2 else [f'S{i}' for i in range(n_classes)]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(n_classes):
            for j in range(n_classes):
                val = float(cm[i, j])
                ax.text(j, i, f'{val:.2f}',
                        ha="center", va="center",
                        color="white" if val > 0.5 else "black")
        acc = accuracies.get(title if title != "CSK-4" else "CSK", float('nan'))
        ax.set_xlabel('Detected Symbol')
        ax.set_ylabel('Transmitted Symbol')
        ax.set_title(f'({chr(97 + idx)}) {title} (Acc: {acc:.1%})')

    plt.suptitle('Figure S5: Detection Confusion Matrices', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------- S6: Energy per bit ------------------------------

def plot_figure_s6_energy_per_bit(results_dir: Path, save_path: Path) -> None:
    """
    Generate Figure S6: Energy per bit analysis including transduction efficiency.
    """
    _ensure_figdir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Load config for parameters
    with open(project_root / "config" / "default.yaml", "r") as f:
        config = yaml.safe_load(f)
    cfg = preprocess_config(config)

    # Parameters
    k_B = 1.381e-23  # Boltzmann constant (J/K)
    T = float(cfg['temperature_K'])  # Temperature (K)
    eta = 0.10  # Transduction efficiency (10%)

    # Data
    Nm_values = np.logspace(2.0, 5.0, 50, dtype=float)

    # Energy per bit (illustrative scaling)
    E_mosk = Nm_values * k_B * T * np.log(2.0) / eta
    E_csk = 0.5 * Nm_values * k_B * T * np.log(2.0) * 2.0 / eta
    E_hybrid = 0.75 * Nm_values * k_B * T * np.log(2.0) * 2.0 / eta

    # Panel (a): Energy per bit vs Nm
    ax1.loglog(_as_float1d(Nm_values), _as_float1d(E_mosk * 1e15), 'b-', label='MoSK', linewidth=2)
    ax1.loglog(_as_float1d(Nm_values), _as_float1d(E_csk * 1e15), 'g--', label='CSK-4', linewidth=2)
    ax1.loglog(_as_float1d(Nm_values), _as_float1d(E_hybrid * 1e15), 'r-.', label='Hybrid', linewidth=2)

    # Landauer limit
    E_landauer = k_B * T * np.log(2.0)
    ax1.axhline(y=_f(E_landauer * 1e15), color='k', linestyle=':', label='Landauer limit', alpha=0.5)

    ax1.set_xlabel('Number of Molecules per Symbol (Nm)')
    ax1.set_ylabel('Energy per Bit (fJ)')
    ax1.set_title(f'(a) Energy Efficiency (η = {eta:.0%})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([1e-3, 1e3])

    # Panel (b): Efficiency factor analysis
    eta_values = np.logspace(-3.0, 0.0, 50, dtype=float)
    Nm_fixed = 1e4
    E_vs_eta = Nm_fixed * k_B * T * np.log(2.0) / eta_values

    ax2.loglog(_as_float1d(eta_values * 100.0), _as_float1d(E_vs_eta * 1e15), 'k-', linewidth=2)
    ax2.axvline(x=_f(eta * 100.0), color='r', linestyle='--', alpha=0.5, label=f'Current η = {eta:.0%}')

    ax2.set_xlabel('Transduction Efficiency η (%)')
    ax2.set_ylabel('Energy per Bit (fJ)')
    ax2.set_title(f'(b) Impact of Transduction Efficiency (Nm = {Nm_fixed:.0e})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    caption = (f"Parameters: T = {T:.0f} K, k_B = {k_B:.3e} J/K, "
               f"η = {eta:.0%} (transduction efficiency)")
    fig.text(0.5, -0.02, caption, ha='center', fontsize=9, style='italic')

    plt.suptitle('Figure S6: Energy per Bit Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# --------------------------------- Main -------------------------------------

def main() -> None:
    """Generate all supplementary figures."""
    import argparse
    
    # NEW: Add argument parsing for strict mode AND data-only gating
    parser = argparse.ArgumentParser(description="Generate supplementary figures S3–S6")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Fail if no real data exists for confusion matrices")
    parser.add_argument("--allow-synthetic", action="store_false", dest="strict",
                        help="Allow synthetic fallback for confusion matrices")
    parser.add_argument("--only-data", action="store_true", default=True,
                        help="Render only data-driven panels (S5). Disables S3/S4/S6 unless explicitly overridden.")
    parser.add_argument("--include-synthetic", action="store_false", dest="only_data",
                        help="Include synthetic panels S3/S4/S6 even in data-only mode")
    args = parser.parse_args()
    
    apply_ieee_style()
    
    print("\n" + "=" * 60)
    print("Generating Supplementary Figures")
    print("=" * 60)

    # Setup paths
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.only_data:
        # S3
        print("\nGenerating Figure S3: Constellation diagrams (synthetic)...")
        plot_figure_s3_constellation(results_dir, figures_dir / "figS3_constellation.png")
        print("✓ Figure S3 saved")

        # S4
        print("\nGenerating Figure S4: SNR analysis (synthetic)...")
        plot_figure_s4_snr_types(results_dir, figures_dir / "figS4_snr_types.png")
        print("✓ Figure S4 saved")
    else:
        print("\n⏭️  Skipping S3/S4 (synthetic panels, use --include-synthetic to enable)")

    # S5 (always generate - data-driven)
    print("\nGenerating Figure S5: Confusion matrices (data-driven)...")
    plot_figure_s5_confusion_matrices(results_dir, figures_dir / "figS5_confusion.png", 
                                     strict_mode=args.strict)
    print("✓ Figure S5 saved")

    if not args.only_data:
        # S6
        print("\nGenerating Figure S6: Energy per bit (synthetic)...")
        plot_figure_s6_energy_per_bit(results_dir, figures_dir / "figS6_energy.png")
        print("✓ Figure S6 saved")
    else:
        print("\n⏭️  Skipping S6 (synthetic panel, use --include-synthetic to enable)")

    print("\n" + "=" * 60)
    mode_desc = "data-only" if args.only_data else "full (including synthetic)"
    print(f"✓ Supplementary figures generated in {mode_desc} mode!")
    print(f"Results saved in: {figures_dir}")
    if args.only_data:
        print("💡 Use --include-synthetic to generate S3/S4/S6 panels")
    print("=" * 60)

if __name__ == "__main__":
    main()
