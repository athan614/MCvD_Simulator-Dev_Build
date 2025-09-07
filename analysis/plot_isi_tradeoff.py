# analysis/plot_isi_tradeoff.py
"""
Stage 6 — ISI trade-off visualization (updated Stage 9 styling).

Loads results/data/isi_tradeoff_{mode}.csv for modes in {MoSK, CSK, Hybrid}
and produces results/figures/fig_isi_tradeoff.png.

CSV expected columns:
- guard_factor (or pipeline.guard_factor)
- ser
- symbol_period_s
- use_ctrl (optional)
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style

def _gf_col(df: pd.DataFrame) -> Optional[str]:
    if 'guard_factor' in df.columns:
        return 'guard_factor'
    if 'pipeline.guard_factor' in df.columns:
        return 'pipeline.guard_factor'
    return None

def _try_load(fp: Path) -> Optional[pd.DataFrame]:
    if fp.exists():
        return pd.read_csv(fp)
    return None

def main() -> None:
    apply_ieee_style()
    results_dir = project_root / "results"
    data_dir = results_dir / "data"
    figs_dir = results_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    dfs: Dict[str, pd.DataFrame] = {}
    for m in ['MoSK', 'CSK', 'Hybrid']:
        df = _try_load(data_dir / f"isi_tradeoff_{m.lower()}.csv")
        if df is not None and not df.empty:
            dfs[m] = df

    plt.figure(figsize=(10.5, 5.8))
    ax = plt.gca()
    plotted = False

    for m, df in dfs.items():
        gfcol = _gf_col(df)
        if gfcol is None or 'ser' not in df.columns:
            continue

        if 'use_ctrl' in df.columns and df['use_ctrl'].nunique() > 1:
            for ctrl_state, grp in df.groupby('use_ctrl'):
                g = grp.groupby(gfcol, as_index=False).agg({
                    'ser': 'median',
                    'symbol_period_s': lambda x: np.median(pd.to_numeric(x, errors='coerce'))
                }).sort_values(gfcol)
                ls = '-' if bool(ctrl_state) else '--'
                lbl = f"{m} • {'CTRL' if bool(ctrl_state) else 'no CTRL'}"
                # Clip SER to prevent log(0) issues
                ser_clipped = np.maximum(g['ser'], 1e-6)
                ax.semilogy(g[gfcol], ser_clipped, marker='o', linewidth=2, linestyle=ls, label=lbl)
                plotted = True
        else:
            g = df.groupby(gfcol, as_index=False).agg({
                'ser': 'median',
                'symbol_period_s': lambda x: np.median(pd.to_numeric(x, errors='coerce'))
            }).sort_values(gfcol)
            # Clip SER to prevent log(0) issues  
            ser_clipped = np.maximum(g['ser'], 1e-6)
            ax.semilogy(g[gfcol], ser_clipped, marker='o', linewidth=2, label=m)
            plotted = True

    ax.set_xlabel('Guard Factor (fraction of Ts)')
    ax.set_ylabel('SER with ISI enabled')
    ax.set_title('ISI Robustness: SER vs Guard Factor at Representative Distance')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.axhline(0.01, color='k', ls=':', alpha=0.6)
    ax.legend(ncol=3)

    # Secondary axis: median Ts, if available (rough guide)
    try:
        guard_union = None
        Ts_median = None
        for m, df in dfs.items():
            gfcol = _gf_col(df)
            if gfcol is None or 'symbol_period_s' not in df.columns:
                continue
            # Use the full dataset for Ts median (regardless of CTRL state)
            g = df.groupby(gfcol, as_index=False).agg({'symbol_period_s': 'median'}).sort_values(gfcol)
            if guard_union is None:
                guard_union = g[gfcol].to_numpy()
                Ts_median = g['symbol_period_s'].to_numpy()
            else:
                if len(g) >= len(guard_union):
                    guard_union = g[gfcol].to_numpy()
                    Ts_median = g['symbol_period_s'].to_numpy()
        if guard_union is not None and Ts_median is not None:
            ax2 = ax.twinx()
            ax2.plot(guard_union, Ts_median, 'k-.', linewidth=1.4, marker='.', ms=6, alpha=0.7)
            ax2.set_ylabel('Median Ts (s)')
    except Exception:
        pass

    plt.tight_layout()
    out_path = figs_dir / "fig_isi_tradeoff.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ ISI trade-off figure saved to {out_path}")

if __name__ == "__main__":
    main()
