"""
Generate comparative plots across all modulation schemes.

Stage 9 additions:
- Applies IEEE style (serif fonts, compact labels; 300+ dpi).
- Adds 95% binomial CIs (Wilson) when symbols_evaluated is available.
- Produces an additional figure comparing CSK SER across NT pairs
  saved as results/figures/fig_nt_pairs_ser.png.

README:
- Expects canonical CSVs in results/data created by run_final_analysis.py.
- If CI columns are absent, error bars are omitted gracefully.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional, List, Tuple

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style  # Stage 9

# ----------------- helpers -----------------

def _nm_col(df: pd.DataFrame) -> Optional[str]:
    if 'pipeline_Nm_per_symbol' in df.columns:
        return 'pipeline_Nm_per_symbol'
    if 'pipeline.Nm_per_symbol' in df.columns:
        return 'pipeline.Nm_per_symbol'
    return None

def _try_load(base: Path) -> Optional[pd.DataFrame]:
    # canonical
    if base.exists():
        print(f"Loaded {base.name}")
        return pd.read_csv(base)
    # fallbacks for older runs (e.g., *_uniform.csv)
    candidates: List[Path] = []
    if base.suffix == ".csv":
        candidates.append(base.with_name(base.stem + "_uniform.csv"))
        candidates.append(base.with_name(base.stem + "_zero.csv"))
    for cand in candidates:
        if cand.exists():
            print(f"Loaded {cand.name} (fallback)")
            return pd.read_csv(cand)
    print(f"Warning: {base.name} not found")
    return None

def load_all_results(data_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    modes = ['MoSK', 'CSK', 'Hybrid']
    sweep_types = ['ser_vs_nm', 'lod_vs_distance']
    for mode in modes:
        results[mode] = {}
        for sweep in sweep_types:
            df = _try_load(data_dir / f"{sweep}_{mode.lower()}.csv")
            if df is not None:
                results[mode][sweep] = df
    return results

# ---- CI helpers (Wilson interval) ----
def _wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    p = np.divide(k, np.maximum(n, 1), out=np.zeros_like(k, dtype=float), where=n>0)
    den = 1 + z**2 / n
    center = (p + z**2/(2*n)) / den
    half = z*np.sqrt((p*(1-p)/n) + (z**2)/(4*n**2)) / den
    low = np.clip(center - half, 1e-8, 1-1e-8)
    high = np.clip(center + half, 1e-8, 1-1e-8)
    return low, high

def _get_ci_for_df(df: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if 'symbols_evaluated' in df.columns and 'ser' in df.columns:
        n = pd.to_numeric(df['symbols_evaluated'], errors='coerce').to_numpy()
        k = (pd.to_numeric(df['ser'], errors='coerce') * n).to_numpy()
        return _wilson_ci(k, np.maximum(n, 1))
    return None

# ----------------- plotting -----------------

def plot_figure_7(results: Dict[str, Dict[str, pd.DataFrame]], save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {'MoSK': '#0072B2', 'CSK': '#009E73', 'Hybrid': '#D55E00'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    linestyles = {'MoSK': '-', 'CSK': '--', 'Hybrid': '-.'}

    # Panel (a): SER curves with CIs when available - CTRL state separation
    ctrl_styles = {True: '-', False: '--'}      # with-CTRL: solid, no-CTRL: dashed
    ctrl_labels = {True: '', False: ' (no CTRL)'}  # suffix for no-CTRL
    
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        if mode in results and 'ser_vs_nm' in results[mode]:
            df = results[mode]['ser_vs_nm'].copy()
            nmcol = _nm_col(df)
            if nmcol and 'ser' in df.columns:
                # Check if CTRL state separation is needed
                if 'use_ctrl' in df.columns and df['use_ctrl'].nunique() > 1:
                    # Separate plotting by CTRL state
                    for ctrl_state, grp in df.groupby('use_ctrl'):
                        if grp.empty:
                            continue
                        grp_sorted = grp.sort_values(nmcol)
                        
                        # Enhanced legend with combiner info for CSK
                        label = mode + ctrl_labels[bool(ctrl_state)]
                        if mode == 'CSK' and 'combiner' in grp.columns:
                            combiners = grp['combiner'].dropna().unique()
                            if len(combiners) == 1:
                                label += f" ({combiners[0]})"
                            elif len(combiners) > 1:
                                label += f" (mixed)"
                        
                        ax1.loglog(grp_sorted[nmcol], grp_sorted['ser'],
                                   color=colors[mode], marker=markers[mode],
                                   linestyle=ctrl_styles[bool(ctrl_state)], 
                                   markersize=6, 
                                   label=label, 
                                   linewidth=2,
                                   markerfacecolor='none' if mode == 'CSK' else colors[mode],
                                   alpha=0.9 if ctrl_state else 0.6)  # Fade no-CTRL slightly
                        
                        ci = _get_ci_for_df(grp_sorted)
                        if ci is not None:
                            low, high = ci
                            yerr = np.vstack([grp_sorted['ser'].to_numpy() - low, high - grp_sorted['ser'].to_numpy()])
                            ax1.errorbar(grp_sorted[nmcol], grp_sorted['ser'], yerr=yerr,
                                         fmt='none', ecolor=colors[mode], alpha=0.25 if ctrl_state else 0.15, capsize=2)
                else:
                    # Single CTRL state or no CTRL column - original behavior
                    df_sorted = df.sort_values(nmcol)
                    
                    # Enhanced legend with combiner info for CSK
                    label = mode
                    if mode == 'CSK' and 'combiner' in df.columns:
                        combiners = df['combiner'].dropna().unique()
                        if len(combiners) == 1:
                            label += f" ({combiners[0]})"
                        elif len(combiners) > 1:
                            label += f" (mixed)"
                    
                    ax1.loglog(df_sorted[nmcol], df_sorted['ser'],
                               color=colors[mode], marker=markers[mode],
                               linestyle=linestyles[mode], markersize=6, label=label, linewidth=2,
                               markerfacecolor='none' if mode == 'CSK' else colors[mode])
                    ci = _get_ci_for_df(df_sorted)
                    if ci is not None:
                        low, high = ci
                        yerr = np.vstack([df_sorted['ser'].to_numpy() - low, high - df_sorted['ser'].to_numpy()])
                        ax1.errorbar(df_sorted[nmcol], df_sorted['ser'], yerr=yerr,
                                     fmt='none', ecolor=colors[mode], alpha=0.35, capsize=2)

    ax1.set_xlabel('Number of Molecules per Symbol (Nm)')
    ax1.set_ylabel('Symbol Error Rate (SER)')
    ax1.set_title('(a) SER Performance Comparison')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend(loc='upper right', ncol=3)
    ax1.set_ylim(1e-4, 1)
    ax1.set_xlim(1e2, 1e5)
    ax1.axhline(y=0.01, color='k', linestyle=':', alpha=0.6)
    ax1.text(2.2e2, 0.013, 'Target SER = 1%', fontsize=8, alpha=0.8)

    # Panel (b): Hybrid error components - ENHANCED with CTRL state separation
    if 'Hybrid' in results and 'ser_vs_nm' in results['Hybrid']:
        dfh = results['Hybrid']['ser_vs_nm']
        nmcol = _nm_col(dfh)
        if nmcol and all(c in dfh.columns for c in ['mosk_ser', 'csk_ser', 'ser']):
            # Check if CTRL state separation is needed
            if 'use_ctrl' in dfh.columns and dfh['use_ctrl'].nunique() > 1:
                # Separate plotting by CTRL state
                for ctrl_state, grp in dfh.groupby('use_ctrl'):
                    if grp.empty:
                        continue
                    g = grp.sort_values(nmcol)
                    ls = ctrl_styles[bool(ctrl_state)]
                    suf = ctrl_labels[bool(ctrl_state)]
                    alpha = 0.9 if ctrl_state else 0.6
                    
                    ax2.loglog(g[nmcol], g['ser'], 'k', linestyle=ls, linewidth=2, 
                            label=f'Total SER{suf}', alpha=alpha, marker='o', markersize=5)
                    ax2.loglog(g[nmcol], g['mosk_ser'], color=colors['MoSK'], 
                            linestyle='--' if ls=='-' else ':', linewidth=2, 
                            label=f'MoSK errors{suf}', marker='^', markersize=5, alpha=alpha)
                    ax2.loglog(g[nmcol], g['csk_ser'], color=colors['CSK'], 
                            linestyle='-.', linewidth=2, 
                            label=f'CSK errors{suf}', marker='s', markersize=5, alpha=alpha)
            else:
                # Single CTRL state or no CTRL column - original behavior
                dfh = dfh.sort_values(nmcol)
                ax2.loglog(dfh[nmcol], dfh['ser'], 'k-', linewidth=2, label='Total SER', marker='o', markersize=5)
                ax2.loglog(dfh[nmcol], dfh['mosk_ser'], color=colors['MoSK'], linestyle='--',
                        linewidth=2, label='MoSK errors', marker='^', markersize=5)
                ax2.loglog(dfh[nmcol], dfh['csk_ser'], color=colors['CSK'], linestyle='-.',
                        linewidth=2, label='CSK errors', marker='s', markersize=5)

    ax2.set_xlabel('Number of Molecules per Symbol (Nm)')
    ax2.set_ylabel('Error Rate')
    ax2.set_title('(b) Hybrid Mode Error Components')
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend()
    ax2.set_ylim(1e-4, 1)
    ax2.set_xlim(1e2, 1e5)

    fig.suptitle('Figure 7: Comparative SER Analysis', fontsize=10, y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_figure_10_11_combined(results: Dict[str, Dict[str, pd.DataFrame]], save_dir: Path):
    # Figure 10 - ENHANCED: Separate CTRL states
    plt.figure(figsize=(10, 5.7))
    colors = {'MoSK': '#0072B2', 'CSK': '#009E73', 'Hybrid': '#D55E00'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    # CTRL state styling
    ctrl_styles = {True: '-', False: '--'}      # with-CTRL: solid, no-CTRL: dashed
    ctrl_labels = {True: '', False: ' (no CTRL)'}  # suffix for no-CTRL
    
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        if mode in results and 'lod_vs_distance' in results[mode]:
            df = results[mode]['lod_vs_distance']
            df_clean = df.dropna(subset=['lod_nm'])
            
            if not df_clean.empty:
                # Check if CTRL state separation is needed
                if 'use_ctrl' in df_clean.columns and df_clean['use_ctrl'].nunique() > 1:
                    # Separate plotting by CTRL state
                    for ctrl_state, grp in df_clean.groupby('use_ctrl'):
                        if grp.empty:
                            continue
                        grp_sorted = grp.sort_values('distance_um')
                        
                        plt.semilogy(grp_sorted['distance_um'], grp_sorted['lod_nm'],
                                     color=colors[mode], 
                                     marker=markers[mode],
                                     markersize=8,
                                     label=f"{mode}{ctrl_labels[bool(ctrl_state)]}",
                                     linewidth=2.2,
                                     linestyle=ctrl_styles[bool(ctrl_state)],
                                     alpha=0.9 if ctrl_state else 0.6)  # Fade no-CTRL slightly
                else:
                    # Single CTRL state or no CTRL column - original behavior
                    df_sorted = df_clean.sort_values('distance_um')
                    plt.semilogy(df_sorted['distance_um'], df_sorted['lod_nm'],
                                 color=colors[mode], marker=markers[mode],
                                 markersize=8, label=mode, linewidth=2.2)
    plt.xlabel('Distance (μm)')
    plt.ylabel('Limit of Detection (molecules)')
    plt.title('Figure 10: Comparative LoD vs. Distance')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "fig10_comparative_lod.png", dpi=300)
    plt.close()

    # Figure 11 - ENHANCED with CTRL state separation AND confidence intervals
    plt.figure(figsize=(10, 5.7))
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        if mode in results and 'lod_vs_distance' in results[mode]:
            df = results[mode]['lod_vs_distance']
            df_clean = df.dropna(subset=['data_rate_bps'])
            
            if not df_clean.empty:
                # Check if CTRL state separation is needed
                if 'use_ctrl' in df_clean.columns and df_clean['use_ctrl'].nunique() > 1:
                    # Separate plotting by CTRL state
                    for ctrl_state, grp in df_clean.groupby('use_ctrl'):
                        if grp.empty:
                            continue
                        grp_sorted = grp.sort_values('distance_um')
                        
                        # Main line plot
                        plt.semilogy(grp_sorted['distance_um'], grp_sorted['data_rate_bps'],
                                    color=colors[mode], 
                                    marker=markers[mode],
                                    markersize=8,
                                    label=f"{mode}{ctrl_labels[bool(ctrl_state)]}",
                                    linewidth=2.2,
                                    linestyle=ctrl_styles[bool(ctrl_state)],
                                    alpha=0.9 if ctrl_state else 0.6)
                        
                        # Confidence intervals if available
                        if all(col in grp_sorted.columns for col in ['data_rate_ci_low', 'data_rate_ci_high']):
                            ci_valid = grp_sorted.dropna(subset=['data_rate_ci_low', 'data_rate_ci_high'])
                            if not ci_valid.empty:
                                plt.fill_between(ci_valid['distance_um'], 
                                            ci_valid['data_rate_ci_low'], 
                                            ci_valid['data_rate_ci_high'],
                                            color=colors[mode], 
                                            alpha=0.15 if ctrl_state else 0.08)
                else:
                    # Single CTRL state or no CTRL column - original behavior
                    grp_sorted = df_clean.sort_values('distance_um')
                    
                    # Main line plot
                    plt.semilogy(grp_sorted['distance_um'], grp_sorted['data_rate_bps'],
                                color=colors[mode], marker=markers[mode],
                                markersize=8, label=mode, linewidth=2.2)
                    
                    # Confidence intervals if available
                    if all(col in grp_sorted.columns for col in ['data_rate_ci_low', 'data_rate_ci_high']):
                        ci_valid = grp_sorted.dropna(subset=['data_rate_ci_low', 'data_rate_ci_high'])
                        if not ci_valid.empty:
                            plt.fill_between(ci_valid['distance_um'], 
                                        ci_valid['data_rate_ci_low'], 
                                        ci_valid['data_rate_ci_high'],
                                        color=colors[mode], alpha=0.2)
                        
    plt.xlabel('Distance (μm)')
    plt.ylabel('Achievable Data Rate (bps)')
    plt.title('Figure 11: Data Rate vs. Distance (with 95% CI)')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(ncol=3)

    # Annotate max rate - unchanged
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        if mode in results and 'lod_vs_distance' in results[mode]:
            df = results[mode]['lod_vs_distance']
            if 'data_rate_bps' in df.columns and not df['data_rate_bps'].empty:
                max_idx = df['data_rate_bps'].idxmax()
                if pd.notna(max_idx) and max_idx in df.index:
                    max_rate = float(np.real(df.at[max_idx, 'data_rate_bps']))
                    max_dist = float(np.real(df.at[max_idx, 'distance_um']))
                    plt.annotate(f'{max_rate:.3f}',
                                 xy=(max_dist, max_rate),
                                 xytext=(5, 5),
                                 textcoords='offset points',
                                 fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "fig11_comparative_data_rate.png", dpi=300)
    plt.close()

def plot_ctrl_ablation_ser(results: Dict[str, Dict[str, pd.DataFrame]], save_path: Path):
    """
    Stage 5: Overlay SER vs Nm with-CTRL vs without-CTRL for each mode, if both exist.
    """
    plt.figure(figsize=(10.5, 5.7))
    colors = {'MoSK': '#0072B2', 'CSK': '#009E73', 'Hybrid': '#D55E00'}
    styles = {True: '-', False: '--'}
    markers = {True: 'o', False: 'x'}
    labels_ctrl = {True: 'with CTRL', False: 'no CTRL'}
    any_plotted = False

    for mode in ['MoSK', 'CSK', 'Hybrid']:
        if mode not in results or 'ser_vs_nm' not in results[mode]:
            continue
        df = results[mode]['ser_vs_nm']
        nmcol = _nm_col(df)
        if nmcol is None or 'ser' not in df.columns or 'use_ctrl' not in df.columns:
            continue

        # Plot both CTRL states if present
        for ctrl_state, grp in df.groupby('use_ctrl'):
            if grp.empty:
                continue
            any_plotted = True
            grp_sorted = grp.sort_values(nmcol)
            plt.loglog(grp_sorted[nmcol], grp_sorted['ser'],
                       linestyle=styles[bool(ctrl_state)],
                       marker=markers[bool(ctrl_state)],
                       color=colors[mode],
                       linewidth=2,
                       markersize=6,
                       label=f"{mode} • {labels_ctrl[bool(ctrl_state)]}")

    plt.xlabel('Number of Molecules per Symbol (Nm)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title('CTRL Ablation: effect of differential CTRL subtraction on SER')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.axhline(y=0.01, color='k', linestyle=':', alpha=0.6)
    if any_plotted:
        plt.legend(ncol=2)
    else:
        plt.text(0.5, 0.5, "No paired with/without CTRL data found.\nRun both conditions to see overlay.",
                 transform=plt.gca().transAxes, ha='center', va='center')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_nt_pairs_ser(data_dir: Path, save_path: Path) -> None:
    """
    Stage 7: Plot CSK SER vs Nm for multiple neurotransmitter pairs.
    Looks for files named ser_vs_nm_csk_<a>_<b>.csv in results/data.
    ENHANCED: Now separates CTRL states when both are present.
    """
    files = sorted(data_dir.glob("ser_vs_nm_csk_*_*.csv"))
    if not files:
        print("No CSK NT-pair CSVs found; skipping fig_nt_pairs_ser.png")
        return

    plt.figure(figsize=(9.5, 5.2))
    for f in files:
        df = pd.read_csv(f)
        nmcol = _nm_col(df)
        if nmcol is None or 'ser' not in df.columns:
            continue
        label = f.stem.replace("ser_vs_nm_csk_", "").replace("_", "–").upper()
        
        # ✅ Add combiner info to label
        if 'combiner' in df.columns:
            combs = df['combiner'].dropna().unique()
            if len(combs) == 1:
                label += f" ({combs[0]})"
        
        # Check if CTRL state separation is needed
        if 'use_ctrl' in df.columns and df['use_ctrl'].nunique() > 1:
            # Separate plotting by CTRL state
            for ctrl_state, grp in df.groupby('use_ctrl'):
                if grp.empty:
                    continue
                grp = grp.sort_values(nmcol)
                ls = '-' if ctrl_state else '--'
                suf = '' if ctrl_state else ' (no CTRL)'
                alpha = 0.9 if ctrl_state else 0.6
                
                plt.loglog(grp[nmcol], grp['ser'], label=f"{label}{suf}", 
                          linewidth=2, marker='o', markersize=5, linestyle=ls, alpha=alpha)
                
                # Add confidence intervals if available
                ci = _get_ci_for_df(grp)
                if ci is not None:
                    low, high = ci
                    yerr = np.vstack([grp['ser'].to_numpy() - low, high - grp['ser'].to_numpy()])
                    plt.errorbar(grp[nmcol], grp['ser'], yerr=yerr, fmt='none', 
                               alpha=0.25 if ctrl_state else 0.15, capsize=2)
        else:
            # Single CTRL state or no CTRL column - original behavior
            df = df.sort_values(nmcol)
            plt.loglog(df[nmcol], df['ser'], label=label, linewidth=2, marker='o', markersize=5)
            ci = _get_ci_for_df(df)
            if ci is not None:
                low, high = ci
                yerr = np.vstack([df['ser'].to_numpy() - low, high - df['ser'].to_numpy()])
                plt.errorbar(df[nmcol], df['ser'], yerr=yerr, fmt='none', alpha=0.35, capsize=2)

    plt.xlabel('Number of Molecules per Symbol (Nm)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title('CSK-4 Versatility: SER across Neurotransmitter Pairs')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.axhline(0.01, color='k', ls=':', alpha=0.6)
    plt.legend(ncol=2)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_summary_table(results: Dict[str, Dict[str, pd.DataFrame]], save_path: Path):
    summary = []
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        row = {'Mode': mode}
        if mode in results and 'ser_vs_nm' in results[mode]:
            df = results[mode]['ser_vs_nm']
            if not df.empty and 'ser' in df.columns:
                min_ser = df['ser'].min()
                nmcol = _nm_col(df)
                if nmcol:
                    nm_at_min = df.loc[df['ser'].idxmin(), nmcol]
                    row['Min SER'] = f"{min_ser:.2e}"
                    row['Nm @ Min SER'] = f"{nm_at_min:.0f}"
        if mode in results and 'lod_vs_distance' in results[mode]:
            df = results[mode]['lod_vs_distance']
            lod_100 = df[df['distance_um'] == 100]['lod_nm'].values
            if len(lod_100) > 0 and pd.notna(lod_100[0]):
                row['LoD @ 100μm'] = f"{lod_100[0]:.0f}"
            if 'data_rate_bps' in df.columns:
                max_rate = df['data_rate_bps'].max()
                if pd.notna(max_rate):
                    row['Max Data Rate (bps)'] = f"{max_rate:.4f}"
        summary.append(row)
    df_summary = pd.DataFrame(summary)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(save_path, index=False)
    latex_path = save_path.with_suffix('.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:  # Add encoding='utf-8'
        f.write(df_summary.to_latex(index=False, escape=False))
    print(f"\nSummary table saved to {save_path}")
    print(df_summary.to_string(index=False))

def main():
    print("\n" + "="*60)
    print("Generating Comparative Plots")
    print("="*60)
    apply_ieee_style()  # Stage 9: consistent styling

    results_dir = project_root / "results"
    data_dir = results_dir / "data"
    figures_dir = results_dir / "figures"
    print("\nLoading results...")
    results = load_all_results(data_dir)
    modes_found = [mode for mode in results if any(results[mode])]
    print(f"\nModes with data: {', '.join(modes_found) if modes_found else '(none)'}")
    if len(modes_found) < 2:
        print("\nWarning: Need at least 2 modes to generate comparative plots.")
    print("\nGenerating plots...")
    plot_figure_7(results, figures_dir / "fig7_comparative_ser.png")
    print("✓ Figure 7 saved")
    plot_figure_10_11_combined(results, figures_dir)
    print("✓ Figures 10 & 11 saved")
    # Stage 5: CTRL ablation overlay
    plot_ctrl_ablation_ser(results, figures_dir / "fig_ctrl_ablation_ser.png")
    print("✓ CTRL ablation figure saved")
    # Stage 7: NT-pair comparison (CSK)
    plot_nt_pairs_ser(data_dir, figures_dir / "fig_nt_pairs_ser.png")
    generate_summary_table(results, data_dir / "performance_summary.csv")
    print("\n" + "="*60)
    print("All comparative plots generated successfully!")
    print(f"Results saved in: {figures_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
