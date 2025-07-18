# analysis/generate_comparative_plots.py
"""
Generate comparative plots across all modulation schemes.
This should be run after running run_final_analysis.py for each mode.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def load_all_results(data_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all CSV results from the data directory."""
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    modes = ['MoSK', 'CSK', 'Hybrid']
    sweep_types = ['ser_vs_nm', 'lod_vs_distance']
    
    for mode in modes:
        results[mode] = {}
        for sweep in sweep_types:
            csv_file = data_dir / f"{sweep}_{mode.lower()}.csv"
            if csv_file.exists():
                results[mode][sweep] = pd.read_csv(csv_file)
                print(f"Loaded {csv_file.name}")
            else:
                print(f"Warning: {csv_file.name} not found")
    
    return results


def plot_figure_7(results: Dict[str, Dict[str, pd.DataFrame]], save_path: Path):
    """Generate Figure 7: Comparative SER vs Nm with panels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    linestyles = {'MoSK': '-', 'CSK': '--', 'Hybrid': '-.'}
    
    # Panel (a): Full SER curves
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        if mode in results and 'ser_vs_nm' in results[mode]:
            df = results[mode]['ser_vs_nm']
            if 'pipeline_Nm_per_symbol' in df.columns and 'ser' in df.columns:
                ax1.loglog(df['pipeline_Nm_per_symbol'], df['ser'], 
                          color=colors[mode],
                          marker=markers[mode],
                          linestyle=linestyles[mode],
                          markersize=8,
                          label=mode,
                          linewidth=2,
                          markerfacecolor='none' if mode == 'CSK' else colors[mode])
    
    ax1.set_xlabel('Number of Molecules per Symbol (Nm)')
    ax1.set_ylabel('Symbol Error Rate (SER)')
    ax1.set_title('(a) SER Performance Comparison')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim(1e-4, 1)
    ax1.set_xlim(1e2, 1e5)
    ax1.axhline(y=0.01, color='k', linestyle=':', alpha=0.5)
    ax1.text(2e2, 0.015, 'Target SER = 1%', fontsize=9, alpha=0.7)
    
    # Panel (b): Hybrid subsymbol errors
    if 'Hybrid' in results and 'ser_vs_nm' in results['Hybrid']:
        df = results['Hybrid']['ser_vs_nm']
        if 'mosk_ser' in df.columns and 'csk_ser' in df.columns:
            # Total SER
            ax2.loglog(df['pipeline_Nm_per_symbol'], df['ser'], 
                      'k-', linewidth=2, label='Total SER', marker='o', markersize=6)
            # MoSK component
            ax2.loglog(df['pipeline_Nm_per_symbol'], df['mosk_ser'], 
                      'b--', linewidth=2, label='MoSK errors', marker='^', markersize=6)
            # CSK component  
            ax2.loglog(df['pipeline_Nm_per_symbol'], df['csk_ser'], 
                      'r-.', linewidth=2, label='CSK errors', marker='s', markersize=6)
    
    ax2.set_xlabel('Number of Molecules per Symbol (Nm)')
    ax2.set_ylabel('Error Rate')
    ax2.set_title('(b) Hybrid Mode Error Components')
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend()
    ax2.set_ylim(1e-4, 1)
    ax2.set_xlim(1e2, 1e5)
    
    plt.suptitle('Figure 7: Comparative SER Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_figure_10_11_combined(results: Dict[str, Dict[str, pd.DataFrame]], save_dir: Path):
    """Generate Figures 10 and 11: LoD and Data Rate comparisons."""
    # Figure 10: LoD vs Distance
    plt.figure(figsize=(10, 6))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        if mode in results and 'lod_vs_distance' in results[mode]:
            df = results[mode]['lod_vs_distance']
            df_valid = df.dropna(subset=['lod_nm'])
            if not df_valid.empty:
                plt.semilogy(df_valid['distance_um'], df_valid['lod_nm'],
                            color=colors[mode],
                            marker=markers[mode],
                            markersize=10,
                            label=mode,
                            linewidth=2.5)
    
    plt.xlabel('Distance (μm)', fontsize=12)
    plt.ylabel('Limit of Detection (molecules)', fontsize=12)
    plt.title('Figure 10: Comparative LoD vs. Distance', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_dir / "fig10_comparative_lod.png", dpi=300)
    plt.close()
    
    # Figure 11: Data Rate vs Distance
    plt.figure(figsize=(10, 6))
    
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        if mode in results and 'lod_vs_distance' in results[mode]:
            df = results[mode]['lod_vs_distance']
            df_valid = df.dropna(subset=['data_rate_bps'])
            if not df_valid.empty:
                plt.semilogy(df_valid['distance_um'], df_valid['data_rate_bps'],
                            color=colors[mode],
                            marker=markers[mode],
                            markersize=10,
                            label=mode,
                            linewidth=2.5)
    
    plt.xlabel('Distance (μm)', fontsize=12)
    plt.ylabel('Achievable Data Rate (bps)', fontsize=12)
    plt.title('Figure 11: Comparative Data Rate vs. Distance', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add annotations for key points
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        if mode in results and 'lod_vs_distance' in results[mode]:
            df = results[mode]['lod_vs_distance']
            # Annotate max data rate
            if 'data_rate_bps' in df.columns:
                max_idx = df['data_rate_bps'].idxmax()
                if not pd.isna(max_idx) and max_idx in df.index:
                    # Extract values and ensure they are Python scalars
                    max_rate_val = df.at[max_idx, 'data_rate_bps']
                    max_dist_val = df.at[max_idx, 'distance_um']
                    # Convert to Python float using numpy
                    max_rate = float(np.real(max_rate_val))
                    max_dist = float(np.real(max_dist_val))
                    plt.annotate(f'{max_rate:.3f}', 
                               xy=(max_dist, max_rate),
                               xytext=(5, 5), 
                               textcoords='offset points',
                               fontsize=9,
                               color=colors[mode])
    
    plt.tight_layout()
    plt.savefig(save_dir / "fig11_comparative_data_rate.png", dpi=300)
    plt.close()


def generate_summary_table(results: Dict[str, Dict[str, pd.DataFrame]], save_path: Path):
    """Generate a summary table of key performance metrics."""
    summary = []
    
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        row = {'Mode': mode}
        
        # Get minimum SER achieved
        if mode in results and 'ser_vs_nm' in results[mode]:
            df = results[mode]['ser_vs_nm']
            min_ser = df['ser'].min()
            nm_at_min = df.loc[df['ser'].idxmin(), 'pipeline_Nm_per_symbol']
            row['Min SER'] = f"{min_ser:.2e}"
            row['Nm @ Min SER'] = f"{nm_at_min:.0f}"
        
        # Get LoD at 100μm
        if mode in results and 'lod_vs_distance' in results[mode]:
            df = results[mode]['lod_vs_distance']
            lod_100 = df[df['distance_um'] == 100]['lod_nm'].values
            if len(lod_100) > 0 and not pd.isna(lod_100[0]):
                row['LoD @ 100μm'] = f"{lod_100[0]:.0f}"
            
            # Get max data rate
            max_rate = df['data_rate_bps'].max()
            if not pd.isna(max_rate):
                row['Max Data Rate (bps)'] = f"{max_rate:.4f}"
        
        summary.append(row)
    
    df_summary = pd.DataFrame(summary)
    
    # Save as CSV
    df_summary.to_csv(save_path, index=False)
    
    # Also create a LaTeX table
    latex_path = save_path.with_suffix('.tex')
    with open(latex_path, 'w') as f:
        f.write(df_summary.to_latex(index=False, escape=False))
    
    print(f"\nSummary table saved to {save_path}")
    print(df_summary.to_string(index=False))


def main():
    """Main function to generate all comparative plots."""
    print("\n" + "="*60)
    print("Generating Comparative Plots")
    print("="*60)
    
    # Setup paths
    results_dir = project_root / "results"
    data_dir = results_dir / "data"
    figures_dir = results_dir / "figures"
    
    # Load all results
    print("\nLoading results...")
    results = load_all_results(data_dir)
    
    # Check if we have results for all modes
    modes_found = [mode for mode in results if any(results[mode])]
    print(f"\nModes with data: {', '.join(modes_found)}")
    
    if len(modes_found) < 2:
        print("\nWarning: Need at least 2 modes to generate comparative plots.")
        print("Please run run_final_analysis.py for each mode first.")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Figure 7: SER comparison
    plot_figure_7(results, figures_dir / "fig7_comparative_ser.png")
    print("✓ Figure 7 saved")
    
    # Figures 10 & 11: LoD and Data Rate
    plot_figure_10_11_combined(results, figures_dir)
    print("✓ Figures 10 & 11 saved")
    
    # Generate summary table
    generate_summary_table(results, data_dir / "performance_summary.csv")
    
    print("\n" + "="*60)
    print("All comparative plots generated successfully!")
    print(f"Results saved in: {figures_dir}")
    print("="*60)


if __name__ == "__main__":
    main()