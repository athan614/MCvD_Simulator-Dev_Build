# analysis/run_ser_sweeps.py
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from copy import deepcopy
import pandas as pd

# --- NEW IMPORTS ---
from joblib import Parallel, delayed    # type: ignore
from scipy.special import erfc  # type: ignore
from statsmodels.stats.proportion import proportion_confint # type: ignore

# Define default BER functions
def default_ber_awgn(snr_db):
    """AWGN BER for binary signaling: Q(sqrt(2*SNR_linear))"""
    snr_linear = 10**(snr_db/10)
    return 0.5 * erfc(np.sqrt(snr_linear))

# Try to import better versions if available
USE_SEP_CSK = False
USE_BER_AWGN = False

try:
    from src.mc_detection import sep_csk_binary  # type: ignore
    USE_SEP_CSK = True
except ImportError:
    pass

if not USE_SEP_CSK:
    try:
        from src.mc_detection import ber_awgn  # type: ignore
        USE_BER_AWGN = True
    except ImportError:
        pass

# Add project root to path to allow importing from 'src'
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline import run_sequence

def preprocess_config(config):
    """
    Prepares the flat YAML config into the nested structure our code expects.
    """
    cfg = deepcopy(config)
    def to_float(key):
        if key in cfg and cfg[key] is not None and not isinstance(cfg[key], (int, float)):
            cfg[key] = float(cfg[key])
    numeric_keys = [
        'temperature_K', 'alpha', 'clearance_rate', 'T_release_ms', 
        'gm_S', 'C_tot_F', 'R_ch_Ohm', 'alpha_H', 'N_c', 'K_d_Hz', 'dt_s'
    ]
    for key in numeric_keys:
        to_float(key)
    if 'pipeline' in cfg:
        if 'distance_um' in cfg['pipeline']: cfg['pipeline']['distance_um'] = float(cfg['pipeline']['distance_um'])
        if 'Nm_per_symbol' in cfg['pipeline']: cfg['pipeline']['Nm_per_symbol'] = float(cfg['pipeline']['Nm_per_symbol'])
    if 'Nm_range' in cfg and cfg['Nm_range'] is not None: cfg['Nm_range'] = [float(x) for x in cfg['Nm_range']]
    if 'distances_um' in cfg and cfg['distances_um'] is not None: cfg['distances_um'] = [float(x) for x in cfg['distances_um']]
    cfg['oect'] = {'gm_S': cfg.get('gm_S'), 'C_tot_F': cfg.get('C_tot_F'), 'R_ch_Ohm': cfg.get('R_ch_Ohm')}
    cfg['noise'] = {'alpha_H': cfg.get('alpha_H'), 'N_c': cfg.get('N_c'), 'K_d_Hz': cfg.get('K_d_Hz')}
    cfg['sim'] = {'dt_s': cfg.get('dt_s'), 'temperature_K': cfg.get('temperature_K')}
    return cfg

def calculate_snr_from_stats(stats_glu, stats_gaba):
    """Calculates empirical SNR from the decision variable distributions."""
    if not stats_glu or not stats_gaba: return 0
    mu_glu, mu_gaba = np.mean(stats_glu), np.mean(stats_gaba)
    var_glu, var_gaba = np.var(stats_glu), np.var(stats_gaba)
    if (var_glu + var_gaba) == 0: return np.inf
    return (mu_glu - mu_gaba)**2 / (var_glu + var_gaba)

def run_single_instance(config, seed):
    """A wrapper function that joblib can call for a single simulation run."""
    cfg_run = deepcopy(config)
    # Fix: Convert numpy uint32 to int for random seed
    cfg_run['pipeline']['random_seed'] = int(seed)
    
    # Run simulation and verify return format
    result = run_sequence(cfg_run)
    
    # Ensure we have the expected keys
    if not all(key in result for key in ['errors', 'stats_glu', 'stats_gaba']):
        raise KeyError(f"run_sequence returned unexpected format. Got keys: {list(result.keys())}")
    
    return result

def main():
    """Main function to run sweeps and generate plots with statistical confidence."""
    print("--- Starting Rigorous SER Sweeps (Task 05 v2) ---")
    
    with open(project_root / "config" / "default.yaml") as f:
        config_base = yaml.safe_load(f)
    cfg_template = preprocess_config(config_base)
    
    # TEST MODE: Add this block for quick testing
    test_mode = False  # Set to False for full run
    if test_mode:
        print("*** RUNNING IN TEST MODE - LIMITED DATA POINTS ***")
        num_seeds = 3  # Just 3 seeds instead of 30
        cfg_template['Nm_range'] = [1e3, 1e4]  # Just 2 points
        # FIX 1: Add intermediate distance points for smoother curve
        cfg_template['distances_um'] = [50, 100, 175]  # Test subset
        cfg_template['pipeline']['sequence_length'] = 200  # Fewer symbols for speed
    else:
        num_seeds = 30  # Full 30 seeds for production
        # FIX 1: Ensure full distance range includes intermediate points
        if 'distances_um' not in cfg_template or len(cfg_template['distances_um']) < 9:
            cfg_template['distances_um'] = [25, 50, 75, 100, 150, 175, 200, 225, 250]
    
    # CRITICAL CHANGE 2: Set sequence_length to 1000 for better error-rate resolution
    if not test_mode:  # Only set to 1000 in full mode
        cfg_template['pipeline']['sequence_length'] = 1000
    print(f"Using sequence_length = {cfg_template['pipeline']['sequence_length']} for better SER resolution")
    
    # CRITICAL CHANGE 3: Set adequate simulation window and diffusion parameters
    # FIX 4: Increase time_window_s to 40 for longer distances
    if 'time_window_s' not in cfg_template['pipeline']:
        cfg_template['pipeline']['time_window_s'] = 40  # Increased to handle r=250µm
    else:
        cfg_template['pipeline']['time_window_s'] = max(40, cfg_template['pipeline']['time_window_s'])
    print(f"Using time_window_s = {cfg_template['pipeline']['time_window_s']} to handle large distances")
    
    results_dir = project_root / "results"
    (results_dir / "figures").mkdir(exist_ok=True)
    (results_dir / "data").mkdir(exist_ok=True)
    
    # CRITICAL CHANGE 1: Multiple runs per data point (already implemented)
    # num_seeds is now set in test_mode block above
    base_seed = 2025
    ss = np.random.SeedSequence(base_seed)
    # Fix: Use spawn() to get proper child sequences
    child_sequences = ss.spawn(num_seeds)
    # Extract integer seeds from the sequences
    child_seeds = [int(seq.generate_state(1)[0]) for seq in child_sequences]

    # --- Sweep 1: SER and SNR vs. Number of Molecules (Nm) ---
    print(f"\nRunning Sweep 1: Varying Nm (with {num_seeds} seeds per point)...")
    nm_range = cfg_template['Nm_range']
    sweep_results_nm = []

    for nm in nm_range:
        print(f"  Testing Nm = {nm:.0e}...")
        cfg = deepcopy(cfg_template)
        cfg['pipeline']['Nm_per_symbol'] = nm
        
        # Use joblib to run simulations in parallel
        results_list = Parallel(n_jobs=-1)(delayed(run_single_instance)(cfg, seed) for seed in child_seeds)
        
        # Ensure results_list is a list (not a generator)
        results_list = list(results_list) if not isinstance(results_list, list) else results_list
        
        # Aggregate results - store n_err and n_sym for transparency
        total_errors = sum(r['errors'] for r in results_list if r is not None)
        total_symbols = len([r for r in results_list if r is not None]) * cfg['pipeline']['sequence_length']
        ser = total_errors / total_symbols if total_symbols > 0 else 0
        
        # Aggregate stats for SNR calculation
        all_stats_glu = sum([r['stats_glu'] for r in results_list if r is not None], [])
        all_stats_gaba = sum([r['stats_gaba'] for r in results_list if r is not None], [])
        snr = calculate_snr_from_stats(all_stats_glu, all_stats_gaba)
        
        # CRITICAL CHANGE 5: Calculate 95% CI for the SER using Wilson bounds
        # Wilson CI works even for zero errors
        ci_low, ci_high = proportion_confint(total_errors, total_symbols, method='wilson')
        ci = ci_high - ser  # Upper error bar
        
        # CRITICAL CHANGE 2: Handle error-rate resolution
        min_measurable_ser = 5 / total_symbols  # Can reliably measure if we see at least 5 errors
        
        sweep_results_nm.append({
            'nm': nm, 
            'ser': ser, 
            'snr': snr, 
            'ci': ci,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_err': total_errors,
            'n_sym': total_symbols,
            'min_measurable_ser': min_measurable_ser
        })

    df_nm = pd.DataFrame(sweep_results_nm)
    df_nm['snr_db'] = 10 * np.log10(df_nm['snr'])
    # Save with test prefix if in test mode
    filename_suffix = "_test" if test_mode else "_rigorous"
    df_nm.to_csv(results_dir / "data" / f"nm_sweep_results{filename_suffix}.csv", index=False)
    print("  ...Nm sweep data saved.")

    # --- Sweep 2: SER vs. Distance ---
    print(f"\nRunning Sweep 2: Varying Distance (with {num_seeds} seeds per point)...")
    dist_range_um = cfg_template['distances_um']
    sweep_results_dist = []

    # CRITICAL CHANGE 3: Filter distances based on simulation window
    # FIX 4: Allow distances up to 250 µm with 40s window
    time_window_s = cfg_template['pipeline'].get('time_window_s', 40)
    max_dist = 250  # Allow up to 250 µm as requested
    
    dist_range_um_filtered = [d for d in dist_range_um if d <= max_dist]
    print(f"  (Keeping distances up to {max_dist} µm with {time_window_s}s simulation window)")

    for dist_um in dist_range_um_filtered:
        print(f"  Testing Distance = {dist_um} um...")
        cfg = deepcopy(cfg_template)
        cfg['pipeline']['distance_um'] = dist_um
        
        results_list = Parallel(n_jobs=-1)(delayed(run_single_instance)(cfg, seed) for seed in child_seeds)
        
        # Ensure results_list is a list
        results_list = list(results_list) if not isinstance(results_list, list) else results_list
        
        total_errors = sum(r['errors'] for r in results_list if r is not None)
        total_symbols = len([r for r in results_list if r is not None]) * cfg['pipeline']['sequence_length']
        ser = total_errors / total_symbols if total_symbols > 0 else 0
        
        # FIX 1: Use Wilson CI for distance sweep too
        ci_low, ci_high = proportion_confint(total_errors, total_symbols, method='wilson')
        ci = ci_high - ser
        
        min_measurable_ser = 5 / total_symbols
        
        # FIX 5: Calculate SNR for distance sweep to enable analytic overlay
        all_stats_glu = sum([r['stats_glu'] for r in results_list if r is not None], [])
        all_stats_gaba = sum([r['stats_gaba'] for r in results_list if r is not None], [])
        snr = calculate_snr_from_stats(all_stats_glu, all_stats_gaba)
        
        sweep_results_dist.append({
            'dist': dist_um, 
            'ser': ser, 
            'snr': snr,
            'ci': ci,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_err': total_errors,
            'n_sym': total_symbols,
            'min_measurable_ser': min_measurable_ser
        })

    df_dist = pd.DataFrame(sweep_results_dist)
    df_dist['snr_db'] = 10 * np.log10(df_dist['snr'])  # FIX 5: Add SNR in dB for distance
    filename_suffix = "_test" if test_mode else "_rigorous"
    df_dist.to_csv(results_dir / "data" / f"dist_sweep_results{filename_suffix}.csv", index=False)
    print("  ...Distance sweep data saved.")

    # --- Plotting Results ---
    print("\nGenerating plots...")
    
    # Add suffix to figure names if in test mode
    fig_suffix = "_test" if test_mode else ""
    
    # Figure 6: SNR vs. Nm
    plt.figure(figsize=(6, 4))
    plt.semilogx(df_nm['nm'].to_numpy(), df_nm['snr_db'].to_numpy(), 'o-', markersize=6)
    plt.xlabel("Number of Molecules ($N_m$)")
    plt.ylabel("Empirical SNR (dB)")
    title = "Figure 6: SNR vs. Number of Molecules" + (" (TEST)" if test_mode else "")
    plt.title(title)
    # FIX 2: Add caption
    total_symbols_per_point = num_seeds * cfg_template['pipeline']['sequence_length']
    plt.text(0.5, -0.15, f"{total_symbols_per_point:,} symbols per point",
             ha='center', transform=plt.gca().transAxes, fontsize=9, style='italic')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / f"fig6_snr_vs_nm{fig_suffix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "figures" / f"fig6_snr_vs_nm{fig_suffix}.pdf", bbox_inches='tight', transparent=True)
    plt.close()

    # Figure 7: SER vs. Nm (with CI and analytical overlay)
    plt.figure(figsize=(6, 4))
    
    # FIX 3: Plot all points, including zero-error points
    # Separate zero and non-zero error points
    zero_mask = df_nm['n_err'] == 0
    
    # Plot non-zero error points with asymmetric error bars
    if not zero_mask.all():
        # FIX: Use asymmetric error bars with cap-ticks
        yerr_lower = df_nm.loc[~zero_mask, 'ser'].to_numpy() - df_nm.loc[~zero_mask, 'ci_low'].to_numpy()
        yerr_upper = df_nm.loc[~zero_mask, 'ci_high'].to_numpy() - df_nm.loc[~zero_mask, 'ser'].to_numpy()
        plt.errorbar(df_nm.loc[~zero_mask, 'nm'].to_numpy(), 
                    df_nm.loc[~zero_mask, 'ser'].to_numpy(), 
                    yerr=[yerr_lower, yerr_upper], 
                    fmt='o-', label="Simulated SER", color='blue',
                    capsize=3, markersize=6)  # FIX 4: Add cap-ticks and larger markers
    
    # Plot zero-error points at measurement floor with downward triangles
    if zero_mask.any():
        plt.loglog(df_nm.loc[zero_mask, 'nm'].to_numpy(), 
                  df_nm.loc[zero_mask, 'min_measurable_ser'].to_numpy(), 
                  'v', color='blue', markersize=8, label="Zero errors (upper bound)")
    
    # CRITICAL CHANGE 4: Add analytical AWGN benchmark
    # FIX 2: Ensure correct SNR format for analytical functions
    if USE_SEP_CSK:
        # Import locally to avoid unbound variable issues
        from src.mc_detection import sep_csk_binary  # type: ignore
        # sep_csk_binary expects Eb/N0 (half of SNR for binary)
        ber_theory = sep_csk_binary(df_nm['snr'].to_numpy() / 2)
    elif USE_BER_AWGN:
        # Use imported ber_awgn with SNR in dB
        from src.mc_detection import ber_awgn  # type: ignore
        ber_theory = ber_awgn(df_nm['snr_db'].to_numpy())
    else:
        # Use default implementation with SNR in dB
        ber_theory = default_ber_awgn(df_nm['snr_db'].to_numpy())
    
    plt.loglog(df_nm['nm'].to_numpy(), ber_theory, 'r--', label="AWGN Benchmark", linewidth=2)
    
    # FIX 6: Plot resolution limit as a shaded region using full array
    plt.fill_between(df_nm['nm'].to_numpy(), 0, df_nm['min_measurable_ser'].to_numpy(), 
                     alpha=0.2, color='gray', label='Measurement floor')
    
    
    plt.xlabel("Number of Molecules ($N_m$)")
    plt.ylabel("Symbol Error Rate (SER)")
    # FIX 2: Add informative caption with sample size
    total_symbols_per_point = num_seeds * cfg_template['pipeline']['sequence_length']
    min_ser = 1 / total_symbols_per_point
    title = f"Figure 7: SER vs. Number of Molecules" + (" (TEST)" if test_mode else "")
    plt.title(title)
    # Add caption as text below plot
    plt.text(0.5, -0.15, f"{total_symbols_per_point:,} symbols per point; triangles denote upper bound when no errors observed (SER < {min_ser:.1e})",
             ha='center', transform=plt.gca().transAxes, fontsize=9, style='italic')
    
    plt.grid(True, which="both", ls="--")
    plt.ylim(1e-4, 1)
    
    # FIX: Reorder legend for clarity
    handles, labels = plt.gca().get_legend_handles_labels()
    # Define desired order
    order = []
    for desired_label in ["Simulated SER", "Zero errors (upper bound)", "AWGN Benchmark", "Measurement floor"]:
        if desired_label in labels:
            order.append(labels.index(desired_label))
    # Reorder and create legend
    if order:
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    else:
        plt.legend()
    
    # FIX 5: Save both PNG and PDF versions
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / f"fig7_ser_vs_nm{fig_suffix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "figures" / f"fig7_ser_vs_nm{fig_suffix}.pdf", bbox_inches='tight', transparent=True)
    plt.close()

    # Figure 8: SER vs. Distance (with CI and analytical overlay)
    plt.figure(figsize=(6, 4))
    
    # FIX 3: Plot all points, including zero-error points
    zero_mask = df_dist['n_err'] == 0
    
    # Plot non-zero error points with asymmetric error bars
    if not zero_mask.all():
        # FIX: Use asymmetric error bars with cap-ticks
        yerr_lower = df_dist.loc[~zero_mask, 'ser'].to_numpy() - df_dist.loc[~zero_mask, 'ci_low'].to_numpy()
        yerr_upper = df_dist.loc[~zero_mask, 'ci_high'].to_numpy() - df_dist.loc[~zero_mask, 'ser'].to_numpy()
        # FIX 4: Larger markers for longer distances
        distances = df_dist.loc[~zero_mask, 'dist'].to_numpy()
        marker_sizes = np.where(distances >= 200, 8, 6)  # Larger markers for d >= 200 µm
        for i, (d, s, yl, yu) in enumerate(zip(distances, 
                                               df_dist.loc[~zero_mask, 'ser'].to_numpy(),
                                               yerr_lower, yerr_upper)):
            plt.errorbar([d], [s], yerr=[[yl], [yu]], 
                        fmt='o', color='blue', capsize=3, markersize=marker_sizes[i],
                        label="Simulated SER" if i == 0 else "")
        # Connect points with line
        plt.plot(distances, df_dist.loc[~zero_mask, 'ser'].to_numpy(), '-', color='blue', alpha=0.5)
    
    # Plot zero-error points at measurement floor
    if zero_mask.any():
        plt.semilogy(df_dist.loc[zero_mask, 'dist'].to_numpy(), 
                    df_dist.loc[zero_mask, 'min_measurable_ser'].to_numpy(), 
                    'v', color='blue', markersize=8, label="Zero errors (upper bound)")
    
    # FIX 5: Add analytical overlay for distance curve
    if USE_BER_AWGN or not USE_SEP_CSK:
        ber_theory_dist = default_ber_awgn(df_dist['snr_db'].to_numpy())
        plt.semilogy(df_dist['dist'].to_numpy(), ber_theory_dist, 
                    '--', color='gray', label="AWGN Benchmark", linewidth=2, alpha=0.7)
    
    # FIX 6: Plot resolution limit as shaded region using full array
    plt.fill_between(df_dist['dist'].to_numpy(), 0, df_dist['min_measurable_ser'].to_numpy(), 
                     alpha=0.2, color='gray', label='Measurement floor')
    
    plt.xlabel("Distance (μm)")
    plt.ylabel("Symbol Error Rate (SER)")
    # FIX 2: Add informative caption
    total_symbols_per_point = num_seeds * cfg_template['pipeline']['sequence_length']
    min_ser = 1 / total_symbols_per_point
    title = "Figure 8: SER vs. Distance" + (" (TEST)" if test_mode else "")
    plt.title(title)
    plt.text(0.5, -0.15, f"{total_symbols_per_point:,} symbols per point; triangles denote upper bound when no errors observed (SER < {min_ser:.1e})",
             ha='center', transform=plt.gca().transAxes, fontsize=9, style='italic')
    
    plt.grid(True, which="both", ls="--")
    plt.ylim(1e-4, 1)
    
    # FIX 3: Reorder legend and ensure no duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate entries
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels and label != "":
            unique_labels.append(label)
            unique_handles.append(handle)
    
    # Define desired order
    order = []
    for desired_label in ["Simulated SER", "Zero errors (upper bound)", "AWGN Benchmark", "Measurement floor"]:
        if desired_label in unique_labels:
            order.append(unique_labels.index(desired_label))
    
    if order:
        plt.legend([unique_handles[idx] for idx in order], [unique_labels[idx] for idx in order])
    else:
        plt.legend()
    
    # FIX 5: Save both PNG and PDF
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / f"fig8_ser_vs_dist{fig_suffix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "figures" / f"fig8_ser_vs_dist{fig_suffix}.pdf", bbox_inches='tight', transparent=True)
    plt.close()
    
    print("\n--- Summary Statistics ---")
    print(f"Mode: {'TEST' if test_mode else 'PRODUCTION'}")
    print(f"Total symbols per data point: {df_nm['n_sym'].iloc[0]:,}")
    print(f"Minimum measurable SER: {df_nm['min_measurable_ser'].iloc[0]:.2e}")
    print(f"Points with zero errors in Nm sweep: {len(df_nm[df_nm['n_err'] == 0])}")
    print(f"Points with zero errors in distance sweep: {len(df_dist[df_dist['n_err'] == 0])}")
    print(f"Distance points tested: {len(df_dist)} values from {df_dist['dist'].min()} to {df_dist['dist'].max()} µm")
    
    if test_mode:
        print("\n*** TEST RUN COMPLETE ***")
        print("To run full simulation, set test_mode = False")
        print("Full run will test 9 distance points: [25, 50, 75, 100, 150, 175, 200, 225, 250] µm")
    else:
        print("\n--- All rigorous plots saved successfully! ---")
        print("Saved 6 files: 3 PNGs + 3 PDFs in results/figures/")

if __name__ == "__main__":
    main()