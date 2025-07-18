# analysis/lod_calc.py
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from copy import deepcopy
import pandas as pd
from joblib import Parallel, delayed # type: ignore
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline import run_sequence

def preprocess_config(config):
    # ... (This function is correct and remains unchanged) ...
    cfg = deepcopy(config)
    def to_float(key):
        if key in cfg and cfg[key] is not None and not isinstance(cfg[key], (int, float)):
            cfg[key] = float(cfg[key])
    numeric_keys = ['temperature_K', 'alpha', 'clearance_rate', 'T_release_ms', 'gm_S', 'C_tot_F', 'R_ch_Ohm', 'alpha_H', 'N_c', 'K_d_Hz', 'dt_s']
    for key in numeric_keys: to_float(key)
    if 'pipeline' in cfg:
        # Also convert new keys
        for pkey in ['distance_um', 'Nm_per_symbol', 'guard_factor', 'lod_nm_min']:
            if pkey in cfg['pipeline'] and cfg['pipeline'][pkey] is not None:
                cfg['pipeline'][pkey] = float(cfg['pipeline'][pkey])
    if 'Nm_range' in cfg and cfg['Nm_range'] is not None: cfg['Nm_range'] = [float(x) for x in cfg['Nm_range']]
    if 'distances_um' in cfg and cfg['distances_um'] is not None: cfg['distances_um'] = [float(x) for x in cfg['distances_um']]
    cfg['oect'] = {'gm_S': cfg.get('gm_S'), 'C_tot_F': cfg.get('C_tot_F'), 'R_ch_Ohm': cfg.get('R_ch_Ohm')}
    cfg['noise'] = {'alpha_H': cfg.get('alpha_H'), 'N_c': cfg.get('N_c'), 'K_d_Hz': cfg.get('K_d_Hz')}
    cfg['sim'] = {'dt_s': cfg.get('dt_s'), 'temperature_K': cfg.get('temperature_K')}
    return cfg

def calculate_snr_from_stats(stats_glu, stats_gaba):
    # ... (This function is correct and remains unchanged) ...
    if not stats_glu or not stats_gaba: return 0
    mu_glu, mu_gaba = np.mean(stats_glu), np.mean(stats_gaba)
    var_glu, var_gaba = np.var(stats_glu), np.var(stats_gaba)
    if (var_glu + var_gaba) == 0: return np.inf
    return (mu_glu - mu_gaba)**2 / (var_glu + var_gaba)

def run_single_instance(config, seed):
    # ... (This function is correct and remains unchanged) ...
    try:
        cfg_run = deepcopy(config)
        cfg_run['pipeline']['random_seed'] = int(seed)
        return run_sequence(cfg_run)
    except Exception as e:
        # FIX: Added warning for silent failures
        print(f"\nWARNING: A simulation run failed with seed {seed}. Error: {e}\n")
        return None

def find_lod_for_ser(config, seeds, target_ser=0.01):
    # FIX: Use configurable lower bound for search
    nm_min = config['pipeline'].get('lod_nm_min', 50)
    nm_max = 100000
    lod_nm, best_ser, final_ser_at_min = np.nan, 1.0, 1.0
    
    for _ in range(14):
        if nm_min > nm_max: break
        nm_mid = int(round((nm_min + nm_max) / 2))
        if nm_mid == 0 or nm_mid > nm_max: break # nm_mid > nm_max can happen if nm_max becomes < nm_min
            
        cfg_run = deepcopy(config)
        cfg_run['pipeline']['Nm_per_symbol'] = nm_mid
        
        results_list = Parallel(n_jobs=-1)(delayed(run_single_instance)(cfg_run, seed) for seed in seeds)
        results_list = [r for r in results_list if r is not None]
        
        if not results_list:
            print(f"Warning: All runs failed for Nm = {nm_mid}. Cannot determine SER.")
            nm_min = nm_mid + 1
            continue
            
        ser = sum(r['errors'] for r in results_list) / (len(results_list) * config['pipeline']['sequence_length'])

        if ser <= target_ser:
            lod_nm, best_ser = nm_mid, ser
            nm_max = nm_mid - 1
        else:
            nm_min = nm_mid + 1
    
    # FIX: Fallback logic for non-convergence
    if np.isnan(lod_nm):
        print(f"  INFO: LoD search did not converge. Checking boundary value Nm={nm_min}...")
        cfg_final = deepcopy(config); cfg_final['pipeline']['Nm_per_symbol'] = nm_min
        results_list = Parallel(n_jobs=-1)(delayed(run_single_instance)(cfg_final, seed) for seed in seeds)
        results_list = [r for r in results_list if r is not None]
        if results_list:
            final_ser = sum(r['errors'] for r in results_list) / (len(results_list) * config['pipeline']['sequence_length'])
            if final_ser <= target_ser:
                print(f"  SUCCESS: Boundary value meets target SER. Setting LoD to {nm_min}.")
                return nm_min, final_ser
            else:
                 print(f"  FAILURE: Boundary value SER ({final_ser:.2%}) > target SER. LoD not found.")
    
    return int(lod_nm) if not np.isnan(lod_nm) else np.nan, best_ser

def main():
    print("--- Starting Final MoSK Analysis (Task 4-I) ---")
    
    with open(project_root / "config" / "default.yaml") as f:
        config_base = yaml.safe_load(f)
    cfg_template = preprocess_config(config_base)
    
    num_seeds = 20
    target_ser = 0.01
    ss = np.random.SeedSequence(2026); child_seeds = [int(s) for s in ss.generate_state(num_seeds)]
    results_dir = project_root / "results"; (results_dir / "figures").mkdir(exist_ok=True); (results_dir / "data").mkdir(exist_ok=True)

    # --- Sweep 1: LoD vs. Distance (with ALL patches) ---
    print(f"\nRunning LoD vs. Distance (Target SER <= {target_ser:.0%}, ISI Enabled)...")
    dist_range_um = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    lod_results = []
    
    for dist_um in tqdm(dist_range_um, desc="LoD vs. Distance", leave=False):
        cfg = deepcopy(cfg_template)
        cfg['pipeline']['distance_um'] = dist_um
        
        # FIX: Correct physics-based dynamic window with guard time
        D_m2_s = cfg['neurotransmitters']['GLU']['D_m2_s']
        # Prefactor 3.0 is for ~95% signal capture time in 3D diffusion
        time_for_95_percent = 3.0 * ((dist_um * 1e-6)**2) / D_m2_s
        guard_time = cfg['pipeline'].get('guard_factor', 0.3) * time_for_95_percent
        dynamic_symbol_period = max(20.0, round(time_for_95_percent + guard_time))
        cfg['pipeline']['symbol_period_s'] = dynamic_symbol_period
        
        lod_nm, ser_at_lod = find_lod_for_ser(cfg, child_seeds, target_ser)
        
        bits_per_symbol = np.log2(2)
        data_rate_bps = (bits_per_symbol / dynamic_symbol_period) * (1 - ser_at_lod)
        lod_results.append({'distance_um': dist_um, 'lod_nm': lod_nm, 'data_rate_bps': data_rate_bps, 'symbol_period_s': dynamic_symbol_period})

    df_lod = pd.DataFrame(lod_results)
    df_lod.to_csv(results_dir / "data" / "lod_distance_results_MoSK_final.csv", index=False)
    df_rate = df_lod.dropna(subset=['lod_nm'])
    df_rate[['distance_um', 'data_rate_bps']].to_csv(results_dir / "data" / "data_rate_results_MoSK_final.csv", index=False)
    print("\nFinal MoSK LoD and Data Rate results saved.")

    # --- Plotting Final MoSK Results ---
    print("\nGenerating final MoSK plots...")
    # Figure 10
    plt.figure(figsize=(6, 4)); plt.semilogy(df_lod['distance_um'], df_lod['lod_nm'], 'o-'); plt.xlabel("Distance (μm)"); plt.ylabel(f"LoD (Nm) for SER ≤ {target_ser:.0%}"); plt.title("Figure 10: MoSK Limit of Detection vs. Distance"); plt.grid(True, which="both", ls="--"); plt.text(0.02, 0.98, f"Symbol periods: {df_lod['symbol_period_s'].min():.0f}s to {df_lod['symbol_period_s'].max():.0f}s", transform=plt.gca().transAxes, va='top', fontsize=9, style='italic'); plt.tight_layout(); plt.savefig(results_dir / "figures" / "fig10_lod_vs_dist_final.png", dpi=300); plt.close()
    print("Figure 10 saved.")

    # Figure 11
    plt.figure(figsize=(8, 5)); plt.semilogy(df_rate['distance_um'], df_rate['data_rate_bps'], 'o-', markersize=8); plt.xlabel("Distance (μm)"); plt.ylabel("Achievable Data Rate (bps)"); plt.title(f"Figure 11: MoSK Achievable Data Rate for SER ≤ {target_ser:.0%}"); plt.grid(True, which="both", ls="--", alpha=0.7)
    rates = df_rate['data_rate_bps']
    # FIX: Robust y-axis limit setting
    if not rates.empty and rates.ptp() > 0: plt.ylim(rates.min() * 0.5, rates.max() * 1.5)
    for i in range(len(df_rate)): plt.annotate(f"{df_rate['data_rate_bps'].iloc[i]:.4f} bps", (df_rate['distance_um'].iloc[i], df_rate['data_rate_bps'].iloc[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.tight_layout(); plt.savefig(results_dir / "figures" / "fig11_data_rate_final.png", dpi=300); plt.close()
    print("Figure 11 saved.")
    
    print("\n--- Task 4-I Complete. MoSK analysis is finalized. ---")

if __name__ == "__main__":
    main()