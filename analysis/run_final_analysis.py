# analysis/run_final_analysis.py
"""
Unified analysis script for all modulation schemes (MoSK, CSK, Hybrid).
Handles calibration, sweeps, and figure generation with CPU optimization.

IMPORTANT: This script uses physics-based dynamic symbol periods to prevent
molecule accumulation and aptamer saturation. The symbol period is automatically
adjusted based on distance to ensure molecules have sufficient time to dissipate
between symbols.

Key features:
- Dynamic symbol period calculation based on diffusion physics
- Automatic ISI memory scaling with symbol period
- Observation window (time_window_s) matched to symbol period
- Automatic threshold recalibration for CSK/Hybrid modes at each distance
- Threshold caching to avoid redundant calibrations
- Initial calibration saved to file, per-distance calibrations kept in memory only
- CPU parallelization for improved performance

For CSK/Hybrid modes, detection thresholds are automatically recalibrated
whenever the symbol period changes (e.g., at different distances) to ensure
optimal detection performance. A cache is used to avoid redundant calibrations
when the same symbol period is encountered multiple times.
"""

'python analysis/run_final_analysis.py --recalibrate --num-seeds 8 --sequence-length 1000 --extreme-mode'

import sys
import json
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from copy import deepcopy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import psutil   #type: ignore[import]
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional, Union

# BEAST MODE CPU OPTIMIZATION SETTINGS
CPU_COUNT = mp.cpu_count()
PHYSICAL_CORES = psutil.cpu_count(logical=False)

# i9-13950HX specific optimization (24 cores, 32 threads)
# Hybrid architecture: 8 P-cores + 16 E-cores = 24 cores, 32 threads total
I9_13950HX_DETECTED = CPU_COUNT == 32 and PHYSICAL_CORES == 24

if I9_13950HX_DETECTED:
    print("🔥 i9-13950HX detected! Using specialized optimization...")
    # Ultra-aggressive settings for i9-13950HX
    OPTIMAL_WORKERS = 40        # 1.25x thread count
    BEAST_MODE_WORKERS = 48     # 1.5x thread count  
    EXTREME_WORKERS = 61        # 2x thread count for 100% utilization
    BATCH_SIZE = 4              # Smaller batches for hybrid architecture
else:
    # Generic settings for other CPUs
    OPTIMAL_WORKERS = min(CPU_COUNT - 4, 24)
    BEAST_MODE_WORKERS = min(CPU_COUNT - 2, 28)
    EXTREME_WORKERS = CPU_COUNT * 2
    BATCH_SIZE = max(8, CPU_COUNT // 4)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline import run_sequence, run_sequence_batch_cpu
from src.config_utils import preprocess_config


def preprocess_config_full(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess configuration to ensure all numeric values are properly typed
    and create nested dictionaries for OECT simulation.
    """
    # First convert all numeric strings
    cfg = preprocess_config(config)
    
    # Create nested dictionaries
    if 'oect' not in cfg:
        cfg['oect'] = {
            'gm_S': cfg.get('gm_S', 0.002),
            'C_tot_F': cfg.get('C_tot_F', 2.4e-7),
            'R_ch_Ohm': cfg.get('R_ch_Ohm', 200)
        }
    
    if 'noise' not in cfg:
        cfg['noise'] = {
            'alpha_H': cfg.get('alpha_H', 3.0e-3),
            'N_c': cfg.get('N_c', 6.0e12),
            'K_d_Hz': cfg.get('K_d_Hz', 1.3e-4)
        }
    
    if 'sim' not in cfg:
        cfg['sim'] = {
            'dt_s': cfg.get('dt_s', 0.01),
            'temperature_K': cfg.get('temperature_K', 310.0)
        }
    
    if 'binding' not in cfg:
        cfg['binding'] = cfg.get('binding', {})
    
    # Optional: clean up top-level keys that were moved
    for key in ['gm_S', 'C_tot_F', 'R_ch_Ohm', 'alpha_H', 'N_c', 'K_d_Hz', 'dt_s', 'temperature_K']:
        cfg.pop(key, None)
    
    return cfg


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run final analysis for molecular communication system with BEAST MODE CPU optimization"
    )
    parser.add_argument(
        "--mode",
        choices=["MoSK", "CSK", "Hybrid"],
        default="MoSK",
        help="Modulation scheme to analyze"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
        help="Number of Monte Carlo seeds per data point"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1000,
        help="Number of symbols per run"
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Force recalibration of thresholds"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=OPTIMAL_WORKERS,
        help=f"Maximum number of CPU workers for parallel processing (default: {OPTIMAL_WORKERS})"
    )
    parser.add_argument(
        "--beast-mode",
        action="store_true",
        help=f"Ultra-aggressive CPU optimization with {BEAST_MODE_WORKERS} workers"
    )
    parser.add_argument(
        "--extreme-mode",
        action="store_true",
        help=f"EXTREME CPU optimization with {EXTREME_WORKERS} workers (i9-13950HX only)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def calculate_dynamic_symbol_period(distance_um: float, cfg: Dict[str, Any]) -> float:
    """
    Calculate physics-based symbol period with guard time.
    
    Parameters
    ----------
    distance_um : float
        Distance in micrometers
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    float
        Symbol period in seconds
    """
    # Use GLU diffusion coefficient as worst-case (slowest diffusion)
    D_glu = cfg['neurotransmitters']['GLU']['D_m2_s']
    
    # Get tortuosity for GLU (brain tissue has high tortuosity)
    lambda_glu = cfg['neurotransmitters']['GLU']['lambda']
    
    # Effective diffusion coefficient in brain tissue
    # D_eff = D / λ²
    D_eff = D_glu / (lambda_glu ** 2)
    
    # Time for 95% of molecules to diffuse away (3D diffusion)
    # Using effective diffusion coefficient that accounts for tortuosity
    time_95 = 3.0 * ((distance_um * 1e-6)**2) / D_eff
    
    # Add guard time for safety
    guard_factor = cfg['pipeline'].get('guard_factor', 0.3)
    guard_time = guard_factor * time_95
    
    # Total symbol period (minimum 20s for practical reasons)
    symbol_period = max(20.0, round(time_95 + guard_time))
    
    return symbol_period


def run_single_instance(config: Dict[str, Any], seed: int) -> Optional[Dict[str, Any]]:
    """Run a single simulation instance with given seed."""
    try:
        cfg_run = deepcopy(config)
        cfg_run['pipeline']['random_seed'] = int(seed)
        return run_sequence(cfg_run)
    except Exception as e:
        print(f"\nWARNING: Simulation failed with seed {seed}. Error: {e}")
        return None


def run_parallel_simulations(cfg: Dict[str, Any], seeds: List[int], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run multiple simulations in parallel using BEAST MODE CPU optimization.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    seeds : list
        List of random seeds
    max_workers : int, optional
        Maximum number of worker processes
        
    Returns
    -------
    list
        List of simulation results
    """
    if max_workers is None:
        max_workers = OPTIMAL_WORKERS
    
    # BEAST MODE: Submit all work at once for maximum CPU utilization
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs immediately (beast mode strategy)
        future_to_seed = {executor.submit(run_single_instance, cfg, seed): seed for seed in seeds}
        
        # Collect results with aggressive timeout
        for future in as_completed(future_to_seed, timeout=600):  # 10 min timeout like beast mode
            try:
                result = future.result(timeout=120)  # 2 min per job
                if result is not None:
                    results.append(result)
            except Exception as exc:
                seed = future_to_seed[future]
                print(f'BEAST MODE: Seed {seed} generated an exception: {exc}')
    
    return results


def calibrate_sigma_parameters(cfg: Dict[str, Any], seeds: List[int], num_samples: int = 200) -> Dict[str, float]:
    """
    Calibrate sigma_glu and sigma_gaba parameters for ML decision statistic.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    seeds : list
        Random seeds for calibration
    num_samples : int
        Number of samples for calibration
        
    Returns
    -------
    dict
        Dictionary with 'sigma_glu' and 'sigma_gaba' values
    """
    print("\nCalibrating sigma parameters for ML decision statistic...")
    
    # Collect control channel statistics from multiple runs
    all_q_ctrl = []
    all_q_glu = []
    all_q_gaba = []
    
    cal_cfg = deepcopy(cfg)
    cal_cfg['pipeline']['sequence_length'] = 50  # Shorter sequences for calibration
    
    dt = cal_cfg['sim']['dt_s']
    
    # Import needed function
    from src.pipeline import _single_symbol_currents
    
    for seed in seeds[:10]:  # Use subset of seeds
        cal_cfg['pipeline']['random_seed'] = seed
        rng = np.random.default_rng(seed)
        tx_history: List[Tuple[int, float]] = []
        
        # Run both GLU and GABA symbols to get noise statistics
        for s_tx in [0, 1]:  # GLU and GABA
            for _ in range(25):  # 25 samples each
                ig, ia, ic, Nm_actual = _single_symbol_currents(s_tx, tx_history, cal_cfg, rng)
                tx_history.append((s_tx, float(Nm_actual)))
                
                # Calculate charges
                q_glu = float(np.trapezoid(ig - ic, dx=dt))
                q_gaba = float(np.trapezoid(ia - ic, dx=dt))
                q_ctrl = float(np.trapezoid(ic, dx=dt))
                
                all_q_ctrl.append(q_ctrl)
                all_q_glu.append(q_glu)
                all_q_gaba.append(q_gaba)
    
    # Calculate sigma values from control channel variations
    sigma_ctrl = float(np.std(all_q_ctrl))
    
    # Estimate noise for each channel (control channel variation + small floor)
    sigma_glu = max(sigma_ctrl * 0.2, 1e-12)  # 20% of control variation
    sigma_gaba = max(sigma_ctrl * 0.2, 1e-12)  # 20% of control variation
    
    print(f"  Calibrated sigma_glu: {sigma_glu:.3e}")
    print(f"  Calibrated sigma_gaba: {sigma_gaba:.3e}")
    print(f"  Control channel std: {sigma_ctrl:.3e}")
    
    return {
        'sigma_glu': sigma_glu,
        'sigma_gaba': sigma_gaba,
        'sigma_ctrl': sigma_ctrl
    }


def calibrate_thresholds(cfg: Dict[str, Any], seeds: List[int], recalibrate: bool = False, 
                        save_to_file: bool = True) -> Dict[str, Union[float, List[float]]]:
    """
    Calibrate detection thresholds and sigma parameters for all modes.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    seeds : list
        Random seeds for calibration runs
    recalibrate : bool
        Force recalibration even if file exists
    save_to_file : bool
        Whether to save calibration results to JSON file
        
    Returns
    -------
    dict
        Dictionary of calibrated thresholds and sigma parameters
    """
    mode = cfg['pipeline']['modulation']
    results_dir = project_root / "results" / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Include symbol period in filename if provided
    symbol_period = cfg['pipeline'].get('symbol_period_s', '')
    if symbol_period and save_to_file:
        threshold_file = results_dir / f"thresholds_{mode.lower()}_ts{int(symbol_period)}.json"
    else:
        threshold_file = results_dir / f"thresholds_{mode.lower()}.json"
    
    # Check if we can load existing thresholds
    if threshold_file.exists() and not recalibrate and save_to_file:
        print(f"Loading existing thresholds from {threshold_file}")
        with open(threshold_file, 'r') as f:
            loaded_thresholds = json.load(f)
            # Ensure proper typing
            typed_thresholds: Dict[str, Union[float, List[float]]] = {}
            for k, v in loaded_thresholds.items():
                typed_thresholds[k] = v
            return typed_thresholds
    
    print(f"\nCalibrating thresholds for {mode} mode...")
    
    # First, calibrate sigma parameters (needed for all modes)
    sigma_params = calibrate_sigma_parameters(cfg, seeds)
    
    # Prepare calibration config
    cal_cfg = deepcopy(cfg)
    cal_cfg['pipeline']['sequence_length'] = 100  # Shorter for calibration
    
    # Ensure time_window_s matches symbol_period_s for calibration
    if 'symbol_period_s' in cal_cfg['pipeline']:
        cal_cfg['pipeline']['time_window_s'] = cal_cfg['pipeline']['symbol_period_s']
    
    thresholds: Dict[str, Union[float, List[float]]] = {}
    
    if mode.startswith("CSK"):
        M = cfg['pipeline']['csk_levels']
        target_channel = cfg['pipeline'].get('csk_target_channel', 'GLU')
        
        # Collect statistics for each level
        level_stats: Dict[int, List[float]] = {level: [] for level in range(M)}
        
        print(f"Collecting statistics for {M} CSK levels...")
        for level in range(M):
            # Generate sequence of only this level
            for seed in seeds[:10]:  # Use subset for calibration
                cal_cfg['pipeline']['random_seed'] = seed
                # Run single symbol multiple times
                result = run_calibration_symbols(cal_cfg, level, mode='CSK')
                if result:
                    level_stats[level].extend(result['q_values'])
        
        # Calculate thresholds as midpoints between levels
        threshold_list: List[float] = []
        for i in range(M - 1):
            if level_stats[i] and level_stats[i + 1]:
                mean_low = float(np.mean(level_stats[i]))
                mean_high = float(np.mean(level_stats[i + 1]))
                threshold = float((mean_low + mean_high) / 2)
                threshold_list.append(threshold)
                print(f"  Threshold {i}: {threshold:.3e} (between levels {i} and {i+1})")
        
        thresholds[f'csk_thresholds_{target_channel.lower()}'] = threshold_list
        
    elif mode == "Hybrid":
        # Collect statistics for four cases
        stats: Dict[str, List[float]] = {
            'glu_low': [],
            'glu_high': [],
            'gaba_low': [],
            'gaba_high': []
        }
        
        print("Collecting statistics for Hybrid mode...")
        # Symbol mapping: 00=GLU_low, 01=GLU_high, 10=GABA_low, 11=GABA_high
        symbol_to_type = {0: 'glu_low', 1: 'glu_high', 2: 'gaba_low', 3: 'gaba_high'}
        
        for symbol in range(4):
            for seed in seeds[:10]:
                cal_cfg['pipeline']['random_seed'] = seed
                result = run_calibration_symbols(cal_cfg, symbol, mode='Hybrid')
                if result:
                    stats[symbol_to_type[symbol]].extend(result['q_values'])
        
        # Calculate thresholds
        if all(stats[k] for k in stats):
            # GLU threshold: midpoint between GLU low and high
            mean_glu_low = np.mean(stats['glu_low'])
            mean_glu_high = np.mean(stats['glu_high'])
            threshold_glu = float((mean_glu_low + mean_glu_high) / 2)
            
            # GABA threshold: midpoint between GABA low and high
            mean_gaba_low = np.mean(stats['gaba_low'])
            mean_gaba_high = np.mean(stats['gaba_high'])
            threshold_gaba = float((mean_gaba_low + mean_gaba_high) / 2)
            
            thresholds['hybrid_threshold_glu'] = threshold_glu
            thresholds['hybrid_threshold_gaba'] = threshold_gaba
            
            print(f"  GLU threshold: {threshold_glu:.3e}")
            print(f"  GABA threshold: {threshold_gaba:.3e}")
    
    # Add sigma parameters to thresholds
    thresholds.update(sigma_params)
    
    # Save thresholds and sigma parameters if requested
    if save_to_file:
        with open(threshold_file, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f"Thresholds and sigma parameters saved to {threshold_file}")
    
    return thresholds


def run_calibration_symbols(cfg: Dict[str, Any], symbol: int, mode: str, num_symbols: int = 50) -> Optional[Dict[str, Any]]:
    """
    Run calibration for a specific symbol value with ML decision statistic support.
    
    Returns dict with 'q_values', 'sigma_glu', 'sigma_gaba' for ML-scaled decision statistic.
    """
    try:
        # Create temporary config
        cal_cfg = deepcopy(cfg)
        cal_cfg['pipeline']['sequence_length'] = num_symbols
        
        # Generate sequence of only this symbol
        tx_symbols = [symbol] * num_symbols
        
        # Run simulation with fixed symbols
        rng = np.random.default_rng(cal_cfg['pipeline']['random_seed'])
        tx_history: List[Tuple[int, float]] = []
        
        # Store all charge values for statistics
        q_glu_values: List[float] = []
        q_gaba_values: List[float] = []
        q_ctrl_values: List[float] = []
        decision_stats: List[float] = []
        
        dt = cal_cfg['sim']['dt_s']
        
        # Import needed function
        from src.pipeline import _single_symbol_currents
        
        for s_tx in tx_symbols:
            ig, ia, ic, Nm_actual = _single_symbol_currents(s_tx, tx_history, cal_cfg, rng)
            tx_history.append((s_tx, float(Nm_actual)))
            
            # Calculate differential charges (channel - control)
            q_glu = float(np.trapezoid(ig - ic, dx=dt))
            q_gaba = float(np.trapezoid(ia - ic, dx=dt))
            q_ctrl = float(np.trapezoid(ic, dx=dt))
            
            q_glu_values.append(q_glu)
            q_gaba_values.append(q_gaba)
            q_ctrl_values.append(q_ctrl)
        
        # Calculate noise statistics from control channel variations
        sigma_glu = max(float(np.std(q_ctrl_values)) * 0.2, 1e-12)  # 20% of control variation
        sigma_gaba = max(float(np.std(q_ctrl_values)) * 0.2, 1e-12)  # 20% of control variation
        
        # Calculate ML-scaled decision statistics
        for i in range(len(q_glu_values)):
            q_glu = q_glu_values[i]
            q_gaba = q_gaba_values[i]
            q_ctrl = q_ctrl_values[i]
            
            # ML-optimal decision statistic
            decision_stat = (q_glu - q_ctrl)/sigma_glu - (q_gaba - q_ctrl)/sigma_gaba
            decision_stats.append(decision_stat)
        
        # Return appropriate values based on mode
        if mode == "MoSK":
            # For MoSK, return decision statistics directly
            return {
                'q_values': decision_stats,
                'sigma_glu': sigma_glu,
                'sigma_gaba': sigma_gaba,
                'q_glu_raw': q_glu_values,
                'q_gaba_raw': q_gaba_values,
                'q_ctrl_raw': q_ctrl_values
            }
        elif mode.startswith("CSK"):
            # For CSK, return target channel values
            target_channel = cal_cfg['pipeline'].get('csk_target_channel', 'GLU')
            target_values = q_glu_values if target_channel == 'GLU' else q_gaba_values
            return {
                'q_values': target_values,
                'sigma_glu': sigma_glu,
                'sigma_gaba': sigma_gaba
            }
        elif mode == "Hybrid":
            # For hybrid, return appropriate channel
            mol_type = symbol >> 1  # 0=GLU, 1=GABA
            target_values = q_glu_values if mol_type == 0 else q_gaba_values
            return {
                'q_values': target_values,
                'sigma_glu': sigma_glu,
                'sigma_gaba': sigma_gaba
            }
        
        return {'q_values': decision_stats, 'sigma_glu': sigma_glu, 'sigma_gaba': sigma_gaba}
        
    except Exception as e:
        print(f"Calibration run failed: {e}")
        return None


def run_sweep(cfg: Dict[str, Any], seeds: List[int], sweep_param: str, 
              sweep_values: List[float], sweep_name: str, max_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Run parameter sweep with MASSIVE BATCH parallelization for 100% CPU utilization.
    
    This creates ALL parameter-seed combinations upfront and submits them simultaneously
    to keep all CPU cores constantly busy, just like the working GPU+CPU version.
    """
    if max_workers is None:
        max_workers = OPTIMAL_WORKERS
    
    # Create ALL parameter-seed combinations upfront (like working version)
    all_combinations = [
        (sweep_param, value, seed)
        for value in sweep_values
        for seed in seeds
    ]
    
    print(f"🚀 MASSIVE BATCH: {len(all_combinations)} jobs ({len(sweep_values)} values × {len(seeds)} seeds)")
    print(f"   Using {max_workers} workers for 100% CPU utilization")
    
    # Force output flushing
    import sys
    sys.stdout.flush()
    
    # MASSIVE BATCH: Submit ALL jobs at once for constant CPU saturation
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all parameter-seed combinations immediately
        future_to_combo = {
            executor.submit(run_param_seed_combo, cfg, param_name, param_value, seed): (param_name, param_value, seed)
            for param_name, param_value, seed in all_combinations
        }
        
        print(f"✅ All {len(all_combinations)} jobs submitted. Starting progress tracking...")
        sys.stdout.flush()
        
        # Collect results with progress bar (force display and flushing)
        for future in tqdm(as_completed(future_to_combo), 
                          total=len(all_combinations), 
                          desc=f"MASSIVE BATCH {sweep_name}",
                          file=sys.stdout,
                          dynamic_ncols=True,
                          leave=True,
                          miniters=1,
                          mininterval=0.1):
            try:
                result = future.result(timeout=600)  # 10 min timeout like working version
                if result:
                    results.append(result)
            except Exception as exc:
                combo = future_to_combo[future]
                print(f'MASSIVE BATCH: Combo {combo} failed: {exc}')
                sys.stdout.flush()
    
    # Process results into DataFrame
    if not results:
        print(f"Warning: No successful results for {sweep_name}")
        return pd.DataFrame()
    
    # Group results by parameter value
    df_data = []
    for value in sweep_values:
        value_results = [r for r in results if r['param_value'] == value]
        
        if not value_results:
            continue
        
        # Aggregate results for this parameter value
        total_symbols = len(value_results) * cfg['pipeline']['sequence_length']
        total_errors = sum(r['errors'] for r in value_results)
        ser = total_errors / total_symbols if total_symbols > 0 else 1.0
        
        # Calculate subsymbol errors for Hybrid
        if cfg['pipeline']['modulation'] == 'Hybrid':
            mosk_errors = sum(r.get('subsymbol_errors', {}).get('mosk', 0) for r in value_results)
            csk_errors = sum(r.get('subsymbol_errors', {}).get('csk', 0) for r in value_results)
            mosk_ser = mosk_errors / total_symbols
            csk_ser = csk_errors / total_symbols
        else:
            mosk_ser = csk_ser = None
        
        # Calculate SNR from statistics
        all_stats_glu = []
        all_stats_gaba = []
        for r in value_results:
            all_stats_glu.extend(r.get('stats_glu', []))
            all_stats_gaba.extend(r.get('stats_gaba', []))
        
        if all_stats_glu and all_stats_gaba:
            snr = calculate_snr_from_stats(all_stats_glu, all_stats_gaba)
        else:
            snr = 0
        
        # Store aggregated results
        row = {
            sweep_param: value,
            'ser': ser,
            'snr_db': snr,
            'num_runs': len(value_results)
        }
        
        if mosk_ser is not None:
            row['mosk_ser'] = mosk_ser
            row['csk_ser'] = csk_ser
        
        df_data.append(row)
    
    return pd.DataFrame(df_data)


def run_param_seed_combo(cfg_base: Dict[str, Any], param_name: str, param_value: float, seed: int) -> Optional[Dict[str, Any]]:
    """
    Run single parameter-seed combination (MASSIVE BATCH worker function).
    
    This is the worker function that gets called for each parameter-seed combination
    in the massive batch parallelization strategy.
    """
    try:
        cfg_run = deepcopy(cfg_base)
        
        # DISABLE PROGRESS BARS in worker functions to prevent conflicts
        cfg_run['disable_progress'] = True
        
        # Set parameter value
        if '.' in param_name:
            keys = param_name.split('.')
            target = cfg_run
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = param_value
        else:
            cfg_run[param_name] = param_value
        
        # Handle distance-specific updates
        if param_name == 'pipeline.distance_um':
            new_symbol_period = calculate_dynamic_symbol_period(param_value, cfg_run)
            cfg_run['pipeline']['symbol_period_s'] = new_symbol_period
            cfg_run['pipeline']['time_window_s'] = new_symbol_period
            
            if cfg_run['pipeline'].get('enable_isi', False):
                D_glu = cfg_run['neurotransmitters']['GLU']['D_m2_s']
                lambda_glu = cfg_run['neurotransmitters']['GLU']['lambda']
                D_eff = D_glu / (lambda_glu ** 2)
                time_95 = 3.0 * ((param_value * 1e-6)**2) / D_eff
                guard_factor = cfg_run['pipeline'].get('guard_factor', 0.3)
                isi_memory = math.ceil((1 + guard_factor) * time_95 / new_symbol_period)
                cfg_run['pipeline']['isi_memory_symbols'] = isi_memory
        
        # Run simulation
        result = run_single_instance(cfg_run, seed)
        
        if result:
            result['param_name'] = param_name
            result['param_value'] = param_value
            result['seed'] = seed
        
        return result
        
    except Exception as e:
        print(f"MASSIVE BATCH worker failed for {param_name}={param_value}, seed={seed}: {e}")
        return None


def calculate_snr_from_stats(stats_glu: List[float], stats_gaba: List[float]) -> float:
    """Calculate SNR from decision statistics."""
    if not stats_glu or not stats_gaba:
        return 0
    
    mu_glu = np.mean(stats_glu)
    mu_gaba = np.mean(stats_gaba)
    var_glu = np.var(stats_glu)
    var_gaba = np.var(stats_gaba)
    
    if (var_glu + var_gaba) == 0:
        return np.inf
    
    return float((mu_glu - mu_gaba)**2 / (var_glu + var_gaba))


def find_lod_for_ser(cfg: Dict[str, Any], seeds: List[int], target_ser: float = 0.01, 
                    max_workers: Optional[int] = None) -> Tuple[Union[int, float], float]:
    """
    Find limit of detection (LoD) using binary search with CPU parallelization.
    
    Returns
    -------
    tuple
        (lod_nm, achieved_ser)
    """
    nm_min = cfg['pipeline'].get('lod_nm_min', 50)
    nm_max = 100000
    lod_nm: float = np.nan
    best_ser: float = 1.0
    
    for _ in range(14):  # Binary search iterations
        if nm_min > nm_max:
            break
            
        nm_mid = int(round((nm_min + nm_max) / 2))
        if nm_mid == 0 or nm_mid > nm_max:
            break
        
        # Test this Nm value
        cfg_test = deepcopy(cfg)
        cfg_test['pipeline']['Nm_per_symbol'] = nm_mid
        
        results_list = run_parallel_simulations(cfg_test, seeds, max_workers)
        
        if not results_list:
            nm_min = nm_mid + 1
            continue
        
        ser = sum(r['errors'] for r in results_list) / (len(results_list) * cfg_test['pipeline']['sequence_length'])
        
        if ser <= target_ser:
            lod_nm = nm_mid
            best_ser = ser
            nm_max = nm_mid - 1
        else:
            nm_min = nm_mid + 1
    
    # Check boundary if search didn't converge
    if np.isnan(lod_nm) and nm_min <= 100000:
        cfg_final = deepcopy(cfg)
        cfg_final['pipeline']['Nm_per_symbol'] = nm_min
        results_list = run_parallel_simulations(cfg_final, seeds, max_workers)
        
        if results_list:
            final_ser = sum(r['errors'] for r in results_list) / (len(results_list) * cfg_final['pipeline']['sequence_length'])
            if final_ser <= target_ser:
                return nm_min, final_ser
    
    return (int(lod_nm) if not np.isnan(lod_nm) else np.nan, best_ser)


def plot_ser_vs_nm(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    """Generate Figure 7: SER vs Nm for all modes."""
    plt.figure(figsize=(10, 6))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    for mode, df in results_dict.items():
        if 'pipeline_Nm_per_symbol' in df.columns and 'ser' in df.columns:
            plt.loglog(df['pipeline_Nm_per_symbol'], df['ser'], 
                      color=colors.get(mode, 'black'),
                      marker=markers.get(mode, 'o'),
                      markersize=8,
                      label=mode,
                      linewidth=2)
    
    plt.xlabel('Number of Molecules per Symbol (Nm)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title('Figure 7: SER vs. Nm for All Modulation Schemes')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.ylim(1e-4, 1)
    plt.xlim(1e2, 1e5)
    
    # Add target SER line
    plt.axhline(y=0.01, color='k', linestyle=':', alpha=0.5, label='Target SER = 1%')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_lod_vs_distance(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    """Generate comparative LoD vs distance plot."""
    plt.figure(figsize=(10, 6))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    for mode, df in results_dict.items():
        if 'distance_um' in df.columns and 'lod_nm' in df.columns:
            df_valid = df.dropna(subset=['lod_nm'])
            plt.semilogy(df_valid['distance_um'], df_valid['lod_nm'],
                        color=colors.get(mode, 'black'),
                        marker=markers.get(mode, 'o'),
                        markersize=8,
                        label=mode,
                        linewidth=2)
    
    plt.xlabel('Distance (μm)')
    plt.ylabel('Limit of Detection (molecules)')
    plt.title('Comparative LoD vs. Distance')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    """Main execution function with i9-13950HX BEAST MODE CPU optimization."""
    # Parse arguments
    args = parse_arguments()
    
    # Override max_workers based on mode
    if args.extreme_mode:
        if I9_13950HX_DETECTED:
            args.max_workers = EXTREME_WORKERS
            print(f"🚀 EXTREME MODE ACTIVATED: Using {EXTREME_WORKERS} workers for i9-13950HX!")
        else:
            print("⚠️  EXTREME MODE only available for i9-13950HX. Using BEAST MODE instead.")
            args.max_workers = BEAST_MODE_WORKERS
    elif args.beast_mode:
        args.max_workers = BEAST_MODE_WORKERS
        print(f"🔥 BEAST MODE ACTIVATED: Using {BEAST_MODE_WORKERS} workers!")
    
    # System optimization for i9-13950HX
    if I9_13950HX_DETECTED and (args.beast_mode or args.extreme_mode):
        try:
            # Set process priority for better performance using psutil
            current_process = psutil.Process()
            current_process.nice(-10)  # Higher priority
            print("🎯 i9-13950HX: Process priority optimized")
        except Exception as e:
            print(f"⚠️  Could not set process priority: {e} (run as admin for better performance)")
    
    print(f"🚀 i9-13950HX BEAST MODE: Using {args.max_workers} worker processes")
    print(f"   Hardware: {CPU_COUNT} threads ({PHYSICAL_CORES} cores)")
    if I9_13950HX_DETECTED:
        print(f"   Architecture: 8 P-cores + 16 E-cores (hybrid)")
        print(f"   Optimization: i9-13950HX specific tuning enabled")
    
    # Initialize threshold cache (will be populated if CSK/Hybrid)
    threshold_cache: Dict[float, Dict[str, Union[float, List[float]]]] = {}
    
    print(f"\n{'='*60}")
    print(f"🔥 BEAST MODE Analysis for {args.mode} Mode")
    print(f"{'='*60}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Recalibrate: {args.recalibrate}")
    print(f"CPU workers: {args.max_workers}")
    print(f"Beast mode: {args.beast_mode}")
    
    # Setup directories
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    data_dir = results_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess configuration
    with open(project_root / "config" / "default.yaml") as f:
        config_base = yaml.safe_load(f)
    
    cfg = preprocess_config_full(config_base)
    
    # Update with command-line arguments
    cfg['pipeline']['modulation'] = args.mode
    cfg['pipeline']['sequence_length'] = args.sequence_length
    cfg['verbose'] = args.verbose
    
    # Generate seeds
    ss = np.random.SeedSequence(2026)
    seeds = [int(s) for s in ss.generate_state(args.num_seeds)]
    
    # Calibrate thresholds if needed
    if args.mode in ["CSK", "Hybrid"]:
        thresholds = calibrate_thresholds(cfg, seeds, args.recalibrate, save_to_file=True)
        # Update config with calibrated thresholds
        for key, value in thresholds.items():
            cfg['pipeline'][key] = value
    
    # Configure mode-specific parameters
    if args.mode.startswith("CSK"):
        cfg['pipeline']['csk_levels'] = 4  # Default to 4-level CSK
        cfg['pipeline']['csk_target_channel'] = 'GLU'  # Default channel
    
    print(f"\n{'='*60}")
    print("Running Performance Sweeps")
    print(f"{'='*60}")
    
    # Print physics parameters
    print(f"\nPhysics Configuration:")
    print(f"  GLU diffusion coefficient: {cfg['neurotransmitters']['GLU']['D_m2_s']:.2e} m²/s")
    print(f"  GABA diffusion coefficient: {cfg['neurotransmitters']['GABA']['D_m2_s']:.2e} m²/s")
    print(f"  Guard factor: {cfg['pipeline'].get('guard_factor', 0.3):.1f}")
    print(f"  ISI enabled: {cfg['pipeline'].get('enable_isi', False)}")
    
    if args.mode in ["CSK", "Hybrid"]:
        print(f"  Threshold calibration: Automatic recalibration at each distance")
    
    # Sweep 1: SER vs Nm
    print("\n1. Running SER vs. Nm sweep...")
    nm_values = [2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]
    
    # Get default distance for symbol period calculation
    default_distance = cfg['pipeline'].get('distance_um', 100)
    
    # Calculate dynamic symbol period based on distance
    symbol_period = calculate_dynamic_symbol_period(default_distance, cfg)
    
    print(f"Using dynamic symbol period: {symbol_period:.0f}s for distance {default_distance}μm")
    cfg['pipeline']['symbol_period_s'] = symbol_period
    cfg['pipeline']['time_window_s'] = symbol_period  # Match observation window to symbol period
    
    df_ser_nm = run_sweep(
        cfg, seeds, 
        'pipeline.Nm_per_symbol', 
        nm_values,
        f"SER vs Nm ({args.mode})",
        args.max_workers
    )
    
    # Save results
    csv_path = data_dir / f"ser_vs_nm_{args.mode.lower()}.csv"
    df_ser_nm.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Sweep 2: LoD vs Distance
    print("\n2. Running LoD vs. Distance sweep...")
    distances = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    lod_results = []
    
    if args.mode in ["CSK", "Hybrid"]:
        print("   Note: Thresholds will be automatically recalibrated at each distance")
    
    for dist_um in tqdm(distances, desc=f"LoD vs Distance ({args.mode})", disable=cfg.get('disable_progress', False)):
        cfg_dist = deepcopy(cfg)
        cfg_dist['pipeline']['distance_um'] = dist_um
        
        # Calculate dynamic symbol period
        symbol_period = calculate_dynamic_symbol_period(dist_um, cfg)
        cfg_dist['pipeline']['symbol_period_s'] = symbol_period
        cfg_dist['pipeline']['time_window_s'] = symbol_period  # Match observation window
        
        # Calculate dynamic ISI memory if ISI is enabled
        if cfg_dist['pipeline'].get('enable_isi', False):
            D_glu = cfg['neurotransmitters']['GLU']['D_m2_s']
            lambda_glu = cfg['neurotransmitters']['GLU']['lambda']
            D_eff = D_glu / (lambda_glu ** 2)  # Effective diffusion
            time_95 = 3.0 * ((dist_um * 1e-6)**2) / D_eff
            guard_factor = cfg['pipeline'].get('guard_factor', 0.3)
            isi_memory = math.ceil((1 + guard_factor) * time_95 / symbol_period)
            cfg_dist['pipeline']['isi_memory_symbols'] = isi_memory
        
        # Re-calibrate thresholds for CSK/Hybrid at this symbol period
        if args.mode in ["CSK", "Hybrid"]:
            if symbol_period not in threshold_cache:
                if cfg.get('verbose', False):
                    print(f"\n  Calibrating thresholds for Ts={symbol_period}s...")
                # Don't save to file during sweep - only use in-memory cache
                threshold_cache[symbol_period] = calibrate_thresholds(
                    cfg_dist, seeds[:10], recalibrate=True, save_to_file=False
                )
            
            # Apply cached thresholds
            for k, v in threshold_cache[symbol_period].items():
                cfg_dist['pipeline'][k] = v
        
        # Find LoD
        lod_nm, ser_at_lod = find_lod_for_ser(cfg_dist, seeds[:10], max_workers=args.max_workers)  # Use subset for speed
        
        # Calculate data rate
        if args.mode == "MoSK":
            bits_per_symbol = 1
        elif args.mode.startswith("CSK"):
            M = cfg['pipeline']['csk_levels']
            bits_per_symbol = np.log2(M)
        else:  # Hybrid
            bits_per_symbol = 2
        
        data_rate = (bits_per_symbol / symbol_period) * (1 - ser_at_lod)
        
        lod_results.append({
            'distance_um': dist_um,
            'lod_nm': lod_nm,
            'ser_at_lod': ser_at_lod,
            'data_rate_bps': data_rate,
            'symbol_period_s': symbol_period
        })
    
    df_lod = pd.DataFrame(lod_results)
    csv_path = data_dir / f"lod_vs_distance_{args.mode.lower()}.csv"
    df_lod.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"\nAll results saved in: {results_dir}")
    print(f"Mode: {args.mode}")
    print(f"Total simulation runs: {len(seeds) * (len(nm_values) + len(distances))}")
    
    # Generate individual plots
    print("\nGenerating plots...")
    
    # Create single-mode plots
    mode_results = {args.mode: df_ser_nm}
    plot_ser_vs_nm(mode_results, figures_dir / f"ser_vs_nm_{args.mode.lower()}.png")
    
    mode_lod_results = {args.mode: df_lod}
    plot_lod_vs_distance(mode_lod_results, figures_dir / f"lod_vs_distance_{args.mode.lower()}.png")
    
    print("\nPlots saved successfully!")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"\nMinimum SER achieved: {df_ser_nm['ser'].min():.2e}")
    print(f"Nm for minimum SER: {df_ser_nm.loc[df_ser_nm['ser'].idxmin(), 'pipeline_Nm_per_symbol']:.0f}")
    
    if threshold_cache and args.mode in ["CSK", "Hybrid"]:
        print(f"\nThreshold calibrations performed: {len(threshold_cache)}")
    
    valid_lod = df_lod.dropna(subset=['lod_nm'])
    if not valid_lod.empty:
        print(f"\nLoD range: {valid_lod['lod_nm'].min():.0f} - {valid_lod['lod_nm'].max():.0f} molecules")
        print(f"Max data rate: {valid_lod['data_rate_bps'].max():.4f} bps")


if __name__ == "__main__":
    main()
