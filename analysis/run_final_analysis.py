# analysis/run_final_analysis.py
"""
Unified analysis script for all modulation schemes (MoSK, CSK, Hybrid).
P-CORE OPTIMIZED VERSION for i9-13950HX and similar hybrid CPUs.

EXPERT PATCH APPLIED: Harmonized calibration with detection logic and fixed performance bottleneck.
"""

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
from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict
from src.detection import calculate_ml_threshold
import traceback
import gc
import os
import platform

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# EXPERT IMPORT: Need calculate_proper_noise_sigma and _single_symbol_currents for calibration
from src.pipeline import run_sequence, run_sequence_batch_cpu, calculate_proper_noise_sigma, _single_symbol_currents
from src.config_utils import preprocess_config
from src.constants import get_nt_params # EXPERT ADDITION


# Type definitions for CPU configuration
class CPUConfig(TypedDict):
    p_cores_physical: List[int]
    p_cores_logical: List[int]
    e_cores_logical: List[int]
    p_core_count: int
    total_p_threads: int

# CPU DETECTION AND P-CORE CONFIGURATION
# (CPU detection logic remains the same as the prompt)
CPU_COUNT = mp.cpu_count()
PHYSICAL_CORES = psutil.cpu_count(logical=False) or 1

I9_13950HX_DETECTED = CPU_COUNT == 32 and PHYSICAL_CORES == 24

HYBRID_CPU_CONFIGS: Dict[str, CPUConfig] = {
    "i9-13950HX": {
        "p_cores_physical": list(range(8)),
        "p_cores_logical": list(range(16)),
        "e_cores_logical": list(range(16, 32)),
        "p_core_count": 8,
        "total_p_threads": 16
    },
    "i9-12900K": {
        "p_cores_physical": list(range(8)),
        "p_cores_logical": list(range(16)),
        "e_cores_logical": list(range(16, 24)),
        "p_core_count": 8,
        "total_p_threads": 16
    }
}

CPU_CONFIG: Optional[CPUConfig] = None
if I9_13950HX_DETECTED:
    CPU_CONFIG = HYBRID_CPU_CONFIGS["i9-13950HX"]
    print("🔥 i9-13950HX detected! P-core optimization available.")


def set_process_affinity_to_p_cores() -> bool:
    # (Implementation remains the same)
    if CPU_CONFIG is None:
        return False
        
    try:
        current_process = psutil.Process()
        
        if platform.system() == "Windows":
            p_core_mask = 0
            for core in CPU_CONFIG["p_cores_logical"]:
                p_core_mask |= (1 << core)
            
            current_process.cpu_affinity([])
            current_process.cpu_affinity(CPU_CONFIG["p_cores_logical"])
            print(f"✅ CPU affinity set to P-cores only (threads 0-{CPU_CONFIG['total_p_threads']-1})")
            print(f"   Using {CPU_CONFIG['p_core_count']} P-cores ({CPU_CONFIG['total_p_threads']} threads with HT)")
            return True
            
        elif platform.system() == "Linux":
            current_process.cpu_affinity(CPU_CONFIG["p_cores_logical"])
            print(f"✅ CPU affinity set to P-cores only (threads 0-{CPU_CONFIG['total_p_threads']-1})")
            return True
            
        else:
            print("⚠️  CPU affinity not supported on this platform")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not set CPU affinity: {e}")
        print("   Try running as administrator/root for CPU affinity control")
        return False


def get_p_core_worker_count(mode: str = "optimal") -> int:
    # (Implementation remains the same)
    if not CPU_CONFIG:
        if mode == "extreme":
            return min(CPU_COUNT, 32)
        elif mode == "beast":
            return min(CPU_COUNT - 2, 28)
        else:
            return min(CPU_COUNT - 4, 24)
    
    p_threads: int = CPU_CONFIG["total_p_threads"]
    
    if mode == "extreme":
        return p_threads
    elif mode == "beast":
        return max(p_threads - 2, 1)
    else:
        return max(p_threads - 4, 1)

# Update worker settings
if CPU_CONFIG is not None:
    OPTIMAL_WORKERS = get_p_core_worker_count("optimal")
    BEAST_MODE_WORKERS = get_p_core_worker_count("beast")
    EXTREME_WORKERS = get_p_core_worker_count("extreme")
    BATCH_SIZE = 2
else:
    OPTIMAL_WORKERS = min(CPU_COUNT - 4, 24)
    BEAST_MODE_WORKERS = min(CPU_COUNT - 2, 28)
    EXTREME_WORKERS = min(CPU_COUNT, 32)
    BATCH_SIZE = max(8, CPU_COUNT // 4)


class PCoreProcessPoolExecutor(ProcessPoolExecutor):
    # (Implementation remains the same)
    def __init__(self, max_workers=None, *args, **kwargs):
        super().__init__(max_workers=max_workers, *args, **kwargs)
        
    def _adjust_process_count(self):
        super()._adjust_process_count()
        if CPU_CONFIG is not None and platform.system() in ["Windows", "Linux"]:
            try:
                pass
            except:
                pass


def worker_init():
    # (Implementation remains the same)
    if CPU_CONFIG is not None:
        try:
            process = psutil.Process()
            process.cpu_affinity(CPU_CONFIG["p_cores_logical"])
        except:
            pass


def run_parallel_simulations_pcore(cfg: Dict[str, Any], seeds: List[int], 
                                  max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    # (Implementation remains the same)
    if max_workers is None:
        max_workers = OPTIMAL_WORKERS
    
    mem_gb, total_gb, available_gb = check_memory_usage()
    print(f"📊 Memory status: {mem_gb:.1f}GB used, {available_gb:.1f}GB available of {total_gb:.1f}GB total")
    
    results = []
    successful_runs = 0
    failed_runs = 0
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init) as executor:
        future_to_seed = {executor.submit(run_single_instance, cfg, seed): seed for seed in seeds}
        
        for future in as_completed(future_to_seed, timeout=600):
            try:
                result = future.result(timeout=120)
                if result is not None:
                    results.append(result)
                    successful_runs += 1
                else:
                    failed_runs += 1
            except Exception as exc:
                seed = future_to_seed[future]
                print(f'❌ P-CORE: Seed {seed} generated an exception: {exc}')
                failed_runs += 1
    
    print(f"✅ Completed: {successful_runs} successful, {failed_runs} failed out of {len(seeds)} total")
    
    return results


def check_memory_usage():
    # (Implementation remains the same)
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)
    
    virtual_mem = psutil.virtual_memory()
    total_gb = virtual_mem.total / (1024**3)
    available_gb = virtual_mem.available / (1024**3)
    
    if mem_gb > 0.8 * total_gb:
        print(f"⚠️  WARNING: High memory usage: {mem_gb:.1f}GB / {total_gb:.1f}GB total")
        print(f"   Available: {available_gb:.1f}GB")
        gc.collect()
        
    return mem_gb, total_gb, available_gb


def preprocess_config_full(config: Dict[str, Any]) -> Dict[str, Any]:
    # (Implementation remains the same)
    cfg = preprocess_config(config)
    
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
    
    for key in ['gm_S', 'C_tot_F', 'R_ch_Ohm', 'alpha_H', 'N_c', 'K_d_Hz', 'dt_s', 'temperature_K']:
        cfg.pop(key, None)
    
    return cfg


def parse_arguments() -> argparse.Namespace:
    # (Implementation remains the same)
    parser = argparse.ArgumentParser(
        description="Run final analysis for molecular communication system with P-CORE optimization"
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
        default=None,
        help=f"Maximum number of CPU workers (default: auto-detect based on P-cores)"
    )
    parser.add_argument(
        "--beast-mode",
        action="store_true",
        help=f"Ultra-aggressive CPU optimization"
    )
    parser.add_argument(
        "--extreme-mode",
        action="store_true",
        help=f"EXTREME CPU optimization using all P-cores"
    )
    parser.add_argument(
        "--force-all-cores",
        action="store_true",
        help="Disable P-core affinity and use all cores (E+P)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--disable-isi",
        action="store_true",
        help="Disable ISI modeling for comparison runs"
    )
    
    return parser.parse_args()


def calculate_dynamic_symbol_period(distance_um: float, cfg: Dict[str, Any]) -> float:
    # (Implementation remains the same)
    D_glu = cfg['neurotransmitters']['GLU']['D_m2_s']
    lambda_glu = cfg['neurotransmitters']['GLU']['lambda']
    D_eff = D_glu / (lambda_glu ** 2)
    time_95 = 3.0 * ((distance_um * 1e-6)**2) / D_eff
    guard_factor = cfg['pipeline'].get('guard_factor', 0.1) # Using 0.1 from Patch 5
    guard_time = guard_factor * time_95
    symbol_period = max(20.0, round(time_95 + guard_time))
    return symbol_period


def run_single_instance(config: Dict[str, Any], seed: int) -> Optional[Dict[str, Any]]:
    # (Implementation remains the same)
    try:
        cfg_run = deepcopy(config)
        cfg_run['pipeline']['random_seed'] = int(seed)
        
        mem_gb, total_gb, available_gb = check_memory_usage()
        if available_gb < 2.0:
            gc.collect()
        
        result = run_sequence(cfg_run)
        gc.collect()
        return result
    except MemoryError as e:
        print(f"\n❌ MEMORY ERROR with seed {seed}: Out of memory!")
        print(f"   Try reducing sequence_length or num_seeds")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: Simulation failed with seed {seed}.")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        if hasattr(e, '__traceback__'):
            print("   Traceback:")
            traceback.print_tb(e.__traceback__)
        return None

# REMOVED: calibrate_sigma_parameters. We now rely entirely on the physics-based calculate_proper_noise_sigma.


def calibrate_thresholds(cfg: Dict[str, Any], seeds: List[int], recalibrate: bool = False, 
                         save_to_file: bool = True) -> Dict[str, Union[float, List[float]]]:
    """
    Calibrate detection thresholds for all modes.
    
    EXPERT PATCH: Harmonized with pipeline detection logic (Unified Statistic D and Signed Q).
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
        # print(f"Loading existing thresholds from {threshold_file}")
        with open(threshold_file, 'r') as f:
            loaded_thresholds = json.load(f)
            typed_thresholds: Dict[str, Union[float, List[float]]] = {}
            for k, v in loaded_thresholds.items():
                typed_thresholds[k] = v
            return typed_thresholds
    
    # print(f"\nCalibrating thresholds for {mode} mode...")
    
    # Prepare calibration config
    cal_cfg = deepcopy(cfg)
    cal_cfg['pipeline']['sequence_length'] = 100
    
    # Ensure detection window matches symbol_period_s for calibration
    if 'symbol_period_s' in cal_cfg['pipeline']:
        cal_cfg['detection']['decision_window_s'] = cal_cfg['pipeline']['symbol_period_s']
    
    thresholds: Dict[str, Union[float, List[float]]] = {}
    
    # MoSK Calibration Block (and Hybrid Stage 1)
    if mode == "MoSK" or mode == "Hybrid":
        # Collect Unified Statistic D for GLU and GABA classes
        mosk_stats: Dict[str, List[float]] = {'glu': [], 'gaba': []}
        
        # Define symbols representing GLU and GABA types
        if mode == "MoSK":
            symbols_to_check = {0: 'glu', 1: 'gaba'}
        else: # Hybrid: 0/1 are GLU, 2/3 are GABA
            symbols_to_check = {0: 'glu', 1: 'glu', 2: 'gaba', 3: 'gaba'}

        for symbol, type_key in symbols_to_check.items():
            for seed in seeds[:10]:
                cal_cfg['pipeline']['random_seed'] = seed
                # CRITICAL FIX: Use mode='MoSK' here to ensure D statistic is returned
                result = run_calibration_symbols(cal_cfg, symbol, mode='MoSK')
                if result:
                    # Collect the Unified Statistic D
                    mosk_stats[type_key].extend(result['q_values'])
        
        # Calculate ML threshold from empirical distributions of D
        if all(mosk_stats[k] for k in mosk_stats):
            mean_D_glu = float(np.mean(mosk_stats['glu']))  # Expected positive
            std_D_glu = max(float(np.std(mosk_stats['glu'])), 1e-15)
            mean_D_gaba = float(np.mean(mosk_stats['gaba'])) # Expected negative
            std_D_gaba = max(float(np.std(mosk_stats['gaba'])), 1e-15)
            
            # Calculate threshold between the distributions of D
            threshold_mosk = calculate_ml_threshold(mean_D_glu, mean_D_gaba, std_D_glu, std_D_gaba)
            thresholds['mosk_threshold'] = threshold_mosk

    
    # CSK Calibration Block
    if mode.startswith("CSK"):
        M = cfg['pipeline']['csk_levels']
        target_channel = cfg['pipeline'].get('csk_target_channel', 'GLU')
        
        # Collect signed differential statistics (Q) for each level
        level_stats: Dict[int, List[float]] = {level: [] for level in range(M)}

        for level in range(M):
            for seed in seeds[:10]:
                cal_cfg['pipeline']['random_seed'] = seed
                result = run_calibration_symbols(cal_cfg, level, mode='CSK')
                if result:
                    # Collect the signed charge Q
                    level_stats[level].extend(result['q_values'])
        
        # Calculate ML thresholds between levels based on Q
        threshold_list: List[float] = []
        for i in range(M - 1):
            if level_stats[i] and level_stats[i + 1]:
                mean_Q_low = float(np.mean(level_stats[i]))
                mean_Q_high = float(np.mean(level_stats[i + 1]))
                std_Q_low = max(float(np.std(level_stats[i])), 1e-15)
                std_Q_high = max(float(np.std(level_stats[i + 1])), 1e-15)
                
                threshold = calculate_ml_threshold(mean_Q_low, mean_Q_high, std_Q_low, std_Q_high)
                threshold_list.append(threshold)

        # CRITICAL FIX: Sort thresholds based on polarity for pipeline.py logic
        q_eff = get_nt_params(cfg, target_channel)['q_eff_e']
        if q_eff > 0:
             threshold_list.sort() # Ascending for positive polarity
        else:
             threshold_list.sort(reverse=True) # Descending for negative polarity
        
        thresholds[f'csk_thresholds_{target_channel.lower()}'] = threshold_list
    
    # Hybrid Stage 2 Calibration Block
    if mode == "Hybrid":
        # Collect signed differential statistics (Q) for the four cases
        stats: Dict[str, List[float]] = {
            'glu_low': [], 'glu_high': [],
            'gaba_low': [], 'gaba_high': []
        }
        symbol_to_type = {0: 'glu_low', 1: 'glu_high', 2: 'gaba_low', 3: 'gaba_high'}
        
        for symbol in range(4):
            for seed in seeds[:10]:
                cal_cfg['pipeline']['random_seed'] = seed
                # Pass mode='Hybrid' to ensure it returns Q
                result = run_calibration_symbols(cal_cfg, symbol, mode='Hybrid')
                if result:
                    # Collect the signed charge Q
                    stats[symbol_to_type[symbol]].extend(result['q_values'])
        
        # Calculate ML thresholds based on Q
        if all(stats[k] for k in stats):
            # GLU threshold
            mean_Q_glu_low = float(np.mean(stats['glu_low']))
            mean_Q_glu_high = float(np.mean(stats['glu_high']))
            std_Q_glu_low = max(float(np.std(stats['glu_low'])), 1e-15)
            std_Q_glu_high = max(float(np.std(stats['glu_high'])), 1e-15)
            threshold_glu = calculate_ml_threshold(mean_Q_glu_low, mean_Q_glu_high, std_Q_glu_low, std_Q_glu_high)

            # GABA threshold (Negative polarity)
            mean_Q_gaba_low = float(np.mean(stats['gaba_low']))
            mean_Q_gaba_high = float(np.mean(stats['gaba_high']))
            std_Q_gaba_low = max(float(np.std(stats['gaba_low'])), 1e-15)
            std_Q_gaba_high = max(float(np.std(stats['gaba_high'])), 1e-15)
            # Note: For negative polarity, mean_high < mean_low. calculate_ml_threshold handles this.
            threshold_gaba = calculate_ml_threshold(mean_Q_gaba_low, mean_Q_gaba_high, std_Q_gaba_low, std_Q_gaba_high)

            thresholds['hybrid_threshold_glu'] = threshold_glu
            thresholds['hybrid_threshold_gaba'] = threshold_gaba
    
    # Save thresholds if requested
    if save_to_file:
        with open(threshold_file, 'w') as f:
            json.dump(thresholds, f, indent=2)
    
    return thresholds


def run_calibration_symbols(cfg: Dict[str, Any], symbol: int, mode: str, num_symbols: int = 50) -> Optional[Dict[str, Any]]:
    """
    Run calibration for a specific symbol value.
    EXPERT PATCH: Returns the exact statistic used by the corresponding detection logic in pipeline.py.
    """
    try:
        # Create temporary config
        cal_cfg = deepcopy(cfg)
        cal_cfg['pipeline']['sequence_length'] = num_symbols
        cal_cfg['disable_progress'] = True

        # Generate sequence of only this symbol
        tx_symbols = [symbol] * num_symbols
        
        rng = np.random.default_rng(cal_cfg['pipeline'].get('random_seed', 42))
        tx_history: List[Tuple[int, float]] = []
        
        # Store results
        q_glu_values: List[float] = []
        q_gaba_values: List[float] = []
        decision_stats: List[float] = []
        
        dt = cal_cfg['sim']['dt_s']
        detection_window_s = cal_cfg['detection'].get('decision_window_s', cal_cfg['pipeline']['symbol_period_s'])

        # Calculate physics-based noise sigma (required for ML statistic)
        # We use the robust implementation imported from src.pipeline
        sigma_glu, sigma_gaba = calculate_proper_noise_sigma(cal_cfg, detection_window_s)
        
        
        for s_tx in tx_symbols:
            ig, ia, ic, Nm_actual = _single_symbol_currents(s_tx, tx_history, cal_cfg, rng)
            tx_history.append((s_tx, float(Nm_actual)))

            n_total_samples = len(ig)
            n_detect_samples = min(int(detection_window_s / dt), n_total_samples)

            if n_detect_samples <= 1: continue
            
            # Calculate differential charges (channel - control)
            q_glu = float(np.trapezoid((ig - ic)[:n_detect_samples], dx=dt))
            q_gaba = float(np.trapezoid((ia - ic)[:n_detect_samples], dx=dt))
            
            q_glu_values.append(q_glu)
            q_gaba_values.append(q_gaba)

        # Calculate the specific decision statistic for the mode
        # This loop processes the collected charges
        for q_glu, q_gaba in zip(q_glu_values, q_gaba_values):

            # EXPERT FIX: Harmonize statistics with detection logic
            if mode == "MoSK":
                # Return the Unified Decision Statistic (Summation) D
                D = q_glu / sigma_glu + q_gaba / sigma_gaba
                decision_stats.append(D)

            elif mode.startswith("CSK"):
                 # Return the signed charge Q of the target channel
                target_channel = cal_cfg['pipeline'].get('csk_target_channel', 'GLU')
                Q = q_glu if target_channel == 'GLU' else q_gaba
                decision_stats.append(Q)

            elif mode == "Hybrid":
                # Return the signed charge Q of the relevant channel for Stage 2 calibration
                mol_type = symbol >> 1  # 0=GLU, 1=GABA
                Q = q_glu if mol_type == 0 else q_gaba
                decision_stats.append(Q)
        
        return {
            'q_values': decision_stats, # This now holds the relevant statistic (D or Q)
            'sigma_glu': sigma_glu,
            'sigma_gaba': sigma_gaba
        }
        
    except Exception as e:
        print(f"Calibration run failed for symbol {symbol} in mode {mode}: {e}")
        traceback.print_exc()
        return None


def run_param_seed_combo(cfg_base: Dict[str, Any], param_name: str, param_value: float, seed: int) -> Optional[Dict[str, Any]]:
    """
    Run single parameter-seed combination (worker function).
    
    EXPERT PATCH: Removed print() statement to fix performance bottleneck.
    """
    try:
        cfg_run = deepcopy(cfg_base)
        
        # DISABLE PROGRESS BARS and Verbose in worker functions
        cfg_run['disable_progress'] = True
        cfg_run['verbose'] = False
        
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
                guard_factor = cfg_run['pipeline'].get('guard_factor', 0.1) # Using 0.1
                isi_memory = math.ceil((1 + guard_factor) * time_95 / new_symbol_period)
                cfg_run['pipeline']['isi_memory_symbols'] = isi_memory
                
        # Dynamic recalibration (Mandatory for varying Nm or Distance)
        if cfg_run['pipeline']['modulation'] in ['MoSK', 'CSK', 'Hybrid'] and param_name in ['pipeline.Nm_per_symbol', 'pipeline.distance_um']:
            # Quick in-memory recalibration with fixed temp seeds
            temp_seeds = [0, 1, 2, 3, 4]
            
            # CRITICAL PERFORMANCE FIX: Removed print() statement here.
            # print(f"Recalibrating thresholds for {param_name}={param_value}...")
            
            thresholds = calibrate_thresholds(cfg_run, temp_seeds, recalibrate=True, save_to_file=False)
            for k, v in thresholds.items():
                cfg_run['pipeline'][k] = v
        
        # Run simulation
        result = run_single_instance(cfg_run, seed)
        
        if result:
            result['param_name'] = param_name
            result['param_value'] = param_value
            result['seed'] = seed
        
        return result
        
    except Exception as e:
        print(f"\n❌ P-CORE worker failed for {param_name}={param_value}, seed={seed}: {e}")
        traceback.print_exc()
        return None

# ... (The rest of the file: run_sweep, calculate_snr_from_stats, find_lod_for_ser, plotting, and main() remains structurally the same as the prompt, relying on the patched functions above) ...
# To ensure completeness, the remaining functions from the original prompt should follow here.
# Due to the length of the original file, I am including the remaining functions below:

def run_sweep(cfg: Dict[str, Any], seeds: List[int], sweep_param: str, 
              sweep_values: List[float], sweep_name: str, max_workers: Optional[int] = None,
              use_p_cores: bool = True) -> pd.DataFrame:
    """
    Run parameter sweep with P-core optimized parallelization.
    """
    if max_workers is None:
        max_workers = OPTIMAL_WORKERS
    
    all_combinations = [
        (sweep_param, value, seed)
        for value in sweep_values
        for seed in seeds
    ]
    
    print(f"🚀 MASSIVE BATCH: {len(all_combinations)} jobs ({len(sweep_values)} values × {len(seeds)} seeds)")
    print(f"   Using {max_workers} workers for 100% CPU utilization")
    
    check_memory_usage()
    
    import sys
    sys.stdout.flush()
    
    results = []
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init if use_p_cores else None) as executor:
        future_to_combo = {
            executor.submit(run_param_seed_combo, cfg, param_name, param_value, seed): (param_name, param_value, seed)
            for param_name, param_value, seed in all_combinations
        }
        
        print(f"✅ All {len(all_combinations)} jobs submitted. Starting progress tracking...")
        sys.stdout.flush()
        
        for future in tqdm(as_completed(future_to_combo), 
                          total=len(all_combinations), 
                          desc=f"MASSIVE BATCH {sweep_name}",
                          file=sys.stdout,
                          dynamic_ncols=True,
                          leave=True,
                          miniters=1,
                          mininterval=0.1):
            try:
                result = future.result(timeout=300)
                if result:
                    results.append(result)
                    successful += 1
                else:
                    failed += 1
            except Exception as exc:
                combo = future_to_combo[future]
                print(f'\n❌ P-CORE BATCH: Combo {combo} failed: {exc}')
                failed += 1
                sys.stdout.flush()
    
    print(f"\n✅ Sweep completed: {successful} successful, {failed} failed")
    
    if not results:
        print(f"⚠️  Warning: No successful results for {sweep_name}")
        return pd.DataFrame()
    
    df_data = []
    for value in sweep_values:
        value_results = [r for r in results if r['param_value'] == value]
        
        if not value_results:
            continue
        
        total_symbols = len(value_results) * cfg['pipeline']['sequence_length']
        total_errors = sum(r['errors'] for r in value_results)
        ser = total_errors / total_symbols if total_symbols > 0 else 1.0
        
        if cfg['pipeline']['modulation'] == 'Hybrid':
            mosk_errors = sum(r.get('subsymbol_errors', {}).get('mosk', 0) for r in value_results)
            csk_errors = sum(r.get('subsymbol_errors', {}).get('csk', 0) for r in value_results)
            mosk_ser = mosk_errors / total_symbols
            csk_ser = csk_errors / total_symbols
        else:
            mosk_ser = csk_ser = None
        
        all_stats_glu = []
        all_stats_gaba = []
        for r in value_results:
            all_stats_glu.extend(r.get('stats_glu', []))
            all_stats_gaba.extend(r.get('stats_gaba', []))
        
        if all_stats_glu and all_stats_gaba:
            snr = calculate_snr_from_stats(all_stats_glu, all_stats_gaba)
        else:
            snr = 0
        
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

def calculate_snr_from_stats(stats_glu: List[float], stats_gaba: List[float]) -> float:
    # (Implementation remains the same)
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
                    max_workers: Optional[int] = None, use_p_cores: bool = True) -> Tuple[Union[int, float], float]:
    # (Implementation remains the same, relying on the patched dynamic recalibration)
    nm_min = cfg['pipeline'].get('lod_nm_min', 50)
    nm_max = 100000
    lod_nm: float = np.nan
    best_ser: float = 1.0
    
    for _ in range(14):
        if nm_min > nm_max:
            break
            
        nm_mid = int(round((nm_min + nm_max) / 2))
        if nm_mid == 0 or nm_mid > nm_max:
            break
        
        cfg_test = deepcopy(cfg)
        cfg_test['pipeline']['Nm_per_symbol'] = nm_mid
        
        # Quick recalib for this Nm (Dynamic calibration is crucial here)
        temp_seeds = [0,1,2,3,4]
        thresholds = calibrate_thresholds(cfg_test, temp_seeds, recalibrate=True, save_to_file=False)
        for k, v in thresholds.items():
            cfg_test['pipeline'][k] = v
        
        if use_p_cores:
            results_list = run_parallel_simulations_pcore(cfg_test, seeds, max_workers)
        else:
            results_list = run_parallel_simulations_pcore(cfg_test, seeds, max_workers)
        
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
    
    if np.isnan(lod_nm) and nm_min <= 100000:
        cfg_final = deepcopy(cfg)
        cfg_final['pipeline']['Nm_per_symbol'] = nm_min
        
        if use_p_cores:
            results_list = run_parallel_simulations_pcore(cfg_final, seeds, max_workers)
        else:
            results_list = run_parallel_simulations_pcore(cfg_final, seeds, max_workers)
        
        if results_list:
            final_ser = sum(r['errors'] for r in results_list) / (len(results_list) * cfg_final['pipeline']['sequence_length'])
            if final_ser <= target_ser:
                return nm_min, final_ser
    
    return (int(lod_nm) if not np.isnan(lod_nm) else np.nan, best_ser)


def plot_ser_vs_nm(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    # (Implementation remains the same)
    plt.figure(figsize=(10, 6))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    for mode, df in results_dict.items():
        if 'pipeline.Nm_per_symbol' in df.columns and 'ser' in df.columns:
            plt.loglog(df['pipeline.Nm_per_symbol'], df['ser'],
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
    
    plt.axhline(y=0.01, color='k', linestyle=':', alpha=0.5, label='Target SER = 1%')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_lod_vs_distance(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    # (Implementation remains the same)
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
    """Main execution function with i9-13950HX P-CORE optimization."""
    # (Implementation remains the same, relying on the patched functions above)
    args = parse_arguments()
    
    use_p_cores = not args.force_all_cores and CPU_CONFIG is not None
    
    if use_p_cores:
        print("\n🎯 P-CORE OPTIMIZATION MODE")
        affinity_set = set_process_affinity_to_p_cores()
        if not affinity_set:
            print("⚠️  Could not set P-core affinity, using all cores")
            use_p_cores = False
    else:
        print("\n📍 Using ALL cores (P+E)")
    
    if args.max_workers is None:
        if args.extreme_mode:
            args.max_workers = EXTREME_WORKERS
            print(f"🚀 EXTREME MODE: Using {EXTREME_WORKERS} workers")
        elif args.beast_mode:
            args.max_workers = BEAST_MODE_WORKERS
            print(f"🔥 BEAST MODE: Using {BEAST_MODE_WORKERS} workers")
        else:
            args.max_workers = OPTIMAL_WORKERS
            print(f"⚡ OPTIMAL MODE: Using {OPTIMAL_WORKERS} workers")
    
    if (args.beast_mode or args.extreme_mode):
        try:
            current_process = psutil.Process()
            current_process.nice(-10)
            print("🎯 Process priority optimized")
        except Exception as e:
            print(f"⚠️  Could not set process priority: {e} (run as admin for better performance)")
    
    print(f"\n🚀 System Configuration:")
    print(f"   Hardware: {CPU_COUNT} threads ({PHYSICAL_CORES} cores)")
    if CPU_CONFIG is not None:
        print(f"   Architecture: {CPU_CONFIG['p_core_count']} P-cores + {PHYSICAL_CORES - CPU_CONFIG['p_core_count']} E-cores")
        if use_p_cores:
            print(f"   Using: P-cores only ({CPU_CONFIG['total_p_threads']} threads)")
        else:
            print(f"   Using: All cores (P+E)")
    print(f"   Workers: {args.max_workers}")
    
    check_memory_usage()
    
    threshold_cache: Dict[float, Dict[str, Union[float, List[float]]]] = {}
    
    print(f"\n{'='*60}")
    print(f"🔥 P-CORE Analysis for {args.mode} Mode")
    print(f"{'='*60}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Recalibrate: {args.recalibrate}")
    print(f"CPU workers: {args.max_workers}")
    print(f"P-core affinity: {use_p_cores}")
    
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    data_dir = results_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(project_root / "config" / "default.yaml") as f:
        config_base = yaml.safe_load(f)
    
    cfg = preprocess_config_full(config_base)
    
    cfg['pipeline']['enable_isi'] = False if args.disable_isi else cfg['pipeline'].get('enable_isi', False)
    print(f"ISI modeling: {'Disabled' if args.disable_isi else 'Enabled'}")
    
    cfg['pipeline']['modulation'] = args.mode
    cfg['pipeline']['sequence_length'] = args.sequence_length
    cfg['verbose'] = args.verbose
    
    ss = np.random.SeedSequence(2026)
    seeds = [int(s) for s in ss.generate_state(args.num_seeds)]
    
    # Calibration is now handled dynamically within the sweeps.
    
    if args.mode.startswith("CSK"):
        cfg['pipeline']['csk_levels'] = 4
        cfg['pipeline']['csk_target_channel'] = 'GLU'
    
    print(f"\n{'='*60}")
    print("Running Performance Sweeps")
    print(f"{'='*60}")
    
    print(f"\nPhysics Configuration:")
    print(f"  GLU diffusion coefficient: {cfg['neurotransmitters']['GLU']['D_m2_s']:.2e} m²/s")
    print(f"  GABA diffusion coefficient: {cfg['neurotransmitters']['GABA']['D_m2_s']:.2e} m²/s")
    print(f"  Guard factor: {cfg['pipeline'].get('guard_factor', 0.3):.1f}")
    print(f"  ISI enabled: {cfg['pipeline'].get('enable_isi', False)}")
    
    # All modes now use dynamic recalibration
    print(f"  Threshold calibration: Automatic dynamic recalibration enabled")
    
    # Sweep 1: SER vs Nm
    print("\n1. Running SER vs. Nm sweep...")
    nm_values = [2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]
    
    default_distance = cfg['pipeline'].get('distance_um', 100)
    
    symbol_period = calculate_dynamic_symbol_period(default_distance, cfg)
    
    print(f"Using dynamic symbol period: {symbol_period:.0f}s for distance {default_distance}μm")
    cfg['pipeline']['symbol_period_s'] = symbol_period
    cfg['pipeline']['time_window_s'] = symbol_period
    
    df_ser_nm = run_sweep(
        cfg, seeds, 
        'pipeline.Nm_per_symbol', 
        nm_values,
        f"SER vs Nm ({args.mode})",
        args.max_workers,
        use_p_cores
    )
    
    csv_path = data_dir / f"ser_vs_nm_{args.mode.lower()}.csv"
    df_ser_nm.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Sweep 2: LoD vs Distance
    print("\n2. Running LoD vs. Distance sweep...")
    distances = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    lod_results = []
        
    for dist_um in tqdm(distances, desc=f"LoD vs Distance ({args.mode})", disable=cfg.get('disable_progress', False)):
        cfg_dist = deepcopy(cfg)
        cfg_dist['pipeline']['distance_um'] = dist_um
        
        symbol_period = calculate_dynamic_symbol_period(dist_um, cfg)
        cfg_dist['pipeline']['symbol_period_s'] = symbol_period
        cfg_dist['pipeline']['time_window_s'] = symbol_period
        
        if cfg_dist['pipeline'].get('enable_isi', False):
            D_glu = cfg['neurotransmitters']['GLU']['D_m2_s']
            lambda_glu = cfg['neurotransmitters']['GLU']['lambda']
            D_eff = D_glu / (lambda_glu ** 2)
            time_95 = 3.0 * ((dist_um * 1e-6)**2) / D_eff
            guard_factor = cfg['pipeline'].get('guard_factor', 0.3)
            isi_memory = math.ceil((1 + guard_factor) * time_95 / symbol_period)
            cfg_dist['pipeline']['isi_memory_symbols'] = isi_memory
        
        # Thresholds are handled dynamically within find_lod_for_ser
        
        # Find LoD
        lod_nm, ser_at_lod = find_lod_for_ser(cfg_dist, seeds[:10], max_workers=args.max_workers, use_p_cores=use_p_cores)
        
        # Calculate data rate
        if args.mode == "MoSK":
            bits_per_symbol = 1
        elif args.mode.startswith("CSK"):
            M = cfg['pipeline']['csk_levels']
            bits_per_symbol = np.log2(M)
        else:
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
    
    # ... (Rest of main() remains the same) ...


if __name__ == "__main__":
    # Add stability for multiprocessing on different OS
    if platform.system() == "Windows":
        mp.freeze_support()
        
    main()