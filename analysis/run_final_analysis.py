# analysis/run_final_analysis.py
"""
FINAL OPTIMIZED VERSION: Fixed CSK calibration with dynamic thresholds.
Implements proper calibration for CSK and Hybrid modes with validation.
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
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from functools import partial
import multiprocessing as mp
import psutil
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict
import traceback
import gc
import os
import platform
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.detection import calculate_ml_threshold
from src.pipeline import run_sequence, calculate_proper_noise_sigma, _single_symbol_currents
from src.config_utils import preprocess_config
from src.constants import get_nt_params

# ============= TYPE DEFINITIONS =============
class CPUConfig(TypedDict):
    p_cores_physical: List[int]
    p_cores_logical: List[int]
    e_cores_logical: List[int]
    p_core_count: int
    total_p_threads: int

# ============= CPU DETECTION & OPTIMIZATION =============
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

def get_optimal_workers(mode: str = "optimal") -> int:
    """Get optimal worker count based on CPU and mode."""
    if not CPU_CONFIG:
        if mode == "extreme":
            return min(CPU_COUNT, 32)
        elif mode == "beast":
            return min(CPU_COUNT - 2, 28)
        else:
            return min(CPU_COUNT - 4, 24)
    
    p_threads = CPU_CONFIG["total_p_threads"]
    if mode == "extreme":
        return p_threads
    elif mode == "beast":
        return max(p_threads - 2, 1)
    else:
        return max(p_threads - 4, 1)

def worker_init():
    """Initialize worker process with P-core affinity if available."""
    if CPU_CONFIG is not None:
        try:
            process = psutil.Process()
            process.cpu_affinity(CPU_CONFIG["p_cores_logical"])
        except:
            pass

# ============= OPTIMIZATION 1: PERSISTENT PROCESS POOL =============
class GlobalProcessPool:
    """Singleton process pool manager for reuse across all operations."""
    _instance = None
    _pool = None
    _max_workers = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_pool(self, max_workers: Optional[int] = None, mode: str = "optimal"):
        resolved_workers = max_workers or get_optimal_workers(mode)
        if self._pool is None or self._max_workers != resolved_workers:
            if self._pool:
                self._pool.shutdown(wait=True)
            self._max_workers = resolved_workers
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=worker_init
            )
            print(f"🚀 Global process pool initialized with {self._max_workers} workers")
        return self._pool
    
    def shutdown(self):
        if self._pool:
            self._pool.shutdown(wait=True)
            self._pool = None
            print("✅ Global process pool shut down")

global_pool = GlobalProcessPool()

# ============= OPTIMIZATION 2: CACHED CALIBRATION =============
calibration_cache: Dict[str, Dict[str, Union[float, List[float]]]] = {}

def get_cache_key(cfg: Dict[str, Any]) -> str:
    """Generate unique cache key for configuration."""
    key_params = [
        cfg['pipeline'].get('modulation'),
        cfg['pipeline'].get('Nm_per_symbol'),
        cfg['pipeline'].get('distance_um'),
        cfg['pipeline'].get('symbol_period_s'),
        cfg['pipeline'].get('csk_levels'),
        cfg['pipeline'].get('csk_target_channel'),
    ]
    return str(hash(tuple(str(p) for p in key_params)))

def calibrate_thresholds_cached(cfg: Dict[str, Any], seeds: List[int]) -> Dict[str, Union[float, List[float]]]:
    """Cached version of calibrate_thresholds to avoid redundant calculations."""
    cache_key = get_cache_key(cfg)
    if cache_key in calibration_cache:
        return calibration_cache[cache_key]
    
    result = calibrate_thresholds(cfg, seeds, recalibrate=True, save_to_file=False)
    calibration_cache[cache_key] = result
    return result

# ============= HELPER FUNCTIONS =============
def check_memory_usage():
    """Check and report memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)
    
    virtual_mem = psutil.virtual_memory()
    total_gb = virtual_mem.total / (1024**3)
    available_gb = virtual_mem.available / (1024**3)
    
    if mem_gb > 0.8 * total_gb:
        print(f"⚠️  High memory usage: {mem_gb:.1f}GB / {total_gb:.1f}GB")
        gc.collect()
    
    return mem_gb, total_gb, available_gb

def preprocess_config_full(config: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess configuration."""
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

def calculate_dynamic_symbol_period(distance_um: float, cfg: Dict[str, Any]) -> float:
    """Calculate dynamic symbol period based on distance."""
    D_glu = cfg['neurotransmitters']['GLU']['D_m2_s']
    lambda_glu = cfg['neurotransmitters']['GLU']['lambda']
    D_eff = D_glu / (lambda_glu ** 2)
    time_95 = 3.0 * ((distance_um * 1e-6)**2) / D_eff
    guard_factor = cfg['pipeline'].get('guard_factor', 0.1)
    guard_time = guard_factor * time_95
    symbol_period = max(20.0, round(time_95 + guard_time))
    return symbol_period

def calculate_snr_from_stats(stats_glu: List[float], stats_gaba: List[float]) -> float:
    """Calculate SNR from statistics."""
    if not stats_glu or not stats_gaba:
        return 0
    
    mu_glu = np.mean(stats_glu)
    mu_gaba = np.mean(stats_gaba)
    var_glu = np.var(stats_glu)
    var_gaba = np.var(stats_gaba)
    
    if (var_glu + var_gaba) == 0:
        return np.inf
    
    return float((mu_glu - mu_gaba)**2 / (var_glu + var_gaba))

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run optimized molecular communication analysis"
    )
    parser.add_argument("--mode", choices=["MoSK", "CSK", "Hybrid"], default="MoSK")
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=1000)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--beast-mode", action="store_true")
    parser.add_argument("--extreme-mode", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable-isi", action="store_true")
    parser.add_argument("--debug-calibration", action="store_true", 
                       help="Print detailed calibration information")
    
    return parser.parse_args()

# ============= IMPROVED CALIBRATION FUNCTIONS =============
def calibrate_thresholds(cfg: Dict[str, Any], seeds: List[int], recalibrate: bool = False, 
                         save_to_file: bool = True, verbose: bool = False) -> Dict[str, Union[float, List[float]]]:
    """
    FIXED: Enhanced calibration with better statistics and validation.
    """
    mode = cfg['pipeline']['modulation']
    results_dir = project_root / "results" / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    symbol_period = cfg['pipeline'].get('symbol_period_s', '')
    if symbol_period and save_to_file:
        threshold_file = results_dir / f"thresholds_{mode.lower()}_ts{int(symbol_period)}.json"
    else:
        threshold_file = results_dir / f"thresholds_{mode.lower()}.json"
    
    if threshold_file.exists() and not recalibrate and save_to_file:
        with open(threshold_file, 'r') as f:
            loaded_thresholds = json.load(f)
            typed_thresholds: Dict[str, Union[float, List[float]]] = {}
            for k, v in loaded_thresholds.items():
                typed_thresholds[k] = v
            return typed_thresholds
    
    cal_cfg = deepcopy(cfg)
    # FIXED: Increase calibration samples for better statistics
    cal_cfg['pipeline']['sequence_length'] = 200  # Increased from 100
    
    if 'symbol_period_s' in cal_cfg['pipeline']:
        cal_cfg['detection']['decision_window_s'] = cal_cfg['pipeline']['symbol_period_s']
    
    thresholds: Dict[str, Union[float, List[float]]] = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CALIBRATING {mode} THRESHOLDS")
        print(f"Distance: {cfg['pipeline'].get('distance_um')}μm")
        print(f"Nm: {cfg['pipeline'].get('Nm_per_symbol')}")
        print(f"{'='*60}")
    
    # MoSK Calibration
    if mode == "MoSK" or mode == "Hybrid":
        mosk_stats: Dict[str, List[float]] = {'glu': [], 'gaba': []}
        
        if mode == "MoSK":
            symbols_to_check = {0: 'glu', 1: 'gaba'}
        else:
            symbols_to_check = {0: 'glu', 1: 'glu', 2: 'gaba', 3: 'gaba'}

        # FIXED: Use more seeds for better statistics
        cal_seeds = seeds[:20] if len(seeds) >= 20 else seeds  # Increased from 10
        
        for symbol, type_key in symbols_to_check.items():
            for seed in cal_seeds:
                cal_cfg['pipeline']['random_seed'] = seed
                result = run_calibration_symbols(cal_cfg, symbol, mode='MoSK' if mode == "MoSK" else mode)
                if result:
                    mosk_stats[type_key].extend(result['q_values'])
        
        if all(mosk_stats[k] for k in mosk_stats):
            mean_D_glu = float(np.mean(mosk_stats['glu']))
            std_D_glu = max(float(np.std(mosk_stats['glu'])), 1e-15)
            mean_D_gaba = float(np.mean(mosk_stats['gaba']))
            std_D_gaba = max(float(np.std(mosk_stats['gaba'])), 1e-15)
            
            threshold_mosk = calculate_ml_threshold(mean_D_glu, mean_D_gaba, std_D_glu, std_D_gaba)
            thresholds['mosk_threshold'] = threshold_mosk
            
            if verbose:
                print(f"\nMoSK Statistics:")
                print(f"  GLU:  μ={mean_D_glu:.3e}, σ={std_D_glu:.3e}")
                print(f"  GABA: μ={mean_D_gaba:.3e}, σ={std_D_gaba:.3e}")
                print(f"  Threshold: {threshold_mosk:.3e}")
    
    # CSK Calibration (FIXED)
    if mode.startswith("CSK"):
        M = cfg['pipeline']['csk_levels']
        target_channel = cfg['pipeline'].get('csk_target_channel', 'GLU')
        
        if verbose:
            print(f"\nCSK Configuration:")
            print(f"  Levels: {M}")
            print(f"  Target Channel: {target_channel}")
        
        level_stats: Dict[int, List[float]] = {level: [] for level in range(M)}

        # FIXED: Use more seeds and samples
        cal_seeds = seeds[:20] if len(seeds) >= 20 else seeds
        
        for level in range(M):
            if verbose:
                print(f"  Calibrating level {level} (Nm fraction: {level/(M-1) if M > 1 else 0:.2f})...")
            
            for seed in cal_seeds:
                cal_cfg['pipeline']['random_seed'] = seed
                result = run_calibration_symbols(cal_cfg, level, mode='CSK', num_symbols=100)  # More symbols
                if result:
                    level_stats[level].extend(result['q_values'])
        
        # Calculate statistics for each level
        if verbose:
            print(f"\nCSK Level Statistics:")
            for level in range(M):
                if level_stats[level]:
                    mean_Q = np.mean(level_stats[level])
                    std_Q = np.std(level_stats[level])
                    print(f"  Level {level}: μ={mean_Q:.3e}, σ={std_Q:.3e}, n={len(level_stats[level])}")
        
        # Calculate thresholds between adjacent levels
        threshold_list: List[float] = []
        for i in range(M - 1):
            if level_stats[i] and level_stats[i + 1]:
                mean_Q_low = float(np.mean(level_stats[i]))
                mean_Q_high = float(np.mean(level_stats[i + 1]))
                std_Q_low = max(float(np.std(level_stats[i])), 1e-15)
                std_Q_high = max(float(np.std(level_stats[i + 1])), 1e-15)
                
                threshold = calculate_ml_threshold(mean_Q_low, mean_Q_high, std_Q_low, std_Q_high)
                threshold_list.append(threshold)
                
                if verbose:
                    print(f"  Threshold {i}->{i+1}: {threshold:.3e}")

        # Sort thresholds based on polarity
        q_eff = get_nt_params(cfg, target_channel)['q_eff_e']
        if q_eff > 0:
            threshold_list.sort()
            if verbose:
                print(f"  Sorted ascending (q_eff={q_eff}>0): {[f'{t:.3e}' for t in threshold_list]}")
        else:
            threshold_list.sort(reverse=True)
            if verbose:
                print(f"  Sorted descending (q_eff={q_eff}<0): {[f'{t:.3e}' for t in threshold_list]}")
        
        # VALIDATION: Check threshold separation
        if len(threshold_list) > 1:
            min_separation = min(abs(threshold_list[i+1] - threshold_list[i]) 
                               for i in range(len(threshold_list)-1))
            if verbose:
                print(f"  Minimum threshold separation: {min_separation:.3e}")
            
            # Warning if thresholds are too close
            if min_separation < 1e-16:
                print(f"⚠️  WARNING: CSK thresholds too close! Min separation: {min_separation:.3e}")
                print(f"   This will cause detection errors. Check signal strength and noise levels.")
        
        thresholds[f'csk_thresholds_{target_channel.lower()}'] = threshold_list
    
    # Hybrid Stage 2 Calibration
    if mode == "Hybrid":
        stats: Dict[str, List[float]] = {
            'glu_low': [], 'glu_high': [],
            'gaba_low': [], 'gaba_high': []
        }
        symbol_to_type = {0: 'glu_low', 1: 'glu_high', 2: 'gaba_low', 3: 'gaba_high'}
        
        cal_seeds = seeds[:20] if len(seeds) >= 20 else seeds
        
        for symbol in range(4):
            for seed in cal_seeds:
                cal_cfg['pipeline']['random_seed'] = seed
                result = run_calibration_symbols(cal_cfg, symbol, mode='Hybrid')
                if result:
                    stats[symbol_to_type[symbol]].extend(result['q_values'])
        
        if all(stats[k] for k in stats):
            mean_Q_glu_low = float(np.mean(stats['glu_low']))
            mean_Q_glu_high = float(np.mean(stats['glu_high']))
            std_Q_glu_low = max(float(np.std(stats['glu_low'])), 1e-15)
            std_Q_glu_high = max(float(np.std(stats['glu_high'])), 1e-15)
            threshold_glu = calculate_ml_threshold(mean_Q_glu_low, mean_Q_glu_high, std_Q_glu_low, std_Q_glu_high)

            mean_Q_gaba_low = float(np.mean(stats['gaba_low']))
            mean_Q_gaba_high = float(np.mean(stats['gaba_high']))
            std_Q_gaba_low = max(float(np.std(stats['gaba_low'])), 1e-15)
            std_Q_gaba_high = max(float(np.std(stats['gaba_high'])), 1e-15)
            threshold_gaba = calculate_ml_threshold(mean_Q_gaba_low, mean_Q_gaba_high, std_Q_gaba_low, std_Q_gaba_high)

            thresholds['hybrid_threshold_glu'] = threshold_glu
            thresholds['hybrid_threshold_gaba'] = threshold_gaba
            
            if verbose:
                print(f"\nHybrid Stage 2 Statistics:")
                print(f"  GLU low:  μ={mean_Q_glu_low:.3e}, σ={std_Q_glu_low:.3e}")
                print(f"  GLU high: μ={mean_Q_glu_high:.3e}, σ={std_Q_glu_high:.3e}")
                print(f"  GLU threshold: {threshold_glu:.3e}")
                print(f"  GABA low:  μ={mean_Q_gaba_low:.3e}, σ={std_Q_gaba_low:.3e}")
                print(f"  GABA high: μ={mean_Q_gaba_high:.3e}, σ={std_Q_gaba_high:.3e}")
                print(f"  GABA threshold: {threshold_gaba:.3e}")
    
    if save_to_file:
        with open(threshold_file, 'w') as f:
            json.dump(thresholds, f, indent=2)
    
    return thresholds

def run_calibration_symbols(cfg: Dict[str, Any], symbol: int, mode: str, num_symbols: int = 100) -> Optional[Dict[str, Any]]:
    """
    FIXED: Improved calibration with better statistics collection.
    """
    try:
        cal_cfg = deepcopy(cfg)
        cal_cfg['pipeline']['sequence_length'] = num_symbols
        cal_cfg['disable_progress'] = True

        tx_symbols = [symbol] * num_symbols
        
        rng = np.random.default_rng(cal_cfg['pipeline'].get('random_seed', 42))
        tx_history: List[Tuple[int, float]] = []
        
        q_glu_values: List[float] = []
        q_gaba_values: List[float] = []
        decision_stats: List[float] = []
        
        dt = cal_cfg['sim']['dt_s']
        detection_window_s = cal_cfg['detection'].get('decision_window_s', cal_cfg['pipeline']['symbol_period_s'])

        sigma_glu, sigma_gaba = calculate_proper_noise_sigma(cal_cfg, detection_window_s)
        
        for s_tx in tx_symbols:
            ig, ia, ic, Nm_actual = _single_symbol_currents(s_tx, tx_history, cal_cfg, rng)
            tx_history.append((s_tx, float(Nm_actual)))

            n_total_samples = len(ig)
            n_detect_samples = min(int(detection_window_s / dt), n_total_samples)

            if n_detect_samples <= 1: continue
            
            # Calculate charges (differential measurement)
            q_glu = float(np.trapezoid((ig - ic)[:n_detect_samples], dx=dt))
            q_gaba = float(np.trapezoid((ia - ic)[:n_detect_samples], dx=dt))
            
            q_glu_values.append(q_glu)
            q_gaba_values.append(q_gaba)

        # Collect appropriate statistics based on mode
        for q_glu, q_gaba in zip(q_glu_values, q_gaba_values):
            if mode == "MoSK":
                # MoSK uses the sum statistic
                D = q_glu / sigma_glu + q_gaba / sigma_gaba
                decision_stats.append(D)
            elif mode.startswith("CSK"):
                # CSK uses the raw charge from target channel
                target_channel = cal_cfg['pipeline'].get('csk_target_channel', 'GLU')
                Q = q_glu if target_channel == 'GLU' else q_gaba
                decision_stats.append(Q)
            elif mode == "Hybrid":
                # Hybrid uses different statistics for each stage
                mol_type = symbol >> 1
                Q = q_glu if mol_type == 0 else q_gaba
                decision_stats.append(Q)
        
        return {
            'q_values': decision_stats,
            'sigma_glu': sigma_glu,
            'sigma_gaba': sigma_gaba
        }
        
    except Exception as e:
        print(f"Calibration failed for symbol {symbol}: {e}")
        return None

# ============= SIMULATION FUNCTIONS =============
def run_single_instance(config: Dict[str, Any], seed: int) -> Optional[Dict[str, Any]]:
    """Run a single simulation instance."""
    try:
        cfg_run = deepcopy(config)
        cfg_run['pipeline']['random_seed'] = int(seed)
        
        mem_gb, total_gb, available_gb = check_memory_usage()
        if available_gb < 2.0:
            gc.collect()
        
        result = run_sequence(cfg_run)
        gc.collect()
        return result
    except MemoryError:
        print(f"❌ Memory error with seed {seed}")
        return None
    except Exception as e:
        print(f"❌ Simulation failed with seed {seed}: {e}")
        return None

# ============= SWEEP FUNCTIONS =============
def run_param_seed_combo(cfg_base: Dict[str, Any], param_name: str, 
                         param_value: Union[float, int], seed: int,
                         debug_calibration: bool = False) -> Optional[Dict[str, Any]]:
    """Worker function for parameter sweep with improved calibration."""
    try:
        cfg_run = deepcopy(cfg_base)
        cfg_run['disable_progress'] = True
        cfg_run['verbose'] = False
        
        # Set parameter
        if '.' in param_name:
            keys = param_name.split('.')
            target = cfg_run
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = param_value
        else:
            cfg_run[param_name] = param_value
        
        # Handle distance updates
        if param_name == 'pipeline.distance_um':
            new_symbol_period = calculate_dynamic_symbol_period(param_value, cfg_run)
            cfg_run['pipeline']['symbol_period_s'] = new_symbol_period
            cfg_run['pipeline']['time_window_s'] = new_symbol_period
            
            if cfg_run['pipeline'].get('enable_isi', False):
                D_glu = cfg_run['neurotransmitters']['GLU']['D_m2_s']
                lambda_glu = cfg_run['neurotransmitters']['GLU']['lambda']
                D_eff = D_glu / (lambda_glu ** 2)
                time_95 = 3.0 * ((param_value * 1e-6)**2) / D_eff
                guard_factor = cfg_run['pipeline'].get('guard_factor', 0.1)
                isi_memory = math.ceil((1 + guard_factor) * time_95 / new_symbol_period)
                cfg_run['pipeline']['isi_memory_symbols'] = isi_memory
        
        # FIXED: Always calibrate with verbose option for CSK debugging
        if cfg_run['pipeline']['modulation'] in ['MoSK', 'CSK', 'Hybrid']:
            if param_name in ['pipeline.Nm_per_symbol', 'pipeline.distance_um']:
                # Use more seeds for calibration
                cal_seeds = list(range(20))
                thresholds = calibrate_thresholds(cfg_run, cal_seeds, 
                                                 recalibrate=True, 
                                                 save_to_file=False,
                                                 verbose=debug_calibration)
                
                # Store thresholds in config
                for k, v in thresholds.items():
                    cfg_run['pipeline'][k] = v
                
                # Debug: Print thresholds for CSK
                if debug_calibration and cfg_run['pipeline']['modulation'] == 'CSK':
                    target_ch = cfg_run['pipeline'].get('csk_target_channel', 'GLU').lower()
                    key = f'csk_thresholds_{target_ch}'
                    if key in cfg_run['pipeline']:
                        print(f"\n[DEBUG] CSK Thresholds for Nm={param_value}: {cfg_run['pipeline'][key]}")
        
        result = run_single_instance(cfg_run, seed)
        
        if result:
            result['param_name'] = param_name
            result['param_value'] = param_value
            result['seed'] = seed
        
        return result
        
    except Exception as e:
        print(f"Error in param_seed_combo: {e}")
        return None

def run_sweep(cfg: Dict[str, Any], seeds: List[int], sweep_param: str, 
              sweep_values: List[float], sweep_name: str,
              debug_calibration: bool = False) -> pd.DataFrame:
    """Run parameter sweep with parallelization."""
    pool = global_pool.get_pool()
    
    all_combinations = [(sweep_param, v, s) for v in sweep_values for s in seeds]
    
    print(f"🚀 SWEEP: {len(all_combinations)} jobs ({len(sweep_values)} values × {len(seeds)} seeds)")
    
    results = []
    future_to_combo = {
        pool.submit(run_param_seed_combo, cfg, p, v, s, debug_calibration): (p, v, s) 
        for p, v, s in all_combinations
    }
    
    for future in tqdm(as_completed(future_to_combo), total=len(all_combinations), desc=sweep_name):
        try:
            result = future.result(timeout=300)
            if result:
                results.append(result)
        except:
            pass
    
    # Process results with debug info for CSK
    df_data = []
    for value in sweep_values:
        value_results = [r for r in results if r['param_value'] == value]
        
        if not value_results:
            continue
        
        total_symbols = len(value_results) * cfg['pipeline']['sequence_length']
        total_errors = sum(r['errors'] for r in value_results)
        ser = total_errors / total_symbols if total_symbols > 0 else 1.0
        
        # Collect statistics
        all_stats_glu = []
        all_stats_gaba = []
        for r in value_results:
            all_stats_glu.extend(r.get('stats_glu', []))
            all_stats_gaba.extend(r.get('stats_gaba', []))
        
        snr = calculate_snr_from_stats(all_stats_glu, all_stats_gaba) if all_stats_glu else 0
        
        row = {
            sweep_param: value,
            'ser': ser,
            'snr_db': snr,
            'num_runs': len(value_results)
        }
        
        # Debug info for CSK
        if cfg['pipeline']['modulation'] == 'CSK' and debug_calibration:
            print(f"CSK @ {sweep_param}={value}: SER={ser:.4f}, SNR={snr:.2f}dB, runs={len(value_results)}")
        
        if cfg['pipeline']['modulation'] == 'Hybrid':
            mosk_errors = sum(r.get('subsymbol_errors', {}).get('mosk', 0) for r in value_results)
            csk_errors = sum(r.get('subsymbol_errors', {}).get('csk', 0) for r in value_results)
            row['mosk_ser'] = mosk_errors / total_symbols
            row['csk_ser'] = csk_errors / total_symbols
        
        df_data.append(row)
    
    return pd.DataFrame(df_data)

# ============= LOD SEARCH FUNCTIONS =============
def find_lod_for_ser(cfg_base: Dict[str, Any], seeds: List[int], 
                     target_ser: float = 0.01,
                     debug_calibration: bool = False) -> Tuple[Union[int, float], float]:
    """Binary search for LoD with improved calibration."""
    nm_min = cfg_base['pipeline'].get('lod_nm_min', 50)
    nm_max = 100000
    lod_nm: float = np.nan
    best_ser: float = 1.0
    
    dist_um = cfg_base['pipeline'].get('distance_um', 0)
    
    for iteration in range(14):
        if nm_min > nm_max:
            break
        
        nm_mid = int((nm_min + nm_max) / 2)
        if nm_mid == 0 or nm_mid > nm_max:
            break
        
        print(f"    [{dist_um}μm] Testing Nm={nm_mid} (iteration {iteration+1}/14, range: {nm_min}-{nm_max})")
        
        cfg_test = deepcopy(cfg_base)
        cfg_test['pipeline']['Nm_per_symbol'] = nm_mid
        
        # FIXED: Use proper calibration with more samples
        cal_seeds = list(range(20))
        thresholds = calibrate_thresholds(cfg_test, cal_seeds, 
                                         recalibrate=True, 
                                         save_to_file=False,
                                         verbose=debug_calibration and iteration == 0)
        for k, v in thresholds.items():
            cfg_test['pipeline'][k] = v
        
        # Run simulations
        results = []
        for i, seed in enumerate(seeds):
            if i % 3 == 0:
                print(f"      [{dist_um}μm] Nm={nm_mid}: seed {i+1}/{len(seeds)}")
            
            cfg_run = deepcopy(cfg_test)
            cfg_run['pipeline']['random_seed'] = seed
            cfg_run['disable_progress'] = True
            cfg_run['verbose'] = False
            try:
                result = run_sequence(cfg_run)
                if result:
                    results.append(result)
            except:
                pass
        
        if not results:
            nm_min = nm_mid + 1
            continue
        
        total_symbols = len(results) * cfg_test['pipeline']['sequence_length']
        total_errors = sum(r['errors'] for r in results)
        ser = total_errors / total_symbols if total_symbols > 0 else 1.0
        
        print(f"      [{dist_um}μm] Nm={nm_mid}: SER={ser:.4f} {'✓ PASS' if ser <= target_ser else '✗ FAIL'}")
        
        if ser <= target_ser:
            lod_nm = nm_mid
            best_ser = ser
            nm_max = nm_mid - 1
        else:
            nm_min = nm_mid + 1
    
    # Final check
    if np.isnan(lod_nm) and nm_min <= 100000:
        print(f"    [{dist_um}μm] Final check at Nm={nm_min}")
        
        cfg_final = deepcopy(cfg_base)
        cfg_final['pipeline']['Nm_per_symbol'] = nm_min
        
        cal_seeds = list(range(20))
        thresholds = calibrate_thresholds(cfg_final, cal_seeds, 
                                         recalibrate=True, 
                                         save_to_file=False,
                                         verbose=False)
        for k, v in thresholds.items():
            cfg_final['pipeline'][k] = v
        
        results = []
        for seed in seeds:
            cfg_run = deepcopy(cfg_final)
            cfg_run['pipeline']['random_seed'] = seed
            cfg_run['disable_progress'] = True
            try:
                result = run_sequence(cfg_run)
                if result:
                    results.append(result)
            except:
                pass
        
        if results:
            total_symbols = len(results) * cfg_final['pipeline']['sequence_length']
            total_errors = sum(r['errors'] for r in results)
            final_ser = total_errors / total_symbols if total_symbols > 0 else 1.0
            if final_ser <= target_ser:
                return nm_min, final_ser
    
    return (int(lod_nm) if not np.isnan(lod_nm) else np.nan, best_ser)

def process_distance_for_lod(dist_um: float, cfg_base: Dict[str, Any], 
                             seeds: List[int], target_ser: float = 0.01,
                             debug_calibration: bool = False) -> Dict[str, Any]:
    """Worker function for parallel LoD sweep with improved calibration."""
    try:
        symbol_period = calculate_dynamic_symbol_period(dist_um, cfg_base)
        est_time_per_symbol = symbol_period * 0.01
        est_time_per_sequence = est_time_per_symbol * cfg_base['pipeline']['sequence_length'] / 60
        est_total_time = est_time_per_sequence * len(seeds) * 5
        
        print(f"  Starting LoD search for {dist_um}μm...")
        print(f"    Symbol period: {symbol_period:.0f}s, Est. time: {est_total_time:.1f} min")
        
        start_time = time.time()
        
        cfg_dist = deepcopy(cfg_base)
        cfg_dist['pipeline']['distance_um'] = dist_um
        cfg_dist['pipeline']['symbol_period_s'] = symbol_period
        cfg_dist['pipeline']['time_window_s'] = symbol_period
        
        if cfg_dist['pipeline'].get('enable_isi', False):
            D_glu = cfg_dist['neurotransmitters']['GLU']['D_m2_s']
            lambda_glu = cfg_dist['neurotransmitters']['GLU']['lambda']
            D_eff = D_glu / (lambda_glu ** 2)
            time_95 = 3.0 * ((dist_um * 1e-6)**2) / D_eff
            guard_factor = cfg_dist['pipeline'].get('guard_factor', 0.1)
            isi_memory = math.ceil((1 + guard_factor) * time_95 / symbol_period)
            cfg_dist['pipeline']['isi_memory_symbols'] = isi_memory
        
        lod_nm, ser_at_lod = find_lod_for_ser(cfg_dist, seeds, target_ser, debug_calibration)
        
        mode = cfg_base['pipeline']['modulation']
        if mode == "MoSK":
            bits_per_symbol = 1
        elif mode.startswith("CSK"):
            M = cfg_base['pipeline']['csk_levels']
            bits_per_symbol = np.log2(M)
        else:  # Hybrid
            bits_per_symbol = 2
        
        data_rate = (bits_per_symbol / symbol_period) * (1 - ser_at_lod)
        
        elapsed = time.time() - start_time
        print(f"  ✓ COMPLETED {dist_um}μm in {elapsed/60:.1f} min: LoD={lod_nm} molecules, SER={ser_at_lod:.4f}")
        
        return {
            'distance_um': dist_um,
            'lod_nm': lod_nm,
            'ser_at_lod': ser_at_lod,
            'data_rate_bps': data_rate,
            'symbol_period_s': symbol_period
        }
        
    except Exception as e:
        print(f"  ✗ Failed {dist_um}μm: {e}")
        return {
            'distance_um': dist_um,
            'lod_nm': np.nan,
            'ser_at_lod': 1.0,
            'data_rate_bps': 0,
            'symbol_period_s': 0
        }

# ============= PLOTTING FUNCTIONS =============
def plot_ser_vs_nm(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    """Plot SER vs Nm."""
    plt.figure(figsize=(10, 6))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    for mode, df in results_dict.items():
        if 'pipeline.Nm_per_symbol' in df.columns and 'ser' in df.columns:
            plt.loglog(df['pipeline.Nm_per_symbol'], df['ser'],
                      color=colors.get(mode, 'black'),
                      marker=markers.get(mode, 'o'),
                      markersize=8, label=mode, linewidth=2)
    
    plt.xlabel('Number of Molecules per Symbol (Nm)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title('SER vs. Nm for All Modulation Schemes')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.ylim(1e-4, 1)
    plt.xlim(1e2, 1e5)
    plt.axhline(y=0.01, color='k', linestyle=':', alpha=0.5, label='Target SER = 1%')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_lod_vs_distance(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    """Plot LoD vs Distance."""
    plt.figure(figsize=(10, 6))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    for mode, df in results_dict.items():
        if 'distance_um' in df.columns and 'lod_nm' in df.columns:
            df_valid = df.dropna(subset=['lod_nm'])
            plt.semilogy(df_valid['distance_um'], df_valid['lod_nm'],
                        color=colors.get(mode, 'black'),
                        marker=markers.get(mode, 'o'),
                        markersize=8, label=mode, linewidth=2)
    
    plt.xlabel('Distance (μm)')
    plt.ylabel('Limit of Detection (molecules)')
    plt.title('LoD vs. Distance')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ============= MAIN FUNCTION =============
def main() -> None:
    """Main execution with fixed CSK calibration."""
    args = parse_arguments()
    
    # Determine worker mode
    if args.max_workers is None:
        if args.extreme_mode:
            mode = "extreme"
        elif args.beast_mode:
            mode = "beast"
        else:
            mode = "optimal"
        args.max_workers = get_optimal_workers(mode)
        print(f"🔥 Using {mode.upper()} mode: {args.max_workers} workers")
    
    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZED ANALYSIS - {args.mode} Mode")
    print(f"{'='*60}")
    print(f"CPU: {CPU_COUNT} threads ({PHYSICAL_CORES} cores)")
    print(f"Workers: {args.max_workers}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Debug calibration: {args.debug_calibration}")
    
    check_memory_usage()
    
    # Setup directories
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    data_dir = results_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(project_root / "config" / "default.yaml") as f:
        config_base = yaml.safe_load(f)
    
    cfg = preprocess_config_full(config_base)
    cfg['pipeline']['enable_isi'] = False if args.disable_isi else cfg['pipeline'].get('enable_isi', False)
    cfg['pipeline']['modulation'] = args.mode
    cfg['pipeline']['sequence_length'] = args.sequence_length
    cfg['verbose'] = args.verbose
    
    if args.mode.startswith("CSK"):
        cfg['pipeline']['csk_levels'] = 4  # Use 4-level CSK
        cfg['pipeline']['csk_target_channel'] = 'GLU'
        print(f"CSK Configuration: {cfg['pipeline']['csk_levels']} levels, {cfg['pipeline']['csk_target_channel']} channel")
    
    # Generate seeds
    ss = np.random.SeedSequence(2026)
    seeds = [int(s) for s in ss.generate_state(args.num_seeds)]
    
    # Initialize global pool
    global_pool.get_pool(args.max_workers)
    
    start_time = time.time()
    
    try:
        print(f"\n{'='*60}")
        print("Running Performance Sweeps")
        print(f"{'='*60}")
        
        print(f"\nConfiguration:")
        print(f"  GLU diffusion: {cfg['neurotransmitters']['GLU']['D_m2_s']:.2e} m²/s")
        print(f"  GABA diffusion: {cfg['neurotransmitters']['GABA']['D_m2_s']:.2e} m²/s")
        print(f"  ISI enabled: {cfg['pipeline'].get('enable_isi', False)}")
        
        # ============= SWEEP 1: SER vs Nm =============
        print("\n1. Running SER vs. Nm sweep...")
        nm_values = [2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]
        
        default_distance = cfg['pipeline'].get('distance_um', 100)
        symbol_period = calculate_dynamic_symbol_period(default_distance, cfg)
        cfg['pipeline']['symbol_period_s'] = symbol_period
        cfg['pipeline']['time_window_s'] = symbol_period
        
        # FIXED: Run initial calibration for CSK/Hybrid modes
        if args.mode in ['CSK', 'Hybrid']:
            print(f"\n📊 Initial calibration for {args.mode} mode...")
            cal_seeds = list(range(20))
            initial_thresholds = calibrate_thresholds(cfg, cal_seeds, 
                                                     recalibrate=True, 
                                                     save_to_file=False,
                                                     verbose=args.debug_calibration)
            print(f"✅ Calibration complete")
            if args.debug_calibration:
                for k, v in initial_thresholds.items():
                    print(f"  {k}: {v}")
        
        df_ser_nm = run_sweep(
            cfg, seeds,
            'pipeline.Nm_per_symbol',
            nm_values,
            f"SER vs Nm ({args.mode})",
            debug_calibration=args.debug_calibration
        )
        
        csv_path = data_dir / f"ser_vs_nm_{args.mode.lower()}.csv"
        df_ser_nm.to_csv(csv_path, index=False)
        print(f"✅ Results saved to {csv_path}")
        
        # Print results summary
        print(f"\nSER vs Nm Results for {args.mode}:")
        print(df_ser_nm[['pipeline.Nm_per_symbol', 'ser', 'snr_db']].to_string(index=False))
        
        # ============= SWEEP 2: LoD vs Distance =============
        print("\n2. Running LoD vs. Distance sweep...")
        distances = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        
        pool = global_pool.get_pool()
        
        print(f"   Submitting {len(distances)} distance searches to pool...")
        future_to_dist = {
            pool.submit(process_distance_for_lod, dist, cfg, seeds[:10], 0.01, args.debug_calibration): dist
            for dist in distances
        }
        
        print(f"   All jobs submitted. Processing in parallel...\n")
        
        lod_results = []
        completed = 0
        
        for future in as_completed(future_to_dist):
            try:
                result = future.result(timeout=7200)
                lod_results.append(result)
                completed += 1
                
                dist = future_to_dist[future]
                if not np.isnan(result['lod_nm']):
                    print(f"  [{completed}/{len(distances)}] Distance {dist}μm complete: "
                          f"LoD={result['lod_nm']:.0f} molecules")
                else:
                    print(f"  [{completed}/{len(distances)}] Distance {dist}μm: No LoD found")
                    
            except TimeoutError:
                dist = future_to_dist[future]
                print(f"  ✗ Timeout for distance {dist}μm")
                lod_results.append({
                    'distance_um': dist,
                    'lod_nm': np.nan,
                    'ser_at_lod': 1.0,
                    'data_rate_bps': 0,
                    'symbol_period_s': 0
                })
            except Exception as e:
                dist = future_to_dist[future]
                print(f"  ✗ Error for distance {dist}μm: {e}")
        
        lod_results.sort(key=lambda x: x['distance_um'])
        
        df_lod = pd.DataFrame(lod_results)
        csv_path = data_dir / f"lod_vs_distance_{args.mode.lower()}.csv"
        df_lod.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to {csv_path}")
        
        # Print LoD results summary
        print(f"\nLoD vs Distance Results for {args.mode}:")
        print(df_lod[['distance_um', 'lod_nm', 'ser_at_lod']].to_string(index=False))
        
        # ============= TIMING REPORT =============
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"   Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        if elapsed > 0:
            speedup = 24 * 3600 / elapsed
            print(f"   Estimated speedup: ~{speedup:.1f}x vs serial")
        print(f"{'='*60}")
        
    finally:
        global_pool.shutdown()

if __name__ == "__main__":
    if platform.system() == "Windows":
        mp.freeze_support()
    main()