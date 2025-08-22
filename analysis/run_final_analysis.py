# analysis/run_final_analysis.py
# RUN FINAL ANALYSIS (Crash-safe resume, ISI fields in CSVs, progress UI, mypy fixes)

from __future__ import annotations

from html import parser
import sys
import json
import argparse
import math
from pathlib import Path
import typing as t
import numpy as np
import matplotlib.pyplot as plt
import yaml
from copy import deepcopy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp
import psutil
from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict, cast, Callable
import gc
import os
import platform
import time
import typing
import hashlib
import threading
import queue as pyqueue
import signal
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent if (Path(__file__).parent.name == "analysis") else Path(__file__).parent
sys.path.append(str(project_root))

# Local modules
try:
    from src.mc_detection.algorithms import calculate_ml_threshold
except ImportError:
    from src.detection import calculate_ml_threshold
from src.pipeline import run_sequence, calculate_proper_noise_sigma, _single_symbol_currents
from src.config_utils import preprocess_config
from src.constants import get_nt_params

# Progress UI
from analysis.ui_progress import ProgressManager
from analysis.log_utils import setup_tee_logging

# ============= TYPE DEFINITIONS =============
class CPUConfig(TypedDict):
    p_cores_physical: List[int]
    p_cores_logical: List[int]
    e_cores_logical: List[int]
    p_core_count: int
    total_p_threads: int
    
class SleepInhibitor:
    """Cross-platform sleep inhibition for long-running simulations."""
    def __init__(self):
        self._proc = None
    def __enter__(self):
        try:
            sysname = platform.system()
            if sysname == "Windows":
                import ctypes
                ES_CONTINUOUS=0x80000000; ES_SYSTEM_REQUIRED=0x00000001; ES_AWAYMODE_REQUIRED=0x00000040; ES_DISPLAY_REQUIRED=0x00000002
                
                # Try with away mode first (preferred for background work)
                flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
                if getattr(self, '_keep_display_on', False):
                    flags |= ES_DISPLAY_REQUIRED
                
                result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
                if result == 0:
                    print("⚠️  Away mode failed, trying fallback without ES_AWAYMODE_REQUIRED...")
                    # Fallback: remove away mode requirement
                    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                    if getattr(self, '_keep_display_on', False):
                        flags |= ES_DISPLAY_REQUIRED
                    result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
                    
                    if result == 0:
                        print("⚠️  Sleep inhibition failed even with fallback flags")
                    else:
                        print("🛡️  Sleep inhibited (Windows fallback mode - no away mode).")
                else:
                    print("🛡️  Sleep inhibited (Windows with away mode).")
                    
            elif sysname == "Darwin":
                self._proc = subprocess.Popen(
                    ["/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())])
                print("🛡️  Sleep inhibited via caffeinate (macOS).")
            else:
                # systemd-inhibit blocks sleep while this child lives
                self._proc = subprocess.Popen(
                    ["systemd-inhibit", "--what=idle:sleep", "--why=MCvD run",
                     "sleep", "infinity"])
                print("🛡️  Sleep inhibited via systemd-inhibit (Linux).")
        except Exception as e:
            print(f"⚠️  Could not inhibit sleep: {e}")
        return self
        
    def __exit__(self, *exc):
        try:
            if platform.system() == "Windows":
                import ctypes
                result = ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS
                if result == 0:
                    print("⚠️  Could not restore Windows power state")
        except Exception:
            pass
        try:
            if self._proc:
                self._proc.terminate()
        except Exception:
            pass

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
    if not CPU_CONFIG:
        if mode == "extreme":
            return min(CPU_COUNT, 32)
        elif mode == "beast":
            return max(1, min(CPU_COUNT - 2, 28))
        else:
            return max(1, min(CPU_COUNT - 4, 24))
    p_threads = CPU_CONFIG["total_p_threads"]
    if mode == "extreme":
        return p_threads
    elif mode == "beast":
        return max(p_threads - 2, 1)
    else:
        return max(p_threads - 4, 1)

def worker_init():
    if CPU_CONFIG is not None:
        try:
            process = psutil.Process()
            process.cpu_affinity(CPU_CONFIG["p_cores_logical"])
        except Exception:
            pass

# ============= PERSISTENT PROCESS POOL =============
class GlobalProcessPool:
    _instance: Optional["GlobalProcessPool"] = None
    _pool: Optional[ProcessPoolExecutor] = None
    _max_workers: Optional[int] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_pool(self, max_workers: Optional[int] = None, mode: str = "optimal") -> ProcessPoolExecutor:
        # If a pool already exists and caller did not request a change, keep it as-is.
        if self._pool is not None and max_workers is None:
            return t.cast(ProcessPoolExecutor, self._pool)
        # Otherwise resolve desired size; prefer the current size if present.
        resolved_workers = max_workers or (self._max_workers or get_optimal_workers(mode))
        if self._pool is None or self._max_workers != resolved_workers:
            if self._pool:
                self._pool.shutdown(wait=True)
            self._max_workers = resolved_workers
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=worker_init
            )
            print(f"🚀 Global process pool initialized with {self._max_workers} workers")
        return t.cast(ProcessPoolExecutor, self._pool)
    
    def cancel_pending(self):
        """Cancel futures that haven't started running yet."""
        if self._pool:
            try:
                # Python 3.9+: cancel queued futures
                self._pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Older Python: best-effort
                self._pool.shutdown(wait=False)
            print("🛑 Pending futures cancelled")
            
    def force_kill(self):
        """Terminate running worker processes (last resort)."""
        if not self._pool:
            return
        procs = list(getattr(self._pool, "_processes", {}).values())  # private but practical
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        self._pool = None
        print("💥 Worker processes terminated")

    def shutdown(self):
        if self._pool:
            self._pool.shutdown(wait=True)
            self._pool = None
            print("✅ Global process pool shut down")

global_pool = GlobalProcessPool()

# ---- Cancellation: first ^C = graceful, second ^C = hard kill
CANCEL = threading.Event()

def _install_signal_handlers():
    state = {"count": 0}
    def _on_sig(_signum, _frame):
        state["count"] += 1
        if state["count"] == 1:
            print("\n^C — cancelling pending work (press Ctrl+C again to abort immediately).", flush=True)
            CANCEL.set()
            try:
                global_pool.cancel_pending()
            except Exception:
                pass
        else:
            print("\n^C again — force terminating workers.", flush=True)
            try:
                global_pool.force_kill()
            finally:
                os._exit(130)
    
    for sig in (getattr(signal, "SIGINT", None),
                getattr(signal, "SIGTERM", None),
                getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            try:
                signal.signal(sig, _on_sig)
            except Exception:
                pass

# ============= CALIBRATION CACHE =============
# Maximum calibration cache entries before cleanup (prevent memory bloat)
MAX_CACHE_SIZE = 50

calibration_cache: Dict[str, Dict[str, Union[float, List[float]]]] = {}

def get_cache_key(cfg: Dict[str, Any]) -> str:
    def _nt_pair_fp(cfg: Dict[str, Any]) -> str:
        # compact hash of the GLU/GABA parameter tuples so cache respects pair identity
        def pick(name: str):
            nt = cfg['neurotransmitters'][name]
            return (
                float(nt.get('k_on_M_s', 0.0)),
                float(nt.get('k_off_s', 0.0)),
                float(nt.get('q_eff_e', 0.0)),
                float(nt.get('D_m2_s', 0.0)),
                float(nt.get('lambda', 1.0)),
            )
        raw = repr((pick('GLU'), pick('GABA'))).encode()
        return hashlib.sha1(raw).hexdigest()[:8]

    key_params = [
        cfg['pipeline'].get('modulation'),
        cfg['pipeline'].get('Nm_per_symbol'),
        cfg['pipeline'].get('distance_um'),
        cfg['pipeline'].get('symbol_period_s'),
        cfg['pipeline'].get('csk_levels'),
        cfg['pipeline'].get('csk_target_channel'),
        cfg['pipeline'].get('csk_level_scheme', 'uniform'),
        cfg['pipeline'].get('guard_factor', 0.0),
        bool(cfg['pipeline'].get('use_control_channel', True)),  # NEW
        _nt_pair_fp(cfg),  # NEW: bind cache to active NT pair
    ]
    return str(hash(tuple(str(p) for p in key_params)))

def _thresholds_filename(cfg: Dict[str, Any]) -> Path:
    """
    Build a filename that captures sweep-dependent parameters so we can safely reuse across runs.
    """
    def _nt_pair_fingerprint(cfg):
        """Generate a stable, unique fingerprint for NT pair based on physical parameters."""
        def pick(nt):
            return (
                float(nt.get('k_on_M_s', 0.0)),
                float(nt.get('k_off_s', 0.0)),
                float(nt.get('q_eff_e', 0.0)),
                float(nt.get('D_m2_s', 0.0)),
                float(nt.get('lambda', 1.0)),
            )
        g = cfg['neurotransmitters']['GLU']
        b = cfg['neurotransmitters']['GABA']
        raw = repr((pick(g), pick(b))).encode()
        return hashlib.sha1(raw).hexdigest()[:8]
    
    # add a short label to disambiguate pairs on disk too
    def _nt_pair_label(cfg):
        g = cfg['neurotransmitters']['GLU']; b = cfg['neurotransmitters']['GABA']
        # fall back to hashed label if 'name' is absent
        name_g = str(g.get('name', 'GLU')); name_b = str(b.get('name', 'GABA'))
        base = f"{name_g}-{name_b}".lower().replace(' ', '')
        return base

    results_dir = project_root / "results" / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    mode = str(cfg['pipeline'].get('modulation', 'unknown')).lower()
    Ts = cfg['pipeline'].get('symbol_period_s', None)
    dist = cfg['pipeline'].get('distance_um', None)
    nm = cfg['pipeline'].get('Nm_per_symbol', None)
    lvl = cfg['pipeline'].get('csk_level_scheme', 'uniform')
    tgt = cfg['pipeline'].get('csk_target_channel', '')
    gf  = cfg['pipeline'].get('guard_factor', None)
    parts = [f"thresholds_{mode}", _nt_pair_label(cfg)]
    parts.append(f"pair{_nt_pair_fingerprint(cfg)}")  # NEW: always disambiguate
    if Ts is not None:   parts.append(f"Ts{Ts:.3g}")
    if dist is not None: parts.append(f"d{dist:.0f}um")
    if nm is not None:   parts.append(f"Nm{nm:.0f}")
    if tgt:              parts.append(f"tgt{tgt}")
    if lvl:              parts.append(f"lvl{lvl}")
    if gf is not None:   parts.append(f"gf{gf:.3g}")
    # Add before the return statement:
    ctrl = 'wctrl' if bool(cfg['pipeline'].get('use_control_channel', True)) else 'noctrl'
    parts.append(ctrl)
    return results_dir / ( "_".join(parts) + ".json" )

def calibrate_thresholds(cfg: Dict[str, Any], seeds: List[int], recalibrate: bool = False,
                         save_to_file: bool = True, verbose: bool = False) -> Dict[str, Union[float, List[float]]]:
    """
    Calibration with ISI off & window=Ts. Returns thresholds dict.
    """
    mode = cfg['pipeline']['modulation']
    threshold_file = _thresholds_filename(cfg)

    if threshold_file.exists() and not recalibrate and save_to_file:
        try:
            with open(threshold_file, 'r') as f:
                loaded_thresholds = json.load(f)
                return {k: v for k, v in loaded_thresholds.items()}
        except Exception:
            pass  # fall through to re-calculate

    cal_cfg = deepcopy(cfg)
    cal_cfg['pipeline']['sequence_length'] = 100
    cal_cfg['pipeline']['enable_isi'] = False
    if 'symbol_period_s' in cal_cfg['pipeline']:
        cal_cfg['detection']['decision_window_s'] = cal_cfg['pipeline']['symbol_period_s']

    thresholds: Dict[str, Union[float, List[float]]] = {}

    if mode == "MoSK" or mode == "Hybrid":
        mosk_stats: Dict[str, List[float]] = {'glu': [], 'gaba': []}
        if mode == "MoSK":
            symbols_to_check = {0: 'glu', 1: 'gaba'}
        else:
            symbols_to_check = {0: 'glu', 1: 'glu', 2: 'gaba', 3: 'gaba'}
        cal_seeds = seeds[:10] if len(seeds) >= 10 else seeds
        for symbol, type_key in symbols_to_check.items():
            for seed in cal_seeds:
                cal_cfg['pipeline']['random_seed'] = seed
                result = run_calibration_symbols(cal_cfg, symbol, mode='MoSK' if mode == "MoSK" else mode)
                if result:
                    mosk_stats[type_key].extend(result['q_values'])
        if all(mosk_stats[k] for k in mosk_stats):
            mean_D_glu = float(np.mean(mosk_stats['glu'])); std_D_glu = max(float(np.std(mosk_stats['glu'])), 1e-15)
            mean_D_gaba = float(np.mean(mosk_stats['gaba'])); std_D_gaba = max(float(np.std(mosk_stats['gaba'])), 1e-15)
            threshold_mosk = calculate_ml_threshold(mean_D_glu, mean_D_gaba, std_D_glu, std_D_gaba)
            thresholds['mosk_threshold'] = threshold_mosk

    if mode.startswith("CSK"):
        M = cfg['pipeline']['csk_levels']
        target_channel = cfg['pipeline'].get('csk_target_channel', 'GLU')
        level_scheme = cfg['pipeline'].get('csk_level_scheme', 'uniform')
        level_stats: Dict[int, List[float]] = {level: [] for level in range(M)}
        cal_seeds = seeds[:10] if len(seeds) >= 10 else seeds
        for level in range(M):
            for seed in cal_seeds:
                cal_cfg['pipeline']['random_seed'] = seed
                result = run_calibration_symbols(cal_cfg, level, mode='CSK', num_symbols=100)
                if result:
                    level_stats[level].extend(result['q_values'])
        threshold_list: List[float] = []
        for i in range(M - 1):
            if level_stats[i] and level_stats[i + 1]:
                m0, s0 = float(np.mean(level_stats[i])), max(float(np.std(level_stats[i])), 1e-15)
                m1, s1 = float(np.mean(level_stats[i+1])), max(float(np.std(level_stats[i+1])), 1e-15)
                threshold_list.append(calculate_ml_threshold(m0, m1, s0, s1))
        q_eff = get_nt_params(cfg, target_channel)['q_eff_e']
        threshold_list.sort(reverse=(q_eff < 0))
        thresholds[f'csk_thresholds_{target_channel.lower()}'] = threshold_list

    if mode == "Hybrid":
        stats: Dict[str, List[float]] = {'glu_low': [], 'glu_high': [], 'gaba_low': [], 'gaba_high': []}
        symbol_to_type = {0: 'glu_low', 1: 'glu_high', 2: 'gaba_low', 3: 'gaba_high'}
        cal_seeds = seeds[:30] if len(seeds) >= 30 else seeds
        for symbol in range(4):
            for seed in cal_seeds:
                cal_cfg['pipeline']['random_seed'] = seed
                result = run_calibration_symbols(cal_cfg, symbol, mode='Hybrid')
                if result:
                    stats[symbol_to_type[symbol]].extend(result['q_values'])
        if all(stats[k] for k in stats):
            m_gl, s_gl = float(np.mean(stats['glu_low'])), max(float(np.std(stats['glu_low'])), 1e-15)
            m_gh, s_gh = float(np.mean(stats['glu_high'])), max(float(np.std(stats['glu_high'])), 1e-15)
            m_bl, s_bl = float(np.mean(stats['gaba_low'])), max(float(np.std(stats['gaba_low'])), 1e-15)
            m_bh, s_bh = float(np.mean(stats['gaba_high'])), max(float(np.std(stats['gaba_high'])), 1e-15)
            thresholds['hybrid_threshold_glu'] = calculate_ml_threshold(m_gl, m_gh, s_gl, s_gh)
            thresholds['hybrid_threshold_gaba'] = calculate_ml_threshold(m_bl, m_bh, s_bl, s_bh)

    if save_to_file:
        try:
            tmp = threshold_file.with_suffix(threshold_file.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(thresholds, f, indent=2)
            os.replace(tmp, threshold_file)
        except Exception:
            # Best-effort; calibration can be recomputed if needed
            try:
                tmp = threshold_file.with_suffix(threshold_file.suffix + ".tmp")
                if tmp.exists(): 
                    tmp.unlink()
            except Exception:
                pass

    return thresholds

def calibrate_thresholds_cached(cfg: Dict[str, Any], seeds: List[int]) -> Dict[str, Union[float, List[float]]]:
    """
    Memory + disk cached calibration. Persist JSON so multiple processes/runs reuse it.
    """
    cache_key = get_cache_key(cfg)
    if cache_key in calibration_cache:
        return calibration_cache[cache_key]
    # IMPORTANT: persist to file (save_to_file=True) and do not force recalibration here
    result = calibrate_thresholds(cfg, seeds, recalibrate=False, save_to_file=True, verbose=False)
    
    # Bound cache size to prevent memory bloat during long sweeps
    if len(calibration_cache) >= MAX_CACHE_SIZE:
        # Remove oldest entries (FIFO eviction)
        oldest_keys = list(calibration_cache.keys())[:len(calibration_cache) - MAX_CACHE_SIZE + 1]
        for old_key in oldest_keys:
            del calibration_cache[old_key]
    
    calibration_cache[cache_key] = result
    return result

# ============= HELPERS =============
def check_memory_usage():
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
            'K_d_Hz': cfg.get('K_d_Hz', 1.3e-4),
            'rho_correlated': cfg.get('rho_correlated', 0.9)
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
    if 'detection' not in cfg:
        cfg['detection'] = {}
    return cfg

def calculate_dynamic_symbol_period(distance_um: float, cfg: Dict[str, Any]) -> float:
    D_glu = cfg['neurotransmitters']['GLU']['D_m2_s']
    lambda_glu = cfg['neurotransmitters']['GLU']['lambda']
    D_eff = D_glu / (lambda_glu ** 2)
    time_95 = 3.0 * ((distance_um * 1e-6)**2) / D_eff
    guard_factor = cfg['pipeline'].get('guard_factor', 0.1)
    guard_time = guard_factor * time_95
    symbol_period = max(20.0, round(time_95 + guard_time))
    return symbol_period

def calculate_snr_from_stats(stats_a: List[float], stats_b: List[float]) -> float:
    if not stats_a or not stats_b:
        return 0.0
    mu_a = np.mean(stats_a)
    mu_b = np.mean(stats_b)
    var_a = np.var(stats_a)
    var_b = np.var(stats_b)
    denom = (var_a + var_b)
    if denom <= 0:
        return float('inf')
    return float((mu_a - mu_b)**2 / denom)

def estimate_isi_overlap_ratio(cfg: Dict[str, Any]) -> float:
    """
    Cheap, physics‑guided ISI tail proxy.
    """
    if not cfg['pipeline'].get('enable_isi', False):
        return 0.0
    d_um = float(cfg['pipeline']['distance_um'])
    Ts = float(cfg['pipeline'].get('symbol_period_s', 1.0))
    D = float(cfg['neurotransmitters']['GLU']['D_m2_s'])
    lam = float(cfg['neurotransmitters']['GLU']['lambda'])
    D_eff = D / (lam ** 2)
    time_95 = 3.0 * ((d_um * 1e-6) ** 2) / D_eff
    if time_95 <= 0:
        return 0.0
    tail = max(0.0, time_95 - Ts)
    return float(min(1.0, tail / time_95))

# ============= CSV PERSIST HELPERS (atomic-ish) =============
def _atomic_write_csv(csv_path: Path, df: pd.DataFrame) -> None:
    """
    Atomic CSV write using temp file pattern (like _atomic_write_json).
    """
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)

def _with_file_lock(path: Path, fn: Callable[[], None], timeout_s: float = 60.0) -> None:
    """Very small cross-platform lock via .lock sentinel file."""
    lock = path.with_suffix(path.suffix + ".lock")
    t0 = time.time()
    while True:
        try:
            # O_CREAT|O_EXCL -> exclusive create
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                fn()
            finally:
                try:
                    lock.unlink(missing_ok=True)
                except Exception:
                    pass
            return
        except FileExistsError:
            if (time.time() - t0) > timeout_s:
                # Improved: Force cleanup stale lock and retry once more
                print(f"⚠️  CSV lock timeout ({timeout_s}s) for {path.name}, attempting cleanup...")
                try:
                    lock.unlink(missing_ok=True)  # Remove potentially stale lock
                    time.sleep(0.1)  # Brief pause
                    # Try once more with the cleaned lock
                    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    try:
                        fn()
                    finally:
                        try:
                            lock.unlink(missing_ok=True)
                        except Exception:
                            pass
                    return
                except Exception:
                    # Last resort: warn and proceed without lock
                    print(f"⚠️  CSV lock cleanup failed for {path.name}, proceeding without lock protection")
                    return fn()
            time.sleep(0.1)

def append_row_atomic(csv_path: Path, row: Dict[str, Any], columns: Optional[List[str]] = None) -> None:
    """
    Append a row with a lock + atomic rename. Read *and* write occur under the lock
    to prevent lost updates when multiple processes append concurrently.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if columns is not None:
        new_row = new_row.reindex(columns=columns)

    def _read_and_write():
        if csv_path.exists():
            try:
                existing = pd.read_csv(csv_path)
                combined = pd.concat([existing, new_row], ignore_index=True)
            except Exception:
                combined = new_row
        else:
            combined = new_row
        _atomic_write_csv(csv_path, combined)

    _with_file_lock(csv_path, _read_and_write)

def load_completed_values(csv_path: Path, key: str, use_ctrl: Optional[bool] = None) -> set:
    """Return the set of completed sweep values for resume, optionally filtered by CTRL state."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        
        # Filter by CTRL state if specified
        if use_ctrl is not None and 'use_ctrl' in df.columns:
            df = df[df['use_ctrl'] == use_ctrl]
        
        for cand in (key, 'pipeline_Nm_per_symbol', 'pipeline.Nm_per_symbol'):
            if cand in df.columns:
                vals = pd.to_numeric(df[cand], errors='coerce').dropna().tolist()
                # Normalize to canonical string keys (avoids float equality pitfalls)
                return set(_value_key(v) for v in vals)
    except Exception:
        pass
    return set()

def _value_key(v):
    """
    Canonicalize numeric values to consistent string representation.
    Examples:
    - 200, 200.0, 2e2 all become "200"
    - 200.5 becomes "200.5"
    - Non-numeric values fall back to str(v)
    """
    try:
        vf = float(v)
        return str(int(vf)) if vf.is_integer() else f"{vf:.6g}"
    except Exception:
        return str(v)

# --- Stage 13 helpers: confidence intervals / screening ---
def _wilson_halfwidth(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("inf")
    p = k / n
    den = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / den
    half = z * math.sqrt((p*(1.0 - p)/n) + (z*z)/(4*n*n)) / den
    return half

def _hoeffding_bounds(k: int, n: int, delta: float = 1e-4) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    eps = math.sqrt((math.log(1.0/delta)) / (2.0 * n))
    return max(0.0, p - eps), min(1.0, p + eps)

def _deterministic_screen(k: int, n_done: int, n_total_planned: int, target: float) -> tuple[bool, bool]:
    """
    Deterministic early-screen: even if all remaining are correct (best-case) or all wrong (worst-case),
    can we still cross/not-cross target?
      return (decide_below, decide_above)
    """
    if n_total_planned <= 0:
        return (False, False)
    # Best-case final SER if all remaining are correct
    p_best = k / n_total_planned
    # Worst-case final SER if all remaining are wrong
    p_worst = (k + (n_total_planned - n_done)) / n_total_planned
    return (p_worst < target, p_best > target)

def seed_cache_path(mode: str, sweep: str, value: Union[float, int], seed: int, 
                    use_ctrl: Optional[bool] = None, cache_tag: Optional[str] = None) -> Path:
    vk = _value_key(value)
    ctrl_seg = "wctrl" if use_ctrl else "noctrl" if use_ctrl is not None else "ctrl_unspecified"
    base = project_root / "results" / "cache" / mode.lower()
    if cache_tag:
        base = base / cache_tag
    return base / f"{sweep}_{ctrl_seg}" / vk / f"seed_{seed}.json"

def read_seed_cache(mode: str, sweep: str, value: Union[float, int], seed: int, 
                    use_ctrl: Optional[bool] = None, cache_tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
    p = seed_cache_path(mode, sweep, value, seed, use_ctrl, cache_tag)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            # Ensure the seed tag is present even for older cache files
            data.setdefault("__seed", int(seed))
            return data
        except Exception:
            return None
    return None

def write_seed_cache(mode: str, sweep: str, value: Union[float, int], seed: int, payload: Dict[str, Any], 
                     use_ctrl: Optional[bool] = None, cache_tag: Optional[str] = None) -> None:
    cache_path = seed_cache_path(mode, sweep, value, seed, use_ctrl, cache_tag)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    # Tag the payload with its seed for later de‑duplication in aggregations
    payload = dict(payload)
    payload.setdefault("__seed", int(seed))
    with open(tmp, "w", encoding='utf-8') as f:
        json.dump(_json_safe(payload), f, indent=2)
    os.replace(tmp, cache_path)

def _lod_state_path(mode: str, dist_um: float, use_ctrl: bool) -> Path:
    ctrl_seg = "wctrl" if use_ctrl else "noctrl"
    base = project_root / "results" / "cache" / mode.lower() / "lod_state" / ctrl_seg
    base.mkdir(parents=True, exist_ok=True)
    return base / f"d{int(dist_um)}um_state.json"

def _lod_state_load(mode: str, dist_um: float, use_ctrl: bool) -> Optional[Dict[str, Any]]:
    p = _lod_state_path(mode, dist_um, use_ctrl)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _lod_state_save(mode: str, dist_um: float, use_ctrl: bool, state: Dict[str, Any]) -> None:
    p = _lod_state_path(mode, dist_um, use_ctrl)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, p)

# ============= ARGUMENTS =============
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run molecular communication analysis (crash-safe w/ resume)")
    # NEW: support both --mode and --modes (orchestrator calls --modes)
    parser.add_argument("--mode", choices=["MoSK", "CSK", "Hybrid", "ALL"], default=None)
    parser.add_argument("--modes", choices=["MoSK", "CSK", "Hybrid", "all"], default=None)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=1000)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--beast-mode", action="store_true")
    parser.add_argument("--extreme-mode", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable-isi", action="store_true", help="Disable ISI (runtime). Default is enabled.")
    parser.add_argument("--debug-calibration", action="store_true", help="Print detailed calibration information")
    parser.add_argument("--csk-level-scheme", choices=["uniform", "zero-based"], default="uniform",
                       help="CSK level mapping scheme")
    parser.add_argument("--resume", action="store_true", help="Resume: skip finished values and append results as we go")
    parser.add_argument("--with-ctrl", dest="use_ctrl", action="store_true", help="Use CTRL differential subtraction")
    parser.add_argument("--no-ctrl", dest="use_ctrl", action="store_false", help="Disable CTRL subtraction (ablation)")
    parser.add_argument("--progress", choices=["tqdm", "rich", "gui", "none"], default="tqdm",
                    help="Progress UI backend")
    parser.add_argument("--nt-pairs", type=str, default="", help="CSV nt-pairs for CSK sweeps")
    parser.add_argument("--watchdog-secs", type=int, default=1800,
                        help="Soft timeout for seed completion before retry hint (default: 1800s/30min)") 
    parser.add_argument("--target-ci", type=float, default=0.0,
                        help="If >0, stop adding seeds once Wilson 95% CI half-width <= target. 0 disables.")
    parser.add_argument("--min-ci-seeds", type=int, default=6,
                        help="Minimum seeds required before adaptive CI stopping can trigger.")
    parser.add_argument("--lod-screen-delta", type=float, default=1e-4,
                        help="Hoeffding screening significance (delta) for early-stop LoD tests.")
    parser.add_argument("--distances",
                        type=str,
                        default="",
                        help="Comma-separated distance grid in µm for LoD (e.g., 25,35,45,55,65,75,85,95,105,115,125,150,175,200)")
    parser.add_argument("--lod-num-seeds",
                        type=str,
                        default=None,
                        help=(
                            "LoD seed schedule. Options:\n"
                            "  N                 -> use fixed N seeds for LoD search\n"
                            "  min,max           -> linearly scale from min at 25µm to max at 200µm\n"
                            "  rules             -> e.g. '<=100:6,<=150:8,>150:10'\n"
                            "Final LoD validation always uses the full seed set."
                        ))
    parser.add_argument("--lod-seq-len",
                        type=int,
                        default=None,
                        help="If set, temporarily override sequence_length during LoD search only.")
    parser.add_argument("--parallel-modes", type=int, default=1,
                        help=">1 to run MoSK/CSK/Hybrid concurrently (e.g., 3).")
    parser.add_argument("--logdir", default=str((project_root / "results" / "logs")),
                        help="Directory for log files")
    parser.add_argument("--no-log", action="store_true", 
                        help="Disable file logging")
    parser.add_argument("--fsync-logs", action="store_true", 
                        help="Force fsync on each write")
    parser.add_argument("--inhibit-sleep", action="store_true",
                        help="Prevent the OS from sleeping while the pipeline runs")
    parser.add_argument("--keep-display-on", action="store_true",
                        help="Also keep the display awake (Windows/macOS)")
    parser.add_argument("--max-ts-for-lod", type=float, default=None,
                        help="If set, skip LoD at distances whose dynamic Ts exceeds this (seconds).")
    parser.add_argument("--max-lod-validation-seeds", type=int, default=None,
                        help="Cap the number of seeds used for LoD validation (default: use all seeds).")
    parser.add_argument("--max-symbol-duration-s", type=float, default=None,
                        help="Skip LoD search at distances where symbol period exceeds this limit (seconds).")
    parser.add_argument("--analytic-lod-bracket", action="store_true",
                    help="Use Gaussian SER approximation for tighter LoD bracketing (experimental).")
    parser.set_defaults(use_ctrl=True)
    args = parser.parse_args()
    
    # Auto-enable sleep inhibition when GUI is requested
    if args.progress == "gui":
        args.inhibit_sleep = True
        
    # Normalize: prefer --modes if provided
    if args.modes is not None:
        args.mode = "ALL" if args.modes.lower() == "all" else args.modes
    elif args.mode is None:
        args.mode = "MoSK"
    return args

# ============= CALIBRATION SAMPLES (helper) =============
def run_calibration_symbols(cfg: Dict[str, Any], symbol: int, mode: str, num_symbols: int = 100) -> Optional[Dict[str, Any]]:
    try:
        cal_cfg = deepcopy(cfg)
        cal_cfg['pipeline']['sequence_length'] = num_symbols
        cal_cfg['disable_progress'] = True
        tx_symbols = [symbol] * num_symbols
        q_glu_values: List[float] = []
        q_gaba_values: List[float] = []
        decision_stats: List[float] = []
        dt = cal_cfg['sim']['dt_s']
        detection_window_s = cal_cfg['detection'].get('decision_window_s', cal_cfg['pipeline']['symbol_period_s'])
        sigma_glu, sigma_gaba = calculate_proper_noise_sigma(cal_cfg, detection_window_s)
        for s_tx in tx_symbols:
            ig, ia, ic, Nm_actual = _single_symbol_currents(s_tx, [], cal_cfg, np.random.default_rng())
            n_total_samples = len(ig)
            n_detect_samples = min(int(detection_window_s / dt), n_total_samples)
            if n_detect_samples <= 1:
                continue
            use_ctrl = bool(cal_cfg['pipeline'].get('use_control_channel', True))
            q_glu = float(np.trapezoid((ig - ic)[:n_detect_samples], dx=dt) if use_ctrl else np.trapezoid(ig[:n_detect_samples], dx=dt))
            q_gaba = float(np.trapezoid((ia - ic)[:n_detect_samples], dx=dt) if use_ctrl else np.trapezoid(ia[:n_detect_samples], dx=dt))
            q_glu_values.append(q_glu); q_gaba_values.append(q_gaba)
        for q_glu, q_gaba in zip(q_glu_values, q_gaba_values):
            if mode == "MoSK":
                D = q_glu / sigma_glu + q_gaba / sigma_gaba
                decision_stats.append(D)
            elif mode.startswith("CSK"):
                target_channel = cal_cfg['pipeline'].get('csk_target_channel', 'GLU')
                Q = q_glu if target_channel == 'GLU' else q_gaba
                decision_stats.append(Q)
            elif mode == "Hybrid":
                mol_type = symbol >> 1
                Q = q_glu if mol_type == 0 else q_gaba
                decision_stats.append(Q)
        return {'q_values': decision_stats, 'sigma_glu': sigma_glu, 'sigma_gaba': sigma_gaba}
    except Exception:
        return None

# ============= RUNTIME WORKERS =============
def run_single_instance(config: Dict[str, Any], seed: int, attach_isi_meta: bool = True) -> Optional[Dict[str, Any]]:
    """Run a single sequence; returns None on failure (so callers must filter)."""
    try:
        cfg_run = deepcopy(config)
        cfg_run['pipeline']['random_seed'] = int(seed)
        mem_gb, total_gb, available_gb = check_memory_usage()
        if available_gb < 2.0:
            gc.collect()
        result = run_sequence(cfg_run)  # returns dict
        if attach_isi_meta:
            # attach light ISI metrics so we can aggregate
            result['isi_enabled'] = bool(cfg_run['pipeline'].get('enable_isi', False))
            result['isi_memory_symbols'] = int(cfg_run['pipeline'].get('isi_memory_symbols', 0)) if result['isi_enabled'] else 0
            result['symbol_period_s'] = float(cfg_run['pipeline'].get('symbol_period_s', np.nan))
            result['decision_window_s'] = float(cfg_run['detection'].get('decision_window_s', result['symbol_period_s']))
            result['isi_overlap_ratio'] = estimate_isi_overlap_ratio(cfg_run)
        gc.collect()
        return result
    except MemoryError:
        print(f"❌ Memory error with seed {seed}")
        return None
    except Exception as e:
        print(f"❌ Simulation failed with seed {seed}: {e}")
        return None

def run_param_seed_combo(cfg_base: Dict[str, Any], param_name: str,
                         param_value: Union[float, int], seed: int,
                         debug_calibration: bool = False,
                         thresholds_override: Optional[Dict[str, Union[float, List[float]]]] = None,
                         sweep_name: str = "ser_vs_nm", cache_tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Worker for parameter sweep with window match; accepts optional precomputed thresholds."""
    try:
        cfg_run = deepcopy(cfg_base)
        cfg_run['disable_progress'] = True
        cfg_run['verbose'] = False

        # Set sweep parameter
        if '.' in param_name:
            keys = param_name.split('.')
            target = cfg_run
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = param_value
        else:
            cfg_run[param_name] = param_value

        # Distance updates: recompute Ts and match window (+ ISI memory)
        if param_name == 'pipeline.distance_um':
            new_symbol_period = calculate_dynamic_symbol_period(cast(float, param_value), cfg_run)
            cfg_run['pipeline']['symbol_period_s'] = new_symbol_period
            cfg_run['pipeline']['time_window_s'] = new_symbol_period
            cfg_run['detection']['decision_window_s'] = new_symbol_period
            if cfg_run['pipeline'].get('enable_isi', False):
                D_glu = cfg_run['neurotransmitters']['GLU']['D_m2_s']
                lambda_glu = cfg_run['neurotransmitters']['GLU']['lambda']
                D_eff = D_glu / (lambda_glu ** 2)
                time_95 = 3.0 * ((cast(float, param_value) * 1e-6)**2) / D_eff
                guard_factor = cfg_run['pipeline'].get('guard_factor', 0.1)
                isi_memory = math.ceil((1 + guard_factor) * time_95 / new_symbol_period)
                cfg_run['pipeline']['isi_memory_symbols'] = isi_memory

        # Guard factor updates (ISI sweep): recompute Ts/window + thresholds
        if param_name == 'pipeline.guard_factor':
            # Update symbol period per the new guard factor, keeping distance fixed
            dist = float(cfg_run['pipeline']['distance_um'])
            new_symbol_period = calculate_dynamic_symbol_period(dist, cfg_run)
            cfg_run['pipeline']['symbol_period_s'] = new_symbol_period
            cfg_run['pipeline']['time_window_s'] = new_symbol_period
            cfg_run['detection']['decision_window_s'] = new_symbol_period
            if cfg_run['pipeline'].get('enable_isi', False):
                D_glu = cfg_run['neurotransmitters']['GLU']['D_m2_s']
                lambda_glu = cfg_run['neurotransmitters']['GLU']['lambda']
                D_eff = D_glu / (lambda_glu ** 2)
                time_95 = 3.0 * ((dist * 1e-6)**2) / D_eff
                guard_factor = float(param_value)
                isi_memory = math.ceil((1 + guard_factor) * time_95 / new_symbol_period)
                cfg_run['pipeline']['isi_memory_symbols'] = isi_memory

        # Thresholds: use override if supplied, else cached calibration
        if thresholds_override is not None:
            for k, v in thresholds_override.items():
                cfg_run['pipeline'][k] = v
        elif cfg_run['pipeline']['modulation'] in ['MoSK', 'CSK', 'Hybrid'] and \
             param_name in ['pipeline.Nm_per_symbol', 'pipeline.distance_um', 'pipeline.guard_factor']:
            cal_seeds = list(range(10))
            thresholds = calibrate_thresholds_cached(cfg_run, cal_seeds)
            for k, v in thresholds.items():
                cfg_run['pipeline'][k] = v
            if debug_calibration and cfg_run['pipeline']['modulation'] == 'CSK':
                target_ch = cfg_run['pipeline'].get('csk_target_channel', 'GLU').lower()
                key = f'csk_thresholds_{target_ch}'
                if key in cfg_run['pipeline']:
                    print(f"[DEBUG] CSK Thresholds @ {param_value}: {cfg_run['pipeline'][key]}")

        # Run the instance and attach per-run ISI metrics
        result = run_single_instance(cfg_run, seed, attach_isi_meta=True)
        if result is not None:
            # Tag the in‑memory result so mixed cached+fresh paths dedupe correctly
            try:
                result["__seed"] = int(seed)
            except Exception:
                pass
            mode = cfg_base['pipeline']['modulation']
            use_ctrl = bool(cfg_base['pipeline'].get('use_control_channel', True))
            result_safe = cast(Dict[str, Any], _json_safe(result))
            write_seed_cache(mode, sweep_name, param_value, seed, result_safe, use_ctrl, cache_tag)
        return result

    except Exception as e:
        print(f"Error in param_seed_combo: {e}")
        return None

# ============= SWEEPS (generic) =============
def run_sweep(cfg: Dict[str, Any],
              seeds: List[int],
              sweep_param: str,
              sweep_values: List[Union[float, int]],
              sweep_name: str,
              progress_mode: str,
              persist_csv: Optional[Path],
              resume: bool,
              debug_calibration: bool = False, 
              cache_tag: Optional[str] = None,
              pm: Optional[ProgressManager] = None,  # NEW: accept progress manager
              sweep_key: Optional[Any] = None,  # NEW: for nested updates
              parent_key: Optional[Any] = None) -> pd.DataFrame:  # NEW: for hierarchical updates
    """
    Parameter sweep with parallelization; returns aggregated df.
    Writes each completed value's row immediately if persist_csv is given.
    """
    pool = global_pool.get_pool()
    # Use provided progress manager or create new one
    local_pm = pm or ProgressManager(progress_mode)
    if not pm:  # only create session meta if we're creating our own PM
        session_meta = {"progress": progress_mode, "resume": False}
        local_pm = ProgressManager(progress_mode, gui_session_meta=session_meta)

    # Resume: filter out done values (by their column in persisted CSV)
    values_to_run = sweep_values
    if resume and persist_csv is not None and persist_csv.exists():
        key = 'pipeline_Nm_per_symbol' if sweep_param == 'pipeline.Nm_per_symbol' else sweep_param
        desired_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
        done = load_completed_values(persist_csv, key, desired_ctrl)
        values_to_run = [v for v in sweep_values if _value_key(v) not in done]
        if done:
            print(f"↩️  Resume: skipping {len(done)} already-completed values for use_ctrl={desired_ctrl}")

    # Pre-calibrate thresholds once per sweep value (hoisted from per-seed)
    thresholds_map: Dict[Union[float, int], Dict[str, Union[float, List[float]]]] = {}
    if cfg['pipeline']['modulation'] in ['MoSK', 'CSK', 'Hybrid'] and \
       sweep_param in ['pipeline.Nm_per_symbol', 'pipeline.distance_um', 'pipeline.guard_factor']:
        cal_seeds = list(range(10))
        for v in values_to_run:
            cfg_v = deepcopy(cfg)
            if '.' in sweep_param:
                keys = sweep_param.split('.')
                tgt = cfg_v
                for k in keys[:-1]:
                    tgt = tgt[k]
                tgt[keys[-1]] = v
            else:
                cfg_v[sweep_param] = v
            # keep detection window consistent with the parameter being swept
            if sweep_param == 'pipeline.distance_um':
                Ts = calculate_dynamic_symbol_period(cast(float, v), cfg_v)
                cfg_v['pipeline']['symbol_period_s'] = Ts
                cfg_v['pipeline']['time_window_s'] = Ts
                cfg_v['detection']['decision_window_s'] = Ts
            elif sweep_param == 'pipeline.guard_factor':
                dist = float(cfg_v['pipeline']['distance_um'])
                Ts = calculate_dynamic_symbol_period(dist, cfg_v)
                cfg_v['pipeline']['symbol_period_s'] = Ts
                cfg_v['pipeline']['time_window_s'] = Ts
                cfg_v['detection']['decision_window_s'] = Ts
            thresholds_map[v] = calibrate_thresholds_cached(cfg_v, cal_seeds)

    # Determine how many seed-jobs remain (for progress accounting)
    # --- NEW: resolve sweep folder early (for seed cache lookups) ---
    if "Nm_per_symbol" in sweep_param:
        sweep_folder = "ser_vs_nm"
    elif "distance_um" in sweep_param:
        sweep_folder = "lod_vs_distance"  # (only used for aggregated CSV; LoD search is handled elsewhere)
    elif "guard_factor" in sweep_param:
        sweep_folder = "isi_tradeoff"
    else:
        sweep_folder = "custom_sweep"

    mode_name = cfg['pipeline']['modulation']
    use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))

    # --- NEW: compute "planned total" and "already done" for prefill ---
    planned_total_jobs = len(sweep_values) * len(seeds)
    done_jobs = 0

    def _value_done_in_csv(val) -> bool:
        if persist_csv is None or not persist_csv.exists():
            return False
        try:
            df = pd.read_csv(persist_csv)
            if sweep_param == 'pipeline.Nm_per_symbol':
                csv_key = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in df.columns else 'pipeline.Nm_per_symbol'
            elif sweep_param == 'pipeline.guard_factor':
                csv_key = 'guard_factor' if 'guard_factor' in df.columns else 'pipeline.guard_factor'
            else:
                csv_key = sweep_param
            if csv_key not in df.columns:
                return False
            df2 = df
            if 'use_ctrl' in df2.columns:
                df2 = df2[df2['use_ctrl'] == use_ctrl]
            vals = pd.to_numeric(df2[csv_key], errors='coerce').dropna().tolist()
            return _value_key(val) in { _value_key(v) for v in vals }
        except Exception:
            return False

    # Count completed seed-jobs across ALL sweep values
    for v in sweep_values:
        if _value_done_in_csv(v):
            done_jobs += len(seeds)
        else:
            for s in seeds:
                if read_seed_cache(mode_name, sweep_folder, v, s, use_ctrl, cache_tag) is not None:
                    done_jobs += 1

    def _seed_done(v, s):
        return read_seed_cache(mode_name, sweep_folder, v, s, use_ctrl, cache_tag) is not None if resume else False

    # Original "total_jobs" only counts remaining (we keep it for batching logic)
    total_jobs_remaining = sum(1 for v in values_to_run for s in seeds if not _seed_done(v, s))

    # --- Create/bind the bar with the FULL total, and prefill ---
    display_name = f"{cfg['pipeline']['modulation']} | {sweep_name} ({'CTRL' if use_ctrl else 'NoCtrl'})"
    if sweep_key and pm:
        # Update totals for the already-created keyed row
        job_bar = pm.task(total=planned_total_jobs, description=display_name,
                        key=sweep_key, parent=parent_key, kind="sweep")
    else:
        job_bar = local_pm.task(total=planned_total_jobs, description=display_name)

    if done_jobs > 0:
        # Jump the bar to reflect completed work
        job_bar.update(done_jobs)

    # Collect rows
    aggregated_rows: List[Dict[str, Any]] = []

    for v in values_to_run:
        # Split cached vs missing seeds
        cached_results: List[Dict[str, Any]] = []
        seeds_to_run: List[int] = []
        if resume:
            for s in seeds:
                r = read_seed_cache(mode_name, sweep_folder, v, s, use_ctrl, cache_tag)
                if r is not None:
                    cached_results.append(r)
                else:
                    seeds_to_run.append(s)
        else:
            seeds_to_run = list(seeds)

        # Submit only missing seeds
        thresholds_override = thresholds_map.get(v)
        
        # Account instantly for cached seeds in the bar
        results: List[Dict[str, Any]] = list(cached_results)

        # Stage 13: adaptive seed batches
        target_ci = float(cfg.get("_stage13_target_ci", 0.0))
        min_ci_seeds = int(cfg.get("_stage13_min_ci_seeds", 6))
        seq_len = int(cfg['pipeline']['sequence_length'])
        
        # internal helper to compute current half-width
        def _current_halfwidth() -> float:
            n = len(results) * seq_len
            k = sum(int(r.get('errors', 0)) for r in results)
            return _wilson_halfwidth(k, n) if n > 0 else float('inf')
        
        # PRE-SUBMISSION CI CHECK: Skip launching futures if cached results already satisfy CI
        if target_ci > 0.0 and len(results) >= min_ci_seeds and len(seeds_to_run) > 0:
            if _current_halfwidth() <= target_ci:
                print(f"        ✓ Pre-submission CI satisfied: halfwidth {_current_halfwidth():.6f} ≤ {target_ci}")
                # Skip all future submissions for this value
                seeds_to_run = []
        
        # batch size: keep pool busy but allow early-stop
        max_workers = getattr(global_pool, "_max_workers", None) or os.cpu_count() or 4
        batch_size = max(1, min(len(seeds_to_run), max_workers))
        
        idx = 0
        pending: set = set()
        # NEW: stable worker-slot mapping (use a real queue to avoid duplicate assignment)
        from collections import deque
        slot_count = max(1, getattr(global_pool, "_max_workers", CPU_COUNT or 4))
        free_slots = deque(range(slot_count))
        fut_slot: Dict[Any, int] = {}
        fut_seed: Dict[Any, int] = {}  # NEW: Track which seed each future handles
        
        while (idx < len(seeds_to_run) or pending) and not CANCEL.is_set():
            # top-up
            while idx < len(seeds_to_run) and len(pending) < batch_size and not CANCEL.is_set():
                s = seeds_to_run[idx]
                fut = pool.submit(
                    run_param_seed_combo, cfg, sweep_param, v, s, debug_calibration, thresholds_override,
                    sweep_name=sweep_folder, cache_tag=cache_tag
                )
                pending.add(fut)
                fut_seed[fut] = s  # NEW: Track which seed this future handles
                
                # Assign a stable slot
                if pm:
                    if not free_slots:
                        # No idle UI slot available (more concurrency than slots); reuse slot 0 visually.
                        slot = 0
                    else:
                        slot = free_slots.popleft()
                    fut_slot[fut] = slot
                    if hasattr(pm, "worker_update"):
                        pm.worker_update(slot, f"{sweep_name} | {sweep_param}={v} | seed {s}")
                idx += 1
            # wait for one to finish
            if pending and not CANCEL.is_set():  # Only process if we have pending futures
                wd = int(cfg.get('_watchdog_secs', 600))  # NEW: Configurable timeout
                try:
                    done_fut = next(as_completed(pending, timeout=wd))
                    # capture seed before removing from map
                    sid = fut_seed.pop(done_fut, -1)
                    pending.remove(done_fut)
                    try:
                        res = done_fut.result(timeout=10)  # Quick result extraction
                        if res is not None:
                            # Ensure tag is present even if worker missed it
                            try:
                                res.setdefault("__seed", int(sid))
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"        ⚠️  Result extraction failed for {sweep_param}={v}: {e}")
                        res = None
                    if res is not None:
                        results.append(res)
                            
                    # Update worker status to idle (stable slot)
                    if pm:
                        slot = fut_slot.pop(done_fut, -1)  # Use -1 as default instead of None
                        if slot >= 0:  # Only update if we had a valid slot
                            if hasattr(pm, "worker_update"):
                                pm.worker_update(slot, "idle")
                            free_slots.append(slot)
                    
                    job_bar.update(1)
                except TimeoutError:
                    print(f"        ⚠️  Timeout ({wd}s) for {sweep_param}={v}, {len(pending)} futures pending")
                    # Pick a reproducible future to retry: the one with the smallest seed
                    if pending:
                        to_retry = min(pending, key=lambda f: fut_seed.get(f, 1<<31))
                        seed_r = fut_seed.get(to_retry, seeds_to_run[0] if seeds_to_run else 12345)  # FIX: Provide default seed
                        
                        # Enhanced logging for timeout retry tracking
                        print(f"        🔍 Timeout details: {sweep_param}={v}, seed={seed_r}, "
                              f"timeout={wd}s, pending_count={len(pending)}, worker_count={getattr(global_pool, '_max_workers', 'unknown')}")
                        
                        # Try to cancel the old future if it hasn't started
                        if to_retry.cancel():
                            print(f"        ✅ Cancelled stale future for seed {seed_r}")
                            pending.remove(to_retry)
                            fut_seed.pop(to_retry, None)
                        else:
                            print(f"        ⚠️  Could not cancel running future for seed {seed_r}")
                        
                        print(f"        🔄 Retrying seed {seed_r} for {sweep_param}={v}")
                        retry_tag = f"{cache_tag}_retry" if cache_tag else "retry"
                        retry_fut = pool.submit(run_param_seed_combo, cfg, sweep_param, v, seed_r,
                                              debug_calibration, thresholds_override,
                                              sweep_name=sweep_folder, cache_tag=retry_tag)
                        pending.add(retry_fut)
                        fut_seed[retry_fut] = seed_r  # NEW: Track retry future too
                    continue  # NEW: Skip to next iteration after timeout handling

                # adaptive early-stop when enough seeds and CI small enough
                if target_ci > 0.0 and len(results) >= min_ci_seeds:
                    if _current_halfwidth() <= target_ci:
                        print(f"        ✓ Early stop: CI halfwidth {_current_halfwidth():.6f} ≤ {target_ci}")
                        # Progress bar nicety: complete the bar when stopping early
                        remaining_not_submitted = len(seeds_to_run) - idx
                        still_pending = len(pending)
                        remaining = remaining_not_submitted + still_pending
                        if remaining > 0:
                            job_bar.update(remaining)
                        # Cancel remaining pending futures
                        for fut in pending:
                            fut.cancel()
                        pending.clear()
                        break
                    
        # If cancellation was requested, try to cancel any still-pending futures
        if CANCEL.is_set():
            global_pool.cancel_pending()  # Optional: immediate GUI feedback
            for f in list(pending):
                f.cancel()
            break  # stop processing more values

        if not results:
            continue

        # --- Aggregate across unique seeds (drop duplicate retries) ---
        if any("__seed" in r for r in results):
            by_seed: Dict[int, Dict[str, Any]] = {}
            for r in results:
                sid = int(r.get("__seed", -1))
                if sid not in by_seed:
                    by_seed[sid] = r
                else:
                    # Prefer the later write (e.g., retry) if you want; either is fine
                    by_seed[sid] = r
            results = list(by_seed.values())

        # Aggregate across seeds for this value
        total_symbols = len(results) * cfg['pipeline']['sequence_length']
        total_errors = sum(cast(int, r['errors']) for r in results)
        ser = total_errors / total_symbols if total_symbols > 0 else 1.0

        # pooled decision stats for SNR proxy
        all_a: List[float] = []
        all_b: List[float] = []
        for r in results:
            all_a.extend(cast(List[float], r.get('stats_glu', [])))
            all_b.extend(cast(List[float], r.get('stats_gaba', [])))
        snr_lin = calculate_snr_from_stats(all_a, all_b) if all_a and all_b else 0.0
        snr_db = (10.0 * float(np.log10(snr_lin))) if snr_lin > 0 else float('nan')

        # ISI context
        isi_enabled = any(bool(r.get('isi_enabled', False)) for r in results)
        isi_memory_symbols = int(np.nanmedian([float(r.get('isi_memory_symbols', np.nan)) for r in results if r is not None])) if isi_enabled else 0
        symbol_period_s = float(np.nanmedian([float(r.get('symbol_period_s', np.nan)) for r in results]))
        decision_window_s = float(np.nanmedian([float(r.get('decision_window_s', np.nan)) for r in results]))
        isi_overlap_mean = float(np.nanmean([float(r.get('isi_overlap_ratio', 0.0)) for r in results]))

        # Stage 14: aggregate noise sigmas across seeds
        ns_glu = [float(r.get('noise_sigma_glu', float('nan'))) for r in results]
        ns_gaba = [float(r.get('noise_sigma_gaba', float('nan'))) for r in results]
        ns_diff = [float(r.get('noise_sigma_I_diff', float('nan'))) for r in results]
        med_sigma_glu = float(np.nanmedian(ns_glu)) if any(np.isfinite(ns_glu)) else float('nan')
        med_sigma_gaba = float(np.nanmedian(ns_gaba)) if any(np.isfinite(ns_gaba)) else float('nan')
        med_sigma_diff = float(np.nanmedian(ns_diff)) if any(np.isfinite(ns_diff)) else float('nan')

        row: Dict[str, Any] = {
            sweep_param: v,
            'ser': ser,
            'snr_db': snr_db,
            'num_runs': len(results),
            'symbols_evaluated': int(total_symbols),  # NEW: for 95% CI calculations
            'sequence_length': int(cfg['pipeline']['sequence_length']),  # NEW: for 95% CI calculations
            'isi_enabled': isi_enabled,
            'isi_memory_symbols': isi_memory_symbols,
            'symbol_period_s': symbol_period_s,
            'decision_window_s': decision_window_s,
            'isi_overlap_ratio': isi_overlap_mean,
            'noise_sigma_glu': med_sigma_glu,           # NEW
            'noise_sigma_gaba': med_sigma_gaba,         # NEW
            'noise_sigma_I_diff': med_sigma_diff,       # NEW
            'use_ctrl': bool(cfg['pipeline'].get('use_control_channel', True)),  # persist CTRL flag
            'mode': cfg['pipeline']['modulation'],
        }
        # Duplicate Nm column name for plotting compatibility
        if sweep_param == 'pipeline.Nm_per_symbol':
            row['pipeline_Nm_per_symbol'] = v
        # Duplicate guard factor to a simple key for easier plotting
        if sweep_param == 'pipeline.guard_factor':
            row['guard_factor'] = float(v)

        if cfg['pipeline']['modulation'] == 'Hybrid':
            mosk_errors = sum(cast(int, r.get('subsymbol_errors', {}).get('mosk', 0)) for r in results)
            csk_errors = sum(cast(int, r.get('subsymbol_errors', {}).get('csk', 0)) for r in results)
            row['mosk_ser'] = mosk_errors / total_symbols
            row['csk_ser'] = csk_errors / total_symbols

        aggregated_rows.append(row)
        
        # Append this value's aggregated row immediately (crash‑safe)
        if persist_csv is not None:
            append_row_atomic(persist_csv, row, list(row.keys()))

    job_bar.close()
    if not pm:  # Only stop if we created our own local_pm
        local_pm.stop()
    # Don't stop pm if it was provided by caller

    return pd.DataFrame(aggregated_rows)

# ============= LOD SEARCH =============
def _analytic_lod_bracket(cfg_base: Dict[str, Any], seeds: List[int], target_ser: float = 0.01) -> Tuple[int, int]:
    """
    Gaussian SER approximation to bracket LoD. Safe + fast; no physics change.
    Returns (nm_min_guess, nm_max_guess) or (0, 0) if we can't estimate.
    """
    try:
        mode = cfg_base['pipeline']['modulation']
        if mode not in ['MoSK', 'CSK']:
            return (0, 0)

        def _Q(x: float) -> float:
            # Q(x) = 0.5 * erfc(x / sqrt(2))
            return 0.5 * math.erfc(x / math.sqrt(2.0))

        # short, ISI-off probes
        cal_cfg = deepcopy(cfg_base)
        cal_cfg['pipeline']['sequence_length'] = 40
        cal_cfg['pipeline']['enable_isi'] = False

        # Adaptive probes to better bracket the target SER
        probes = [1000, 10000]
        estimates: List[Tuple[int, float]] = []

        # Adaptive probe expansion (up to 3 iterations)
        for expansion in range(3):  # initial + 2 expansions max
            estimates.clear()
            
            for nm_probe in probes:
                cfg_p = deepcopy(cal_cfg)
                cfg_p['pipeline']['Nm_per_symbol'] = int(nm_probe)

                # thresholds at this Nm (few seeds, cached)
                th = calibrate_thresholds_cached(cfg_p, seeds[:3])
                for k, v in th.items():
                    cfg_p['pipeline'][k] = v

                if mode == 'MoSK':
                    stats_0, stats_1 = [], []
                    for sd in seeds[:3]:
                        r0 = run_calibration_symbols(cfg_p, 0, mode='MoSK', num_symbols=20)
                        r1 = run_calibration_symbols(cfg_p, 1, mode='MoSK', num_symbols=20)
                        if r0 and 'q_values' in r0: stats_0.extend(r0['q_values'])
                        if r1 and 'q_values' in r1: stats_1.extend(r1['q_values'])
                    
                    if len(stats_0) >= 5 and len(stats_1) >= 5:
                        mu0, mu1 = float(np.mean(stats_0)), float(np.mean(stats_1))
                        s0, s1 = max(1e-15, float(np.std(stats_0))), max(1e-15, float(np.std(stats_1)))
                        # MoSK threshold should be a single float value
                        threshold_raw = th.get('mosk_threshold', (mu0 + mu1) / 2.0)
                        if isinstance(threshold_raw, list):
                            threshold_val = float(threshold_raw[0]) if threshold_raw else (mu0 + mu1) / 2.0
                        else:
                            threshold_val = float(threshold_raw)
                        ser_est = 0.5 * (_Q((threshold_val - mu0) / s0) + _Q((mu1 - threshold_val) / s1))
                        estimates.append((nm_probe, max(1e-8, min(1.0, ser_est))))

                elif mode == 'CSK':
                    M = int(cfg_p['pipeline'].get('csk_levels', 4))
                    level_stats: List[List[float]] = [[] for _ in range(M)]
                    for sd in seeds[:3]:
                        for i in range(M):
                            r = run_calibration_symbols(cfg_p, i, mode='CSK', num_symbols=20)
                            if r and 'q_values' in r:
                                level_stats[i].extend(r['q_values'])
                    
                    means, stds = [], []
                    for i in range(M):
                        if len(level_stats[i]) < 5:
                            break
                        means.append(float(np.mean(level_stats[i])))
                        stds.append(max(1e-15, float(np.std(level_stats[i]))))
                    if len(means) == M:
                        target_ch = cfg_p['pipeline'].get('csk_target_channel', 'GLU').lower()
                        tau = th.get(f'csk_thresholds_{target_ch}', [])
                        tau = [float(x) for x in (tau if isinstance(tau, list) else [])]
                        if len(tau) == M - 1:
                            # SER ≈ average over classes of tail probabilities across adjacent thresholds
                            ser_sum = 0.0
                            for i in range(M):
                                # lower tail
                                if i > 0:
                                    ser_sum += 0.5 * math.erfc((tau[i-1] - means[i]) / (stds[i] * math.sqrt(2)))
                                # upper tail
                                if i < M - 1:
                                    ser_sum += 0.5 * math.erfc((means[i] - tau[i]) / (stds[i] * math.sqrt(2)))
                            ser_est = ser_sum / M
                            estimates.append((nm_probe, max(1e-8, min(1.0, ser_est))))

            # Check if we have good estimates and whether to expand probes
            if len(estimates) == 2:
                ser1, ser2 = estimates[0][1], estimates[1][1]
                # Check if both probes are far from target
                if ser1 > 10*target_ser and ser2 > 10*target_ser:
                    # Both too high - increase Nm (shift probes up)
                    probes = [p * 3 for p in probes]
                    print(f"    🔄 Analytic probes too high ({ser1:.1e}, {ser2:.1e} >> {target_ser:.1e}), expanding up: {probes}")
                    continue
                elif ser1 < 0.1*target_ser and ser2 < 0.1*target_ser:
                    # Both too low - decrease Nm (shift probes down)  
                    probes = [max(50, p // 3) for p in probes]
                    print(f"    🔄 Analytic probes too low ({ser1:.1e}, {ser2:.1e} << {target_ser:.1e}), expanding down: {probes}")
                    continue
                else:
                    # Good bracket - proceed with interpolation
                    print(f"    ✓ Good analytic bracket found: {probes} -> SER [{ser1:.1e}, {ser2:.1e}]")
                    break
            else:
                # Failed to get estimates - break and return (0,0)
                break

        # If we have good estimates, interpolate in log-space to find target
        if len(estimates) == 2 and estimates[0][1] != estimates[1][1]:
            nm1, ser1 = estimates[0]
            nm2, ser2 = estimates[1]
            
            # Interpolate in log-space for more stable results
            lnm1, lnm2 = math.log(nm1), math.log(nm2)
            lser1, lser2 = math.log(ser1), math.log(ser2)
            lser_t = math.log(target_ser)
            
            # Linear interpolation in log-space
            alpha = (lser_t - lser1) / (lser2 - lser1)
            lnm_t = lnm1 + alpha * (lnm2 - lnm1)
            nm_t = int(math.exp(lnm_t))
            
            # Conservative bracket: ±50% around interpolated point
            nm_min_est = max(50, int(nm_t * 0.5))
            nm_max_est = min(100000, int(nm_t * 1.5))
            
            print(f"    📊 Analytic interpolation: target SER {target_ser:.1e} → Nm ≈ {nm_t}, bracket [{nm_min_est}-{nm_max_est}]")
            return (nm_min_est, nm_max_est)
        else:
            print(f"    ⚠️  Analytic bracketing failed: insufficient data")
            return (0, 0)
            
    except Exception as e:
        print(f"    ⚠️  Analytic bracketing failed: {e}")
        return (0, 0)

def find_lod_for_ser(cfg_base: Dict[str, Any], seeds: List[int],
                     target_ser: float = 0.01,
                     debug_calibration: bool = False,
                     progress_cb: Optional[Any] = None,
                     resume: bool = False,
                     cache_tag: Optional[str] = None) -> Tuple[Union[int, float], float, int]:
    nm_min = cfg_base['pipeline'].get('lod_nm_min', 50)
    nm_max = 100000
    nm_max_default = nm_max  # Store original default

    # NEW: Try analytic bracketing if enabled (experimental feature)
    analytic_bracket_cache = None  # Cache for analytic bracket result
    if cfg_base.get('_analytic_lod_bracket', False):
        analytic_bracket_cache = _analytic_lod_bracket(cfg_base, seeds, target_ser)
        nm_min_analytic, nm_max_analytic = analytic_bracket_cache
        if nm_min_analytic > 0 and nm_max_analytic > nm_min_analytic:
            nm_min = max(nm_min, nm_min_analytic)
            nm_max = min(nm_max, nm_max_analytic)
            print(f"    📊 Using analytic bracket: [{nm_min} - {nm_max}]")

    # NEW: warm-start bracket if provided with analytic intersection
    warm = int(cfg_base.get('_warm_lod_guess', 0))
    if warm > 0:
        # Simplified nm_min calculation (removes redundant max)
        nm_min = max(nm_min, int(warm * 0.5))
        
        # Distance-aware upper bound: longer distances need wider brackets
        dist_um = cfg_base['pipeline'].get('distance_um', 0)
        if dist_um >= 175:
            mult = 10.0
        elif dist_um > 150:
            mult = 8.0
        elif dist_um > 100:
            mult = 4.0
        else:
            mult = 2.0
        
        # Intersect analytic and warm brackets when both available
        if cfg_base.get('_analytic_lod_bracket', False) and analytic_bracket_cache:
            # Reuse cached analytic bounds
            nm_min_analytic, nm_max_analytic = analytic_bracket_cache
            if nm_min_analytic > 0 and nm_max_analytic > nm_min_analytic:
                # Take intersection: analytic upper bound constrains warm expansion  
                upper_from_warm = int(mult * warm)
                upper_from_analytic = int(1.25 * nm_max_analytic)   # optional +25% safety
                
                # intersect "warm high" with (padded) analytic high, then cap to the global ceiling
                nm_max = min(max(nm_max_default, upper_from_warm), upper_from_analytic)
                nm_min = max(nm_min, nm_min_analytic)
                nm_max = min(nm_max, nm_max_default)  # enforce ≤ 100_000
                
                print(f"    🔄 Warm + analytic intersect: [{nm_min} - {nm_max}] (capped at 100k)")
        else:
            # Pure warm-start without analytic constraints
            nm_max = max(nm_max_default, int(mult * warm))
            print(f"    🔥 Warm-start bracket: [{nm_min} - {nm_max}]")
    
    lod_nm: float = float('nan')
    best_ser: float = 1.0
    dist_um = cfg_base['pipeline'].get('distance_um', 0)
    mode_name = cfg_base['pipeline']['modulation']
    use_ctrl = bool(cfg_base['pipeline'].get('use_control_channel', True))
    
    # Track actual progress increments
    progress_count = 0
    
    # Load prior state if resuming
    state = _lod_state_load(mode_name, float(dist_um), use_ctrl) if resume else None
    if state:
        nm_min = int(state.get("nm_min", nm_min))
        nm_max = int(state.get("nm_max", nm_max))
        print(f"    ↩️  Resuming LoD search @ {dist_um}μm: range {nm_min}-{nm_max}")
    
    # Extract CTRL state for debug logging
    ctrl_str = "CTRL" if use_ctrl else "NoCtrl"

    # NEW: Cache thresholds during bisection to reduce calibration overhead
    th_cache: Dict[int, Dict[str, Union[float, List[float]]]] = {}  # nm -> thresholds dict
    
    def _get_th(nm: int):
        if nm in th_cache:
            return th_cache[nm]
        cfg_tmp = deepcopy(cfg_base)
        cfg_tmp['pipeline']['Nm_per_symbol'] = nm
        th = calibrate_thresholds_cached(cfg_tmp, list(range(6)))  # faster with fewer seeds
        th_cache[nm] = th
        return th

    for iteration in range(20):
        if CANCEL.is_set():
            break
        if nm_min > nm_max:
            break
        nm_mid = int((nm_min + nm_max) / 2)
        if nm_mid == 0 or nm_mid > nm_max:
            break

        print(f"    [{dist_um}μm|{ctrl_str}] Testing Nm={nm_mid} (iteration {iteration+1}/20, range: {nm_min}-{nm_max})")

        cfg_test = deepcopy(cfg_base)
        cfg_test['pipeline']['Nm_per_symbol'] = nm_mid

        # Apply cached thresholds for the current test point
        for k, v in _get_th(nm_mid).items():
            cfg_test['pipeline'][k] = v

        # --- Gather cached seed results first (if any) ---
        results: List[Dict[str, Any]] = []
        k_err = 0
        n_seen = 0
        seq_len = int(cfg_test['pipeline']['sequence_length'])
        tested_nm = {}
        if state and "tested" in state and str(nm_mid) in state["tested"]:
            tested_nm = state["tested"][str(nm_mid)]
        
        # pull cached seeds from disk
        cached = []
        for sd in seeds:
            r = read_seed_cache(mode_name, "lod_search", nm_mid, sd, use_ctrl,
                                cache_tag=cache_tag)
            if r is not None:
                cached.append((sd, r))
        
        for sd, r in cached:
            results.append(r)
            k_err += int(r.get('errors', 0))
            n_seen += seq_len
            if progress_cb is not None:
                try: 
                    progress_cb.put_nowait(1)
                except Exception: 
                    pass
        
        # Fallback: if nothing was cached but we have a saved tally, reuse it
        if not cached and tested_nm:
            k_err = int(tested_nm.get("k_err", 0))
            n_seen = int(tested_nm.get("n_seen", 0))
        
        delta = float(cfg_base.get("_stage13_lod_delta", 1e-4))
        
        # loop remaining (non-cached) seeds with screening
        for i, seed in enumerate(seeds):
            # skip if already cached
            if any(sd == seed for sd, _ in cached):
                continue
            
            # reuse our generic worker to also persist the per-seed result
            res = run_param_seed_combo(cfg_test, 'pipeline.Nm_per_symbol', nm_mid, seed,
                                       debug_calibration=False,
                                       thresholds_override=_get_th(nm_mid),
                                       sweep_name="lod_search",
                                       cache_tag=cache_tag)
            if res is not None:
                results.append(res)
                k_err += int(res.get('errors', 0))
                n_seen += seq_len
                if progress_cb is not None:
                    try: 
                        progress_cb.put_nowait(1)
                    except Exception: 
                        pass
                
                # persist search checkpoint after each seed
                checkpoint = {
                    "nm_min": nm_min, "nm_max": nm_max, "iteration": iteration,
                    "last_nm": nm_mid,
                    "tested": {
                        **(state.get("tested", {}) if state else {}),
                        str(nm_mid): {"k_err": k_err, "n_seen": n_seen}
                    }
                }
                _lod_state_save(mode_name, float(dist_um), use_ctrl, checkpoint)
                
                # Deterministic screen: if even worst/best remaining cannot cross target
                decide_below, decide_above = _deterministic_screen(
                    k_err, n_seen, len(seeds)*seq_len, target_ser
                )
                if decide_below or decide_above:
                    break
                # Hoeffding bounds: if CI fully below/above target
                low, high = _hoeffding_bounds(k_err, n_seen, delta=delta)
                if high < target_ser or low > target_ser:
                    break
            # light heartbeat
            if (len(results) % 3) == 0:
                print(f"      [{dist_um}μm] Nm={nm_mid}: {len(results)}/{len(seeds)} seeds")

        # Compute SER using either real per-seed results or checkpoint tallies
        if results:
            total_symbols = len(results) * cfg_test['pipeline']['sequence_length']
            total_errors = sum(cast(int, r['errors']) for r in results)
            ser = total_errors / total_symbols if total_symbols > 0 else 1.0
        elif n_seen > 0:
            # Resume path: use previously saved (k_err, n_seen) for this Nm
            ser = k_err / n_seen
            print(f"      [{dist_um}μm|{ctrl_str}] Nm={nm_mid}: SER≈{ser:.4f} (checkpoint)")
        else:
            # No information collected for this Nm; move to higher Nm
            nm_min = nm_mid + 1
            continue

        print(f"      [{dist_um}μm|{ctrl_str}] Nm={nm_mid}: SER={ser:.4f} {'✓ PASS' if ser <= target_ser else '✗ FAIL'}")

        if ser <= target_ser:
            lod_nm = nm_mid
            best_ser = ser
            nm_max = nm_mid - 1
        else:
            nm_min = nm_mid + 1
            
        # Count each binary search iteration
        progress_count += 1
        
        # persist bounds after each iteration
        _lod_state_save(mode_name, float(dist_um), use_ctrl,
                        {"nm_min": nm_min, "nm_max": nm_max, "iteration": iteration,
                         "last_nm": nm_mid})

    # OPTIMIZATION 1: Cap LoD validation retries
    if math.isnan(lod_nm) and nm_min <= 100000:
        print(f"    [{dist_um}μm|{ctrl_str}] Final check at Nm={nm_min}")
        cfg_final = deepcopy(cfg_base)
        cfg_final['pipeline']['Nm_per_symbol'] = nm_min
        cal_seeds = list(range(10))
        thresholds = calibrate_thresholds(cfg_final, cal_seeds, recalibrate=False, save_to_file=True, verbose=False)
        for k, v in thresholds.items():
            cfg_final['pipeline'][k] = v

        # NEW: Cap validation seeds for performance
        max_validation_seeds = cfg_base.get('max_lod_validation_seeds', len(seeds))
        validation_seeds = seeds[:max_validation_seeds] if max_validation_seeds < len(seeds) else seeds
        if len(validation_seeds) < len(seeds):
            print(f"    [{dist_um}μm|{ctrl_str}] Validation capped at {max_validation_seeds}/{len(seeds)} seeds")

        results2: List[Dict[str, Any]] = []
        for i, sd in enumerate(validation_seeds):
            cfg_run2 = deepcopy(cfg_final)
            cfg_run2['pipeline']['random_seed'] = sd
            cfg_run2['disable_progress'] = True
            try:
                r2 = run_sequence(cfg_run2)
            except Exception:
                r2 = None
            if r2 is not None:
                results2.append(r2)
                progress_count += 1  # Count validation seeds
                # Progress callback for completed validation seed
                if progress_cb is not None:
                    try:
                        progress_cb.put(1)
                    except Exception:
                        pass
                
                # NEW: Early success detection for statistical efficiency
                if len(results2) >= 5:  # Minimum for statistical validity
                    interim_errors = sum(cast(int, r['errors']) for r in results2)
                    interim_symbols = len(results2) * cfg_final['pipeline']['sequence_length']
                    interim_ser = interim_errors / interim_symbols if interim_symbols > 0 else 1.0
                    if interim_ser <= target_ser * 0.8:  # Clear success margin
                        print(f"    [{dist_um}μm|{ctrl_str}] Early validation success after {len(results2)} seeds (SER={interim_ser:.4f})")
                        break

        if results2:
            total_symbols = len(results2) * cfg_final['pipeline']['sequence_length']
            total_errors = sum(cast(int, r['errors']) for r in results2)
            final_ser = total_errors / total_symbols if total_symbols > 0 else 1.0
            if final_ser <= target_ser:
                # Track actual progress - binary search iterations + final check seeds + overhead
                actual_progress = 20 + len(seeds) + 5  # max 20 iterations + validation seeds + overhead
                return nm_min, final_ser, actual_progress

    # Return actual count instead of constant
    return lod_nm, best_ser, progress_count

def _validate_lod_point_with_full_seeds(cfg_base: Dict[str, Any],
                                        lod_nm: int,
                                        full_seeds: List[int]) -> Tuple[float, float, float, float]:
    """
    Run one pass at the chosen LoD using the FULL seeds + FULL sequence_length
    to report paper-grade SER and data-rate with 95% CI.
    Returns: (ser_at_lod, data_rate_bps, ci_low, ci_high)
    """
    cfg = deepcopy(cfg_base)
    # Ensure symbol period and decision window are consistent for this distance
    Ts = calculate_dynamic_symbol_period(float(cfg['pipeline']['distance_um']), cfg)
    cfg['pipeline']['symbol_period_s'] = Ts
    cfg['pipeline']['time_window_s'] = Ts
    cfg.setdefault('detection', {})
    cfg['detection']['decision_window_s'] = Ts

    cfg['pipeline']['Nm_per_symbol'] = int(lod_nm)

    # Apply thresholds at this exact operating point
    th = calibrate_thresholds_cached(cfg, list(range(10)))
    for k, v in th.items():
        cfg['pipeline'][k] = v

    # Determine bits/symbol
    mode = cfg['pipeline']['modulation']
    if mode == 'MoSK':
        bpsym = 1.0
    elif mode == 'CSK':
        M = int(cfg['pipeline'].get('csk_levels', 4))
        bpsym = float(math.log2(max(M, 2)))
    else:
        bpsym = 2.0

    per_seed_rates, per_seed_ser = [], []
    Ts_list = []

    for seed in full_seeds:
        res = run_single_instance(cfg, seed, attach_isi_meta=True)
        if res is None:
            continue
        L = int(cfg['pipeline']['sequence_length'])
        e = res.get('errors', None)
        ser_seed = float(res.get('ser', res.get('SER', (e / L) if (e is not None and L > 0) else 1.0)))
        Ts_seed = float(res.get('symbol_period_s', Ts))
        per_seed_rates.append((bpsym / Ts_seed) * (1.0 - ser_seed))
        per_seed_ser.append(ser_seed)
        Ts_list.append(Ts_seed)

    if per_seed_rates:
        mean_rate = float(np.mean(per_seed_rates))
        std_rate = float(np.std(per_seed_rates, ddof=1)) if len(per_seed_rates) > 1 else 0.0
        n = max(1, len(per_seed_rates))
        try:
            from scipy.stats import t
            t_val = t.ppf(0.975, n - 1) if n > 1 else 1.96
        except Exception:
            t_val = 1.96
        ci_half = t_val * std_rate / math.sqrt(n)
        Ts_mean = float(np.mean(Ts_list)) if Ts_list else Ts
        return (float(np.mean(per_seed_ser)),
                mean_rate,
                max(0.0, mean_rate - ci_half),
                min(bpsym / Ts_mean, mean_rate + ci_half))
    else:
        # Conservative fallback
        Ts_mean = float(np.mean(Ts_list)) if Ts_list else Ts
        ser_fallback = 1.0
        rate = (bpsym / Ts_mean) * (1.0 - ser_fallback)
        return (ser_fallback, rate, rate, rate)

def process_distance_for_lod(dist_um: float, cfg_base: Dict[str, Any],
                             seeds: List[int], target_ser: float = 0.01,
                             debug_calibration: bool = False,
                             progress_cb: Optional[Any] = None,
                             resume: bool = False, args: Optional[Any] = None,
                             warm_lod_guess: Optional[int] = None) -> Dict[str, Any]:
    """
    Process a single distance for LoD calculation.
    Returns dict with lod_nm, ser_at_lod, and data_rate_bps.
    """
    actual_progress = 0  # Initialize before try block
    skipped_reason = None  # ✅ Initialize skipped_reason variable
    cfg = deepcopy(cfg_base)
    Ts_dyn = calculate_dynamic_symbol_period(dist_um, cfg)
    
    # NEW: Check for Ts explosion and skip if too large (OPTIMIZATION 2)
    max_symbol_duration_s = cfg_base.get('max_symbol_duration_s', None)
    if max_symbol_duration_s is not None and Ts_dyn > max_symbol_duration_s:
        print(f"⚠️  Skipping distance {dist_um}μm: symbol period {Ts_dyn:.1f}s exceeds limit {max_symbol_duration_s}s")
        return {
            'distance_um': dist_um,
            'lod_nm': float('nan'),
            'ser_at_lod': float('nan'),
            'data_rate_bps': 0.0,
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'symbol_period_s': Ts_dyn,
            'skipped_reason': f'Ts_explosion_{Ts_dyn:.1f}s'
        }
    
    # Continue with existing logic for args.max_ts_for_lod (keep this too)
    if args and getattr(args, "max_ts_for_lod", None) and Ts_dyn > float(args.max_ts_for_lod):
        return {
            'distance_um': dist_um, 'lod_nm': float('nan'), 'ser_at_lod': float('nan'),
            'data_rate_bps': float('nan'), 'data_rate_ci_low': float('nan'), 'data_rate_ci_high': float('nan'),
            'symbol_period_s': Ts_dyn, 'isi_enabled': bool(cfg['pipeline'].get('enable_isi', False)),
            'isi_memory_symbols': 0, 'decision_window_s': Ts_dyn, 'isi_overlap_ratio': estimate_isi_overlap_ratio(cfg),
            'use_ctrl': bool(cfg['pipeline'].get('use_control_channel', True)),
            'mode': cfg['pipeline']['modulation'], 'noise_sigma_I_diff': float('nan'),
            'actual_progress': 0
        }
    cfg['pipeline']['symbol_period_s'] = Ts_dyn
    
    # LoD search can use shorter sequences to bracket quickly
    if args and getattr(args, "lod_seq_len", None):
        cfg['pipeline']['sequence_length'] = args.lod_seq_len
    
    # NEW: Propagate warm-start guess
    if warm_lod_guess and warm_lod_guess > 0:
        cfg['_warm_lod_guess'] = int(warm_lod_guess)
    
    try:
        cache_tag = f"d{int(dist_um)}um"
        lod_nm, ser_at_lod, actual_progress = find_lod_for_ser(
            cfg, seeds, target_ser, debug_calibration, progress_cb,
            resume=resume, cache_tag=cache_tag
        )
        
        # Calculate data rate at LoD
        if not np.isnan(lod_nm):
            # Update config with LoD
            cfg['pipeline']['Nm_per_symbol'] = lod_nm
            
            # Ensure detection window matches Ts
            cfg['pipeline']['time_window_s'] = cfg['pipeline']['symbol_period_s']
            cfg.setdefault('detection', {})
            cfg['detection']['decision_window_s'] = cfg['pipeline']['symbol_period_s']

            # Ensure the same thresholds used during LoD search are active here
            cal_seeds = list(range(10))
            th = calibrate_thresholds_cached(cfg, cal_seeds)
            for k, v in th.items():
                cfg['pipeline'][k] = v
            
            # --- Calculate data rate at LoD using per-seed SER (recommended) ---
            per_seed_rates = []
            per_seed_ser = []
            per_seed_Ts = []

            # Bits per symbol by mode
            if cfg['pipeline']['modulation'] == 'MoSK':
                bits_per_symbol = 1.0
            elif cfg['pipeline']['modulation'] == 'CSK':
                M = int(cfg['pipeline'].get('csk_levels', 4))
                bits_per_symbol = float(math.log2(max(M, 2)))
            else:  # Hybrid
                bits_per_symbol = 2.0

            for seed in seeds:
                try:
                    res = run_single_instance(cfg, seed, attach_isi_meta=True)
                    if res is None:
                        continue

                    # Symbol period per seed (usually constant, but read if present)
                    Ts_seed = float(res.get('symbol_period_s', cfg['pipeline']['symbol_period_s']))

                    # SER per seed: prefer explicit keys, fall back to errors/sequence_length,
                    # and finally to ser_at_lod as a last resort.
                    L = int(cfg['pipeline']['sequence_length'])
                    e = res.get('errors', None)
                    ser_seed = res.get('ser', res.get('SER', (e / L) if (e is not None and L > 0) else ser_at_lod))
                    ser_seed = float(min(max(ser_seed, 0.0), 1.0))

                    per_seed_rates.append((bits_per_symbol / Ts_seed) * (1.0 - ser_seed))
                    per_seed_ser.append(ser_seed)
                    per_seed_Ts.append(Ts_seed)

                    if progress_cb is not None:
                        try: 
                            progress_cb.put(1)
                        except Exception: 
                            pass
                except Exception:
                    continue

            # Aggregate data rate with 95% t-CI (fallback to normal if SciPy missing)
            if per_seed_rates:
                mean_rate = float(np.mean(per_seed_rates))
                std_rate = float(np.std(per_seed_rates, ddof=1)) if len(per_seed_rates) > 1 else 0.0
                n = max(1, len(per_seed_rates))
                try:
                    from scipy.stats import t  # type: ignore
                    t_val = t.ppf(0.975, n - 1) if n > 1 else 1.96
                except Exception:
                    t_val = 1.96
                ci_half = t_val * std_rate / math.sqrt(n)

                data_rate_bps = mean_rate
                data_rate_ci_low = max(0.0, float(mean_rate - ci_half))  # Clamp to physical bounds
                # Optional: clamp upper bound to theoretical maximum
                Ts = float(cfg['pipeline']['symbol_period_s'])
                theoretical_max = float(bits_per_symbol / Ts)
                data_rate_ci_high = min(theoretical_max, float(mean_rate + ci_half))
            else:
                # Fallback (deterministic)
                Ts = float(cfg['pipeline']['symbol_period_s'])
                data_rate_bps = (bits_per_symbol / Ts) * (1.0 - ser_at_lod)
                data_rate_ci_low = data_rate_bps
                data_rate_ci_high = data_rate_bps
        else:
            data_rate_bps = float('nan')
            data_rate_ci_low = float('nan')
            data_rate_ci_high = float('nan')
    
    except Exception as e:
        print(f"Error processing distance {dist_um}: {e}")
        lod_nm = float('nan')
        ser_at_lod = float('nan')
        data_rate_bps = float('nan')
        data_rate_ci_low = float('nan')
        data_rate_ci_high = float('nan')
        skipped_reason = 'lod_search_failed'  # ✅ Set in exception block
    
    # NEW: Re-validate final LoD with full seeds for publication-grade statistics
    full_seeds = getattr(args, "full_seeds", seeds)  # default to reduced set if absent
    if not np.isnan(lod_nm) and lod_nm > 0:
        # Build a cfg with the chosen distance baked in (so Ts is recomputed inside helper)
        cfg_base_with_distance = deepcopy(cfg_base)
        cfg_base_with_distance['pipeline']['distance_um'] = dist_um
        ser_at_lod, data_rate_bps, data_rate_ci_low, data_rate_ci_high = \
            _validate_lod_point_with_full_seeds(cfg_base_with_distance, int(lod_nm), full_seeds)
        
        # NEW: Enforce 1% target after validation - adjust upward if needed
        target_ser = 0.01
        if ser_at_lod > target_ser:
            nm_try = int(max(lod_nm, 1))
            for _ in range(6):   # hard cap safety
                nm_try = int(math.ceil(nm_try * 1.25))
                ser2, rate2, lo2, hi2 = _validate_lod_point_with_full_seeds(cfg_base_with_distance, nm_try, full_seeds)
                if ser2 <= target_ser:
                    lod_nm, ser_at_lod = nm_try, ser2
                    data_rate_bps, data_rate_ci_low, data_rate_ci_high = rate2, lo2, hi2
                    break
    
    # NEW: Optional noise_sigma persistence aggregation
    if lod_nm > 0:
        # Re-simulate at the LoD point to collect noise_sigma_I_diff
        cfg_lod = deepcopy(cfg_base)
        cfg_lod['pipeline']['distance_um'] = dist_um
        # Recompute Ts and decision window for this distance
        Ts_lod = calculate_dynamic_symbol_period(dist_um, cfg_lod)
        cfg_lod['pipeline']['symbol_period_s'] = Ts_lod
        cfg_lod['pipeline']['time_window_s'] = Ts_lod
        cfg_lod.setdefault('detection', {})
        cfg_lod['detection']['decision_window_s'] = Ts_lod

        cfg_lod['pipeline']['Nm_per_symbol'] = lod_nm

        # Apply thresholds at this exact (distance, Ts, Nm)
        th_lod = calibrate_thresholds_cached(cfg_lod, list(range(4)))
        for k, v in th_lod.items():
            cfg_lod['pipeline'][k] = v
        
        sigma_values = []
        for seed in seeds[:5]:  # Limited seeds for efficiency
            result = run_param_seed_combo(cfg_lod, 'pipeline.Nm_per_symbol', lod_nm, seed, 
                                        debug_calibration=debug_calibration, 
                                        sweep_name="lod_validation", cache_tag="lod_sigma")
            if result and 'noise_sigma_I_diff' in result:
                sigma_values.append(result['noise_sigma_I_diff'])
            # Progress callback for completed noise sigma seed
            if progress_cb is not None:
                try: 
                    progress_cb.put(1)
                except Exception: 
                    pass
        
        lod_sigma_median = float(np.median(sigma_values)) if sigma_values else float('nan')
    else:
        lod_sigma_median = float('nan')

    # mark LoD state as done (clean checkpoint)
    try:
        done_state = {"done": True, "nm_min": lod_nm, "nm_max": lod_nm}
        _lod_state_save(cfg['pipeline']['modulation'], float(dist_um),
                        bool(cfg['pipeline'].get('use_control_channel', True)), done_state)
    except Exception:
        pass
    
    return {
        'distance_um': dist_um,
        'lod_nm': lod_nm,
        'ser_at_lod': ser_at_lod,
        'data_rate_bps': data_rate_bps,
        'data_rate_ci_low': data_rate_ci_low,
        'data_rate_ci_high': data_rate_ci_high,
        'symbol_period_s': cfg['pipeline']['symbol_period_s'],
        'isi_enabled': bool(cfg['pipeline'].get('enable_isi', False)),
        'isi_memory_symbols': int(cfg['pipeline'].get('isi_memory_symbols', 0)) if cfg['pipeline'].get('enable_isi', False) else 0,
        'decision_window_s': float(cfg.get('detection', {}).get(
            'decision_window_s', cfg['pipeline']['symbol_period_s'])),
        'isi_overlap_ratio': estimate_isi_overlap_ratio(cfg),
        'use_ctrl': bool(cfg['pipeline'].get('use_control_channel', True)),
        'mode': cfg['pipeline']['modulation'],
        'noise_sigma_I_diff': lod_sigma_median,
        'actual_progress': int(actual_progress),
        'skipped_reason': skipped_reason,  # ✅ Use the variable instead of None
    }

# ============= MAIN PLOTTING HELPERS (unchanged visuals) =============
def plot_ser_vs_nm(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    plt.figure(figsize=(10, 6))
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    for mode, df in results_dict.items():
        nm_col = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in df.columns else (
            'pipeline.Nm_per_symbol' if 'pipeline.Nm_per_symbol' in df.columns else None
        )
        if nm_col and 'ser' in df.columns:
            plt.loglog(df[nm_col], df['ser'],
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

def _canonical_nt_key(cfg: Dict[str, Any], name: str) -> Optional[str]:
    """
    Find canonical neurotransmitter key supporting common aliases.
    
    Args:
        cfg: Configuration dict containing neurotransmitters
        name: NT name or alias (case-insensitive)
    
    Returns:
        Canonical key if found, None otherwise
        
    Examples:
        'glu' -> 'GLU', 'acetylcholine' -> 'ACh', 'DA' -> 'DA'
    """
    nts = cfg.get('neurotransmitters', {})
    
    # Stage 1: Exact match (case-insensitive)
    for k in nts.keys():
        if k.lower() == name.lower():
            return k
    
    # Stage 2: Alias lookup
    aliases = {
        'ach': 'ACh', 'acetylcholine': 'ACh',
        'da': 'DA', 'dopamine': 'DA', 
        'glu': 'GLU', 'glutamate': 'GLU',
        'gaba': 'GABA',  # redundant but explicit
        # Add more as needed:
        'serotonin': '5HT', '5ht': '5HT',
        'norepinephrine': 'NE', 'ne': 'NE'
    }
    
    canonical = aliases.get(name.lower())
    return canonical if canonical and canonical in nts else None

def _apply_nt_pair(cfg: Dict[str, Any], first: str, second: str) -> Dict[str, Any]:
    """Swap underlying molecule dicts into GLU/GABA slots so the tri-channel interface stays stable."""
    cfg_new = deepcopy(cfg)
    nts = cfg.get('neurotransmitters', {})
    k1 = _canonical_nt_key(cfg, first)
    k2 = _canonical_nt_key(cfg, second)
    if k1 is None or k2 is None:
        nts = cfg.get('neurotransmitters', {})
        available = list(nts.keys())
        aliases = ['ach', 'acetylcholine', 'da', 'dopamine', 'glu', 'glutamate', 'gaba', 'serotonin', '5ht', 'norepinephrine', 'ne']
        raise ValueError(f"Unknown neurotransmitter key(s): {first}, {second}. "
                        f"Available: {available}. "
                        f"Supported aliases: {aliases}")

    cfg_new['neurotransmitters']['GLU'] = dict(nts[k1])   # first
    cfg_new['neurotransmitters']['GABA'] = dict(nts[k2])  # second
    if cfg_new['pipeline'].get('modulation') == 'CSK':
        cfg_new['pipeline']['csk_target_channel'] = 'GLU'  # measure 'first'
    return cfg_new

def run_csk_nt_pair_sweeps(args, cfg_base: Dict[str, Any], seeds: List[int], nm_values: List[float]) -> None:
    pairs_arg = (args.nt_pairs or "").strip()
    if not pairs_arg:
        return
    data_dir = project_root / "results" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pairs = [p.strip() for p in pairs_arg.split(",") if p.strip()]
    print(f"\n🔁 NT-pair sweeps ({len(pairs)}): {pairs}")
    for pair in pairs:
        if "-" not in pair:
            print(f"  • Skipping malformed pair '{pair}' (expected FIRST-SECOND)")
            continue
        first, second = [s.strip() for s in pair.split("-", 1)]
        try:
            cfg_pair = _apply_nt_pair(cfg_base, first, second)
        except ValueError as e:
            print(f"  • {e}; skipping")
            continue
        # Calibrate for this pair (short, cached)
        cal_seeds = list(range(10))
        thresholds = calibrate_thresholds_cached(cfg_pair, cal_seeds)
        for k, v in thresholds.items():
            cfg_pair['pipeline'][k] = v
        out_csv = data_dir / f"ser_vs_nm_csk_{first.lower()}_{second.lower()}.csv"
        print(f"  • Running SER vs Nm for pair {first}-{second} → {out_csv.name}")
        
        # NEW: Create cache tag for this NT pair
        pair_tag = f"pair_{first.lower()}_{second.lower()}"
        print(f"    Cache tag: {pair_tag}")
        
        df_pair = run_sweep(
            cfg_pair, seeds,
            'pipeline.Nm_per_symbol',
            nm_values,
            f"SER vs Nm (CSK {first}-{second})",
            progress_mode=args.progress,
            persist_csv=out_csv,
            resume=args.resume,
            debug_calibration=args.debug_calibration,
            cache_tag=pair_tag  # NEW: Add cache tag to prevent collisions
        )
        # De-dupe on resume
        if out_csv.exists() and not df_pair.empty:
            prev = pd.read_csv(out_csv)
            nm_key = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in prev.columns else 'pipeline.Nm_per_symbol'
            combined = pd.concat([prev, df_pair], ignore_index=True).drop_duplicates(subset=[nm_key], keep='last')
            _atomic_write_csv(out_csv, combined)

    print("✓ NT-pair sweeps complete; comparative figure will be generated by generate_comparative_plots.py")

def _json_safe(obj):
    """Convert NumPy types to JSON-serializable types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj

def _write_hybrid_isi_distance_grid(cfg_base: Dict[str, Any], 
                                    distances_um: List[float], 
                                    guard_grid: List[float], 
                                    out_csv: Path, 
                                    seeds: List[int]) -> None:
    """
    Generate ISI-distance grid for Hybrid mode 2D visualization.
    Creates a CSV with (distance_um, guard_factor, symbol_period_s, ser, use_ctrl) for heatmaps.
    """
    print(f"🔧 Generating Hybrid ISI-distance grid: {len(distances_um)} distances × {len(guard_grid)} guard factors")
    
    # Extract CTRL state from base config
    use_ctrl = bool(cfg_base['pipeline'].get('use_control_channel', True))
    
    rows = []
    for d in distances_um:
        for g in guard_grid:
            # Use median across a few seeds for stability
            seed_results = []
            for seed in seeds[:3]:  # Use first 3 seeds for speed
                try:
                    cfg = deepcopy(cfg_base)
                    cfg['pipeline']['distance_um'] = d
                    cfg['pipeline']['guard_factor'] = g
                    cfg['pipeline']['sequence_length'] = 200  # Faster for grid generation
                    cfg['pipeline']['enable_isi'] = True
                    cfg['pipeline']['random_seed'] = int(seed)  # Set seed in config
                    
                    # NEW: Recompute symbol period and decision window for this distance
                    Ts = calculate_dynamic_symbol_period(d, cfg)
                    cfg['pipeline']['symbol_period_s'] = Ts
                    cfg['pipeline']['time_window_s'] = Ts
                    cfg.setdefault('detection', {})
                    cfg['detection']['decision_window_s'] = Ts
                    
                    # NEW: Calibrate thresholds for this specific (distance, guard_factor) point
                    try:
                        cal_seeds = list(range(4))  # Fast calibration with 4 seeds
                        th = calibrate_thresholds_cached(cfg, cal_seeds)
                        for k, v in th.items():
                            cfg['pipeline'][k] = v
                    except Exception:
                        pass  # Fallback to default thresholds if calibration fails
                    
                    # FIXED: Use run_single_instance for normalized keys and metadata
                    res = run_single_instance(cfg, seed, attach_isi_meta=True)
                    if res is not None:
                        # Use normalized 'ser' key (run_single_instance converts 'SER' → 'ser')
                        # and attached 'symbol_period_s' metadata
                        ser_val = float(res.get('ser', res.get('SER', 1.0)))  # fallback chain
                        mosk_err = int(res.get('subsymbol_errors', {}).get('mosk', 0))
                        csk_err  = int(res.get('subsymbol_errors', {}).get('csk', 0))
                        L        = int(cfg['pipeline']['sequence_length'])
                        # Best-effort: convert component error counts to SER fractions
                        mosk_ser_val = (mosk_err / L) if L > 0 else float('nan')
                        csk_ser_val  = (csk_err  / L) if L > 0 else float('nan')
                        ts_val = float(res.get('symbol_period_s', cfg['pipeline']['symbol_period_s']))
                        seed_results.append({
                            'ser': ser_val,
                            'mosk_ser': mosk_ser_val,
                            'csk_ser': csk_ser_val,
                            'symbol_period_s': ts_val
                        })
                except Exception as e:
                    print(f"⚠️  Grid point failed (d={d}, g={g}, seed={seed}): {e}")
                    continue
            
            if seed_results:
                # Take median across seeds
                median_ser = np.median([r['ser'] for r in seed_results])
                median_ts = np.median([r['symbol_period_s'] for r in seed_results])
                
                rows.append({
                    'distance_um': d,
                    'guard_factor': g,
                    'symbol_period_s': median_ts,
                    'ser': median_ser,
                    'mosk_ser': float(np.nanmedian([r['mosk_ser'] for r in seed_results])),
                    'csk_ser':  float(np.nanmedian([r['csk_ser']  for r in seed_results])),
                    'use_ctrl': use_ctrl  # NEW: Track CTRL state
                })
    
    if rows:
        # Load existing data and merge with current CTRL state
        existing_df = pd.DataFrame()
        if out_csv.exists():
            try:
                existing_df = pd.read_csv(out_csv)
                # Filter out rows with the same CTRL state (we're updating them)
                if 'use_ctrl' in existing_df.columns:
                    existing_df = existing_df[existing_df['use_ctrl'] != use_ctrl]
            except Exception as e:
                print(f"⚠️  Could not read existing grid CSV: {e}")
                existing_df = pd.DataFrame()
        
        # Combine existing data with new data
        new_df = pd.DataFrame(rows)
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
            
        # Sort and ensure no duplicates
        combined_df = combined_df.drop_duplicates(
            subset=['distance_um', 'guard_factor', 'use_ctrl'], 
            keep='last'
        ).sort_values(['use_ctrl', 'distance_um', 'guard_factor'])
        
        _atomic_write_csv(out_csv, combined_df)
        print(f"✓ Saved ISI-distance grid: {out_csv} ({len(rows)} new points, {len(combined_df)} total)")
    else:
        print("⚠️  No valid grid points generated")

# ============= MAIN =============
def run_one_mode(args: argparse.Namespace, mode: str) -> None:
    # Build session metadata for GUI context
    session_meta = {
        "modes": [mode],
        "progress": args.progress,
        "resume": bool(args.resume),
        "with_ctrl": bool(args.use_ctrl),
        "isi": not args.disable_isi,
        "flags": [f"--num-seeds={args.num_seeds}", f"--sequence-length={args.sequence_length}"] +
                ([f"--nt-pairs={args.nt_pairs}"] if args.nt_pairs else []) +
                (["--recalibrate"] if args.recalibrate else [])
    }
    
    _install_signal_handlers()
    
    # Determine worker count
    maxw = args.max_workers
    if maxw is None:
        if args.extreme_mode:
            m = "extreme"
        elif args.beast_mode:
            m = "beast"
        else:
            m = "optimal"
        maxw = get_optimal_workers(m)
        print(f"🔥 Using {m.upper()} mode: {maxw} workers")

    print(f"\n{'='*60}\n🚀 ANALYSIS - {mode} Mode (ISI {'OFF' if args.disable_isi else 'ON'})\n{'='*60}")
    print(f"CPU: {CPU_COUNT} threads ({PHYSICAL_CORES} cores)")
    print(f"Workers: {maxw}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Progress UI: {args.progress}")
    if mode == "CSK":
        print(f"CSK Level Scheme: {args.csk_level_scheme}")

    check_memory_usage()

    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    data_dir = results_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(project_root / "config" / "default.yaml") as f:
        config_base = yaml.safe_load(f)

    cfg = preprocess_config_full(config_base)
    cfg['pipeline']['enable_isi'] = not args.disable_isi
    cfg['pipeline']['modulation'] = mode
    cfg['pipeline']['sequence_length'] = args.sequence_length
    cfg['pipeline']['use_control_channel'] = bool(args.use_ctrl)
    print(f"CTRL subtraction: {'ON' if cfg['pipeline']['use_control_channel'] else 'OFF'}")
    cfg['verbose'] = args.verbose
    # Stage 13: pass adaptive-CI config via cfg (so workers see it)
    cfg['_stage13_target_ci'] = float(args.target_ci)
    cfg['_stage13_min_ci_seeds'] = int(args.min_ci_seeds)
    cfg['_stage13_lod_delta'] = float(args.lod_screen_delta)
    cfg['_watchdog_secs'] = int(args.watchdog_secs)
    cfg['_analytic_lod_bracket'] = getattr(args, 'analytic_lod_bracket', False)
    # make LoD skip/limit flags visible to workers via cfg
    if getattr(args, "max_lod_validation_seeds", None) is not None:
        cfg["max_lod_validation_seeds"] = int(args.max_lod_validation_seeds)
    if getattr(args, "max_symbol_duration_s", None) is not None:
        cfg["max_symbol_duration_s"] = float(args.max_symbol_duration_s)
    if getattr(args, "max_ts_for_lod", None) is not None:
        cfg["max_ts_for_lod"] = float(args.max_ts_for_lod)

    if mode.startswith("CSK"):
        cfg['pipeline']['csk_levels'] = 4
        cfg['pipeline']['csk_target_channel'] = 'GLU'
        cfg['pipeline']['csk_level_scheme'] = args.csk_level_scheme
        print(f"CSK Configuration: {cfg['pipeline']['csk_levels']} levels, "
              f"{cfg['pipeline']['csk_target_channel']} channel, "
              f"{args.csk_level_scheme} scheme")

    ss = np.random.SeedSequence(2026)
    seeds = [int(s) for s in ss.generate_state(args.num_seeds)]

    global_pool.get_pool(maxw)
    start_time = time.time()

    print(f"\n{'='*60}\nRunning Performance Sweeps\n{'='*60}")

    # Shared GUI for this mode
    pm = ProgressManager(args.progress, gui_session_meta=session_meta)
    pm.set_status(mode=mode, sweep="SER vs Nm")

    # Pre-compute job counts for consistent totals
    nm_values = [2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]
    distances = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    guard_values = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
    ser_jobs = len(nm_values) * args.num_seeds
    lod_seed_cap = 10
    lod_jobs = len(distances) * (lod_seed_cap * 8 + lod_seed_cap + 5)  # initial estimate only
    isi_jobs = (len(guard_values) * args.num_seeds) if (not args.disable_isi) else 0

    # Hierarchy
    # Create overall progress bar first
    overall_key = ("overall", mode)
    # Create hierarchy only for GUI backend (avoids duplicate bars in rich/tqdm)
    hierarchy_supported = (args.progress == "gui")
    
    # Initialize variables to prevent unbound issues
    overall_manual = None
    mode_bar = None
    overall = None
    
    # Initialize variables with proper types
    mode_key: Optional[Tuple[str, str]] = None
    ser_key: Optional[Tuple[str, str, str]] = None
    lod_key: Optional[Tuple[str, str, str]] = None
    isi_key: Optional[Tuple[str, str, str]] = None
    
    if hierarchy_supported:
        overall_key = ("overall", mode)
        overall = pm.task(total=ser_jobs + lod_jobs + isi_jobs, 
                         description=f"Overall ({mode})", 
                         key=overall_key, kind="overall")
        
        mode_key = ("mode", mode)
        mode_bar = pm.task(total=ser_jobs + lod_jobs + isi_jobs,
                          description=f"{mode} Mode",
                          parent=overall_key, key=mode_key, kind="mode")

        ser_key = ("sweep", mode, "SER_vs_Nm")
        lod_key = ("sweep", mode, "LoD_vs_distance") 
        isi_key = ("sweep", mode, "ISI_vs_guard")

    # Always create the ser_bar with appropriate parent
    if hierarchy_supported:
        ser_bar = pm.task(total=ser_jobs, description="SER vs Nm",
                          parent=mode_key, key=("sweep", mode, "SER_vs_Nm"), kind="sweep")
    else:
        ser_bar = None

    # CSV file paths
    ser_csv = data_dir / f"ser_vs_nm_{mode.lower()}.csv"

    # ---------- 1) SER vs Nm ----------
    print("\n1. Running SER vs. Nm sweep...")

    # initial calibration (kept; thresholds hoisted per Nm in run_sweep)
    if mode in ['CSK', 'Hybrid']:
        print(f"\n📊 Initial calibration for {mode} mode...")
        cal_seeds = list(range(10))
        # store to disk so subsequent processes reuse quickly
        initial_thresholds = calibrate_thresholds(cfg, cal_seeds, recalibrate=False, save_to_file=True, verbose=args.debug_calibration)
        print("✅ Calibration complete")
        for k, v in initial_thresholds.items():
            cfg['pipeline'][k] = v

    df_ser_nm = run_sweep(
        cfg, seeds,
        'pipeline.Nm_per_symbol', nm_values,
        f"SER vs Nm ({mode})",
        progress_mode=args.progress,
        persist_csv=ser_csv,
        resume=args.resume,
        debug_calibration=args.debug_calibration,
        pm=pm,                                # always share one PM
        sweep_key=ser_key if hierarchy_supported else None,
        parent_key=mode_key if hierarchy_supported else None,
    )
    # advance the aggregate mode bar by however many jobs actually ran
    if ser_bar: ser_bar.close()

    # --- Finalize SER CSV (de‑dupe by (Nm, use_ctrl)) to support ablation overlays ---
    if ser_csv.exists():
        existing = pd.read_csv(ser_csv)
        nm_key: Optional[str] = None
        if 'pipeline_Nm_per_symbol' in existing.columns:
            nm_key = 'pipeline_Nm_per_symbol'
        elif 'pipeline.Nm_per_symbol' in existing.columns:
            nm_key = 'pipeline.Nm_per_symbol'
        if nm_key is not None:
            # Combine prior rows with new rows from this run (if any)
            combined = existing if df_ser_nm.empty else pd.concat([existing, df_ser_nm], ignore_index=True)
            # Keep both CTRL states; last write wins per pair
            if 'use_ctrl' in combined.columns:
                combined = combined.drop_duplicates(subset=[nm_key, 'use_ctrl'], keep='last').sort_values(by=[nm_key, 'use_ctrl'])
            else:
                combined = combined.drop_duplicates(subset=[nm_key], keep='last').sort_values(by=[nm_key])
            _atomic_write_csv(ser_csv, combined)
    elif not df_ser_nm.empty:
        _atomic_write_csv(ser_csv, df_ser_nm)

    print(f"✅ SER vs Nm results saved to {ser_csv}")
    
    # Manual parent update for non-GUI backends  
    if not hierarchy_supported:
        # For rich/tqdm, create simple overall progress tracker
        if overall_manual is None:
            overall_manual = pm.task(total=3, description=f"{mode} Progress")
        overall_manual.update(1, description=f"{mode} - SER vs Nm completed")

    # ---------- 1′) HDS grid (Hybrid only): Nm × distance with component errors ----------
    if mode == "Hybrid":
        print("\n1′. Building Hybrid HDS grid (Nm × distance)…")
        grid_csv = data_dir / "hybrid_hds_grid.csv"
        
        # Use the distances configured for LoD (or fallback to a small set)
        try:
            distances = list(cfg['pipeline'].get('distances_um', []))
        except Exception:
            distances = []
        if not distances:
            # Fallback to a representative subset for the grid
            distances = [25, 50, 100, 150, 200]
        
        # Use a subset of Nm values for the grid (to keep computation manageable)
        grid_nm_values = [5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4]  # subset of nm_values
        
        rows = []
        for d in distances:
            cfg_d = deepcopy(cfg)
            cfg_d['pipeline']['distance_um'] = int(d)
            
            # Recompute symbol period if your model depends on distance
            try:
                Ts = calculate_dynamic_symbol_period(int(d), cfg_d)
                cfg_d['pipeline']['symbol_period_s'] = Ts
                cfg_d['pipeline']['time_window_s'] = Ts
                cfg_d['detection']['decision_window_s'] = Ts
            except Exception:
                pass
            
            # Run SER sweep for this distance
            df_d = run_sweep(
                cfg_d, seeds,
                'pipeline.Nm_per_symbol',
                grid_nm_values,
                f"HDS grid Hybrid (d={d} μm)",
                progress_mode=args.progress,
                persist_csv=None,  # Don't persist intermediate results
                resume=args.resume,
                debug_calibration=args.debug_calibration
            )
            
            if not df_d.empty:
                df_d = df_d.copy()
                df_d['distance_um'] = int(d)
                rows.append(df_d)
        
        if rows:
            grid = pd.concat(rows, ignore_index=True)
            # Keep both CTRL states if present; de-dup by (d, Nm, use_ctrl)
            nm_key = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in grid.columns else 'pipeline.Nm_per_symbol'
            subset = ['distance_um', nm_key] + (['use_ctrl'] if 'use_ctrl' in grid.columns else [])
            grid = grid.drop_duplicates(subset=subset, keep='last').sort_values(subset)
            _atomic_write_csv(grid_csv, grid)
            print(f"✅ HDS grid saved to {grid_csv}")
        else:
            print("⚠️ HDS grid: no rows produced (skipping).")

    if not df_ser_nm.empty:
        nm_col_print = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in df_ser_nm.columns else 'pipeline.Nm_per_symbol'
        cols_to_show = [c for c in [nm_col_print, 'ser', 'snr_db', 'use_ctrl'] if c in df_ser_nm.columns]
        print(f"\nSER vs Nm Results (head) for {mode}:")
        print(df_ser_nm[cols_to_show].head().to_string(index=False))

    # After the standard CSK SER vs Nm sweep finishes and nm_values are known:
    if mode == "CSK" and (args.nt_pairs or ""):
        run_csk_nt_pair_sweeps(args, cfg, seeds, nm_values)

    # ---------- 2) LoD vs Distance ----------
    print("\n2. Building LoD vs distance curve…")
    if args.distances:
        d_run = [int(x) for x in args.distances.split(",") if x.strip()]
    else:
        if 'distances_um' in cfg:
            d_run = [int(x) for x in cfg['distances_um']]
        else:
            d_run = [25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 150, 175, 200]
    lod_csv = data_dir / f"lod_vs_distance_{mode.lower()}.csv"
    pm.set_status(mode=mode, sweep="LoD vs distance")
    optimal_workers = get_optimal_workers("beast" if args.beast_mode else "optimal")
    pool = global_pool.get_pool(max_workers=optimal_workers)
    maxw = optimal_workers
    use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))

    estimated_per_distance = args.num_seeds * 8 + args.num_seeds + 5

    # --- NEW: find fully-completed distances for this CTRL state ---
    done_distances: set[int] = set()
    if args.resume and lod_csv.exists():
        df_prev = None
        try:
            df_prev = pd.read_csv(lod_csv)
        except Exception as e:
            print(f"⚠️  Could not read existing LoD CSV ({e}); will recompute all distances")
            df_prev = None

        if df_prev is not None:
            # Respect CTRL state if present
            if 'use_ctrl' in df_prev.columns:
                df_prev = df_prev[df_prev['use_ctrl'] == bool(cfg['pipeline'].get('use_control_channel', True))]

            if 'lod_nm' in df_prev.columns:
                # Base success mask: finite & > 0
                lod_nm_num = pd.to_numeric(df_prev['lod_nm'], errors='coerce')
                success_mask = lod_nm_num.gt(0) & np.isfinite(lod_nm_num)

                # Optional: require SER success when available
                if 'ser_at_lod' in df_prev.columns:
                    ser_num = pd.to_numeric(df_prev['ser_at_lod'], errors='coerce')
                    # 0.01 is the default target in this script; change if you use a different target
                    success_mask &= ser_num.le(0.01)

                df_successful = df_prev[success_mask].copy()

                # If multiple rows per distance exist, keep the last
                if 'distance_um' in df_successful.columns:
                    df_successful = (
                        df_successful
                        .sort_index()                # assumes later appends have larger index
                        .drop_duplicates(['distance_um'], keep='last')
                    )

                    done_distances = {
                        int(x) for x in pd.to_numeric(
                            df_successful['distance_um'], errors='coerce'
                        ).dropna().tolist()
                    }

                if done_distances:
                    print(f"↩️  Resume: {len(done_distances)} LoD distance(s) already complete: "
                          f"{sorted(done_distances)} μm")
            else:
                # Old CSV without lod_nm → just don't prefill
                print("ℹ️  Existing LoD CSV has no 'lod_nm' column; will recompute all distances.")

    # Worklist excludes done distances
    d_run_work = [int(d) for d in d_run if int(d) not in done_distances]

    lod_jobs = len(d_run) * estimated_per_distance
    lod_key = ("sweep", mode, "LoD")
    if hierarchy_supported:
        lod_bar = pm.task(total=lod_jobs, description="LoD vs distance",
                          parent=mode_key, key=lod_key, kind="sweep")
        # Keep the overall headline consistent with the true LoD total
        if hasattr(pm, "update_total") and overall is not None:
            pm.update_total(key=("overall", mode), total=ser_jobs + lod_jobs + isi_jobs,
                            label=f"Overall ({mode})", kind="overall")
    else:
        lod_bar = None
    
    # --- NEW: Setup multiprocessing-safe progress queues ---
    mgr = mp.Manager()
    progress_queues = {}
    drainers = {}

    def _start_drain_thread(dist, bar, q):
        stop = threading.Event()
        def _drain():
            while not stop.is_set():
                try:
                    inc = q.get(timeout=0.25)
                    if inc is None:
                        break
                    if bar is not None:
                        bar.update(int(inc))
                except pyqueue.Empty:
                    pass
        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        return t, stop

    # Create individual distance progress bars FIRST with accurate totals
    distance_bars = {} # binary search + data-rate + sigma sampling
    dist_totals = {}  # NEW: track actual totals

    for d_um in d_run:
        dist_key = ("dist", mode, "LoD", float(d_um))
        if hierarchy_supported:
            dist_bar = pm.task(total=estimated_per_distance,
                            description=f"LoD @ {float(d_um):.0f}μm",
                            parent=lod_key, key=dist_key, kind="dist")
        else:
            dist_bar = None
        distance_bars[int(d_um)] = dist_bar
        dist_totals[int(d_um)] = estimated_per_distance

        if int(d_um) in done_distances:
            # Prefill 100% and close; also bubble progress to parents
            if dist_bar:
                dist_bar.update(estimated_per_distance)
                dist_bar.close()
            # No queue/drainer for completed distances
            continue

        # Not done yet: set up queue & drainer
        q = mgr.Queue(maxsize=1000)
        progress_queues[int(d_um)] = q
        t, stop_evt = _start_drain_thread(int(d_um), dist_bar, q)
        drainers[int(d_um)] = (t, stop_evt)

    # --- NEW: Create per-worker progress bars ---
    worker_bars = {}
    if hierarchy_supported:
        for wid in range(maxw):
            worker_bars[wid] = pm.worker_task(
                worker_id=wid, 
                total=estimated_per_distance, 
                label=f"Worker {wid:02d}",
                parent=lod_key
            )

    def _choose_seeds_for_distance(distance_um: float, all_seeds: List[int], lod_num_seeds_arg: Optional[Union[int, str]]) -> List[int]:
        """Auto-tune LoD search seed count based on distance."""
        if lod_num_seeds_arg is None:
            return all_seeds

        # Parse schedule if it's a string
        if isinstance(lod_num_seeds_arg, str):
            if ',' in lod_num_seeds_arg:
                try:
                    # Format: "min,max" or "<=100:6,<=150:8,>150:10"
                    if lod_num_seeds_arg.count(',') == 1 and all(x.isdigit() for x in lod_num_seeds_arg.split(',')):
                        # Simple min,max format
                        min_seeds, max_seeds = map(int, lod_num_seeds_arg.split(','))
                        # Linear interpolation by distance
                        ratio = min(1.0, max(0.0, (distance_um - 25) / (200 - 25)))  # 25-200μm range
                        seed_count = int(min_seeds + ratio * (max_seeds - min_seeds))
                        return all_seeds[:seed_count]
                    else:
                        # Rich schedule format: "<=100:6,<=150:8,>150:10"
                        for rule in lod_num_seeds_arg.split(','):
                            if ':' in rule:
                                condition, count_str = rule.split(':')
                                count = int(count_str)  # Ensure it's an integer
                                if condition.startswith('<='):
                                    threshold = float(condition[2:])
                                    if distance_um <= threshold:
                                        return all_seeds[:count]
                                elif condition.startswith('>='):
                                    threshold = float(condition[2:])
                                    if distance_um >= threshold:
                                        return all_seeds[:count]
                                elif condition.startswith('>'):
                                    threshold = float(condition[1:])
                                    if distance_um > threshold:
                                        return all_seeds[:count]
                                elif condition.startswith('<'):
                                    threshold = float(condition[1:])
                                    if distance_um < threshold:
                                        return all_seeds[:count]
                        # Fallback if no rule matches
                        return all_seeds[:8]
                except Exception:
                    # Parse error - fallback to simple integer
                    return all_seeds[:8]
            else:
                # Single number as string
                try:
                    return all_seeds[:int(lod_num_seeds_arg)]
                except Exception:
                    return all_seeds
        else:
            # Integer argument
            return all_seeds[:lod_num_seeds_arg]

    # Submit distance jobs in batches for effective warm-start
    future_to_dist: Dict[Any, float] = {}
    future_worker: Dict[Any, int] = {}
    last_lod_guess: Optional[int] = None
    batch_size = 3
    lod_results: List[Dict[str, Any]] = []

    # Ensure we only process distances that have progress queues
    d_run_work_with_queues = [d for d in d_run_work if d in progress_queues]

    for i in range(0, len(d_run_work_with_queues), batch_size):
        batch_d = d_run_work_with_queues[i:i+batch_size]
        future_to_dist.clear()
        future_worker.clear()

        for j, dist in enumerate(batch_d):
            wid = (i + j) % max(1, maxw)
            if hasattr(pm, "worker_update"):
                pm.worker_update(wid, f"LoD | d={dist} μm")
            
            # Now we know this queue exists
            q = progress_queues[dist]  # Safe to use direct access
            
            seeds_for_lod = _choose_seeds_for_distance(float(dist), seeds, args.lod_num_seeds)
            args.full_seeds = seeds

            future = pool.submit(
                process_distance_for_lod, float(dist), cfg, seeds_for_lod, 0.01,
                args.debug_calibration, q, args.resume, args,
                warm_lod_guess=last_lod_guess
            )
            future_to_dist[future] = float(dist)
            future_worker[future] = wid

        # Process batch results and update warm-start guess
        for fut in as_completed(list(future_to_dist)):
            dist_float = future_to_dist[fut]
            dist = int(dist_float)  # Convert to int for consistency
            wid = future_worker.pop(fut, 0)  # stable id assignment
            
            res = {}  # Initialize to prevent unbound variable
            try:
                res = fut.result(timeout=7200)
                # Store actual progress for accurate parent counting  
                res['actual_progress'] = res.get('actual_progress', estimated_per_distance)
                
                # NEW: Update warm-start guess if we got a valid LoD (feeds next batch)
                if res and not pd.isna(res.get('lod_nm', np.nan)):
                    last_lod_guess = int(res['lod_nm'])
            except TimeoutError:
                print(f"⚠️  LoD timeout at {dist}μm (mode={mode}, use_ctrl={use_ctrl}, timeout=7200s), skipping")
                res = {'distance_um': dist, 'lod_nm': float('nan'), 'ser_at_lod': float('nan')}
            except Exception as ex:
                print(f"💥 Distance processing failed for {dist}μm: {ex}")
                res = {'distance_um': dist, 'lod_nm': float('nan'), 'ser_at_lod': float('nan')}
            finally:
                # Mark this worker as idle
                if hasattr(pm, "worker_update"):
                    pm.worker_update(wid, "idle")
                # NEW: increment worker bar by the actual work performed for that distance
                actual_total = res.get('actual_progress', estimated_per_distance)
                if hierarchy_supported and wid in worker_bars:
                    worker_bars[wid].update(actual_total)
                # NEW: stop the per-distance drainer cleanly
                try:
                    q_cleanup = progress_queues.get(dist)
                    if q_cleanup is not None:
                        try:
                            q_cleanup.put_nowait(None)  # sentinel for drainer
                        except Exception:
                            pass
                    t, stop_evt = drainers.get(dist, (None, None))
                    if stop_evt is not None:
                        stop_evt.set()
                except Exception:
                    pass

            # Append atomically per distance (only if we have a real result)
            if res and len(res.keys()) > 0:
                append_row_atomic(lod_csv, res, list(res.keys()))
            
            lod_results.append(res)
            
            if res and not pd.isna(res.get('lod_nm', np.nan)):
                print(f"  [{len(lod_results)}/{len(d_run)}] {dist}μm done: LoD={res['lod_nm']:.0f} molecules")
            else:
                print(f"  ⚠️  [{len(lod_results)}/{len(d_run)}] {dist}μm failed")
            
            # Update distance bar with actual progress before closing
            if dist in distance_bars:
                bar = distance_bars[dist]
                actual_total = res.get('actual_progress', estimated_per_distance)
                
                # Create the correct dist_key for this specific distance
                current_dist_key = ("dist", mode, "LoD", float(dist))
                
                if hierarchy_supported and bar and hasattr(pm, "update_total"):
                    pm.update_total(key=current_dist_key, total=actual_total,
                                    label=f"LoD @ {dist:.0f}μm", kind="dist", parent=lod_key)
                    
                    # NEW: remember actual total
                    dist_totals[dist] = actual_total
                    
                    # NEW: update the parent LoD row with sum of actual totals
                    if hierarchy_supported and lod_bar and hasattr(pm, "update_total"):
                        new_lod_total = sum(dist_totals.values())
                        pm.update_total(key=lod_key, total=new_lod_total,
                                        label="LoD vs distance", kind="sweep", parent=mode_key)
                        
                        # Also update overall total to reflect actual work
                        if overall_key:
                            # Calculate difference between estimated and actual LoD work
                            original_lod_estimate = len(d_run) * estimated_per_distance
                            actual_lod_total = new_lod_total
                            lod_diff = actual_lod_total - original_lod_estimate
                            
                            # Update overall total if there's a significant difference
                            if abs(lod_diff) > 0:
                                new_overall_total = ser_jobs + actual_lod_total + isi_jobs
                                pm.update_total(key=overall_key, total=new_overall_total,
                                                label=f"Overall ({mode})", kind="overall")
                    
                    remaining = max(0, actual_total - int(getattr(bar, "completed", 0)))
                    if remaining > 0: bar.update(remaining)
                    if bar:
                        bar.close()

    # Close remaining distance bars and stop any drainers just in case
    for bar in distance_bars.values():
        if bar:
            bar.close()
    for dist, (t, stop_evt) in drainers.items():
        try:
            if stop_evt is not None:
                stop_evt.set()
            cleanup_queue: Optional[Any] = progress_queues.get(dist)
            if cleanup_queue is not None:
                cleanup_queue.put_nowait(None)
        except Exception:
            pass

    # NEW: Close worker bars
    if hierarchy_supported:
        for bar in worker_bars.values():
            if bar:
                bar.close()
    
    # NEW: Clean up the multiprocessing manager
    try:
        mgr.shutdown()
    except Exception:
        pass
    # DO NOT stop pm here; continue to ISI sweep

    # Filter out empty LoD results to prevent DataFrame errors
    real_lod_results = [r for r in lod_results if isinstance(r, dict) and 'distance_um' in r and not pd.isna(r.get('distance_um', np.nan))]
    # More precise failed distance detection for NaN LoDs
    failed_distances = sorted(
        int(r['distance_um'])
        for r in lod_results
        if (isinstance(r, dict) and 'distance_um' in r and 
            (('lod_nm' not in r) or pd.isna(r.get('lod_nm'))))
    )

    if failed_distances:
        print(f"⚠️  No LoD found at distances: {failed_distances} μm")

    # Merge with any prior rows (for a clean sorted CSV at the end)
    if lod_csv.exists():
        prior = pd.read_csv(lod_csv)
        # Keep both CTRL states; sort by distance, then use_ctrl
        if 'use_ctrl' in prior.columns:
            df_lod = prior.drop_duplicates(subset=['distance_um', 'use_ctrl'], keep='last').sort_values(['distance_um', 'use_ctrl'])
        else:
            df_lod = prior.drop_duplicates(subset=['distance_um'], keep='last').sort_values('distance_um')
    else:
        if real_lod_results:
            df_lod = pd.DataFrame(real_lod_results).sort_values('distance_um')
        else:
            print("⚠️  No LoD points were produced in this run.")
            df_lod = pd.DataFrame()  # Empty DataFrame with no columns

    if not df_lod.empty:
        _atomic_write_csv(lod_csv, df_lod)
        print(f"\n✅ LoD vs distance saved to {lod_csv} ({len(df_lod)} points)")
    else:
        print(f"\n⚠️  No valid LoD data to save to {lod_csv}")
    
    # Manual parent update for non-GUI backends
    if not hierarchy_supported:
        if overall_manual is None:
            overall_manual = pm.task(total=3, description=f"{mode} Progress")
        overall_manual.update(1, description=f"{mode} - LoD vs Distance completed")

    if not df_lod.empty:
        cols_to_show = [c for c in ['distance_um', 'lod_nm', 'ser_at_lod', 'use_ctrl'] if c in df_lod.columns]
        print(f"\nLoD vs Distance (head) for {mode}:")
        print(df_lod[cols_to_show].head().to_string(index=False))

    # ---------- 3) ISI trade-off (guard-factor sweep) ----------
    if cfg['pipeline'].get('enable_isi', True):
        print("\n3. Running ISI trade-off sweep (guard factor)…")
        # Representative distance: use current distance from cfg
        # (user can change default distance in config/default.yaml)
        guard_values = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
        isi_csv = data_dir / f"isi_tradeoff_{mode.lower()}.csv"
        # Ensure ISI is enabled for this sweep
        cfg['pipeline']['enable_isi'] = True
        pm.set_status(mode=mode, sweep="ISI trade-off")
        isi_key = ("sweep", mode, "ISI_tradeoff")
        isi_bar = None
        if hierarchy_supported:
            isi_bar = pm.task(total=isi_jobs, description="ISI trade-off (guard)",
                            parent=mode_key, key=isi_key, kind="sweep")
        df_isi = run_sweep(
            cfg, seeds,
            'pipeline.guard_factor',
            guard_values,
            f"ISI trade-off ({mode})",
            progress_mode=args.progress,
            persist_csv=isi_csv,
            resume=args.resume,
            debug_calibration=args.debug_calibration,
            pm=pm,                       # share PM
            sweep_key=isi_key if hierarchy_supported else None,
            parent_key=mode_key if hierarchy_supported else None
        )
        if isi_bar: isi_bar.close()
        # De-duplicate by (guard_factor, use_ctrl)
        if isi_csv.exists():
            existing = pd.read_csv(isi_csv)
            gf_key = 'guard_factor' if 'guard_factor' in existing.columns else 'pipeline.guard_factor'
            if gf_key in existing.columns:
                combined = existing if df_isi.empty else pd.concat([existing, df_isi], ignore_index=True)
                if 'use_ctrl' in combined.columns:
                    combined = combined.drop_duplicates(subset=[gf_key, 'use_ctrl'], keep='last').sort_values([gf_key, 'use_ctrl'])
                else:
                    combined = combined.drop_duplicates(subset=[gf_key], keep='last').sort_values([gf_key])
                _atomic_write_csv(isi_csv, combined)
        elif not df_isi.empty:
            _atomic_write_csv(isi_csv, df_isi)
        print(f"✅ ISI trade-off saved to {isi_csv}")
        
        # Manual parent update for non-GUI backends
        if not hierarchy_supported:
            if overall_manual is None:
                overall_manual = pm.task(total=3, description=f"{mode} Progress")
            overall_manual.update(1, description=f"{mode} - ISI Trade-off completed")

        # NEW: Generate ISI-distance grid for Hybrid mode 2D visualization
        if mode.lower() == "hybrid":
            print("\n4. Generating Hybrid ISI-distance grid for 2D visualization…")
            guard_grid = np.round(np.linspace(0.0, 1.0, 11), 2).tolist()
            dist_grid = [25.0, 50.0, 75.0, 100.0, 150.0, 200.0]
            isi_grid_csv = data_dir / "isi_grid_hybrid.csv"
            
            # NEW: Check for current CTRL state data instead of just file existence
            use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
            skip_grid = False
            
            if args.resume and isi_grid_csv.exists():
                try:
                    existing_grid = pd.read_csv(isi_grid_csv)
                    if 'use_ctrl' in existing_grid.columns:
                        # Check if we already have data for this CTRL state
                        current_ctrl_data = existing_grid[existing_grid['use_ctrl'] == use_ctrl]
                        expected_points = len(guard_grid) * len(dist_grid)
                        if len(current_ctrl_data) >= expected_points:
                            skip_grid = True
                            print(f"    ↩️  Resume: ISI grid already complete for use_ctrl={use_ctrl}")
                except Exception:
                    pass
            
            if not skip_grid:
                _write_hybrid_isi_distance_grid(
                    cfg_base=cfg,
                    distances_um=dist_grid,
                    guard_grid=guard_grid,
                    out_csv=isi_grid_csv,
                    seeds=seeds
                )
            else:
                print(f"✓ ISI grid exists: {isi_grid_csv}")
    else:
        print("\n3. ISI trade-off sweep skipped (ISI disabled).")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}\n✅ ANALYSIS COMPLETE ({mode})")
    print(f"   Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"{'='*60}")
    
    if hierarchy_supported:
        if mode_bar is not None:
            mode_bar.close()
        if overall is not None:
            overall.close()
    pm.stop()

def main() -> None:
    # Install signal handlers for graceful cancellation
    _install_signal_handlers()
    
    args = parse_arguments()
    
    # Auto-enable sleep inhibition when GUI is requested
    if args.progress == "gui":
        args.inhibit_sleep = True
    
    # Guard: avoid multiple Tk windows when interleaving modes (macOS only)
    if platform.system() == "Darwin" and args.progress == "gui" and args.parallel_modes and args.parallel_modes > 1:
        print("⚠️  macOS Tkinter limitation → falling back to 'rich'.")
        args.progress = "rich"
    
    # Setup logging with the tee approach
    if not args.no_log:
        setup_tee_logging(Path(args.logdir), prefix="run_final_analysis", fsync=args.fsync_logs)
    
    # Determine which modes to run
    modes = (["MoSK", "CSK", "Hybrid"] if args.mode == "ALL" else [args.mode])
    
    # Initialize sleep inhibition context
    ctx = None
    
    try:
        ctx = SleepInhibitor() if args.inhibit_sleep else None
        if ctx and args.keep_display_on:
            setattr(ctx, "_keep_display_on", True)
        if ctx:
            ctx.__enter__()
        
        # Existing parallel/sequential execution logic
        if args.parallel_modes > 1 and len(modes) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            n = min(args.parallel_modes, len(modes))
            print(f"🔀 Interleaving modes with {n} thread(s): {modes}")
            with ThreadPoolExecutor(max_workers=n) as tpool:
                futs = [tpool.submit(run_one_mode, args, m) for m in modes]
                for f in as_completed(futs):
                    f.result()
        else:
            for m in modes:
                run_one_mode(args, m)
                
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted")
        sys.exit(130)
    finally:
        try:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        except Exception:
            pass
        global_pool.shutdown()
        
def integration_test_noise_correlation() -> None:
    """
    Step 5: Integration test for cross-channel noise correlation enhancement.
    
    This test validates that:
    1. The rho_between_channels_after_ctrl parameter is properly read from config
    2. Different correlation values produce expected sigma_diff calculations
    3. The enhanced noise model integrates seamlessly with existing pipeline
    """
    print("🧪 Running Cross-Channel Noise Correlation Integration Test...")
    
    # Test configurations with different correlation values
    test_cases: List[Dict[str, Union[float, str]]] = [  # FIX: Add explicit type annotation
        {"rho_cc": 0.0, "description": "Independent channels (backward compatibility)"},
        {"rho_cc": 0.3, "description": "Moderate cross-channel correlation"},
        {"rho_cc": 0.7, "description": "Strong cross-channel correlation"},
    ]
    
    base_config = yaml.safe_load(open(project_root / "config" / "default.yaml"))
    base_config = preprocess_config_full(base_config)
    
    # Configure test parameters
    base_config['pipeline']['modulation'] = 'MoSK'
    base_config['pipeline']['sequence_length'] = 10  # Small for quick test
    base_config['pipeline']['Nm_per_symbol'] = 1e4
    base_config['pipeline']['distance_um'] = 100
    
    for case in test_cases:
        rho_cc_value = float(case['rho_cc'])  # FIX: Extract and cast early
        description = str(case['description'])  # FIX: Extract description with proper type
        
        print(f"  Testing {description} (ρcc = {rho_cc_value})...")
        
        # Set cross-channel correlation parameter
        test_config = deepcopy(base_config)
        test_config['noise']['rho_between_channels_after_ctrl'] = rho_cc_value
        
        try:
            # Ensure a symbol period is present (match your dynamic Ts logic)
            test_config['pipeline']['symbol_period_s'] = calculate_dynamic_symbol_period(
                float(test_config['pipeline']['distance_um']), test_config
            )
            # Test 1: Verify sigma calculation
            detection_window_s = 1.0  # Default value
            test_config['pipeline']['symbol_period_s'] = detection_window_s
            test_config['pipeline']['time_window_s'] = detection_window_s
            test_config['detection']['decision_window_s'] = detection_window_s
            sigma_glu, sigma_gaba = calculate_proper_noise_sigma(test_config, detection_window_s)
            
            # Manual calculation for verification
            expected_sigma_diff = math.sqrt(sigma_glu**2 + sigma_gaba**2 - 2*rho_cc_value*sigma_glu*sigma_gaba)
            
            # Test 2: Run short simulation to verify integration
            result = run_sequence(test_config)
            actual_sigma_diff = result.get('noise_sigma_I_diff', 0.0)
            
            # Verify calculations match
            tolerance = 1e-12
            if abs(actual_sigma_diff - expected_sigma_diff) < tolerance:
                print(f"    ✅ σ_diff calculation: {actual_sigma_diff:.2e} (expected: {expected_sigma_diff:.2e})")
            else:
                print(f"    ❌ σ_diff mismatch: {actual_sigma_diff:.2e} vs expected {expected_sigma_diff:.2e}")
                
            # Test 3: Verify correlation reduces differential noise (when rho_cc > 0)
            if rho_cc_value > 0:
                independent_sigma = math.sqrt(sigma_glu**2 + sigma_gaba**2)
                noise_reduction = (independent_sigma - actual_sigma_diff) / independent_sigma * 100
                print(f"    📉 Noise reduction: {noise_reduction:.1f}% vs independent assumption")
                
        except Exception as e:
            print(f"    ❌ Test failed: {e}")
    
    print("🎯 Cross-channel noise correlation integration test completed!\n")

if __name__ == "__main__":
    # Windows multiprocessing support
    if platform.system() == "Windows":
        mp.freeze_support()
    
    # Handle integration test option
    if len(sys.argv) > 1 and sys.argv[1] == "--test-noise-correlation":
        integration_test_noise_correlation()
    else:
        main()
