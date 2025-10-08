# analysis/run_final_analysis.py
# RUN FINAL ANALYSIS (Crash-safe resume, ISI fields in CSVs, progress UI, mypy fixes)

from __future__ import annotations

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
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError, Future
import multiprocessing as mp
import psutil
from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict, cast, Callable, Set
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
import logging

# Disable BLAS/OpenMP oversubscription for optimal process-level parallelism
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") 
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Add project root to path
project_root = Path(__file__).parent.parent if (Path(__file__).parent.name == "analysis") else Path(__file__).parent
sys.path.append(str(project_root))

# Local modules
try:
    from src.mc_detection.algorithms import calculate_ml_threshold
except ImportError:
    from src.detection import calculate_ml_threshold
from src.pipeline import run_sequence, calculate_proper_noise_sigma, _single_symbol_currents, _csk_dual_channel_Q, _resolve_decision_window
from src.config_utils import preprocess_config

# Progress UI
from analysis.ui_progress import ProgressManager
from analysis.log_utils import setup_tee_logging

MODE_NAME_ALIASES: Dict[str, str] = {
    "MOSK": "MoSK",
    "CSK": "CSK",
    "HYBRID": "Hybrid",
    "ALL": "ALL",
    "*": "ALL",
}

DEFAULT_MODE_DISTANCES: Dict[str, List[int]] = {
    "MoSK": [25, 35, 45, 55, 65, 75, 85, 95, 105, 125, 150, 175, 200],
    "CSK": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125],
    "Hybrid": [15, 20, 25, 30, 35, 40, 45, 55, 65, 75, 85, 95, 105, 115, 125, 150],
}

DEFAULT_GUARD_SAMPLES_CAP: float = 4.0e7  # per-seed limit on total time samples (sequence_length * Ts/dt)

def _canonical_mode_name(mode: str) -> str:
    if not mode:
        return "MoSK"
    key = mode.strip().upper()
    return MODE_NAME_ALIASES.get(key, mode.strip())

def _parse_distance_list(spec: str) -> List[int]:
    tokens: List[str] = []
    cleaned = spec.strip()
    if cleaned.startswith('[') and cleaned.endswith(']'):
        cleaned = cleaned[1:-1]
    for chunk in cleaned.replace(';', ',').split(','):
        for part in chunk.split():
            token = part.strip()
            if token:
                tokens.append(token.strip('[]'))
    seen: Set[int] = set()
    result: List[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            value = int(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid distance '{token}' in --distances") from exc
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

def _coerce_distance_sequence(values: Any) -> List[int]:
    if isinstance(values, str):
        return _parse_distance_list(values)
    result: List[int] = []
    seen: Set[int] = set()
    if isinstance(values, (list, tuple)):
        for item in values:
            try:
                value = int(float(item))
            except (TypeError, ValueError):
                continue
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result

def parse_distance_overrides(specs: Optional[List[str]]) -> Dict[str, List[int]]:
    overrides: Dict[str, List[int]] = {}
    if not specs:
        return overrides
    for raw in specs:
        if raw is None:
            continue
        text = raw.strip()
        if not text:
            continue
        if '=' in text:
            mode_part, values_part = text.split('=', 1)
        elif ':' in text:
            mode_part, values_part = text.split(':', 1)
        else:
            mode_part, values_part = 'ALL', text
        mode_key = _canonical_mode_name(mode_part)
        distances = _parse_distance_list(values_part)
        if not distances:
            raise ValueError(f"--distances entry '{raw}' did not provide any numeric distances")
        overrides[mode_key] = distances
    return overrides

def resolve_mode_distance_grid(mode: str, cfg: Dict[str, Any], overrides: Dict[str, List[int]]) -> List[int]:
    canonical_mode = _canonical_mode_name(mode)
    if canonical_mode in overrides:
        return list(overrides[canonical_mode])
    if 'ALL' in overrides:
        return list(overrides['ALL'])
    config_map: Dict[str, List[int]] = {}
    cfg_lod = cfg.get('lod_distances_um')
    if isinstance(cfg_lod, dict):
        for key, seq in cfg_lod.items():
            distances = _coerce_distance_sequence(seq)
            if distances:
                config_map[_canonical_mode_name(str(key))] = distances
    elif cfg_lod is not None:
        distances = _coerce_distance_sequence(cfg_lod)
        if distances:
            config_map['ALL'] = distances
    legacy = cfg.get('distances_um')
    if isinstance(legacy, dict):
        for key, seq in legacy.items():
            distances = _coerce_distance_sequence(seq)
            if distances:
                config_map.setdefault(_canonical_mode_name(str(key)), distances)
    elif legacy is not None:
        distances = _coerce_distance_sequence(legacy)
        if distances:
            config_map.setdefault('ALL', distances)
    if canonical_mode in config_map:
        return list(config_map[canonical_mode])
    if 'ALL' in config_map:
        return list(config_map['ALL'])
    return list(DEFAULT_MODE_DISTANCES.get(canonical_mode, DEFAULT_MODE_DISTANCES['MoSK']))

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
I9_14900KS_DETECTED = CPU_COUNT == 32 and PHYSICAL_CORES == 24  # Same topology as 13950HX
I9_14900K_DETECTED = CPU_COUNT == 32 and PHYSICAL_CORES == 24   # Same topology as 13950HX/14900KS

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
    },
    "i9-14900KS": {
        "p_cores_physical": list(range(8)),
        "p_cores_logical": list(range(16)),
        "e_cores_logical": list(range(16, 32)),
        "p_core_count": 8,
        "total_p_threads": 16
    },
    "i9-14900K": {
        "p_cores_physical": list(range(8)),
        "p_cores_logical": list(range(16)),
        "e_cores_logical": list(range(16, 32)),
        "p_core_count": 8,
        "total_p_threads": 16
    }
}

CPU_CONFIG: Optional[CPUConfig] = None
if I9_13950HX_DETECTED or I9_14900KS_DETECTED or I9_14900K_DETECTED:
    # Need to distinguish between CPUs since they have identical core topology
    import platform
    cpu_name = platform.processor().upper()
    if "14900KS" in cpu_name:
        CPU_CONFIG = HYBRID_CPU_CONFIGS["i9-14900KS"]
        print("🔥 i9-14900KS detected! P-core optimization available.")
    elif "14900K" in cpu_name:
        CPU_CONFIG = HYBRID_CPU_CONFIGS["i9-14900K"]
        print("🔥 i9-14900K detected! P-core optimization available.")
    elif "13950HX" in cpu_name:
        CPU_CONFIG = HYBRID_CPU_CONFIGS["i9-13950HX"]
        print("🔥 i9-13950HX detected! P-core optimization available.")
    else:
        # Fallback to generic detection if specific model isn't found in processor string
        CPU_CONFIG = HYBRID_CPU_CONFIGS["i9-13950HX"]  # Use as default for this topology
        print("🔥 Intel 13th/14th gen hybrid CPU detected! P-core optimization available.")

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
            # Get current process and available P-cores
            process = psutil.Process()
            p_cores = CPU_CONFIG["p_cores_logical"]
            
            # Cross-platform CPU affinity setting
            if hasattr(process, 'cpu_affinity'):
                # Windows/Linux via psutil (preferred method)
                process.cpu_affinity(p_cores)
                pinned_cores = process.cpu_affinity()
                print(f"🎯 Worker {os.getpid()}: P-core affinity set to {pinned_cores}")
            else:
                print(f"⚠️  Worker {os.getpid()}: CPU affinity not supported on this platform")
                
        except Exception as e:
            # Graceful degradation - worker continues without affinity
            print(f"⚠️  Worker {os.getpid()}: P-core affinity failed ({e}), continuing without optimization")

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

calibration_cache: Dict[str, Dict[str, Union[float, List[float], str]]] = {}

def get_cache_key(cfg: Dict[str, Any]) -> str:
    def _nt_pair_fp(cfg: Dict[str, Any]) -> str:
        # compact hash of the DA/SERO parameter tuples so cache respects pair identity
        def pick(name: str):
            nt = cfg['neurotransmitters'][name]
            return (
                float(nt.get('k_on_M_s', 0.0)),
                float(nt.get('k_off_s', 0.0)),
                float(nt.get('q_eff_e', 0.0)),
                float(nt.get('D_m2_s', 0.0)),
                float(nt.get('lambda', 1.0)),
            )
        raw = repr((pick('DA'), pick('SERO'))).encode()
        return hashlib.sha1(raw).hexdigest()[:8]

    def _qeff_signs(cfg: Dict[str, Any]) -> str:
        s_da = 1 if float(cfg['neurotransmitters']['DA'].get('q_eff_e', 0.0)) > 0 else -1 if float(cfg['neurotransmitters']['DA'].get('q_eff_e', 0.0)) < 0 else 0
        s_se = 1 if float(cfg['neurotransmitters']['SERO'].get('q_eff_e', 0.0)) > 0 else -1 if float(cfg['neurotransmitters']['SERO'].get('q_eff_e', 0.0)) < 0 else 0
        return f"{s_da}:{s_se}"

    # Decision window intended for calibration (will be Ts in calibration)
    dw = None
    try:
        dw = float(cfg.get('detection', {}).get('decision_window_s',
                   cfg['pipeline'].get('symbol_period_s', float('nan'))))
    except Exception:
        dw = cfg['pipeline'].get('symbol_period_s', None)

    key_params = [
        cfg['pipeline'].get('modulation'),
        cfg['pipeline'].get('Nm_per_symbol'),
        cfg['pipeline'].get('distance_um'),
        cfg['pipeline'].get('symbol_period_s'),
        dw,  # NEW: decision_window used in calibration
        cfg['pipeline'].get('csk_levels'),
        cfg['pipeline'].get('csk_target_channel'),
        cfg['pipeline'].get('csk_level_scheme', 'uniform'),
        cfg['pipeline'].get('guard_factor', 0.0),
        bool(cfg['pipeline'].get('use_control_channel', True)),
        cfg['pipeline'].get('channel_profile', 'tri'),
        _nt_pair_fp(cfg),
        _qeff_signs(cfg),  # NEW: signs guard
        cfg['pipeline'].get('csk_combiner', 'zscore'),
        cfg['pipeline'].get('csk_leakage_frac', 0.0),
    ]
    return str(hash(tuple(str(p) for p in key_params)))

def _thresholds_filename(cfg: Dict[str, Any]) -> Path:
    """
    Build a filename that captures sweep-dependent parameters so we can safely reuse across runs.
    """
    def _nt_pair_fingerprint(cfg):
        def pick(nt):
            return (
                float(nt.get('k_on_M_s', 0.0)),
                float(nt.get('k_off_s', 0.0)),
                float(nt.get('q_eff_e', 0.0)),
                float(nt.get('D_m2_s', 0.0)),
                float(nt.get('lambda', 1.0)),
            )
        g = cfg['neurotransmitters']['DA']
        b = cfg['neurotransmitters']['SERO']
        raw = repr((pick(g), pick(b))).encode()
        return hashlib.sha1(raw).hexdigest()[:8]
    
    def _nt_pair_label(cfg):
        g = cfg['neurotransmitters']['DA']; b = cfg['neurotransmitters']['SERO']
        name_g = str(g.get('name', 'DA')); name_b = str(b.get('name', 'SERO'))
        base = f"{name_g}-{name_b}".lower().replace(' ', '')
        return base

    results_dir = project_root / "results" / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    mode = str(cfg['pipeline'].get('modulation', 'unknown')).lower()
    Ts = cfg['pipeline'].get('symbol_period_s', None)
    dist = cfg['pipeline'].get('distance_um', None)
    nm = cfg['pipeline'].get('Nm_per_symbol', None)
    lvl = cfg['pipeline'].get('csk_level_scheme', 'uniform')
    profile = cfg['pipeline'].get('channel_profile', 'tri')
    tgt = cfg['pipeline'].get('csk_target_channel', '')
    M   = cfg['pipeline'].get('csk_levels', None)
    gf  = cfg['pipeline'].get('guard_factor', None)
    # Decision window intended for calibration
    win = cfg.get('detection', {}).get('decision_window_s', Ts)

    parts = [f"thresholds_{mode}", _nt_pair_label(cfg)]
    parts.append(f"pair{_nt_pair_fingerprint(cfg)}")
    if Ts is not None:   parts.append(f"Ts{float(Ts):.3g}")
    if win is not None:  parts.append(f"win{float(win):.3g}")  # NEW
    if dist is not None: parts.append(f"d{float(dist):.0f}um")
    if nm is not None:   parts.append(f"Nm{float(nm):.0f}")
    if tgt:              parts.append(f"tgt{tgt}")
    if lvl:              parts.append(f"lvl{lvl}")
    if mode.startswith("csk") and M is not None:
        parts.append(f"M{int(M)}")
    if gf is not None:   parts.append(f"gf{float(gf):.3g}")
    cmb = cfg['pipeline'].get('csk_combiner', 'zscore')
    parts.append(f"cmb{cmb}")
    lf = cfg['pipeline'].get('csk_leakage_frac', None)
    if cmb == 'leakage' and lf is not None:
        parts.append(f"leak{float(lf):.3g}")
    ctrl = 'wctrl' if bool(cfg['pipeline'].get('use_control_channel', True)) else 'noctrl'
    parts.append(f"profile{profile}")
    parts.append(ctrl)
    return results_dir / ( "_".join(parts) + ".json" )

def run_sequence_wrapper(cfg: Dict[str, Any], seed: int, attach_isi_meta: bool = False) -> Optional[Dict[str, Any]]:
    """
    Wrapper around run_sequence to handle configuration validation and threshold management.
    """
    from src.pipeline import run_sequence
    
    if cfg['pipeline']['modulation'].startswith('CSK'):
        current_M = int(cfg['pipeline'].get('csk_levels', 4))
        current_target = cfg['pipeline'].get('csk_target_channel', 'DA')
        current_combiner = cfg['pipeline'].get('csk_combiner', 'zscore')
        should_preserve = False

        if '_resume_active' in cfg and cfg['_resume_active']:
            threshold_key = f'csk_thresholds_{current_target.lower()}'
            cached_thresholds = cfg['pipeline'].get(threshold_key, [])
            if (isinstance(cached_thresholds, list) and 
                len(cached_thresholds) == (current_M - 1)):
                cache_meta = cfg.get('_threshold_cache_meta', {})
                if (cache_meta.get('M') == current_M and
                    cache_meta.get('target_channel') == current_target and
                    cache_meta.get('combiner') == current_combiner):
                    should_preserve = True

        if not should_preserve:
            keys_to_clear = [k for k in cfg['pipeline'].keys() if k.startswith('csk_thresholds_')]
            for key in keys_to_clear:
                if key in cfg['pipeline']:
                    del cfg['pipeline'][key]
                    print(f"🧹 Cleared stale threshold key: {key}")
            if '_threshold_cache_meta' in cfg:
                del cfg['_threshold_cache_meta']

    if attach_isi_meta:
        cfg['collect_isi_metrics'] = True
    
    return run_sequence(cfg)

def calibrate_thresholds(cfg: Dict[str, Any], seeds: List[int], recalibrate: bool = False,
                         save_to_file: bool = True, verbose: bool = False) -> Dict[str, Union[float, List[float], str]]:
    """
    Calibration with ISI off & decision window = Ts (or enforced minimum). Returns thresholds dict.
    Incorporates robust cache compatibility, finite filtering, and optional early-stop for MoSK/Hybrid.
    """
    mode = cfg['pipeline']['modulation']
    threshold_file = _thresholds_filename(cfg)

    # ---------- small helpers ----------
    def _fingerprint_nt_pair(c: Dict[str, Any]) -> str:
        def pick(name: str):
            nt = c['neurotransmitters'][name]
            return (
                float(nt.get('k_on_M_s', 0.0)),
                float(nt.get('k_off_s', 0.0)),
                float(nt.get('q_eff_e', 0.0)),
                float(nt.get('D_m2_s', 0.0)),
                float(nt.get('lambda', 1.0)),
            )
        raw = repr((pick('DA'), pick('SERO'))).encode()
        return hashlib.sha1(raw).hexdigest()[:8]

    def _qeff_signs(c: Dict[str, Any]) -> Tuple[int, int]:
        q_da = float(c['neurotransmitters']['DA'].get('q_eff_e', 0.0))
        q_se = float(c['neurotransmitters']['SERO'].get('q_eff_e', 0.0))
        s_da = 1 if q_da > 0 else -1 if q_da < 0 else 0
        s_se = 1 if q_se > 0 else -1 if q_se < 0 else 0
        return s_da, s_se

    def _clean(vals: List[float]) -> List[float]:
        out: List[float] = []
        for x in vals:
            try:
                xf = float(x)
            except Exception:
                continue
            if np.isfinite(xf):
                out.append(xf)
        return out

    # ---------- try cache unless recalibrate ----------
    if threshold_file.exists() and not recalibrate and save_to_file:
        try:
            with open(threshold_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            meta = cached.get("__meta__", cached.get("_metadata", {})) or {}
            if verbose:
                print(f"📁 Loaded cached thresholds from {threshold_file}")

            # Compatibility checks (invalidate on any mismatch)
            want_use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
            have_use_ctrl = bool(meta.get('use_ctrl', want_use_ctrl))
            if have_use_ctrl != want_use_ctrl:
                if verbose: print("  ↪︎ cache invalid: CTRL mismatch")
                raise RuntimeError("cache: use_ctrl mismatch")

            # NT pair + q_eff signs
            if meta.get('nt_pair_fp') != _fingerprint_nt_pair(cfg):
                if verbose: print("  ↪︎ cache invalid: NT pair fingerprint mismatch")
                raise RuntimeError("cache: nt_pair_fp mismatch")
            s_da, s_se = _qeff_signs(cfg)
            if tuple(meta.get('q_eff_signs', (s_da, s_se))) != (s_da, s_se):
                if verbose: print("  ↪︎ cache invalid: q_eff sign mismatch")
                raise RuntimeError("cache: q_eff_signs mismatch")

            # Mode-specific checks
            if mode.startswith("CSK"):
                M = int(cfg['pipeline'].get('csk_levels', 4))
                tgt = str(cfg['pipeline'].get('csk_target_channel', 'DA')).upper()
                comb = str(cfg['pipeline'].get('csk_combiner', 'zscore'))
                leak = float(cfg['pipeline'].get('csk_leakage_frac', 0.0))
                if meta.get('M') != M or meta.get('target') != tgt:
                    if verbose: print("  ↪︎ cache invalid: CSK M/target mismatch")
                    raise RuntimeError("cache: M/target mismatch")
                if meta.get('combiner') != comb or abs(float(meta.get('leakage_frac', 0.0)) - leak) > 1e-12:
                    if verbose: print("  ↪︎ cache invalid: CSK combiner/leakage mismatch")
                    raise RuntimeError("cache: combiner/leakage mismatch")
                # enforce M−1 thresholds length
                key = f"csk_thresholds_{tgt.lower()}"
                tau = cached.get(key, [])
                if not isinstance(tau, list) or len(tau) != (M - 1):
                    if verbose: print("  ↪︎ cache invalid: CSK threshold length mismatch")
                    raise RuntimeError("cache: csk threshold length mismatch")

            # Window & Ts guard
            Ts_now = float(cfg['pipeline'].get('symbol_period_s', float('nan')))
            win_used = float(meta.get('decision_window_used', Ts_now))
            if not np.isfinite(win_used) or abs(win_used - Ts_now) > 1e-9:
                if verbose: print("  ↪︎ cache invalid: decision window/Ts mismatch")
                raise RuntimeError("cache: decision window mismatch")

            # If we get here, cache is acceptable
            if verbose:
                for k, v in cached.items():
                    if k == "__meta__": 
                        continue
                    print(f"   {k}: {v if not isinstance(v, list) else f'list[{len(v)}]'}")
            return {k: v for k, v in cached.items() if k not in ("_metadata", "__meta__")}
        except Exception:
            # fall through to re-calculate
            if verbose:
                print("⚠️  Threshold cache invalid or incompatible — recalibrating…")

    # ---------- clean calibration environment ----------
    cal_cfg = deepcopy(cfg)
    cal_cfg['pipeline']['sequence_length'] = int(cfg.get('_cal_symbols_per_seed', 100))
    cal_cfg['pipeline']['enable_isi'] = False

    # Force decision window = Ts (≥ enforced minimum); also keep time_window ≥ Ts
    Ts = float(cal_cfg['pipeline']['symbol_period_s'])
    min_win = _enforce_min_window(cal_cfg, Ts)
    cal_cfg.setdefault('detection', {})['decision_window_s'] = float(min_win)
    cal_cfg['pipeline']['time_window_s'] = max(float(cal_cfg['pipeline'].get('time_window_s', 0.0)), float(min_win))

    thresholds: Dict[str, Union[float, List[float], str]] = {}

    # ----- shared ES knobs -----
    eps          = float(cfg.get('_cal_eps_rel', 0.01))
    patience     = int(cfg.get('_cal_patience', 2))
    min_seeds    = int(cfg.get('_cal_min_seeds', 4))
    max_seeds    = int(cfg.get('_cal_max_seeds', 0)) or len(seeds)
    min_per_cls  = int(cfg.get('_cal_min_samples_per_class', 50))
    es_mosk      = bool(cfg.get('_cal_enable_es_mosk', True))
    es_hybrid    = bool(cfg.get('_cal_enable_es_hybrid', True))

    def _rel_delta(prev, curr):
        if prev is None: return float('inf')
        if isinstance(curr, (list, tuple)):
            a = np.asarray(prev, dtype=float)
            b = np.asarray(curr, dtype=float)
            denom = np.maximum(np.abs(a), 1e-12)
            return float(np.max(np.abs(b - a) / denom))
        denom = max(abs(prev), 1e-12)
        return float(abs(curr - prev) / denom)

    # ---------- MoSK threshold (also used by Hybrid molecule bit) ----------
    if mode in ("MoSK", "Hybrid"):
        mosk_stats: Dict[str, List[float]] = {'da': [], 'sero': []}
        prev_tau: Optional[float] = None
        streak = 0

        # Which way should the comparator go? Provide *hints* (non-breaking).
        s_da, s_se = _qeff_signs(cfg)
        mosk_dir_hint = ">" if s_da >= s_se else "<"  # heuristic; also provide empirical below

        used = 0
        # For Hybrid sweeps, the generator must be MoSK to avoid mapping 0/1 to (DA,DA)
        # with different amplitudes. This ensures DA-only vs SERO-only samples.
        for seed in seeds[:max_seeds]:
            used += 1
            # IMPORTANT: force MoSK generator to produce true DA-only / SERO-only symbols
            cal_cfg_mosk = deepcopy(cal_cfg)
            cal_cfg_mosk['pipeline']['modulation'] = 'MoSK'      # <—— key line
            cal_cfg_mosk['pipeline']['random_seed'] = seed

            r_da = run_calibration_symbols(cal_cfg_mosk, 0, mode='MoSK')  # DA class
            r_se = run_calibration_symbols(cal_cfg_mosk, 1, mode='MoSK')  # SERO class
            if r_da and 'q_values' in r_da:
                mosk_stats['da'].extend(_clean(r_da['q_values']))
            if r_se and 'q_values' in r_se:
                mosk_stats['sero'].extend(_clean(r_se['q_values']))

            # Early-stop only when enough data per class
            if es_mosk and used >= min_seeds and \
               len(mosk_stats['da']) >= min_per_cls and len(mosk_stats['sero']) >= min_per_cls:
                m0, s0 = float(np.mean(mosk_stats['da'])), max(float(np.std(mosk_stats['da'])), 1e-15)
                m1, s1 = float(np.mean(mosk_stats['sero'])), max(float(np.std(mosk_stats['sero'])), 1e-15)
                tau = float(calculate_ml_threshold(m0, m1, s0, s1))
                if _rel_delta(prev_tau, tau) <= eps:
                    streak += 1
                    if streak >= patience:
                        if verbose:
                            print(f"🎯 MoSK calibration converged after {used} seeds (delta≤{eps:.3g})")
                        prev_tau = tau
                        break
                else:
                    streak = 0
                prev_tau = tau

        # Final threshold (either converged or all data)
        if prev_tau is None:
            if mosk_stats['da'] and mosk_stats['sero']:
                m0, s0 = float(np.mean(mosk_stats['da'])), max(float(np.std(mosk_stats['da'])), 1e-15)
                m1, s1 = float(np.mean(mosk_stats['sero'])), max(float(np.std(mosk_stats['sero'])), 1e-15)
                prev_tau = float(calculate_ml_threshold(m0, m1, s0, s1))
            else:
                prev_tau = 0.0  # safe fallback
                if verbose:
                    print("⚠️  MoSK calibration collected no finite samples; using 0.0 fallback")

        thresholds['mosk_threshold'] = float(prev_tau)

        # --- NEW: persist MoSK detector metadata so decoding matches calibration ---
        # Decision statistic used during calibration:
        thresholds['mosk_statistic'] = 'sign_aware_diff'  # D = (sgn(qeff_DA)*q_da - sgn(qeff_SERO)*q_sero) / sigma_diff
        # Calculate empirical direction from collected data
        emp_dir = None
        if mosk_stats['da'] and mosk_stats['sero']:
            if float(np.mean(mosk_stats['da'])) > float(np.mean(mosk_stats['sero'])):
                emp_dir = ">"
            else:
                emp_dir = "<"
        
        # Use empirical direction from data; fall back to sign hint
        chosen_dir = emp_dir if emp_dir else mosk_dir_hint
        thresholds['mosk_direction'] = chosen_dir     # DA wins when stat > threshold
        
        # Comparator direction hint:
        thresholds['mosk_comparator'] = chosen_dir

        # Persist q_eff signs for sanity/debug
        try:
            q_da = float(cfg['neurotransmitters']['DA'].get('q_eff_e', 0.0))
            q_se = float(cfg['neurotransmitters']['SERO'].get('q_eff_e', 0.0))
        except Exception:
            q_da, q_se = 0.0, 0.0
        thresholds['mosk_decision_meta'] = json.dumps({
            'qeff_signs': {'DA': (q_da >= 0.0), 'SERO': (q_se >= 0.0)},
            'normalization': 'sigma_diff'
        })

        thresholds['mosk_direction_hint'] = mosk_dir_hint
        thresholds['mosk_direction_empirical'] = emp_dir if emp_dir else mosk_dir_hint

    # ---------- CSK thresholds (adjacent ML; sign‑aware ordering) ----------
    if mode.startswith("CSK"):
        M = int(cfg['pipeline'].get('csk_levels', 4))
        target_channel = str(cfg['pipeline'].get('csk_target_channel', 'DA')).upper()
        level_stats: Dict[int, List[float]] = {i: [] for i in range(M)}
        prev_tau_list: Optional[List[float]] = None
        streak = 0
        used = 0

        # FIX: Measure correlation first, then update config to use it
        use_ctrl = bool(cal_cfg['pipeline'].get('use_control_channel', True))
        if use_ctrl and mode.startswith('CSK'):
            # Quick measurement with first few seeds
            for test_seed in seeds[:min(3, len(seeds))]:
                cal_cfg_test = deepcopy(cal_cfg)
                cal_cfg_test['pipeline']['random_seed'] = test_seed
                test_result = run_calibration_symbols(cal_cfg_test, 0, mode='CSK', num_symbols=20)
                if test_result and 'rho_cc_measured' in test_result:
                    rho_cc_measured = float(test_result['rho_cc_measured'])
                    if np.isfinite(rho_cc_measured):
                        # Update calibration config to use measured value
                        cal_cfg.setdefault('noise', {})['rho_between_channels_after_ctrl'] = rho_cc_measured
                        break

        for seed in seeds[:max_seeds]:
            used += 1
            cal_cfg['pipeline']['random_seed'] = seed
            for level in range(M):
                r = run_calibration_symbols(cal_cfg, level, mode='CSK', num_symbols=int(cfg.get('_cal_symbols_per_seed', 100)))
                if r and 'q_values' in r:
                    level_stats[level].extend(_clean(r['q_values']))

            # Only check stability when each class has enough samples
            if used >= min_seeds and all(len(level_stats[i]) >= min_per_cls for i in range(M)):
                tau_list: List[float] = []
                for i in range(M - 1):
                    a = level_stats[i]; b = level_stats[i + 1]
                    m0, s0 = float(np.mean(a)), max(float(np.std(a)), 1e-15)
                    m1, s1 = float(np.mean(b)), max(float(np.std(b)), 1e-15)
                    tau_list.append(float(calculate_ml_threshold(m0, m1, s0, s1)))

                # Sign‑aware ordering for target channel
                qeff = float(cfg['neurotransmitters'][target_channel]['q_eff_e'])
                tau_list.sort(reverse=(qeff < 0))

                if _rel_delta(prev_tau_list, tau_list) <= eps:
                    streak += 1
                    if streak >= patience:
                        if verbose:
                            print(f"🎯 CSK calibration converged after {used} seeds (delta≤{eps:.3g})")
                        prev_tau_list = tau_list
                        break
                else:
                    streak = 0
                prev_tau_list = tau_list

        # Final thresholds
        final_tau = prev_tau_list
        if final_tau is None:
            # Compute once from whatever samples we have (may be sparse)
            tau_list = []
            ok = True
            for i in range(M - 1):
                a = level_stats[i]; b = level_stats[i + 1]
                if not a or not b:
                    ok = False
                    break
                m0, s0 = float(np.mean(a)), max(float(np.std(a)), 1e-15)
                m1, s1 = float(np.mean(b)), max(float(np.std(b)), 1e-15)
                tau_list.append(float(calculate_ml_threshold(m0, m1, s0, s1)))
            if ok:
                qeff = float(cfg['neurotransmitters'][target_channel]['q_eff_e'])
                tau_list.sort(reverse=(qeff < 0))
                final_tau = tau_list
            else:
                final_tau = [0.0] * (M - 1)
                if verbose:
                    print("⚠️  CSK calibration incomplete; using 0.0 thresholds")

        thresholds[f'csk_thresholds_{target_channel.lower()}'] = final_tau

        # provenance for resume
        cfg['_threshold_cache_meta'] = {
            'M': M,
            'target_channel': target_channel,
            'combiner': str(cfg['pipeline'].get('csk_combiner', 'zscore'))
        }

    # ---------- Hybrid amplitude thresholds (+ optional early‑stop) ----------
    if mode == "Hybrid":
        stats: Dict[str, List[float]] = {'da_low': [], 'da_high': [], 'sero_low': [], 'sero_high': []}
        prev_da_tau: Optional[float] = None
        prev_se_tau: Optional[float] = None
        streak_da = 0
        streak_se = 0
        used = 0

        for seed in seeds[:max_seeds]:
            used += 1
            cal_cfg['pipeline']['random_seed'] = seed
            for sym in range(4):
                r = run_calibration_symbols(cal_cfg, sym, mode='Hybrid')
                if r and 'q_values' in r:
                    vals = _clean(r['q_values'])
                    if sym == 0:   stats['da_low'].extend(vals)
                    elif sym == 1: stats['da_high'].extend(vals)
                    elif sym == 2: stats['sero_low'].extend(vals)
                    elif sym == 3: stats['sero_high'].extend(vals)

            if es_hybrid and used >= min_seeds and \
               min(len(stats['da_low']), len(stats['da_high'])) >= min_per_cls and \
               min(len(stats['sero_low']), len(stats['sero_high'])) >= min_per_cls:
                # DA amplitude threshold
                m_gl, s_gl = float(np.mean(stats['da_low'])),  max(float(np.std(stats['da_low'])), 1e-15)
                m_gh, s_gh = float(np.mean(stats['da_high'])), max(float(np.std(stats['da_high'])), 1e-15)
                tau_da = float(calculate_ml_threshold(m_gl, m_gh, s_gl, s_gh))
                if _rel_delta(prev_da_tau, tau_da) <= eps:
                    streak_da += 1
                else:
                    streak_da = 0
                prev_da_tau = tau_da

                # SERO amplitude threshold
                m_sl, s_sl = float(np.mean(stats['sero_low'])),  max(float(np.std(stats['sero_low'])), 1e-15)
                m_sh, s_sh = float(np.mean(stats['sero_high'])), max(float(np.std(stats['sero_high'])), 1e-15)
                tau_se = float(calculate_ml_threshold(m_sl, m_sh, s_sl, s_sh))
                if _rel_delta(prev_se_tau, tau_se) <= eps:
                    streak_se += 1
                else:
                    streak_se = 0
                prev_se_tau = tau_se

                if streak_da >= patience and streak_se >= patience:
                    if verbose:
                        print(f"🎯 Hybrid amplitude thresholds converged after {used} seeds")
                    break

        # Final thresholds (compute if ES didn't fill them)
        if prev_da_tau is None or prev_se_tau is None:
            if all(stats[k] for k in stats):
                m_gl, s_gl = float(np.mean(stats['da_low'])),  max(float(np.std(stats['da_low'])), 1e-15)
                m_gh, s_gh = float(np.mean(stats['da_high'])), max(float(np.std(stats['da_high'])), 1e-15)
                m_sl, s_sl = float(np.mean(stats['sero_low'])),  max(float(np.std(stats['sero_low'])), 1e-15)
                m_sh, s_sh = float(np.mean(stats['sero_high'])), max(float(np.std(stats['sero_high'])), 1e-15)
                prev_da_tau = float(calculate_ml_threshold(m_gl, m_gh, s_gl, s_gh))
                prev_se_tau = float(calculate_ml_threshold(m_sl, m_sh, s_sl, s_sh))
            else:
                prev_da_tau = prev_da_tau or 0.0
                prev_se_tau = prev_se_tau or 0.0
                if verbose:
                    print("⚠️  Hybrid calibration incomplete; using 0.0 fallbacks")

        thresholds['hybrid_threshold_da'] = float(prev_da_tau)
        thresholds['hybrid_threshold_sero'] = float(prev_se_tau)

        # Persist the learned orientation (on Q_amp it may be "increasing" even if q_eff < 0)
        thresholds['hybrid_threshold_da_increasing']   = bool(np.mean(stats['da_high']) > np.mean(stats['da_low']))
        thresholds['hybrid_threshold_sero_increasing'] = bool(np.mean(stats['sero_high']) > np.mean(stats['sero_low']))

        # Keep existing hints for logging
        s_da, s_se = _qeff_signs(cfg)
        thresholds['hybrid_direction_da_hint']   = ">" if s_da > 0 else "<" if s_da < 0 else "="
        thresholds['hybrid_direction_sero_hint'] = ">" if s_se > 0 else "<" if s_se < 0 else "="

    # ---------- persist to disk with rich __meta__ (atomic) ----------
    if save_to_file:
        try:
            threshold_file.parent.mkdir(parents=True, exist_ok=True)

            # JSON‑safe thresholds
            payload: Dict[str, Any] = {}
            for k, v in thresholds.items():
                if isinstance(v, np.ndarray):
                    payload[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    payload[k] = float(v)
                elif isinstance(v, list):
                    payload[k] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v]
                else:
                    payload[k] = v

            # Metadata
            s_da, s_se = _qeff_signs(cfg)
            meta = {
                "mode": str(cfg['pipeline'].get('modulation')),
                "M": int(cfg['pipeline'].get('csk_levels', 0)),
                "target": str(cfg['pipeline'].get('csk_target_channel', 'DA')).upper(),
                "combiner": str(cfg['pipeline'].get('csk_combiner', 'zscore')),
                "leakage_frac": float(cfg['pipeline'].get('csk_leakage_frac', 0.0)),
                "Nm": float(cfg['pipeline'].get('Nm_per_symbol', 0.0)),
                "distance_um": float(cfg['pipeline'].get('distance_um', 0.0)),
                "Ts": float(Ts),
                "decision_window_used": float(min_win),
                "guard_factor": float(cfg['pipeline'].get('guard_factor', 0.0)),
                "use_ctrl": bool(cfg['pipeline'].get('use_control_channel', True)),
                "nt_pair_label": f"{cfg['neurotransmitters']['DA'].get('name','DA')}–{cfg['neurotransmitters']['SERO'].get('name','SERO')}",
                "nt_pair_fp": _fingerprint_nt_pair(cfg),
                "q_eff_signs": (s_da, s_se),
                "version": "2.0",
                "timestamp": time.time()
            }
            payload["__meta__"] = meta

            tmp_file = threshold_file.with_suffix('.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_file, threshold_file)

            if verbose:
                print(f"💾 Saved thresholds to {threshold_file}")
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to save thresholds: {e}")

    # Per-point combiner - propagate measured correlation to runtime
    if mode.startswith("CSK") or mode == "Hybrid":
        # Extract rho_cc_measured from calibration results
        rho_cc_measured = np.nan
        if mode.startswith("CSK"):
            # For CSK, get it from the last calibration run
            for level in range(cfg['pipeline'].get('csk_levels', 4)):
                cal_result = run_calibration_symbols(cal_cfg, level, mode='CSK', num_symbols=int(cfg.get('_cal_symbols_per_seed', 100)))
                if cal_result and 'rho_cc_measured' in cal_result:
                    measured_val = cal_result['rho_cc_measured']
                    if np.isfinite(measured_val):
                        rho_cc_measured = float(measured_val)
                        break
        elif mode == "Hybrid":
            # For Hybrid, get it from any symbol run
            cal_result = run_calibration_symbols(cal_cfg, 0, mode='Hybrid', num_symbols=int(cfg.get('_cal_symbols_per_seed', 100)))
            if cal_result and 'rho_cc_measured' in cal_result:
                measured_val = cal_result['rho_cc_measured']
                if np.isfinite(measured_val):
                    rho_cc_measured = float(measured_val)
        
        # Propagate measured correlation to runtime
        rho_cc = cal_cfg.get('noise', {}).get('rho_between_channels_after_ctrl', 0.5)
        rho_used = float(rho_cc_measured) if (np.isfinite(rho_cc_measured)) else float(rho_cc)
        thresholds['noise.rho_between_channels_after_ctrl'] = rho_used
        thresholds['rho_cc_measured'] = float(rho_cc_measured) if np.isfinite(rho_cc_measured) else float('nan')
        
        # Adaptive CTRL gating decision
        ctrl_use = bool(cfg['pipeline'].get('use_control_channel', True))
        if bool(cfg.get('_ctrl_auto', False)) and (mode.startswith('CSK') or mode == 'Hybrid'):
            rho_abs = abs(rho_cc_measured) if np.isfinite(rho_cc_measured) else 0.0
            # Simple correlation rule
            if rho_abs < float(cfg.get('_ctrl_auto_rho_min_abs', 0.10)):
                ctrl_use = False
                
            # Optional SNR gain rule (conservative approximation)
            if ctrl_use and bool(cfg.get('_ctrl_auto', False)):
                # For differential measurement: SNR gain ~ 1/(1-rho^2) vs single-ended
                if rho_abs > 0:
                    gain_linear = 1.0 / (1.0 - rho_abs**2)
                    gain_db = 10.0 * np.log10(gain_linear) if gain_linear > 1.0 else 0.0
                    min_gain_db = float(cfg.get('_ctrl_auto_min_gain_db', 0.0))
                    if gain_db < min_gain_db:
                        ctrl_use = False

        # Emit decisions into the thresholds override
        thresholds['use_control_channel'] = bool(ctrl_use)
        thresholds['ctrl_auto_applied'] = bool(cfg.get('_ctrl_auto', False))

    return thresholds

def calibrate_thresholds_cached(cfg: Dict[str, Any], seeds: List[int], recalibrate: bool = False) -> Dict[str, Union[float, List[float], str]]:
    """
    Memory + disk cached calibration. Persist JSON so multiple processes/runs reuse it.
    """
    cache_key = get_cache_key(cfg)
    
    # If recalibrating, clear both memory and disk cache
    if recalibrate:
        if cache_key in calibration_cache:
            del calibration_cache[cache_key]
        threshold_file = _thresholds_filename(cfg)
        if threshold_file.exists():
            try:
                threshold_file.unlink()
                print(f"🗑️  Cleared threshold cache: {threshold_file.name}")
            except Exception:
                pass  # Best effort
    
    # Check memory cache first
    if cache_key in calibration_cache:
        return calibration_cache[cache_key]
    
    # Compute thresholds (respecting the recalibrate flag)
    result = calibrate_thresholds(cfg, seeds, recalibrate=recalibrate, save_to_file=True, verbose=False)
    
    # Bound cache size to prevent memory bloat
    if len(calibration_cache) >= MAX_CACHE_SIZE:
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
    
    # Add missing defaults for optional OECT parameters to prevent KeyError
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
    # Add binding
    cfg['binding'] = cfg.get('binding', {})
    
    # ---- Calibration knobs (back‑compatible defaults) ----
    cfg['_cal_symbols_per_seed']       = int(cfg.get('_cal_symbols_per_seed', 100))
    cfg['_cal_min_samples_per_class']  = int(cfg.get('_cal_min_samples_per_class', 50))
    cfg['_cal_min_seeds']              = int(cfg.get('_cal_min_seeds', 4))
    cfg['_cal_max_seeds']              = int(cfg.get('_cal_max_seeds', 0))  # 0 → resolve to len(seeds) at runtime
    cfg['_cal_eps_rel']                = float(cfg.get('_cal_eps_rel', 0.01))
    cfg['_cal_patience']               = int(cfg.get('_cal_patience', 2))
    cfg['_cal_enable_es_mosk']         = bool(cfg.get('_cal_enable_es_mosk', True))
    cfg['_cal_enable_es_hybrid']       = bool(cfg.get('_cal_enable_es_hybrid', True))

    # Window guard tuning (existing)
    cfg['_min_decision_points'] = int(cfg.get('_min_decision_points', 4))

    # NEW: Dual‑channel CSK configuration defaults
    cfg['pipeline']['csk_dual_channel'] = cfg['pipeline'].get('csk_dual_channel', True)
    cfg['pipeline']['csk_combiner'] = cfg['pipeline'].get('csk_combiner', 'zscore')
    cfg['pipeline']['csk_leakage_frac'] = cfg['pipeline'].get(
        'csk_leakage_frac', cfg['pipeline'].get('non_specific_binding_factor', 0.0)
    )
    cfg['pipeline']['csk_store_combiner_meta'] = cfg['pipeline'].get('csk_store_combiner_meta', True)

    lod_cfg = cfg.get('lod_distances_um')
    normalized: Dict[str, List[int]] = {}
    if isinstance(lod_cfg, dict):
        for key, seq in lod_cfg.items():
            distances = _coerce_distance_sequence(seq)
            if distances:
                normalized[_canonical_mode_name(str(key))] = distances
    elif lod_cfg is not None:
        distances = _coerce_distance_sequence(lod_cfg)
        if distances:
            normalized['ALL'] = distances
    final_map: Dict[str, List[int]] = {mode: list(vals) for mode, vals in DEFAULT_MODE_DISTANCES.items()}
    if 'ALL' in normalized:
        final_map = {mode: list(normalized['ALL']) for mode in final_map}
    for mode_key, distances in normalized.items():
        if mode_key == 'ALL':
            continue
        final_map[mode_key] = list(distances)
    cfg['lod_distances_um'] = final_map

    return cfg

def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI argument overrides to configuration."""
    
    # Decision window overrides
    if args.decision_window_policy is not None:
        cfg.setdefault('detection', {})['decision_window_policy'] = args.decision_window_policy
    
    if args.decision_window_frac is not None:
        cfg.setdefault('detection', {})['decision_window_fraction'] = float(args.decision_window_frac)
        # Validate range
        if not (0.1 <= args.decision_window_frac <= 1.0):
            raise ValueError(f"--decision-window-frac must be between 0.1 and 1.0, got {args.decision_window_frac}")
    
    # Analysis overrides
    if args.allow_ts_exceed:
        cfg.setdefault('analysis', {})['allow_ts_exceed'] = True
    
    if args.ts_cap_s is not None:
        cfg.setdefault('analysis', {})['ts_cap_s'] = float(args.ts_cap_s)
    
    # Pipeline overrides
    if args.isi_memory_cap is not None:
        cfg.setdefault('pipeline', {})['isi_memory_cap_symbols'] = int(args.isi_memory_cap)
    
    if args.guard_factor is not None:
        cfg.setdefault('pipeline', {})['guard_factor'] = float(args.guard_factor)
        # Validate range
        if not (0.0 <= args.guard_factor <= 1.0):
            raise ValueError(f"--guard-factor must be between 0.0 and 1.0, got {args.guard_factor}")
    
    if getattr(args, 'guard_samples_cap', None) is not None:
        cfg.setdefault('analysis', {})['guard_samples_cap'] = float(args.guard_samples_cap)
    
    if getattr(args, 'csk_target', None) is not None:
        cfg.setdefault('pipeline', {})['csk_target_channel'] = str(args.csk_target)

    if getattr(args, 'csk_dual', None) is not None:
        cfg.setdefault('pipeline', {})['csk_dual_channel'] = (args.csk_dual == 'on')

    profile = str(getattr(args, 'channel_profile', 'tri')).lower()
    cfg.setdefault('pipeline', {})['channel_profile'] = profile
    if profile in ('single', 'dual'):
        cfg.setdefault('pipeline', {})['use_control_channel'] = False

    # NEW: LoD maximum Nm override
    if hasattr(args, 'lod_max_nm') and args.lod_max_nm is not None:
        cfg.setdefault('pipeline', {})['lod_nm_max'] = int(args.lod_max_nm)
    
    return cfg

def calculate_dynamic_symbol_period(distance_um: float, cfg: Dict[str, Any]) -> float:
    D_da = cfg['neurotransmitters']['DA']['D_m2_s']
    lambda_da = cfg['neurotransmitters']['DA']['lambda']
    D_eff = D_da / (lambda_da ** 2)
    time_95 = 3.0 * ((distance_um * 1e-6)**2) / D_eff
    guard_factor = cfg['pipeline'].get('guard_factor', 0.1)
    guard_time = guard_factor * time_95
    dt = float(cfg['sim']['dt_s'])
    raw = float(time_95 + guard_time)
    # NEW: configurable minimum symbol period (default 5s, was hardcoded 20s)
    min_Ts = float(cfg['pipeline'].get('min_symbol_period_s', 5.0))
    symbol_period = max(min_Ts, math.ceil(raw / dt) * dt)
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
    D = float(cfg['neurotransmitters']['DA']['D_m2_s'])
    lam = float(cfg['neurotransmitters']['DA']['lambda'])
    D_eff = D / (lam ** 2)
    time_95 = 3.0 * ((d_um * 1e-6) ** 2) / D_eff
    if time_95 <= 0:
        return 0.0
    tail = max(0.0, time_95 - Ts)
    return float(min(1.0, tail / time_95))



def _resolve_isi_overlap_warn_threshold(cfg: Dict[str, Any]) -> float:
    pipeline_cfg = cfg.get('pipeline', {})
    if isinstance(pipeline_cfg, dict):
        val = pipeline_cfg.get('isi_overlap_warn_threshold')
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    analysis_cfg = cfg.get('analysis')
    if isinstance(analysis_cfg, dict):
        val = analysis_cfg.get('isi_overlap_warn_threshold')
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.2


def _maybe_warn_isi_overlap(cfg: Dict[str, Any], ratio: float, context: str = "") -> None:
    if not np.isfinite(ratio):
        return
    threshold = _resolve_isi_overlap_warn_threshold(cfg)
    if threshold <= 0:
        return
    if ratio >= threshold:
        suffix = f" ({context})" if context else ""
        print(f"??  ISI overlap {ratio:.1%} exceeds {threshold:.0%} threshold{suffix}")


def _coerce_float(value: Any, default: float = float('nan')) -> float:
    """Best-effort float coercion that is type-checker friendly."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

def _enforce_min_window(cfg: Dict[str, Any], Ts: float) -> float:
    """
    Centralized minimum decision window enforcement.
    
    Args:
        cfg: Configuration dictionary
        Ts: Symbol period in seconds
        
    Returns:
        Minimum window satisfying all constraints
    """
    dt = float(cfg['sim']['dt_s'])
    min_pts = int(cfg.get('_min_decision_points', 4))
    # Prefer root-level override when set, else pipeline key, else 0
    min_win_cfg = float(cfg.get('_min_decision_window_s',
                          cfg['pipeline'].get('min_decision_window_s', 0.0)))
    return max(Ts, min_pts * dt, min_win_cfg)


def _calculate_guard_sampling_load(
    cfg: Dict[str, Any],
    guard_pairs: List[Tuple[float, float]]
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]], float]:
    """
    Evaluate total time-sample load for guard-factor candidates.

    Returns
    -------
    keep : list of tuples (guard_factor, Ts, total_samples)
        Combinations that satisfy the configured sample cap.
    skipped : list of tuples (guard_factor, Ts, total_samples)
        Combinations rejected because they exceed the cap.
    cap : float
        Active sample cap (<=0 means unlimited).
    """
    dt = float(cfg['sim']['dt_s'])
    seq_len = int(cfg['pipeline'].get('sequence_length', 1000))
    samples_cap = float(cfg.get('_guard_total_samples_cap', 0.0))

    keep: List[Tuple[float, float, float]] = []
    skipped: List[Tuple[float, float, float]] = []

    for guard, Ts in guard_pairs:
        n_samples = max(1, math.ceil(Ts / dt))
        total_samples = float(n_samples * seq_len)
        if samples_cap > 0 and total_samples > samples_cap:
            skipped.append((guard, Ts, total_samples))
        else:
            keep.append((guard, Ts, total_samples))

    return keep, skipped, samples_cap

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
    
    ENHANCED: Persists CSK configuration triple (M, target_channel, combiner) for plot provenance.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ✅ Add combiner metadata columns BEFORE creating DataFrame
    if row.get('mode', '').startswith('CSK') and row.get('csk_store_combiner_meta', True):
        if 'combiner' not in row:
            row['combiner'] = row.get('csk_selected_combiner', row.get('csk_combiner', 'zscore'))
        if 'sigma_da' not in row:
            row['sigma_da'] = row.get('noise_sigma_da', np.nan)
        if 'sigma_sero' not in row:
            row['sigma_sero'] = row.get('noise_sigma_sero', np.nan)
        if 'rho_cc' not in row:
            # Note: This should be populated from the config when creating the row
            # For now, use a fallback or remove if not available
            row['rho_cc'] = row.get('rho_cc', 0.0)  # Use direct key or default
        if 'leakage_frac' not in row:
            combiner = row.get('combiner', 'zscore')
            row['leakage_frac'] = row.get('csk_leakage_frac', 0.0) if combiner == 'leakage' else np.nan
        
        # NEW: Add CSK configuration triple for plot provenance
        row.setdefault('csk_levels', row.get('M', 4))
        row.setdefault('csk_target_channel', 'DA')
    
    # ✅ NOW create DataFrame with complete metadata
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

# ENHANCEMENT: Export the canonical value key formatter for consistency across modules
def canonical_value_key(v):
    """Convert a numeric parameter value to a standard string key for CSV/cache lookups."""
    if isinstance(v, (int, float)):
        return f"{float(v):.10g}"
    return str(v)

def _auto_refine_nm_points_from_df(df: pd.DataFrame,
                                   target: float = 0.01,
                                   extra_points: int = 2,
                                   nm_min: int = 50,
                                   nm_max: int = 500_000) -> List[int]:
    """
    Given a SER vs Nm dataframe, find the first (lo, hi) pair that brackets `target`
    (SER decreases with Nm), and return up to `extra_points` **log-spaced** Nm's
    between them. Already-existing Nm's are filtered out by the caller.

    Expects columns: one Nm column ('pipeline_Nm_per_symbol' or 'pipeline.Nm_per_symbol')
    and 'ser'.
    """
    if df is None or df.empty:
        return []

    # Resolve Nm column
    nm_col = None
    for c in ("pipeline_Nm_per_symbol", "pipeline.Nm_per_symbol"):
        if c in df.columns:
            nm_col = c
            break
    if nm_col is None or 'ser' not in df.columns:
        return []

    # Clean + sort + drop duplicate Nm's (keep last)
    d = df[[nm_col, 'ser']].copy()
    d[nm_col] = pd.to_numeric(d[nm_col], errors='coerce')
    d['ser'] = pd.to_numeric(d['ser'], errors='coerce')
    d = d.dropna().drop_duplicates(subset=[nm_col], keep='last').sort_values(nm_col)

    if d.empty:
        return []

    # Find first bracket: previous SER > target and current SER <= target
    nms = d[nm_col].to_numpy(dtype=float)
    sers = d['ser'].to_numpy(dtype=float)

    lo, hi = None, None
    for i in range(1, len(d)):
        prev, curr = float(sers[i-1]), float(sers[i])
        if (prev > target) and (curr <= target):
            lo = int(round(nms[i-1]))
            hi = int(round(nms[i]))
            break

    if lo is None or hi is None:
        # No bracket in coarse grid -> nothing to refine (you could optionally
        # schedule an out-of-range probe here, but we keep it minimal).
        return []

    # Clamp and sanity
    if lo > hi:
        lo, hi = hi, lo
    lo = max(nm_min, lo)
    hi = min(nm_max, hi)
    if hi - lo <= 1:
        return []

    # Generate log-spaced splits between lo and hi:
    #   extra_points=1 -> one midpoint (geom. mean)
    #   extra_points=2 -> thirds in log space, etc.
    k = max(1, int(extra_points))
    mids: List[int] = []
    for j in range(1, k + 1):
        alpha = j / (k + 1)  # 1/(k+1), 2/(k+1), ...
        mid = int(round((lo ** (1.0 - alpha)) * (hi ** alpha)))
        # Keep strictly inside the bracket
        mid = min(max(mid, lo + 1), hi - 1)
        mids.append(mid)

    # Unique, sorted
    mids = sorted({m for m in mids if lo < m < hi})
    return mids

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
    parser.add_argument("--isi-sweep", choices=["auto", "always", "never"],
                        default="always",
                        help="Run ISI trade-off sweep: always (default), auto (only when ISI enabled), or never.")
    parser.add_argument("--debug-calibration", action="store_true", help="Print detailed calibration information")
    parser.add_argument("--channel-profile", choices=["tri", "dual", "single"], default="tri",
                        help="Physical channel setup: tri (DA+SERO+CTRL), dual (DA+SERO), single (DA only).")
    parser.add_argument("--csk-level-scheme", choices=["uniform", "zero-based"], default="uniform",
                       help="CSK level mapping scheme")
    parser.add_argument("--resume", action="store_true", help="Resume: skip finished values and append results as we go")
    parser.add_argument("--with-ctrl", dest="use_ctrl", action="store_true", help="Use CTRL differential subtraction")
    parser.add_argument("--no-ctrl", dest="use_ctrl", action="store_false", help="Disable CTRL subtraction (ablation)")
    parser.add_argument("--progress", choices=["tqdm", "rich", "gui", "none"], default="tqdm",
                    help="Progress UI backend")
    parser.add_argument("--nt-pairs", type=str, default="", help="CSV nt-pairs for CSK sweeps")

    # --- Baseline / variant helpers ---
    parser.add_argument(
        "--variant", type=str, default="",
        help="Suffix appended to CSV basenames (e.g., _single_DA, _dual)."
    )
    parser.add_argument(
        "--csk-target", choices=["DA", "SERO"], default=None,
        help="Override CSK target channel (single-channel baselines)."
    )
    parser.add_argument(
        "--csk-dual", choices=["on", "off"], default=None,
        help="Force dual-channel CSK combiner on/off for this run."
    )
    parser.add_argument("--watchdog-secs", type=int, default=1800,
                        help="Soft timeout for seed completion before retry hint (default: 1800s/30min)")
    parser.add_argument("--nonlod-watchdog-secs", type=int, default=3600,
                        help="Timeout for non-LoD sweeps (guard/frontier, SER vs Nm, etc.); <=0 disables.")
    parser.add_argument("--guard-max-ts", type=float, default=0.0,
                        help="Cap guard-factor sweeps when symbol period exceeds this many seconds (0 disables).")
    parser.add_argument("--target-ci", type=float, default=0.004,
                        help="If >0, stop adding seeds once Wilson 95% CI half-width <= target. 0 disables.")
    parser.add_argument("--min-ci-seeds", type=int, default=8,
                        help="Minimum seeds required before adaptive CI stopping can trigger.")
    parser.add_argument("--lod-screen-delta", type=float, default=1e-4,
                        help="Hoeffding screening significance (delta) for early-stop LoD tests.")
    parser.add_argument(
        "--distances",
        action="append",
        default=None,
        metavar="MODE=LIST",
        help=("Override LoD distance grid per mode. Example: --distances MoSK=25,35,45 --distances CSK=15,25. Use 'ALL=' to apply to all modes."),
    )
    parser.add_argument("--lod-num-seeds",
                        type=str,
                        default="<=100:6,<=150:8,>150:10",
                        help=(
                            "LoD seed schedule. Options:\n"
                            "  N                 -> use fixed N seeds for LoD search\n"
                            "  min,max           -> linearly scale from min at 25µm to max at 200µm\n"
                            "  rules             -> e.g. '<=100:6,<=150:8,>150:10'\n"
                            "Final LoD validation always uses the full seed set."
                        ))
    parser.add_argument("--lod-seq-len",
                        type=int,
                        default=250,
                        help="If set, temporarily override sequence_length during LoD search only.")
    parser.add_argument("--lod-validate-seq-len", 
                        type=int, 
                        default=None,
                        help="If set, override sequence_length during final LoD validation only (not search).")
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
    parser.add_argument("--max-lod-validation-seeds", type=int, default=12,
                        help="Cap the number of seeds used for LoD validation (default: use all seeds).")
    parser.add_argument("--max-symbol-duration-s", type=float, default=None,
                        help="Skip LoD search at distances where symbol period exceeds this limit (seconds).")
    parser.add_argument("--analytic-lod-bracket", action="store_true",
                    help="Use Gaussian SER approximation for tighter LoD bracketing (experimental).")
    parser.set_defaults(use_ctrl=True, analytic_lod_bracket=True)
    # Adaptive calibration tuning (Issue 1 optimization)
    parser.add_argument("--cal-eps-rel", type=float, default=0.01,
                        help="Adaptive calibration convergence threshold (relative change, default: 0.01)")
    parser.add_argument("--cal-patience", type=int, default=2,
                        help="Wait N iterations before stopping convergence (default: 2)")
    parser.add_argument("--cal-min-seeds", type=int, default=4,
                        help="Minimum seeds before early stopping can trigger (default: 4)")
    parser.add_argument("--cal-min-samples", type=int, default=50,
                        help="Minimum samples per class for stable thresholds (default: 50)")
    parser.add_argument("--nm-grid", type=str, default="",
                        help="Comma-separated Nm values for SER sweeps (e.g., 200,500,1000,2000). "
                             "If not provided, uses cfg['Nm_range'] from YAML.")
    # NEW: Decision window and ISI optimization toggles
    parser.add_argument("--decision-window-policy", choices=["fixed", "fraction_of_Ts", "full_Ts"], default=None,
                   help="Override decision window policy")
    parser.add_argument("--decision-window-frac", type=float, default=None,
                   help="Decision window fraction for fraction_of_Ts policy")
    parser.add_argument("--allow-ts-exceed", action="store_true",
                   help="Allow Ts to exceed limits during LoD sweeps")
    parser.add_argument("--ts-cap-s", type=float, default=None,
                   help="Symbol period cap in seconds")
    parser.add_argument("--isi-memory-cap", type=int, default=None,
                   help="ISI memory cap in symbols")
    parser.add_argument("--guard-factor", type=float, default=None,
                   help="Override guard factor for ISI calculations")
    parser.add_argument("--guard-samples-cap", type=float, default=None,
                   help="Per-seed sample cap for guard-factor sweeps (0 disables cap)")
    parser.add_argument("--lod-distance-timeout-s", type=float, default=7200.0,
                        help="Per-distance time budget during LoD analysis. <=0 disables timeout.")
    parser.add_argument("--lod-distance-concurrency", type=int, default=8,
                        help="How many distances to run concurrently in LoD sweep (default: 8).")
    parser.add_argument("--lod-max-nm", type=int, default=1000000,
                        help="Upper bound for Nm during LoD search (default: 1000000).")
    parser.add_argument("--ts-warn-only", action="store_true",
                        help="Issue warnings for long Ts instead of skipping (overrides all Ts limits)")
    parser.add_argument("--lod-skip-retry", action="store_true",
                        help="On resume, do not retry distances whose previous LoD attempt failed (keep NaN).")
    
    # ------ SER auto-refine near target SER ------
    parser.add_argument("--ser-refine", action="store_true",
                   help="After coarse SER vs Nm sweep, auto-run a few Nm points that bracket the target SER.")
    parser.add_argument("--ser-target", type=float, default=0.01,
                   help="Target SER for auto-refine (default: 0.01).")
    parser.add_argument("--ser-refine-points", type=int, default=4,
                   help="How many log-spaced Nm points to add between the bracketing Nm pair (default: 4).")

    # Window guard tuning (Issue 2 optimization)
    parser.add_argument("--min-decision-points", type=int, default=4,
                        help="Minimum time points for window guard (default: 4)")
    
    # Adaptive CTRL control
    parser.add_argument("--ctrl-auto", action="store_true",
                       help="Enable adaptive CTRL on/off based on measured correlation")
    parser.add_argument("--ctrl-rho-min-abs", type=float, default=0.10,
                       help="Minimum absolute correlation threshold for CTRL (default: 0.10)")
    parser.add_argument("--ctrl-snr-min-gain-db", type=float, default=0.0,
                       help="Minimum SNR gain in dB to keep CTRL enabled (default: 0.0)")
    
    args = parser.parse_args()
    
    raw_distance_specs = args.distances or []
    if raw_distance_specs is None:
        raw_distance_specs = []
    try:
        args.distances_by_mode = parse_distance_overrides(raw_distance_specs)
    except ValueError as exc:
        parser.error(str(exc))
    args.distances = list(raw_distance_specs)
    
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
def _pava_monotone_means(mu: List[float], w: Optional[List[float]] = None) -> List[float]:
    """Pool-adjacent-violators for nondecreasing means. Returns adjusted means."""
    import numpy as np
    
    if not mu:
        return []
    
    # Convert to numpy for computation but ensure proper types
    x_array = np.array(mu, dtype=float)
    if w is None:
        w_array = np.ones_like(x_array, dtype=float)
    else:
        w_array = np.array(w, dtype=float)
    
    # Make working copies to avoid assignment type issues
    x = x_array.copy()
    w_weights = w_array.copy()
    
    # Nondecreasing PAVA
    i = 0
    while i < len(x) - 1:
        if x[i] <= x[i+1] + 1e-15:
            i += 1
            continue
        # merge pools
        total_w = w_weights[i] + w_weights[i+1]
        if total_w > 0:
            merged = (w_weights[i]*x[i] + w_weights[i+1]*x[i+1]) / total_w
            # Assign to both positions
            x[i] = merged
            x[i+1] = merged
            w_weights[i] = total_w
            w_weights[i+1] = total_w
        
        j = i
        while j > 0 and x[j-1] > x[j] + 1e-15:
            total_w = w_weights[j-1] + w_weights[j]
            if total_w > 0:
                merged = (w_weights[j-1]*x[j-1] + w_weights[j]*x[j]) / total_w
                x[j-1] = merged
                x[j] = merged
                w_weights[j-1] = total_w
                w_weights[j] = total_w
            j -= 1
        i = max(j, 0)
    
    # Convert back to list of floats to match return type
    return [float(val) for val in x.tolist()]

def _adjacent_threshold_ml(mu0: float, mu1: float, s0: float, s1: float) -> float:
    try:
        from src.mc_detection.algorithms import calculate_ml_threshold
    except ImportError:
        from src.detection import calculate_ml_threshold
    return float(calculate_ml_threshold(float(mu0), float(mu1), max(float(s0),1e-15), max(float(s1),1e-15)))

def _adjacent_threshold_map(mu0: float, mu1: float, s0: float, s1: float, p0: float, p1: float) -> float:
    import numpy as np
    # ML threshold shifted by log-prior ratio (Gaussian log-likelihoods differ by log p)
    # Solve t* for unequal variance via numeric adjust around ML; fallback to midpoint.
    t_ml = _adjacent_threshold_ml(mu0, mu1, s0, s1)
    # Approximate shift for priors in equal-variance case
    if abs(s0 - s1) / max(s0, s1) < 1e-3:
        # equal variance: shift by Δ = (s^2)*log(p1/p0)/(mu1-mu0)
        if abs(mu1 - mu0) > 1e-15:
            return float(t_ml + (s0**2) * (np.log(p1/p0)) / (mu1 - mu0))
    return float(t_ml)

def run_calibration_symbols(cfg: Dict[str, Any], symbol: int, mode: str, num_symbols: int = 100) -> Optional[Dict[str, Any]]:
    try:
        # Import the dual-channel helper
        from src.pipeline import _csk_dual_channel_Q
        
        cal_cfg = deepcopy(cfg)
        cal_cfg['pipeline']['sequence_length'] = num_symbols
        cal_cfg['disable_progress'] = True
        tx_symbols = [symbol] * num_symbols
        q_da_values: List[float] = []
        q_sero_values: List[float] = []
        decision_stats: List[float] = []
        dt = cal_cfg['sim']['dt_s']
        d = cal_cfg.setdefault('detection', {})
        detection_window_s = d.get('decision_window_s', cal_cfg['pipeline']['symbol_period_s'])
        sigma_da, sigma_sero = calculate_proper_noise_sigma(cal_cfg, detection_window_s)
        
        # Initialize CSK dual-channel parameters with defaults
        target_channel = cal_cfg['pipeline'].get('csk_target_channel', 'DA')
        combiner = cal_cfg['pipeline'].get('csk_combiner', 'zscore')
        use_dual = bool(cal_cfg['pipeline'].get('csk_dual_channel', True))
        leakage = float(cal_cfg['pipeline'].get('csk_leakage_frac', 0.0))
        
        # Enhancement 2: Use 0.5 fallback if noise model is symmetric
        rho_pre = float(cal_cfg.get('noise', {}).get('rho_corr',
                        cal_cfg.get('noise', {}).get('rho_correlated', 0.9)))
        use_ctrl = bool(cal_cfg['pipeline'].get('use_control_channel', True))

        noise_cfg_cal = cal_cfg.get('noise', {})
        # FIX: Logic should check if rho_correlated is NOT explicitly set (i.e., using default)
        # If rho_between_channels_after_ctrl is not set AND model is symmetric, use 0.5
        if 'rho_between_channels_after_ctrl' not in noise_cfg_cal:
            # Only use 0.5 if this is clearly a symmetric triplet setup
            rho_post_default = 0.5 if use_ctrl else 0.0
        else:
            rho_post_default = noise_cfg_cal['rho_between_channels_after_ctrl']
        rho_post = float(noise_cfg_cal.get('rho_between_channels_after_ctrl', rho_post_default))
        rho_cc   = rho_post if (use_ctrl and mode != "MoSK") else rho_pre
        rho_cc   = max(-1.0, min(1.0, rho_cc))
        
        # Use deterministic seeding for calibration consistency
        seed = cal_cfg['pipeline'].get('random_seed', 0)
        rng = np.random.default_rng(seed)
        
        # Auto-measure ρ after CTRL for validation (Enhancement 1)
        rho_cc_measured = np.nan
        if use_ctrl and (mode.startswith('CSK') or mode == 'Hybrid'):
            try:
                # Generate noise-only samples for correlation measurement
                # FIX D: Improve ρ measurement stability at low Nm
                base_noise_samples = 20
                # For very low Nm analysis, use more samples for stable correlation estimation
                nm_value = cfg['pipeline'].get('Nm_per_symbol', 1e6)
                if nm_value < 1000:  # Very low Nm regime
                    noise_samples = 100
                elif nm_value < 10000:  # Low Nm regime
                    noise_samples = 50
                else:
                    noise_samples = base_noise_samples
                q_da_noise, q_sero_noise = [], []
                
                # Temporarily set Nm to zero for noise-only measurement
                cal_cfg_noise = deepcopy(cal_cfg)
                cal_cfg_noise['pipeline']['Nm_per_symbol'] = 1e-6  # Minimal signal
                
                # Calculate detection samples for noise measurement
                n_total_samples_noise = int(cal_cfg_noise['pipeline']['symbol_period_s'] / dt)
                n_detect_samples_noise = min(int(detection_window_s / dt), n_total_samples_noise)
                
                for _ in range(noise_samples):
                    ig_n, ia_n, ic_n, _ = _single_symbol_currents(0, [], cal_cfg_noise, rng)
                    
                    # Apply CTRL subtraction (same as main path)
                    sig_da_n = ig_n - ic_n
                    sig_sero_n = ia_n - ic_n
                    
                    # Integrate over decision window
                    q_da_n = float(np.trapezoid(sig_da_n[:n_detect_samples_noise], dx=dt))
                    q_sero_n = float(np.trapezoid(sig_sero_n[:n_detect_samples_noise], dx=dt))
                    
                    q_da_noise.append(q_da_n)
                    q_sero_noise.append(q_sero_n)
                
                # Measure empirical correlation
                if len(q_da_noise) >= 3:
                    rho_cc_measured = float(np.corrcoef(q_da_noise, q_sero_noise)[0, 1])
                    if not np.isfinite(rho_cc_measured):
                        rho_cc_measured = np.nan
                        
            except Exception:
                rho_cc_measured = np.nan
        else:
            rho_cc_measured = np.nan
        
        # Main symbol generation loop
        for s_tx in tx_symbols:
            ig, ia, ic, Nm_actual = _single_symbol_currents(s_tx, [], cal_cfg, rng)
            n_total_samples = len(ig)
            n_detect_samples = min(int(detection_window_s / dt), n_total_samples)
            if n_detect_samples <= 1:
                continue
                
            # Tail-gated integration
            tail = float(cal_cfg['pipeline'].get('csk_tail_fraction', 1.0))
            tail = min(max(tail, 0.1), 1.0)
            i0 = int((1.0 - tail) * n_detect_samples)

            use_ctrl = bool(cal_cfg['pipeline'].get('use_control_channel', True))
            # For MoSK, do NOT subtract CTRL from the charges; for CSK/Hybrid keep the existing behavior.
            subtract_for_q = (mode != "MoSK") and use_ctrl
            
            q_da_series   = (ig - ic) if subtract_for_q else ig
            q_sero_series = (ia - ic) if subtract_for_q else ia
            
            q_da   = float(np.trapezoid(q_da_series[i0:n_detect_samples],   dx=dt))
            q_sero = float(np.trapezoid(q_sero_series[i0:n_detect_samples], dx=dt))
            
            q_da_values.append(q_da); q_sero_values.append(q_sero)
        
        for q_da, q_sero in zip(q_da_values, q_sero_values):
            if mode == "MoSK":
                # Get q_eff signs to match runtime logic
                q_eff_da = float(cal_cfg['neurotransmitters']['DA']['q_eff_e'])
                q_eff_sero = float(cal_cfg['neurotransmitters']['SERO']['q_eff_e'])
                sign_da = 1.0 if q_eff_da >= 0 else -1.0
                sign_sero = 1.0 if q_eff_sero >= 0 else -1.0
                
                # FIX: For MoSK, never subtract CTRL from the statistic itself.
                # Use CTRL only via the (reduced) cross-correlation in the denominator.
                cal_use_ctrl = bool(cal_cfg['pipeline'].get('use_control_channel', True))
                # For MoSK, use pre-CTRL correlation since we don't subtract CTRL from charges
                if mode == "MoSK":
                    rho_cc = float(cal_cfg.get('noise', {}).get('rho_corr', 0.9))
                else:
                    rho_cc = float(cal_cfg.get('noise', {}).get(
                        'rho_between_channels_after_ctrl' if cal_use_ctrl else 'rho_corr',
                        0.9
                    ))
                sigma_diff = math.sqrt(
                    sigma_da*sigma_da + sigma_sero*sigma_sero - 2.0*rho_cc*sigma_da*sigma_sero
                )
                if sigma_diff <= 1e-15:
                    sigma_diff = 1e-15
                # Always use optimal sign-aware difference (no CTRL subtraction)
                D = (sign_da * q_da - sign_sero * q_sero) / max(sigma_diff, 1e-15)
                decision_stats.append(float(D))
            elif mode.startswith("CSK"):
                if use_dual:
                    Q = _csk_dual_channel_Q(
                        q_da=q_da, q_sero=q_sero,
                        sigma_da=sigma_da, sigma_sero=sigma_sero,
                        rho_cc=rho_cc, combiner=combiner, leakage_frac=leakage,
                        target=target_channel,
                        cfg=cal_cfg  # FIX: Pass cfg for shrinkage logic
                    )
                else:
                    # Legacy single-channel
                    Q = q_da if target_channel == 'DA' else q_sero
                decision_stats.append(Q)
            elif mode == "Hybrid":
                mol_type = symbol >> 1  # 0: DA, 1: SERO (which channel carries amplitude)
                combiner = str(cal_cfg['pipeline'].get('hybrid_combiner', cal_cfg['pipeline'].get('csk_combiner','zscore')))
                leakage  = float(cal_cfg['pipeline'].get('hybrid_leakage_frac', cal_cfg['pipeline'].get('csk_leakage_frac', 0.0)))
                target   = 'DA' if mol_type == 0 else 'SERO'
                Q_amp = _csk_dual_channel_Q(
                    q_da=q_da, q_sero=q_sero,
                    sigma_da=sigma_da, sigma_sero=sigma_sero,
                    rho_cc=rho_cc, combiner=combiner, leakage_frac=leakage,
                    target=target, cfg=cal_cfg
                )
                decision_stats.append(float(Q_amp))
        
        # Enhanced return with combiner metadata for CSK
        # Keep aux raw charges for flexible combiners
        aux_q = list(zip(q_da_values, q_sero_values))
        result = {
            "q_values": decision_stats,
            "aux_q": aux_q
        }
        
        # Add combiner metadata for CSV traceability if enabled
        if (mode.startswith('CSK') and 
            cal_cfg['pipeline'].get('csk_store_combiner_meta', True)):
            result.update({
                'combiner': combiner,
                'rho_cc': rho_cc,
                'rho_cc_measured': rho_cc_measured,  # Enhancement 1: Store measured correlation
                'leakage_frac': leakage if combiner == 'leakage' else np.nan
            })
        
        # FIX A: Always include rho_cc_measured for Hybrid mode (not just CSK)
        try:
            if 'rho_cc_measured' in locals() and np.isfinite(rho_cc_measured):
                result['rho_cc_measured'] = float(rho_cc_measured)
        except Exception:
            pass
            
        return result
    except Exception:
        return None

def _measure_rho_for_seed(cfg: Dict[str, Any], mode: str) -> float:
    """Measure cross-channel correlation for a specific seed."""
    if not cfg['pipeline'].get('use_control_channel', True):
        return float('nan')
    if mode not in ('CSK', 'Hybrid'):
        return float('nan')
    
    cal = deepcopy(cfg)
    cal['pipeline']['Nm_per_symbol'] = 1e-6  # Minimal signal
    dt = float(cal['sim']['dt_s'])
    win = float(cal.get('detection', {}).get('decision_window_s',
                  cal['pipeline'].get('symbol_period_s', dt)))
    n = max(1, int(win / dt))
    qd, qs = [], []
    rng = np.random.default_rng(cal['pipeline'].get('random_seed', 0))
    
    for _ in range(20):
        ig, ia, ic, _ = _single_symbol_currents(0, [], cal, rng)
        sig_da = ig - ic
        sig_se = ia - ic
        qd.append(float(np.trapezoid(sig_da[:n], dx=dt)))
        qs.append(float(np.trapezoid(sig_se[:n], dx=dt)))
    
    r = float(np.corrcoef(qd, qs)[0, 1]) if len(qd) >= 3 else float('nan')
    return r if np.isfinite(r) else float('nan')

# ============= RUNTIME WORKERS =============
def run_single_instance(config: Dict[str, Any], seed: int, attach_isi_meta: bool = True) -> Optional[Dict[str, Any]]:
    """Run a single sequence; returns None on failure (so callers must filter)."""
    try:
        cfg_run = deepcopy(config)
        cfg_run['pipeline']['random_seed'] = int(seed)
        mode = cfg_run['pipeline']['modulation']

        # Add per-seed correlation measurement
        rho_seed = _measure_rho_for_seed(cfg_run, mode)
        if np.isfinite(rho_seed):
            cfg_run.setdefault('noise', {})['rho_between_channels_after_ctrl'] = float(np.clip(rho_seed, -1.0, 1.0))

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
                         thresholds_override: Optional[Dict[str, Union[float, List[float], str]]] = None,
                         sweep_name: str = "ser_vs_nm", cache_tag: Optional[str] = None,
                         recalibrate: bool = False) -> Optional[Dict[str, Any]]:
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
            dt = float(cfg_run['sim']['dt_s'])
            min_pts = int(cfg_run.get('_min_decision_points', 4))
            min_win = _enforce_min_window(cfg_run, new_symbol_period)
            cfg_run['pipeline']['symbol_period_s'] = new_symbol_period
            cfg_run['pipeline']['time_window_s'] = max(cfg_run['pipeline'].get('time_window_s', 0.0), min_win)
            cfg_run.setdefault('detection', {})['decision_window_s'] = min_win
            if cfg_run['pipeline'].get('enable_isi', False):
                D_da = cfg_run['neurotransmitters']['DA']['D_m2_s']
                lambda_da = cfg_run['neurotransmitters']['DA']['lambda']
                D_eff = D_da / (lambda_da ** 2)
                time_95 = 3.0 * ((cast(float, param_value) * 1e-6)**2) / D_eff
                guard_factor = cfg_run['pipeline'].get('guard_factor', 0.1)
                isi_memory = math.ceil((1 + guard_factor) * time_95 / new_symbol_period)
                cfg_run['pipeline']['isi_memory_symbols'] = isi_memory

        # Guard factor updates (ISI sweep): recompute Ts/window + thresholds
        if param_name == 'pipeline.guard_factor':
            # Update symbol period per the new guard factor, keeping distance fixed
            dist = float(cfg_run['pipeline']['distance_um'])
            new_symbol_period = calculate_dynamic_symbol_period(dist, cfg_run)
            dt = float(cfg_run['sim']['dt_s'])
            min_pts = int(cfg_run.get('_min_decision_points', 4))
            min_win = _enforce_min_window(cfg_run, new_symbol_period)
            cfg_run['pipeline']['symbol_period_s'] = new_symbol_period
            cfg_run['pipeline']['time_window_s'] = max(cfg_run['pipeline'].get('time_window_s', 0.0), min_win)
            cfg_run.setdefault('detection', {})['decision_window_s'] = min_win
            if cfg_run['pipeline'].get('enable_isi', False):
                D_da = cfg_run['neurotransmitters']['DA']['D_m2_s']
                lambda_da = cfg_run['neurotransmitters']['DA']['lambda']
                D_eff = D_da / (lambda_da ** 2)
                time_95 = 3.0 * ((dist * 1e-6)**2) / D_eff
                guard_factor = float(param_value)
                isi_memory = math.ceil((1 + guard_factor) * time_95 / new_symbol_period)
                cfg_run['pipeline']['isi_memory_symbols'] = isi_memory

        # Apply consistent window guard for Nm_per_symbol sweeps (symmetry with distance sweeps)
        elif param_name == 'pipeline.Nm_per_symbol':
            # Apply consistent window guard for symmetry
            dt = float(cfg_run['sim']['dt_s'])
            min_pts = int(cfg_run.get('_min_decision_points', 4))
            min_win = _enforce_min_window(cfg_run, cfg_run['pipeline']['symbol_period_s'])
            cfg_run['pipeline']['time_window_s'] = max(cfg_run['pipeline'].get('time_window_s', 0.0), min_win)
            cfg_run.setdefault('detection', {})['decision_window_s'] = min_win
        elif param_name in ['oect.gm_S', 'oect.C_tot_F']:
            min_win = _enforce_min_window(cfg_run, cfg_run['pipeline']['symbol_period_s'])
            cfg_run['pipeline']['time_window_s'] = max(cfg_run['pipeline'].get('time_window_s', 0.0), min_win)
            cfg_run.setdefault('detection', {})['decision_window_s'] = min_win

        # Thresholds: use override if supplied, else cached calibration
        if thresholds_override is not None:
            for k, v in thresholds_override.items():
                if isinstance(k, str) and k.startswith('noise.'):
                    cfg_run.setdefault('noise', {})[k.split('.', 1)[1]] = v
                else:
                    cfg_run['pipeline'][k] = v
        elif cfg_run['pipeline']['modulation'] in ['MoSK', 'CSK', 'Hybrid'] and \
             param_name in ['pipeline.Nm_per_symbol', 'pipeline.distance_um', 'pipeline.guard_factor', 'oect.gm_S', 'oect.C_tot_F']:
            cal_seeds = list(range(10))
            thresholds = calibrate_thresholds_cached(cfg_run, cal_seeds, recalibrate)
            for k, v in thresholds.items():
                if not str(k).startswith("__"):
                    cfg_run['pipeline'][k] = v
            if debug_calibration and cfg_run['pipeline']['modulation'] == 'CSK':
                target_ch = cfg_run['pipeline'].get('csk_target_channel', 'DA').lower()
                key = f'csk_thresholds_{target_ch}'
                if key in cfg_run['pipeline']:
                    print(f"[DEBUG] CSK Thresholds @ {param_value}: {cfg_run['pipeline'][key]}")

        # Run the instance and attach per-run ISI metrics
        result = run_single_instance(cfg_run, seed, attach_isi_meta=True)
        if result is not None:
            # Tag the in-memory result so mixed cached+fresh paths dedupe correctly
            try:
                result["__seed"] = int(seed)
            except Exception:
                pass
            mode = cfg_base['pipeline']['modulation']
            # FIX: Use effective CTRL setting from thresholds_override
            eff_use_ctrl = bool((thresholds_override or {}).get('use_control_channel',
                                cfg_run['pipeline'].get('use_control_channel', True)))
            result_safe = cast(Dict[str, Any], _json_safe(result))
            write_seed_cache(mode, sweep_name, param_value, seed, result_safe, eff_use_ctrl, cache_tag)
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
              pm: Optional[ProgressManager] = None,
              sweep_key: Optional[Any] = None,
              parent_key: Optional[Any] = None,
              recalibrate: bool = False) -> pd.DataFrame:
    """
    ENHANCED: Marks resume context for threshold management.
    """
    """
    Parameter sweep with parallelization; returns aggregated df.
    Writes each completed value's row immediately if persist_csv is given.
    """
    # Mark resume context for threshold management
    if resume:
        cfg['_resume_active'] = True
    
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
    thresholds_map: Dict[Union[float, int], Dict[str, Union[float, List[float], str]]] = {}
    cal_seeds: List[int] = []
    if cfg['pipeline']['modulation'] in ['MoSK', 'CSK', 'Hybrid'] and \
       sweep_param in ['pipeline.Nm_per_symbol', 'pipeline.distance_um', 'pipeline.guard_factor', 'oect.gm_S', 'oect.C_tot_F']:
        cal_seeds = list(range(10))
    job_bar: Optional[Any] = None
    try:
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
                    dt = float(cfg_v['sim']['dt_s'])
                    min_pts = int(cfg_v.get('_min_decision_points', 4))
                    min_win = _enforce_min_window(cfg_v, Ts)
                    cfg_v['pipeline']['time_window_s'] = max(cfg_v['pipeline'].get('time_window_s', 0.0), min_win)
                    cfg_v['detection']['decision_window_s'] = min_win
                elif sweep_param == 'pipeline.guard_factor':
                    dist = float(cfg_v['pipeline']['distance_um'])
                    Ts = calculate_dynamic_symbol_period(dist, cfg_v)
                    cfg_v['pipeline']['symbol_period_s'] = Ts
                    dt = float(cfg_v['sim']['dt_s'])
                    min_pts = int(cfg_v.get('_min_decision_points', 4))
                    min_win = _enforce_min_window(cfg_v, Ts)
                    cfg_v['pipeline']['time_window_s'] = max(cfg_v['pipeline'].get('time_window_s', 0.0), min_win)
                    cfg_v['detection']['decision_window_s'] = min_win
                elif sweep_param in ['oect.gm_S', 'oect.C_tot_F']:
                    Ts = float(cfg_v['pipeline'].get('symbol_period_s', calculate_dynamic_symbol_period(float(cfg_v['pipeline'].get('distance_um', 50.0)), cfg_v)))
                    min_win = _enforce_min_window(cfg_v, Ts)
                    cfg_v['pipeline']['symbol_period_s'] = Ts
                    cfg_v['pipeline']['time_window_s'] = max(cfg_v['pipeline'].get('time_window_s', 0.0), min_win)
                    cfg_v.setdefault('detection', {})['decision_window_s'] = min_win
                thresholds_map[v] = calibrate_thresholds_cached(cfg_v, cal_seeds, recalibrate)
    
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
        is_lod_sweep = (sweep_param == 'pipeline.distance_um')
        if is_lod_sweep:
            timeout_s = int(cfg.get('_watchdog_secs', 600))
        else:
            timeout_s = int(cfg.get('_nonlod_watchdog_secs', cfg.get('_watchdog_secs', 600)))
        timeout_wait = timeout_s if timeout_s and timeout_s > 0 else None
    
        for v in values_to_run:
            # FIX: Get effective CTRL state for this value
            use_ctrl_for_v = bool(thresholds_map.get(v, {}).get('use_control_channel',
                                cfg['pipeline'].get('use_control_channel', True)))
            
            # Split cached vs missing seeds
            cached_results: List[Dict[str, Any]] = []
            seeds_to_run: List[int] = []
            if resume:
                for s in seeds:
                    r = read_seed_cache(mode_name, sweep_folder, v, s, use_ctrl_for_v, cache_tag)  # FIX: use effective
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
                        sweep_name=sweep_folder, cache_tag=cache_tag, recalibrate=recalibrate
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
                    try:
                        done_fut = next(as_completed(pending, timeout=timeout_wait))
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
                            print(f"        ??  Result extraction failed for {sweep_param}={v}: {e}")
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
                        timeout_desc = f"{timeout_s}s" if timeout_wait is not None else "inf"
                        print(f"        ??  Timeout ({timeout_desc}) for {sweep_param}={v}, {len(pending)} futures pending")
                        # Pick a reproducible future to retry: the one with the smallest seed
                        if pending:
                            to_retry = min(pending, key=lambda f: fut_seed.get(f, 1<<31))
                            seed_r = fut_seed.get(to_retry, seeds_to_run[0] if seeds_to_run else 12345)  # FIX: Provide default seed
                            # Enhanced logging for timeout retry tracking
                            print(f"        ?? Timeout details: {sweep_param}={v}, seed={seed_r}, timeout={timeout_desc}, pending_count={len(pending)}, worker_count={getattr(global_pool, '_max_workers', 'unknown')}")
                            # Try to cancel the old future if it hasn't started
                            if to_retry.cancel():
                                print(f"        ? Cancelled stale future for seed {seed_r}")
                                pending.remove(to_retry)
                                fut_seed.pop(to_retry, None)
                            else:
                                print(f"        ??  Could not cancel running future for seed {seed_r}")
                            print(f"        ?? Retrying seed {seed_r} for {sweep_param}={v}")
                            retry_tag = f"{cache_tag}_retry" if cache_tag else "retry"
                            retry_fut = pool.submit(run_param_seed_combo, cfg, sweep_param, v, seed_r,
                                                    debug_calibration, thresholds_override,
                                                    sweep_name=sweep_folder, cache_tag=retry_tag, recalibrate=recalibrate)
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
                all_a.extend(cast(List[float], r.get('stats_da', [])))
                all_b.extend(cast(List[float], r.get('stats_sero', [])))
            snr_lin = calculate_snr_from_stats(all_a, all_b) if all_a and all_b else 0.0
            snr_db = (10.0 * float(np.log10(snr_lin))) if snr_lin > 0 else float('nan')
    
            # ISI context
            isi_enabled = any(bool(r.get('isi_enabled', False)) for r in results)
            isi_memory_symbols = int(np.nanmedian([float(r.get('isi_memory_symbols', np.nan)) for r in results if r is not None])) if isi_enabled else 0
            symbol_period_s = float(np.nanmedian([float(r.get('symbol_period_s', np.nan)) for r in results]))
            decision_window_s = float(np.nanmedian([float(r.get('decision_window_s', np.nan)) for r in results]))
            isi_overlap_mean = float(np.nanmean([float(r.get('isi_overlap_ratio', 0.0)) for r in results]))
    
            # Stage 14: aggregate noise sigmas across seeds
            ns_da = [float(r.get('noise_sigma_da', float('nan'))) for r in results]
            ns_sero = [float(r.get('noise_sigma_sero', float('nan'))) for r in results]
            ns_diff = [float(r.get('noise_sigma_I_diff', float('nan'))) for r in results]
            ns_thermal = [float(r.get('noise_sigma_thermal', float('nan'))) for r in results]
            ns_flicker = [float(r.get('noise_sigma_flicker', float('nan'))) for r in results]
            ns_drift = [float(r.get('noise_sigma_drift', float('nan'))) for r in results]
            thermal_fracs = [float(r.get('noise_thermal_fraction', float('nan'))) for r in results]
            arr_da = np.asarray(ns_da, dtype=float)
            arr_sero = np.asarray(ns_sero, dtype=float)
            arr_diff = np.asarray(ns_diff, dtype=float)
            arr_thermal = np.asarray(ns_thermal, dtype=float)
            arr_flicker = np.asarray(ns_flicker, dtype=float)
            arr_drift = np.asarray(ns_drift, dtype=float)
            arr_thermal_frac = np.asarray(thermal_fracs, dtype=float)
            med_sigma_da = float(np.nanmedian(arr_da)) if np.isfinite(arr_da).any() else float('nan')
            med_sigma_sero = float(np.nanmedian(arr_sero)) if np.isfinite(arr_sero).any() else float('nan')
            med_sigma_diff = float(np.nanmedian(arr_diff)) if np.isfinite(arr_diff).any() else float('nan')
            med_sigma_thermal = float(np.nanmedian(arr_thermal)) if np.isfinite(arr_thermal).any() else float('nan')
            med_sigma_flicker = float(np.nanmedian(arr_flicker)) if np.isfinite(arr_flicker).any() else float('nan')
            med_sigma_drift = float(np.nanmedian(arr_drift)) if np.isfinite(arr_drift).any() else float('nan')
            med_thermal_frac = float(np.nanmedian(arr_thermal_frac)) if np.isfinite(arr_thermal_frac).any() else float('nan')
    
            delta_stat_values: List[float] = []
            for r in results:
                stats_da = np.asarray(r.get('stats_da', []), dtype=float)
                stats_sero = np.asarray(r.get('stats_sero', []), dtype=float)
                if stats_da.size == 0 or stats_sero.size == 0:
                    continue
                mean_da = float(np.nanmean(stats_da))
                mean_sero = float(np.nanmean(stats_sero))
                if not (np.isfinite(mean_da) and np.isfinite(mean_sero)):
                    continue
                delta_val = mean_da - mean_sero
                if np.isfinite(delta_val):
                    delta_stat_values.append(delta_val)
            med_delta_stat = float(np.nanmedian(np.asarray(delta_stat_values, dtype=float))) if delta_stat_values else float('nan')
            delta_I_diff = float(med_delta_stat * med_sigma_diff) if (np.isfinite(med_delta_stat) and np.isfinite(med_sigma_diff)) else float('nan')
    
            i_dc_vals = np.asarray([r.get('I_dc_used_A', float('nan')) for r in results], dtype=float)
            v_g_vals = np.asarray([r.get('V_g_bias_V_used', float('nan')) for r in results], dtype=float)
            gm_vals = np.asarray([r.get('gm_S', float('nan')) for r in results], dtype=float)
            c_tot_vals = np.asarray([r.get('C_tot_F', float('nan')) for r in results], dtype=float)
    
            def _finite_median(arr: np.ndarray) -> float:
                finite = arr[np.isfinite(arr)]
                return float(np.median(finite)) if finite.size else float('nan')
    
            med_I_dc = _finite_median(i_dc_vals)
            med_V_g_bias = _finite_median(v_g_vals)
            med_gm = _finite_median(gm_vals)
            med_c_tot = _finite_median(c_tot_vals)
    
            mode_name = cfg['pipeline']['modulation']
            snr_semantics = ("MoSK contrast statistic (sign-aware DA vs SERO)"
                            if mode_name in ("MoSK", "Hybrid") else
                            "CSK Q-statistic (dual-channel combiner)")
    
            # Extract rho_cc_measured safely before creating the dictionary
            rho_cc_raw = thresholds_map.get(v, {}).get('rho_cc_measured', float('nan'))
            
            current_distance = float(cfg['pipeline'].get('distance_um', float('nan')))
            current_nm = float(cfg['pipeline'].get('Nm_per_symbol', float('nan')))
            if sweep_param == 'pipeline.distance_um':
                current_distance = float(v)
            if sweep_param == 'pipeline.Nm_per_symbol':
                current_nm = float(v)
    
            row: Dict[str, Any] = {
                sweep_param: v,
                'ser': ser,
                'snr_db': snr_db,
                'snr_semantics': snr_semantics,
                'num_runs': len(results),
                'symbols_evaluated': int(total_symbols),
                'sequence_length': int(cfg['pipeline']['sequence_length']),
                'isi_enabled': isi_enabled,
                'isi_memory_symbols': isi_memory_symbols,
                'symbol_period_s': symbol_period_s,
                'decision_window_s': decision_window_s,
                'isi_overlap_ratio': isi_overlap_mean,
                'noise_sigma_da': med_sigma_da,
                'noise_sigma_sero': med_sigma_sero,
                'noise_sigma_I_diff': med_sigma_diff,
                'noise_sigma_thermal': med_sigma_thermal,
                'noise_sigma_flicker': med_sigma_flicker,
                'noise_sigma_drift': med_sigma_drift,
                'noise_thermal_fraction': med_thermal_frac,
                'I_dc_used_A': med_I_dc,
                'V_g_bias_V_used': med_V_g_bias,
                'gm_S': med_gm,
                'C_tot_F': med_c_tot,
                'delta_over_sigma': med_delta_stat,
                'delta_I_diff': delta_I_diff,
                'distance_um': current_distance,
                'Nm_per_symbol': current_nm,
                'rho_cc': float(cfg.get('noise', {}).get('rho_between_channels_after_ctrl', 0.0)),
                'use_ctrl': bool(thresholds_map.get(v, {}).get('use_control_channel',
                                cfg['pipeline'].get('use_control_channel', True))),
                'ctrl_auto_applied': bool(thresholds_map.get(v, {}).get('ctrl_auto_applied', False)),
                'rho_cc_measured': float(rho_cc_raw) if isinstance(rho_cc_raw, (int, float)) else float('nan'),
                'mode': mode_name,
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
    
                # Enhancement A: Add conditional CSK error for Hybrid mode
                if mode_name == 'Hybrid':
                    total_hybrid_errors = mosk_errors + csk_errors
                    
                    # PATCH 2: Enhanced MoSK exposure tracking and conditional CSK aggregation
                    # Extract MoSK correct count from results for exposure analysis
                    mosk_correct_total = sum(cast(int, r.get('n_mosk_correct', 0)) for r in results)
                    
                    # FIX: Move the Option A changes HERE, inside the Hybrid block
                    row['conditional_csk_ser'] = csk_errors / mosk_correct_total if mosk_correct_total > 0 else 0.0
                    row['mosk_exposure_frac'] = mosk_correct_total / total_symbols if total_symbols > 0 else 0.0
                    
                    # Track conditional CSK errors given MoSK exposure
                    row['mosk_correct_total'] = mosk_correct_total
                    row['csk_exposure_rate'] = mosk_correct_total / total_symbols if total_symbols > 0 else 0.0
                    row['conditional_csk_error_given_exposure'] = csk_errors / mosk_correct_total if mosk_correct_total > 0 else 0.0
                    
                    # Compute hybrid error attribution percentages
                    if total_symbols > 0:
                        row['mosk_error_pct'] = (mosk_errors / total_symbols) * 100.0
                        row['csk_error_pct'] = (csk_errors / total_symbols) * 100.0
                        row['hybrid_total_error_pct'] = ((mosk_errors + csk_errors) / total_symbols) * 100.0
                    bits_per_symbol_csk_branch = float(math.log2(max(cfg['pipeline'].get('csk_levels', 4), 2)))
                    if total_symbols > 0:
                        bits_mosk_realized = (total_symbols - mosk_errors) / total_symbols
                        bits_csk_realized = ((mosk_correct_total - csk_errors) / total_symbols) * bits_per_symbol_csk_branch
                    else:
                        bits_mosk_realized = float('nan')
                        bits_csk_realized = float('nan')
                    row['hybrid_bits_per_symbol_mosk'] = bits_mosk_realized
                    row['hybrid_bits_per_symbol_csk'] = bits_csk_realized
                    if math.isnan(bits_mosk_realized) or math.isnan(bits_csk_realized):
                        row['hybrid_bits_per_symbol_total'] = float('nan')
                    else:
                        row['hybrid_bits_per_symbol_total'] = bits_mosk_realized + bits_csk_realized
    
                    # Enhancement C: Optional assertion during sweeps
                    if total_symbols > 0:
                        total_reported_errors = sum(cast(int, r['errors']) for r in results)
                        
                        # Relaxed assertion allowing for potential edge cases
                        if total_hybrid_errors > total_reported_errors * 1.1:  # 10% tolerance
                            print(f"Warning: Component errors ({total_hybrid_errors}) exceed total errors "
                                  f"({total_reported_errors}) by >10% at {sweep_param}={v}")
    
            aggregated_rows.append(row)
            
            # Append this value's aggregated row immediately (crash‑safe)
            if persist_csv is not None:
                append_row_atomic(persist_csv, row, list(row.keys()))
    
    finally:
        if job_bar is not None:
            try:
                job_bar.close()
            except Exception:
                pass
        if not pm:
            try:
                local_pm.stop()
            except Exception:
                pass
    return pd.DataFrame(aggregated_rows)

# ============= LOD SEARCH =============
def _analytic_lod_bracket(cfg_base: Dict[str, Any], seeds: List[int], target_ser: float = 0.01, nm_ceiling: int = 1000000) -> Tuple[int, int]:
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

                # NEW: Enforce consistent minimum window (Fix A - minimal, localized)
                Ts = float(cfg_p['pipeline'].get('symbol_period_s',
                        calculate_dynamic_symbol_period(float(cfg_p['pipeline']['distance_um']), cfg_p)))
                min_win = _enforce_min_window(cfg_p, Ts)
                cfg_p.setdefault('detection', {})
                cfg_p['pipeline']['time_window_s'] = max(cfg_p['pipeline'].get('time_window_s', 0.0), min_win)
                cfg_p['detection']['decision_window_s'] = min_win

                # thresholds at this Nm (few seeds, cached)
                th = calibrate_thresholds_cached(cfg_p, list(range(min(3, len(seeds)))))
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
                        target_ch = cfg_p['pipeline'].get('csk_target_channel', 'DA').lower()
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
                    # Cap BEFORE multiplication to prevent overflow
                    new_probes = []
                    for p in probes:
                        new_p = min(p * 3, nm_ceiling)  # Cap first
                        new_probes.append(max(50, new_p))
                    probes = new_probes
                    
                    if all(p >= nm_ceiling for p in probes):
                        print("    ⚠️ Analytic probes saturated at ceiling; aborting analytic bracket.")
                        break
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
            
            # Linear interpolation in log-space (clamped to prevent extrapolation)
            alpha = max(0.0, min(1.0, (lser_t - lser1) / (lser2 - lser1)))
            lnm_t = lnm1 + alpha * (lnm2 - lnm1)
            # Prevent overflow
            if lnm_t > 25:  # exp(25) ≈ 7×10^10
                nm_t = nm_ceiling
            else:
                nm_t = int(math.exp(lnm_t))
            
            # Conservative bracket: ±50% around interpolated point
            nm_min_est = max(50, int(nm_t * 0.5))
            nm_max_est = min(nm_ceiling, int(nm_t * 1.5))
            
            print(f"    📊 Analytic interpolation: target SER {target_ser:.1e} → Nm ≈ {nm_t}, bracket [{nm_min_est}-{nm_max_est}]")
            return (nm_min_est, nm_max_est)
        else:
            print(f"    ⚠️  Analytic bracketing failed: insufficient data")
            return (0, 0)
            
    except Exception as e:
        print(f"    ⚠️  Analytic bracketing failed: {e}")
        return (0, 0)

def _validate_and_fix_bracket(cfg_base: Dict[str, Any], nm_min: int, nm_max: int, 
                              nm_ceiling: int, target_ser: float, seeds: List[int], 
                              cache_tag: Optional[str] = None) -> Tuple[int, int, Optional[str]]:
    """
    Validate and fix the LoD bracket to ensure SER(nm_min) > target >= SER(nm_max).
    Returns (corrected_nm_min, corrected_nm_max, skip_reason_if_any).
    """
    # Quick validation with 3 seeds
    quick_seeds = seeds[:min(3, len(seeds))]
    
    def _quick_ser(nm: int) -> float:
        cfg_test = deepcopy(cfg_base)
        cfg_test['pipeline']['Nm_per_symbol'] = nm
        cfg_test['pipeline']['sequence_length'] = 100  # Short for speed
        
        results = []
        for seed in quick_seeds:
            res = run_param_seed_combo(cfg_test, 'pipeline.Nm_per_symbol', nm, seed,
                                    sweep_name="bracket_validation",
                                    cache_tag=cache_tag)
            if res:
                results.append(res)
        
        if not results:
            return 1.0  # Assume failure if no results
        
        total_symbols = len(results) * cfg_test['pipeline']['sequence_length']
        total_errors = sum(int(r.get('errors', 0)) for r in results)
        return total_errors / total_symbols if total_symbols > 0 else 1.0
    
    # Check lower bound
    ser_min = _quick_ser(nm_min)
    if ser_min <= target_ser:
        # Lower bound too good, push it down
        while nm_min > 50 and ser_min <= target_ser:
            nm_min = max(50, int(nm_min / 2))
            ser_min = _quick_ser(nm_min)
    
    # Check upper bound
    ser_max = _quick_ser(nm_max)
    if ser_max > target_ser:
        # Upper bound not good enough, try to grow it
        while nm_max < nm_ceiling and ser_max > target_ser:
            nm_max = min(nm_ceiling, nm_max * 2)
            ser_max = _quick_ser(nm_max)
        
        # If we hit ceiling and still can't achieve target
        if nm_max >= nm_ceiling and ser_max > target_ser:
            return nm_min, nm_max, "nm_ceiling_exhausted"
    
    # Final validation
    if ser_min <= target_ser or ser_max > target_ser:
        # Still invalid bracket
        return nm_min, nm_max, "bracket_validation_failed"
    
    return nm_min, nm_max, None

def find_lod_for_ser(cfg_base: Dict[str, Any], seeds: List[int],
                     target_ser: float = 0.01,
                     debug_calibration: bool = False,
                     progress_cb: Optional[Any] = None,
                     resume: bool = False,
                     cache_tag: Optional[str] = None) -> Tuple[Union[int, float], float, int]:
    nm_min = int(cfg_base['pipeline'].get('lod_nm_min', 50))
    # NEW: configurable ceiling with fallback for backward compatibility
    nm_ceiling = int(cfg_base.get('pipeline', {}).get('lod_nm_max', 
                    cfg_base.get('lod_max_nm', 1000000)))
    nm_max = nm_ceiling
    nm_max_default = nm_ceiling

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
                
                # FIXED: respect both caps and the global 100k ceiling
                nm_max = min(upper_from_warm, upper_from_analytic, nm_max_default)
                nm_min = max(nm_min, nm_min_analytic)
                
                print(f"    🔄 Warm + analytic intersect: [{nm_min} - {nm_max}] (capped at {nm_ceiling})")
        else:
            # Pure warm-start without analytic constraints
            nm_max = min(nm_max_default, int(mult * warm))
            print(f"    🔥 Warm-start bracket: [{nm_min} - {nm_max}]")
    
    lod_nm: float = float('nan')
    best_ser: float = 1.0
    best_nm: Optional[int] = None  # NEW: Track the Nm that gave best_ser
    dist_um = cfg_base['pipeline'].get('distance_um', 0)
    mode_name = cfg_base['pipeline']['modulation']
    use_ctrl = bool(cfg_base['pipeline'].get('use_control_channel', True))
    
    # Track actual progress increments
    progress_count = 0
    
    # Load prior state if resuming
    state = _lod_state_load(mode_name, float(dist_um), use_ctrl) if resume else None

    if state:
        # 1) Fast exit when a previous run already marked 'done' (robust to NaN)
        nm_min_state = state.get("nm_min")
        nm_max_state = state.get("nm_max")
        if (state.get("done") and 
            nm_min_state is not None and nm_max_state is not None and
            all(isinstance(x, (int, float)) and math.isfinite(x) for x in (nm_min_state, nm_max_state)) and
            int(nm_min_state) == int(nm_max_state) and int(nm_min_state) > 0):
            lod_nm = int(nm_min_state)
            print(f"    ✔ Resume: LoD already found in previous run → {lod_nm}")
            return lod_nm, target_ser, 0

        # 2) Try to reconstruct a sane bracket from per-Nm tallies
        t = state.get("tested", {})
        if isinstance(t, dict) and t:
            succ: list[int] = []
            fail: list[int] = []
            for k, v in t.items():
                try:
                    nm = int(k)
                    n = int(v.get("n_seen", 0))
                    kerr = int(v.get("k_err", 0))
                except Exception:
                    continue
                if n <= 0:
                    continue
                (succ if (kerr / n) <= target_ser else fail).append(nm)

            succ.sort(); fail.sort()
            # failure bound below / success bound above -> narrow the bracket
            if succ and fail:
                nm_min = max(int(state.get("nm_min", nm_min)), max(fail) + 1)
                nm_max = min(int(state.get("nm_max", nm_max)), min(succ))
            elif fail and not succ:
                nm_min = max(int(state.get("nm_min", nm_min)), max(fail) + 1)
                nm_max = min(nm_ceiling, max(int(state.get("nm_max", nm_max)), max(fail) * 2))
            elif succ and not fail:
                nm_min = max(50, int(min(succ) * 0.5))
                nm_max = min(int(state.get("nm_max", nm_max)), min(succ))

        # 3) Guard: never continue with an invalid bracket
        if nm_min > nm_max:
            print("    ⚠️  Stale LoD state (nm_min>nm_max). Clearing state and restarting bracket.")
            _lod_state_save(mode_name, float(dist_um), use_ctrl, {"tested": {}})
            nm_min = cfg_base['pipeline'].get('lod_nm_min', 50)
            nm_max = nm_ceiling
        else:
            print(f"    ↩️  Resuming LoD search @ {dist_um}μm: range {nm_min}-{nm_max}")
    
    # Extract CTRL state for debug logging
    ctrl_str = "CTRL" if use_ctrl else "NoCtrl"

    # NEW: Cache thresholds during bisection to reduce calibration overhead
    th_cache: Dict[int, Dict[str, Union[float, List[float], str]]] = {}  # nm -> thresholds dict
    
    def _get_th(nm: int):
        if nm in th_cache:
            return th_cache[nm]
        cfg_tmp = deepcopy(cfg_base)
        cfg_tmp['pipeline']['Nm_per_symbol'] = nm
        th = calibrate_thresholds_cached(cfg_tmp, list(range(6)))  # faster with fewer seeds
        th_cache[nm] = th
        return th
    
    # NEW: Validate and fix bracket before proceeding to bisection
    tag = f"d{int(dist_um)}um"
    nm_min, nm_max, skip_reason = _validate_and_fix_bracket(
        cfg_base, nm_min, nm_max, nm_ceiling, target_ser, seeds, cache_tag=tag
    )

    if skip_reason:
        print(f"    [{dist_um}μm|{ctrl_str}] Skipping LoD search: {skip_reason}")
        # Clear any misleading state when ceiling is exhausted
        if skip_reason == "nm_ceiling_exhausted":
            _lod_state_save(mode_name, float(dist_um), use_ctrl, {"tested": {}})
            print(f"    🧹 Cleared LoD state for future runs with higher ceiling")
        return float('nan'), 1.0, 0

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

        # Track best attempt regardless of whether it passes
        if ser < best_ser:
            best_ser = ser
            best_nm = nm_mid

        if ser <= target_ser:
            lod_nm = nm_mid
            
            # NEW: LoD down-step confirmation accelerator
            # Try a more aggressive value with minimal seeds for fast screening
            nm_probe = max(nm_min, int(0.60 * nm_mid))  # was int(nm_mid / sqrt(2))
            if nm_probe < nm_mid and nm_probe >= nm_min:
                print(f"      [{dist_um}μm|{ctrl_str}] 🚀 Down-step probe: testing Nm={nm_probe} with {min(3, len(seeds))} seeds")
                
                # Use minimal seeds for fast screening
                probe_seeds = seeds[:min(3, len(seeds))]
                probe_k_err = 0
                probe_n_seen = 0
                
                cfg_probe = deepcopy(cfg_base)
                cfg_probe['pipeline']['Nm_per_symbol'] = nm_probe
                for k, v in _get_th(nm_probe).items():
                    cfg_probe['pipeline'][k] = v
                
                for probe_seed in probe_seeds:
                    # Check cache first
                    cached_probe = read_seed_cache(mode_name, "lod_search", nm_probe, probe_seed, use_ctrl, cache_tag=cache_tag)
                    if cached_probe:
                        probe_k_err += int(cached_probe.get('errors', 0))
                        probe_n_seen += seq_len
                    else:
                        # Run minimal simulation
                        res_probe = run_param_seed_combo(cfg_probe, 'pipeline.Nm_per_symbol', nm_probe, probe_seed,
                                                        debug_calibration=False, thresholds_override=_get_th(nm_probe),
                                                        sweep_name="lod_search", cache_tag=cache_tag)
                        if res_probe:
                            probe_k_err += int(res_probe.get('errors', 0))
                            probe_n_seen += seq_len
                    
                    # Early deterministic screen after each seed
                    total_planned = len(probe_seeds) * seq_len
                    decide_below, decide_above = _deterministic_screen(probe_k_err, probe_n_seen, total_planned, target_ser)
                    if decide_below:
                        # Probe passes! Skip bisection iterations
                        probe_ser = probe_k_err / probe_n_seen if probe_n_seen > 0 else 1.0
                        if probe_ser < best_ser:
                            best_ser = probe_ser
                            best_nm = nm_probe
                        print(f"      [{dist_um}μm|{ctrl_str}] ✓ Down-step probe SUCCESS → skip to Nm={nm_probe}")
                        lod_nm = nm_probe
                        nm_max = nm_probe - 1
                        progress_count += len(probe_seeds)  # Count probe work
                        break
                    elif decide_above:
                        # Probe fails decisively, stick with original nm_mid
                        print(f"      [{dist_um}μm|{ctrl_str}] ✗ Down-step probe FAIL → continue bisection")
                        progress_count += len(probe_seeds)  # Count probe work  
                        break
                else:
                    # All probe seeds completed, check final SER
                    if probe_n_seen > 0:
                        probe_ser = probe_k_err / probe_n_seen
                        if probe_ser <= target_ser:
                            print(f"      [{dist_um}μm|{ctrl_str}] ✓ Down-step probe SUCCESS (SER={probe_ser:.4f}) → skip to Nm={nm_probe}")
                            lod_nm = nm_probe
                            nm_max = nm_probe - 1
                        else:
                            print(f"      [{dist_um}μm|{ctrl_str}] ✗ Down-step probe FAIL (SER={probe_ser:.4f}) → continue bisection")
                        progress_count += len(probe_seeds)
            
            nm_max = nm_mid - 1  # Standard bisection update
        else:
            nm_min = nm_mid + 1
            
        # Count each binary search iteration
        progress_count += 1
        
        # persist bounds after each iteration
        _lod_state_save(mode_name, float(dist_um), use_ctrl,
                        {"nm_min": nm_min, "nm_max": nm_max, "iteration": iteration,
                         "last_nm": nm_mid})

    # OPTIMIZATION 1: Cap LoD validation retries
    # Only run a single-point validation when the bracket collapsed to one point
    if math.isnan(lod_nm) and nm_min <= nm_ceiling and nm_min == nm_max:
        print(f"    [{dist_um}μm|{ctrl_str}] Final validation at Nm={nm_min}")
        cfg_final = deepcopy(cfg_base)
        cfg_final['pipeline']['Nm_per_symbol'] = nm_min
        
        # NEW: enforce minimum decision window here
        Ts = float(cfg_final['pipeline'].get(
            'symbol_period_s',
            calculate_dynamic_symbol_period(float(cfg_final['pipeline']['distance_um']), cfg_final)
        ))
        min_win = _enforce_min_window(cfg_final, Ts)
        cfg_final.setdefault('detection', {})
        cfg_final['pipeline']['time_window_s'] = max(cfg_final['pipeline'].get('time_window_s', 0.0), min_win)
        cfg_final['detection']['decision_window_s'] = min_win
        
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

    # Return best attempt if no solution found, otherwise return found solution
    final_lod_nm = lod_nm if not math.isnan(lod_nm) else float('nan')
    final_ser = best_ser  # Always return the best SER seen (either successful or closest attempt)
    
    # Return actual count instead of constant
    return final_lod_nm, final_ser, progress_count

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
    dt = float(cfg['sim']['dt_s'])
    min_pts = int(cfg.get('_min_decision_points', 4))
    min_win = _enforce_min_window(cfg, Ts)
    cfg['pipeline']['time_window_s'] = max(cfg['pipeline'].get('time_window_s', 0.0), min_win)
    cfg.setdefault('detection', {})
    cfg['detection']['decision_window_s'] = min_win

    cfg['pipeline']['Nm_per_symbol'] = int(lod_nm)

    # NEW: Apply LoD validation sequence length override if specified
    lod_validate_seq_len = cfg_base.get('_lod_validate_seq_len', None)
    if lod_validate_seq_len:
        cfg['pipeline']['sequence_length'] = int(lod_validate_seq_len)
        print(f"    📏 LoD validation using shorter sequences: {lod_validate_seq_len} symbols/seed")

    # Apply thresholds at this exact operating point
    th = calibrate_thresholds_cached(cfg, list(range(10)))
    for k, v in th.items():
        cfg['pipeline'][k] = v

    # NEW: Read adaptive stopping configuration
    target_ci = float(cfg_base.get('_stage13_target_ci', 0.0))  # reuse runner knob
    min_ci_seeds = int(cfg_base.get('_stage13_min_ci_seeds', 8))

    def _wilson_halfwidth(k_err: int, n_tot: int, z: float = 1.96) -> float:
        if n_tot <= 0:
            return 1.0
        p = k_err / n_tot
        denom = 1 + z*z/n_tot
        center = (p + z*z/(2*n_tot)) / denom
        rad = z * math.sqrt(p*(1-p)/n_tot + z*z/(4*n_tot*n_tot)) / denom
        return rad

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
    
    # NEW: Adaptive stopping variables
    k_err, n_tot = 0, 0

    for i, seed in enumerate(full_seeds):
        res = run_single_instance(cfg, seed, attach_isi_meta=True)
        if res is None:
            continue
        L = int(cfg['pipeline']['sequence_length'])
        e = res.get('errors', None)
        
        # NEW: Accumulate errors for adaptive stopping
        if e is not None:
            k_err += int(e)
            n_tot += L
        
        ser_seed = float(res.get('ser', res.get('SER', (e / L) if (e is not None and L > 0) else 1.0)))
        Ts_seed = float(res.get('symbol_period_s', Ts))
        per_seed_rates.append((bpsym / Ts_seed) * (1.0 - ser_seed))
        per_seed_ser.append(ser_seed)
        Ts_list.append(Ts_seed)

        # NEW: Adaptive stop once CI is tight enough (after min seeds)
        if target_ci > 0 and (i + 1) >= min_ci_seeds:
            if _wilson_halfwidth(k_err, n_tot) <= target_ci:
                print(f"    ✓ Early CI stop: {i+1}/{len(full_seeds)} seeds (CI half-width ≤ {target_ci:.3f})")
                break

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
    
    # ✅ FIX: Set distance before LoD search and rebuild window consistently
    cfg['pipeline']['distance_um'] = float(dist_um)  # Bake distance into cfg
    
    # Recompute Ts and enforce a consistent minimum decision window
    Ts_dyn = calculate_dynamic_symbol_period(float(dist_um), cfg)
    cfg['pipeline']['symbol_period_s'] = Ts_dyn
    min_win = _enforce_min_window(cfg, Ts_dyn)
    cfg['pipeline']['time_window_s'] = max(cfg['pipeline'].get('time_window_s', 0.0), min_win)
    cfg.setdefault('detection', {})
    cfg['detection']['decision_window_s'] = min_win

    isi_overlap_ratio_initial = estimate_isi_overlap_ratio(cfg)
    _maybe_warn_isi_overlap(cfg, isi_overlap_ratio_initial, context=f"LoD distance {dist_um:.0f} um")

    # Check if Ts exceed flags are disabled
    allow_ts = bool(getattr(args, "allow_ts_exceed", False))
    if not allow_ts:
        allow_ts = bool(cfg_base.get('analysis', {}).get('allow_ts_exceed', False))
    # Check if we should only warn instead of skip
    warn_only = bool(getattr(args, "ts_warn_only", False))
    # NEW: Check for Ts explosion and skip if too large (OPTIMIZATION 2)
    max_symbol_duration_s = cfg_base.get('max_symbol_duration_s', None)
    if (not allow_ts) and (max_symbol_duration_s is not None) and (max_symbol_duration_s > 0) and (Ts_dyn > max_symbol_duration_s):
        if warn_only:
            print(f"⚠️  WARNING: distance {dist_um}μm has long symbol period {Ts_dyn:.1f}s (exceeds {max_symbol_duration_s}s), continuing anyway")
            # Continue with LoD analysis instead of returning NaN
        else:
            print(f"⚠️  Skipping distance {dist_um}μm: symbol period {Ts_dyn:.1f}s exceeds limit {max_symbol_duration_s}s")
            return {
                'distance_um': dist_um,
                'lod_nm': float('nan'),
                'ser_at_lod': float('nan'),
                'data_rate_bps': 0.0,
                'data_rate_ci_low': float('nan'),
                'data_rate_ci_high': float('nan'),
                'symbol_period_s': Ts_dyn,
                'isi_enabled': bool(cfg['pipeline'].get('enable_isi', False)),
                'isi_memory_symbols': 0,
                'decision_window_s': Ts_dyn,
                'isi_overlap_ratio': isi_overlap_ratio_initial,
                'use_ctrl': bool(cfg['pipeline'].get('use_control_channel', True)),
                'mode': cfg['pipeline']['modulation'],
                'noise_sigma_I_diff': float('nan'),
                'noise_sigma_thermal': float('nan'),
                'noise_sigma_flicker': float('nan'),
                'noise_sigma_drift': float('nan'),
                'noise_thermal_fraction': float('nan'),
                'I_dc_used_A': float('nan'),
                'V_g_bias_V_used': float('nan'),
                'gm_S': float('nan'),
                'C_tot_F': float('nan'),
                'actual_progress': 0,
                'skipped_reason': f'Ts_explosion_{Ts_dyn:.1f}s'
            }
    # Continue with existing logic for args.max_ts_for_lod (keep this too)
    cap_cli = getattr(args, "max_ts_for_lod", None) if args else None
    if (not allow_ts) and (cap_cli is not None) and (float(cap_cli) > 0) and (Ts_dyn > float(cap_cli)):
        if warn_only:
            print(f"⚠️  WARNING: distance {dist_um}μm has long symbol period {Ts_dyn:.1f}s (exceeds CLI limit {cap_cli}s), continuing anyway")
            # Continue with LoD analysis instead of returning NaN
        else:
            return {
                'distance_um': dist_um, 'lod_nm': float('nan'), 'ser_at_lod': float('nan'),
                'data_rate_bps': float('nan'), 'data_rate_ci_low': float('nan'), 'data_rate_ci_high': float('nan'),
                'symbol_period_s': Ts_dyn, 'isi_enabled': bool(cfg['pipeline'].get('enable_isi', False)),
                'isi_memory_symbols': 0, 'decision_window_s': Ts_dyn, 'isi_overlap_ratio': isi_overlap_ratio_initial,
                'use_ctrl': bool(cfg['pipeline'].get('use_control_channel', True)),
                'mode': cfg['pipeline']['modulation'], 'noise_sigma_I_diff': float('nan'),
                'noise_sigma_thermal': float('nan'), 'noise_sigma_flicker': float('nan'),
                'noise_sigma_drift': float('nan'), 'noise_thermal_fraction': float('nan'),
                'I_dc_used_A': float('nan'), 'V_g_bias_V_used': float('nan'), 'gm_S': float('nan'), 'C_tot_F': float('nan'),
                'actual_progress': 0, 'skipped_reason': f'Ts>{cap_cli}s'
            }
    
    # LoD search can use shorter sequences to bracket quickly
    if args and getattr(args, "lod_seq_len", None):
        cfg['pipeline']['sequence_length'] = args.lod_seq_len
    
    # NEW: Propagate warm-start guess
    if warm_lod_guess and warm_lod_guess > 0:
        cfg['_warm_lod_guess'] = int(warm_lod_guess)
    
    # NEW: Set LoD max from args (use consistent key)
    cfg['pipeline']['lod_nm_max'] = int(getattr(args, "lod_max_nm", 1000000))
    
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
            
            # Ensure detection window matches Ts with consistent guard
            dt = float(cfg['sim']['dt_s'])
            min_pts = int(cfg.get('_min_decision_points', 4))
            Ts = cfg['pipeline']['symbol_period_s']
            min_win = _enforce_min_window(cfg, Ts)
            cfg['pipeline']['time_window_s'] = max(cfg['pipeline'].get('time_window_s', 0.0), min_win)
            cfg.setdefault('detection', {})
            cfg['detection']['decision_window_s'] = min_win

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
        # Recompute Ts and enforce a consistent minimum decision window
        Ts_lod = calculate_dynamic_symbol_period(dist_um, cfg_lod)

        dt = float(cfg_lod['sim']['dt_s'])
        min_pts = int(cfg_lod.get('_min_decision_points', 4))
        min_win = _enforce_min_window(cfg_lod, Ts_lod)

        cfg_lod['pipeline']['symbol_period_s'] = Ts_lod           # keep the true dynamic Ts
        cfg_lod['pipeline']['time_window_s'] = max(cfg_lod['pipeline'].get('time_window_s', 0.0), min_win)
        cfg_lod.setdefault('detection', {})
        cfg_lod['detection']['decision_window_s'] = min_win       # <- align

        cfg_lod['pipeline']['Nm_per_symbol'] = lod_nm

        # Apply thresholds at this exact (distance, Ts, Nm)
        th_lod = calibrate_thresholds_cached(cfg_lod, list(range(4)))
        for k, v in th_lod.items():
            cfg_lod['pipeline'][k] = v
        
        sigma_values = []
        sigma_thermal_values = []
        sigma_flicker_values = []
        sigma_drift_values = []
        thermal_fraction_values = []
        i_dc_values = []
        v_g_bias_values = []
        gm_values = []
        c_tot_values = []
        for seed in seeds[:5]:  # Limited seeds for efficiency
            result = run_param_seed_combo(cfg_lod, 'pipeline.Nm_per_symbol', lod_nm, seed, 
                                        debug_calibration=debug_calibration, 
                                        sweep_name="lod_validation", cache_tag="lod_sigma")
            if result:
                if 'noise_sigma_I_diff' in result:
                    sigma_values.append(result['noise_sigma_I_diff'])
                if 'noise_sigma_thermal' in result:
                    sigma_thermal_values.append(result['noise_sigma_thermal'])
                if 'noise_sigma_flicker' in result:
                    sigma_flicker_values.append(result['noise_sigma_flicker'])
                if 'noise_sigma_drift' in result:
                    sigma_drift_values.append(result['noise_sigma_drift'])
                if 'noise_thermal_fraction' in result:
                    thermal_fraction_values.append(result['noise_thermal_fraction'])
                if 'I_dc_used_A' in result:
                    i_dc_values.append(result['I_dc_used_A'])
                if 'V_g_bias_V_used' in result:
                    v_g_bias_values.append(result['V_g_bias_V_used'])
                if 'gm_S' in result:
                    gm_values.append(result['gm_S'])
                if 'C_tot_F' in result:
                    c_tot_values.append(result['C_tot_F'])
            # Progress callback for completed noise sigma seed
            if progress_cb is not None:
                try: 
                    progress_cb.put(1)
                except Exception: 
                    pass
        
        def _finite_list_median(values):
            arr = np.asarray(values, dtype=float)
            finite = arr[np.isfinite(arr)]
            return float(np.median(finite)) if finite.size else float('nan')

        lod_sigma_median = _finite_list_median(sigma_values)
        lod_sigma_thermal = _finite_list_median(sigma_thermal_values)
        lod_sigma_flicker = _finite_list_median(sigma_flicker_values)
        lod_sigma_drift = _finite_list_median(sigma_drift_values)
        lod_thermal_fraction = _finite_list_median(thermal_fraction_values)
        lod_I_dc = _finite_list_median(i_dc_values)
        lod_V_g_bias = _finite_list_median(v_g_bias_values)
        lod_gm = _finite_list_median(gm_values)
        lod_c_tot = _finite_list_median(c_tot_values)
    else:
        lod_sigma_median = float('nan')
        lod_sigma_thermal = float('nan')
        lod_sigma_flicker = float('nan')
        lod_sigma_drift = float('nan')
        lod_thermal_fraction = float('nan')
        lod_I_dc = float('nan')
        lod_V_g_bias = float('nan')
        lod_gm = float('nan')
        lod_c_tot = float('nan')

    # mark LoD state as done ONLY if valid result
    try:
        if isinstance(lod_nm, (int, float)) and math.isfinite(lod_nm) and lod_nm > 0:
            done_state = {"done": True, "nm_min": int(lod_nm), "nm_max": int(lod_nm)}
            _lod_state_save(cfg['pipeline']['modulation'], float(dist_um),
                            bool(cfg['pipeline'].get('use_control_channel', True)), done_state)
    except Exception:
        pass
    
    # If LoD was not found, remove stale state so the next --resume starts clean
    try:
        if (isinstance(lod_nm, float) and (math.isnan(lod_nm) or lod_nm <= 0)) or (isinstance(lod_nm, int) and lod_nm <= 0):
            p = _lod_state_path(cfg['pipeline']['modulation'], float(dist_um), bool(cfg['pipeline'].get('use_control_channel', True)))
            if p.exists():
                p.unlink()
    except Exception:
        pass
    
    # Handle not found case
    if math.isnan(lod_nm) or lod_nm <= 0:
        skipped_reason = skipped_reason or 'not_bracketed'

    isi_overlap_ratio_final = estimate_isi_overlap_ratio(cfg)

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
        'isi_overlap_ratio': isi_overlap_ratio_final,
        'use_ctrl': bool(cfg['pipeline'].get('use_control_channel', True)),
        'mode': cfg['pipeline']['modulation'],
        'noise_sigma_I_diff': lod_sigma_median,
        'noise_sigma_thermal': lod_sigma_thermal,
        'noise_sigma_flicker': lod_sigma_flicker,
        'noise_sigma_drift': lod_sigma_drift,
        'noise_thermal_fraction': lod_thermal_fraction,
        'I_dc_used_A': float(lod_I_dc),
        'V_g_bias_V_used': float(lod_V_g_bias),
        'gm_S': float(lod_gm),
        'C_tot_F': float(lod_c_tot),
        'actual_progress': int(actual_progress),
        'skipped_reason': skipped_reason,  # ✅ Use the variable instead of None
        'lod_found': bool(not (isinstance(lod_nm, float) and (math.isnan(lod_nm) or lod_nm <= 0))),
        'best_seen_ser': float(ser_at_lod),
        'lod_nm_ceiling': int(cfg.get('lod_max_nm', 1000000))
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
    plt.xlim(1e2, 5e5)
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
        'da' -> 'DA', 'acetylcholine' -> 'ACh', 'DA' -> 'DA'
    """
    nts = cfg.get('neurotransmitters', {})
    
    # Stage 1: Exact match (case-insensitive)
    for k in nts.keys():
        if k.lower() == name.lower():
            return k
    
    # Stage 2: Alias lookup
    aliases = {
        'ach': 'ACh', 'acetylcholine': 'ACh',
        'da': 'DA', 'dopamine': 'DA', 'datamate': 'DA',
        'sero': 'SERO', 'serotonin': 'SERO',
        'norepinephrine': 'NE', 'ne': 'NE'
    }
    
    canonical = aliases.get(name.lower())
    return canonical if canonical and canonical in nts else None

def _apply_nt_pair(cfg: Dict[str, Any], first: str, second: str) -> Dict[str, Any]:
    """Swap underlying molecule dicts into DA/SERO slots so the tri-channel interface stays stable."""
    cfg_new = deepcopy(cfg)
    nts = cfg.get('neurotransmitters', {})
    k1 = _canonical_nt_key(cfg, first)
    k2 = _canonical_nt_key(cfg, second)
    if k1 is None or k2 is None:
        nts = cfg.get('neurotransmitters', {})
        available = list(nts.keys())
        aliases = ['ach', 'acetylcholine', 'da', 'dopamine', 'da', 'datamate', 'sero', 'serotonin', 'sero', 'norepinephrine', 'ne']
        raise ValueError(f"Unknown neurotransmitter key(s): {first}, {second}. "
                        f"Available: {available}. "
                        f"Supported aliases: {aliases}")

    cfg_new['neurotransmitters']['DA'] = dict(nts[k1])   # first
    cfg_new['neurotransmitters']['SERO'] = dict(nts[k2])  # second
    if cfg_new['pipeline'].get('modulation') == 'CSK':
        cfg_new['pipeline']['csk_target_channel'] = 'DA'  # measure 'first'
    return cfg_new

def run_csk_nt_pair_sweeps(args, cfg_base: Dict[str, Any], seeds: List[int], nm_values: List[Union[float, int]]) -> None:
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
        thresholds = calibrate_thresholds_cached(cfg_pair, cal_seeds, args.recalibrate)
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
            # Include use_ctrl in deduplication when both states are present
            combined = pd.concat([prev, df_pair], ignore_index=True)
            subset = [nm_key] + (['use_ctrl'] if 'use_ctrl' in combined.columns else [])
            combined = combined.drop_duplicates(subset=subset, keep='last')
            _atomic_write_csv(out_csv, combined)

            # --- Apply SER auto-refine to this NT-pair if enabled ---
        if args.ser_refine:
            try:
                # Read the final CSV back for this pair to get the most recent results
                df_pair_final = pd.read_csv(out_csv) if out_csv.exists() else df_pair

                # Find refine candidates for this pair around the target SER
                refine_candidates = _auto_refine_nm_points_from_df(
                    df_pair_final,
                    target=float(args.ser_target),
                    extra_points=int(args.ser_refine_points)
                )

                # Filter out any Nm that are already present for this CTRL state
                if refine_candidates:
                    desired_ctrl = bool(cfg_pair['pipeline'].get('use_control_channel', True))
                    done_pair = load_completed_values(out_csv, 'pipeline_Nm_per_symbol', desired_ctrl)
                    refine_candidates = [n for n in refine_candidates if canonical_value_key(n) not in done_pair]

                if refine_candidates:
                    print(f"    🔎 SER auto-refine for {first}-{second} around {args.ser_target:.2%}: {refine_candidates}")

                    # Run the refine points for this specific NT pair
                    df_refined_pair = run_sweep(
                        cfg_pair, seeds,
                        'pipeline.Nm_per_symbol',
                        [float(n) for n in refine_candidates],
                        f"SER refine near {args.ser_target:.2%} (CSK {first}-{second})",
                        progress_mode=args.progress,
                        persist_csv=out_csv,
                        resume=args.resume,
                        debug_calibration=args.debug_calibration,
                        cache_tag=f"{pair_tag}_refine"  # Separate cache tag for refine
                    )

                    # Re-de-dupe the CSV again
                    if out_csv.exists():
                        existing_pair = pd.read_csv(out_csv)
                        nm_key = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in existing_pair.columns else 'pipeline.Nm_per_symbol'
                        combined_pair = existing_pair if df_refined_pair.empty else pd.concat([existing_pair, df_refined_pair], ignore_index=True)
                        if 'use_ctrl' in combined_pair.columns:
                            combined_pair = combined_pair.drop_duplicates(subset=[nm_key, 'use_ctrl'], keep='last').sort_values([nm_key, 'use_ctrl'])
                        else:
                            combined_pair = combined_pair.drop_duplicates(subset=[nm_key], keep='last').sort_values([nm_key])
                        _atomic_write_csv(out_csv, combined_pair)
                        print(f"    ✅ SER auto-refine for {first}-{second} completed; CSV updated")
                else:
                    print(f"    ℹ️  SER auto-refine for {first}-{second}: no bracket found or all refine Nm already present.")
            except Exception as e:
                print(f"    ⚠️  SER auto-refine for {first}-{second} failed: {e}")

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

def _run_device_fom_sweeps(
    cfg_base: Dict[str, Any],
    seeds: List[int],
    mode: str,
    data_dir: Path,
    suffix: str,
    df_lod: Optional[pd.DataFrame],
    resume: bool,
    args: argparse.Namespace,
    pm: Optional[ProgressManager] = None,
    mode_key: Optional[Any] = None,
) -> None:
    """Sweep OECT parameters to capture delta-over-sigma device figures of merit."""
    if not seeds:
        return

    desired_ctrl = bool(cfg_base['pipeline'].get('use_control_channel', True))
    gm_grid = [1e-3, 3e-3, 5e-3, 1e-2]
    c_grid = [1e-8, 3e-8, 5e-8, 1e-7]

    device_csv = data_dir / f"device_fom_{mode.lower()}{suffix}.csv"
    try:
        existing = pd.read_csv(device_csv)
    except Exception:
        existing = pd.DataFrame()

    def _filter(values: List[float], param_type: str) -> List[float]:
        vals = [float(v) for v in values]
        if not resume or existing.empty or 'param_type' not in existing.columns:
            return vals
        df_match = existing[existing['param_type'] == param_type]
        if 'use_ctrl' in df_match.columns:
            df_match = df_match[df_match['use_ctrl'] == desired_ctrl]
        done = {
            round(float(v), 12)
            for v in pd.to_numeric(df_match.get('param_value', pd.Series(dtype=float)), errors='coerce').dropna()
        }
        return [v for v in vals if round(v, 12) not in done]

    gm_values = _filter(gm_grid, 'gm_S') if gm_grid else []
    c_values = _filter(c_grid, 'C_tot_F') if c_grid else []

    df_ref = df_lod if df_lod is not None else pd.DataFrame()
    d_ref: Optional[int] = None
    nm_ref: Optional[float] = None
    if not df_ref.empty and {'distance_um', 'lod_nm'}.issubset(df_ref.columns):
        df_match = df_ref.copy()
        if 'use_ctrl' in df_match.columns:
            df_match = df_match[df_match['use_ctrl'] == desired_ctrl]
        df_valid = df_match.dropna(subset=['lod_nm'])
        if not df_valid.empty:
            df_valid = df_valid.assign(
                __dist=pd.to_numeric(df_valid['distance_um'], errors='coerce')
            ).dropna(subset=['__dist'])
            if not df_valid.empty:
                dist_vals = df_valid['__dist'].to_numpy(dtype=float)
                d_ref = int(np.median(dist_vals))
                dist_delta = np.abs(dist_vals - float(d_ref))
                idx = int(np.argmin(dist_delta))
                nm_candidate = float(pd.to_numeric(df_valid.iloc[idx]['lod_nm'], errors='coerce'))
                if math.isfinite(nm_candidate) and nm_candidate > 0:
                    nm_ref = nm_candidate

    if d_ref is None:
        d_ref = int(float(cfg_base['pipeline'].get('distance_um', 50.0)))
    if nm_ref is None or not math.isfinite(nm_ref) or nm_ref <= 0:
        nm_ref = float(cfg_base['pipeline'].get('Nm_per_symbol', 1e4))

    device_seed_count = min(len(seeds), 6) if len(seeds) > 0 else 0
    if device_seed_count == 0:
        return
    device_seeds = seeds[:device_seed_count]

    cfg_template = deepcopy(cfg_base)
    cfg_template['pipeline']['distance_um'] = d_ref
    cfg_template['pipeline']['Nm_per_symbol'] = nm_ref
    Ts_ref = calculate_dynamic_symbol_period(d_ref, cfg_template)
    min_win = _enforce_min_window(cfg_template, Ts_ref)
    cfg_template['pipeline']['symbol_period_s'] = Ts_ref
    cfg_template['pipeline']['time_window_s'] = max(cfg_template['pipeline'].get('time_window_s', 0.0), min_win)
    cfg_template.setdefault('detection', {})['decision_window_s'] = min_win

    seq_len_base = int(cfg_template['pipeline'].get('sequence_length', args.sequence_length))
    device_sequence_length = min(seq_len_base, 400)

    print(f"?? Device FoM anchor ({mode}): distance={d_ref} um, Nm={nm_ref:.3g}, seeds={device_seed_count}")

    frames: List[pd.DataFrame] = []
    if gm_values:
        cfg_gm = deepcopy(cfg_template)
        cfg_gm['pipeline']['sequence_length'] = device_sequence_length
        df_gm = run_sweep(
            cfg_gm,
            device_seeds,
            'oect.gm_S',
            gm_values,
            f"Device FoM gm ({mode})",
            progress_mode=args.progress,
            persist_csv=None,
            resume=False,
            debug_calibration=args.debug_calibration,
            cache_tag=f"device_fom_gm_{mode.lower()}",
            pm=pm,
            sweep_key=("sweep", mode, "DeviceFoM_gm") if (pm and mode_key is not None) else None,
            parent_key=mode_key if (pm and mode_key is not None) else None,
            recalibrate=args.recalibrate,
        )
        if not df_gm.empty:
            df_gm = df_gm.copy()
            df_gm['param_type'] = 'gm_S'
            df_gm['param_value'] = pd.to_numeric(df_gm['oect.gm_S'], errors='coerce')
            frames.append(df_gm)
    else:
        print("?? Device FoM gm sweep already up-to-date (resume)")

    if c_values:
        cfg_c = deepcopy(cfg_template)
        cfg_c['pipeline']['sequence_length'] = device_sequence_length
        df_c = run_sweep(
            cfg_c,
            device_seeds,
            'oect.C_tot_F',
            c_values,
            f"Device FoM C_tot ({mode})",
            progress_mode=args.progress,
            persist_csv=None,
            resume=False,
            debug_calibration=args.debug_calibration,
            cache_tag=f"device_fom_c_{mode.lower()}",
            pm=pm,
            sweep_key=("sweep", mode, "DeviceFoM_C") if (pm and mode_key is not None) else None,
            parent_key=mode_key if (pm and mode_key is not None) else None,
            recalibrate=args.recalibrate,
        )
        if not df_c.empty:
            df_c = df_c.copy()
            df_c['param_type'] = 'C_tot_F'
            df_c['param_value'] = pd.to_numeric(df_c['oect.C_tot_F'], errors='coerce')
            frames.append(df_c)
    else:
        print("?? Device FoM C_tot sweep already up-to-date (resume)")

    if frames:
        new_data = pd.concat(frames, ignore_index=True)
    else:
        new_data = pd.DataFrame()

    if not existing.empty and not new_data.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
    elif existing.empty:
        combined = new_data
    else:
        combined = existing

    if combined.empty:
        if not device_csv.exists() and not new_data.empty:
            _atomic_write_csv(device_csv, combined)
        return

    if 'param_value' not in combined.columns:
        combined['param_value'] = pd.to_numeric(combined.get('param_value', pd.Series(dtype=float)), errors='coerce')
    subset_cols = ['param_type', 'param_value']
    if 'use_ctrl' in combined.columns:
        subset_cols.append('use_ctrl')
    combined = combined.drop_duplicates(subset=subset_cols, keep='last').sort_values(by=subset_cols)
    _atomic_write_csv(device_csv, combined)
    print(f"?? Device FoM sweep saved to {device_csv} ({len(combined)} rows)")
def _run_guard_frontier(
    cfg_base: Dict[str, Any],
    seeds: List[int],
    mode: str,
    data_dir: Path,
    suffix: str,
    df_lod: Optional[pd.DataFrame],
    distances: List[float],
    guard_values: List[float],
    args: argparse.Namespace,
    pm: Optional[ProgressManager] = None,
    mode_key: Optional[Any] = None,
) -> None:
    """Compute guard-factor frontiers by maximizing IRT at each distance."""
    if not guard_values:
        return

    desired_ctrl = bool(cfg_base['pipeline'].get('use_control_channel', True))
    resume_flag = bool(getattr(args, 'resume', False))
    tradeoff_csv = data_dir / f"guard_tradeoff_{mode.lower()}{suffix}.csv"
    frontier_csv = data_dir / f"guard_frontier_{mode.lower()}{suffix}.csv"

    try:
        existing_tradeoff = pd.read_csv(tradeoff_csv)
    except Exception:
        existing_tradeoff = pd.DataFrame()
    try:
        existing_frontier = pd.read_csv(frontier_csv)
    except Exception:
        existing_frontier = pd.DataFrame()

    mode_lower = mode.lower()
    if mode_lower == "mosk":
        bits_per_symbol = 1.0
    elif mode_lower == "csk":
        levels = int(cfg_base['pipeline'].get('csk_levels', 4))
        bits_per_symbol = math.log2(max(levels, 2))
    else:
        bits_per_symbol = 2.0

    df_lod_ref = df_lod if df_lod is not None else pd.DataFrame()
    distance_candidates: List[float] = []
    if not df_lod_ref.empty and 'distance_um' in df_lod_ref.columns:
        dist_vals = pd.to_numeric(df_lod_ref['distance_um'], errors='coerce').dropna()
        if not dist_vals.empty:
            distance_candidates = sorted(dist_vals.unique().tolist())
    if not distance_candidates:
        distance_candidates = [float(d) for d in distances] if distances else []
    if not distance_candidates:
        distance_candidates = [float(cfg_base['pipeline'].get('distance_um', 50.0))]

    guard_values_sorted = [round(float(g), 4) for g in guard_values]
    guard_max_ts = float(cfg_base.get('_guard_max_ts', 0.0))
    guard_seed_count = min(len(seeds), 6) if len(seeds) > 0 else 0
    if guard_seed_count == 0:
        return
    guard_seeds = seeds[:guard_seed_count]
    distance_task = None
    if pm:
        try:
            pm.set_status(mode=mode, sweep='Guard Frontier')
        except Exception:
            pass
        try:
            distance_task = pm.task(total=len(distance_candidates), description=f'{mode} Guard frontier',
                                      parent=mode_key if mode_key is not None else None, kind='stage')
        except Exception:
            distance_task = None

    def _nm_for_distance(distance: float) -> float:
        if df_lod_ref.empty or 'lod_nm' not in df_lod_ref.columns:
            return float(cfg_base['pipeline'].get('Nm_per_symbol', 1e4))
        df_match = df_lod_ref.copy()
        if 'use_ctrl' in df_match.columns:
            df_match = df_match[df_match['use_ctrl'] == desired_ctrl]
        if df_match.empty:
            return float(cfg_base['pipeline'].get('Nm_per_symbol', 1e4))
        df_match = df_match.assign(__dist=pd.to_numeric(df_match['distance_um'], errors='coerce'))
        df_match = df_match.dropna(subset=['__dist'])
        if df_match.empty:
            return float(cfg_base['pipeline'].get('Nm_per_symbol', 1e4))
        dist_arr = df_match['__dist'].to_numpy(dtype=float)
        idx = int(np.argmin(np.abs(dist_arr - float(distance))))
        nm_candidate = float(pd.to_numeric(df_match.iloc[idx]['lod_nm'], errors='coerce'))
        if math.isfinite(nm_candidate) and nm_candidate > 0:
            return nm_candidate
        return float(cfg_base['pipeline'].get('Nm_per_symbol', 1e4))

    frames_tradeoff: List[pd.DataFrame] = []
    frontier_rows: List[Dict[str, Any]] = []
    distances_recomputed: set = set()
    distances_considered: set = set()

    try:
        for dist in distance_candidates:
            dist_float = float(dist)
            nm_target = _nm_for_distance(dist_float)
    
            existing_for_dist = existing_tradeoff.copy()
            if not existing_for_dist.empty:
                existing_for_dist['distance_um'] = pd.to_numeric(existing_for_dist['distance_um'], errors='coerce')
                existing_for_dist = existing_for_dist.dropna(subset=['distance_um'])
                existing_for_dist = existing_for_dist[np.isclose(existing_for_dist['distance_um'], dist_float, atol=1e-6)]
                if 'use_ctrl' in existing_for_dist.columns:
                    existing_for_dist = existing_for_dist[existing_for_dist['use_ctrl'] == desired_ctrl]
            else:
                existing_for_dist = pd.DataFrame()
    
            if not resume_flag:
                run_values = guard_values_sorted
                distances_recomputed.add(dist_float)
            else:
                done_values = {
                    round(float(v), 4)
                    for v in pd.to_numeric(
                        existing_for_dist.get('guard_factor', existing_for_dist.get('pipeline.guard_factor', pd.Series(dtype=float))),
                        errors='coerce',
                    ).dropna()
                }
                run_values = [g for g in guard_values_sorted if round(g, 4) not in done_values]
                if run_values:
                    distances_recomputed.add(dist_float)
    
            df_guard_new = pd.DataFrame()
            guard_pairs: List[Tuple[float, float]] = []
            for g in run_values:
                cfg_probe = deepcopy(cfg_base)
                cfg_probe['pipeline']['distance_um'] = dist_float
                cfg_probe['pipeline']['guard_factor'] = g
                ts_est = calculate_dynamic_symbol_period(dist_float, cfg_probe)
                guard_pairs.append((g, ts_est))

            filtered_pairs = guard_pairs
            if guard_max_ts > 0.0:
                filtered_pairs = [(g, ts) for g, ts in guard_pairs if ts <= guard_max_ts]
                skipped_ts = [(g, ts) for g, ts in guard_pairs if ts > guard_max_ts]
                if skipped_ts:
                    preview = ', '.join(f'g={g:.1f}(Ts={ts:.1f}s)' for g, ts in skipped_ts[:5])
                    more = '...' if len(skipped_ts) > 5 else ''
                    print(f'??  Guard frontier: skipping guard factors {preview}{more} at {dist_float:.0f}um (Ts limit {guard_max_ts:.1f}s).')
            else:
                warn_values = [(g, ts) for g, ts in guard_pairs if ts > 900.0]
                if warn_values:
                    warn_g, warn_ts = warn_values[0]
                    print(f'??  Guard frontier: guard {warn_g:.1f} at {dist_float:.0f}um yields Ts={warn_ts:.1f}s; consider --guard-max-ts to cap runtime.')
            allowed_pairs, skipped_samples, samples_cap = _calculate_guard_sampling_load(cfg_base, filtered_pairs)
            if skipped_samples and samples_cap > 0:
                preview = ', '.join(
                    f'g={g:.1f}(Ts={ts:.1f}s, ~{total/1e6:.1f}M samples)'
                    for g, ts, total in skipped_samples[:5]
                )
                more = '...' if len(skipped_samples) > 5 else ''
                cap_desc = f"{samples_cap/1e6:.1f}M"
                print(f'??  Guard frontier: skipping guard factors {preview}{more} at {dist_float:.0f}um (>~{cap_desc} samples per seed).')

            run_values = [g for g, _, _ in allowed_pairs]
            if not run_values:
                if guard_max_ts > 0.0:
                    print(f'??  Guard frontier: all guard factors skipped at {dist_float:.0f}um (Ts limit {guard_max_ts:.1f}s).')
                elif samples_cap > 0:
                    print(f'??  Guard frontier: all guard factors skipped at {dist_float:.0f}um (sample cap ~{samples_cap/1e6:.1f}M exceeded).')
                if distance_task:
                    try:
                        distance_task.update(1, description=f'{mode} d={dist_float:.0f}um (skipped)')
                    except Exception:
                        pass
                continue
            if run_values:
                cfg_dist = deepcopy(cfg_base)
                cfg_dist['pipeline']['distance_um'] = dist_float
                cfg_dist['pipeline']['Nm_per_symbol'] = nm_target
                cfg_dist['pipeline']['enable_isi'] = True
                Ts_ref = calculate_dynamic_symbol_period(dist_float, cfg_dist)
                min_win = _enforce_min_window(cfg_dist, Ts_ref)
                cfg_dist['pipeline']['symbol_period_s'] = Ts_ref
                cfg_dist['pipeline']['time_window_s'] = max(cfg_dist['pipeline'].get('time_window_s', 0.0), min_win)
                cfg_dist.setdefault('detection', {})['decision_window_s'] = min_win
    
                df_guard_new = run_sweep(
                    cfg_dist,
                    guard_seeds,
                    'pipeline.guard_factor',
                    run_values,
                    f"Guard sweep ({mode}, d={dist_float:.0f} um)",
                    progress_mode=args.progress,
                    persist_csv=tradeoff_csv,
                    resume=resume_flag,
                    debug_calibration=args.debug_calibration,
                    cache_tag=f"guard_frontier_{mode_lower}_{int(round(dist_float))}",
                    pm=pm,
                    sweep_key=("sweep", mode, f"GuardFrontier_d{int(round(dist_float))}") if (pm and mode_key is not None) else None,
                    parent_key=mode_key if (pm and mode_key is not None) else None,
                    recalibrate=args.recalibrate,
                )
                if not df_guard_new.empty:
                    df_guard_new = df_guard_new.copy()
                    df_guard_new['distance_um'] = dist_float
                    frames_tradeoff.append(df_guard_new)
    
            combined_parts: List[pd.DataFrame] = []
            if not existing_for_dist.empty:
                if 'distance_um' not in existing_for_dist.columns:
                    existing_for_dist['distance_um'] = dist_float
                combined_parts.append(existing_for_dist)
            if not df_guard_new.empty:
                combined_parts.append(df_guard_new)
            if not combined_parts:
                if distance_task:
                    try:
                        distance_task.update(1, description=f'{mode} d={dist_float:.0f}um (no data)')
                    except Exception:
                        pass
                continue
    
            df_guard_combined = pd.concat(combined_parts, ignore_index=True)
            if 'distance_um' not in df_guard_combined.columns:
                df_guard_combined['distance_um'] = dist_float
    
            guard_col = 'guard_factor' if 'guard_factor' in df_guard_combined.columns else 'pipeline.guard_factor'
            df_guard_combined[guard_col] = pd.to_numeric(df_guard_combined[guard_col], errors='coerce')
            df_guard_combined['symbol_period_s'] = pd.to_numeric(df_guard_combined['symbol_period_s'], errors='coerce')
            df_guard_combined['ser'] = pd.to_numeric(df_guard_combined['ser'], errors='coerce')
            df_guard_combined = df_guard_combined.dropna(subset=[guard_col, 'symbol_period_s', 'ser'])
            if df_guard_combined.empty:
                continue
    
            df_guard_combined['IRT'] = (
                bits_per_symbol / df_guard_combined['symbol_period_s']
            ) * (1.0 - df_guard_combined['ser'])
    
            best_idx = int(df_guard_combined['IRT'].idxmax())
    
            guard_val_raw = df_guard_combined.at[best_idx, guard_col]
            guard_val = _coerce_float(guard_val_raw)
            irt_val = _coerce_float(df_guard_combined.at[best_idx, 'IRT'])
            ser_val = _coerce_float(df_guard_combined.at[best_idx, 'ser'])
            symbol_period_val = _coerce_float(df_guard_combined.at[best_idx, 'symbol_period_s'])
            if 'Nm_per_symbol' in df_guard_combined.columns:
                nm_val_raw = df_guard_combined.at[best_idx, 'Nm_per_symbol']
                nm_val = _coerce_float(nm_val_raw, float(nm_target))
            else:
                nm_val = float(nm_target)
    
            distances_considered.add(dist_float)
            frontier_rows.append({
                'distance_um': dist_float,
                'best_guard_factor': guard_val,
                'max_irt_bps': irt_val,
                'ser_at_best_guard': ser_val,
                'symbol_period_s': symbol_period_val,
                'Nm_per_symbol': nm_val,
                'use_ctrl': desired_ctrl,
                'mode': mode,
                'bits_per_symbol': bits_per_symbol,
            })
            if distance_task:
                try:
                    distance_task.update(1, description=f'{mode} d={dist_float:.0f}um')
                except Exception:
                    pass
    
    finally:
        if distance_task is not None:
            try:
                distance_task.close()
            except Exception:
                pass
    if frames_tradeoff:
        df_new_tradeoff = pd.concat(frames_tradeoff, ignore_index=True)
        if not existing_tradeoff.empty and distances_recomputed:
            mask = pd.Series(True, index=existing_tradeoff.index)
            if 'distance_um' in existing_tradeoff.columns:
                dist_series = pd.to_numeric(existing_tradeoff['distance_um'], errors='coerce')
                mask &= ~dist_series.apply(lambda x: any(abs(x - d) < 1e-6 for d in distances_recomputed))
            if 'use_ctrl' in existing_tradeoff.columns:
                mask |= existing_tradeoff['use_ctrl'] != desired_ctrl
            existing_tradeoff = existing_tradeoff[mask]
        combined_tradeoff = pd.concat([existing_tradeoff, df_new_tradeoff], ignore_index=True)
    else:
        combined_tradeoff = existing_tradeoff

    if not combined_tradeoff.empty:
        if 'distance_um' not in combined_tradeoff.columns:
            combined_tradeoff['distance_um'] = pd.to_numeric(combined_tradeoff.get('distance_um', pd.Series(dtype=float)), errors='coerce')
        else:
            combined_tradeoff['distance_um'] = pd.to_numeric(combined_tradeoff['distance_um'], errors='coerce')
        if 'guard_factor' not in combined_tradeoff.columns and 'pipeline.guard_factor' in combined_tradeoff.columns:
            combined_tradeoff['guard_factor'] = pd.to_numeric(combined_tradeoff['pipeline.guard_factor'], errors='coerce')
        elif 'guard_factor' in combined_tradeoff.columns:
            combined_tradeoff['guard_factor'] = pd.to_numeric(combined_tradeoff['guard_factor'], errors='coerce')
        subset_cols = ['distance_um', 'guard_factor']
        if 'use_ctrl' in combined_tradeoff.columns:
            subset_cols.append('use_ctrl')
        combined_tradeoff = combined_tradeoff.dropna(subset=subset_cols)
        combined_tradeoff = combined_tradeoff.drop_duplicates(subset=subset_cols, keep='last').sort_values(subset_cols)
        _atomic_write_csv(tradeoff_csv, combined_tradeoff)
        print(f"?? Guard trade-off data saved to {tradeoff_csv} ({len(combined_tradeoff)} rows)")

    if frontier_rows:
        df_frontier_new = pd.DataFrame(frontier_rows)
        if not existing_frontier.empty and distances_considered:
            mask_front = pd.Series(True, index=existing_frontier.index)
            if 'distance_um' in existing_frontier.columns:
                dist_series = pd.to_numeric(existing_frontier['distance_um'], errors='coerce')
                mask_front &= ~dist_series.apply(lambda x: any(abs(x - d) < 1e-6 for d in distances_considered))
            if 'use_ctrl' in existing_frontier.columns:
                mask_front |= existing_frontier['use_ctrl'] != desired_ctrl
            existing_frontier = existing_frontier[mask_front]
        combined_frontier = pd.concat([existing_frontier, df_frontier_new], ignore_index=True)
    else:
        combined_frontier = existing_frontier

    if not combined_frontier.empty:
        combined_frontier['distance_um'] = pd.to_numeric(combined_frontier['distance_um'], errors='coerce')
        subset_cols = ['distance_um']
        if 'use_ctrl' in combined_frontier.columns:
            subset_cols.append('use_ctrl')
        combined_frontier = combined_frontier.dropna(subset=subset_cols)
        combined_frontier = combined_frontier.drop_duplicates(subset=subset_cols, keep='last').sort_values(subset_cols)
        _atomic_write_csv(frontier_csv, combined_frontier)
        print(f"?? Guard frontier saved to {frontier_csv} ({len(combined_frontier)} rows)")
def _write_hybrid_isi_distance_grid(cfg_base: Dict[str, Any], 
                                    distances_um: List[float], 
                                    guard_grid: List[float], 
                                    out_csv: Path, 
                                    seeds: List[int], 
                                    pm: Optional[ProgressManager] = None, 
                                    parent_key: Optional[Any] = None) -> None:
    """
    Generate ISI-distance grid for Hybrid mode 2D visualization.
    Creates a CSV with (distance_um, guard_factor, symbol_period_s, ser, use_ctrl) for heatmaps.
    """
    print(f"🔧 Generating Hybrid ISI-distance grid: {len(distances_um)} distances × {len(guard_grid)} guard factors")
    
    # Extract CTRL state from base config
    use_ctrl = bool(cfg_base['pipeline'].get('use_control_channel', True))
    
    rows = []
    guard_max_ts = float(cfg_base.get('_guard_max_ts', 0.0))
    mode_label = str(cfg_base.get('pipeline', {}).get('modulation', 'Hybrid'))
    combos: List[Tuple[float, float, float]] = []
    skipped: List[Tuple[float, float, float]] = []
    for d in distances_um:
        for g in guard_grid:
            cfg_probe = deepcopy(cfg_base)
            cfg_probe['pipeline']['distance_um'] = d
            cfg_probe['pipeline']['guard_factor'] = g
            ts_est = calculate_dynamic_symbol_period(d, cfg_probe)
            if guard_max_ts > 0.0 and ts_est > guard_max_ts:
                skipped.append((d, g, ts_est))
            else:
                combos.append((d, g, ts_est))
    if guard_max_ts > 0.0 and skipped:
        preview = ', '.join(f'{dist:.0f}um:g={gf:.1f}(Ts={ts:.1f}s)' for dist, gf, ts in skipped[:5])
        more = '...' if len(skipped) > 5 else ''
        print(f'??  Hybrid grid: skipping guard factors {preview}{more} (Ts limit {guard_max_ts:.1f}s).')
    elif guard_max_ts <= 0.0:
        warn_cases = [(dist, gf, ts) for dist, gf, ts in combos if ts > 900.0]
        if warn_cases:
            sample_dist, sample_gf, sample_ts = warn_cases[0]
            print(f'??  Hybrid grid: guard {sample_gf:.1f} at {sample_dist:.0f}um yields Ts={sample_ts:.1f}s; consider --guard-max-ts to cap runtime.')
    total_points = len(combos)
    if total_points == 0:
        if guard_max_ts > 0.0:
            print(f'??  Hybrid ISI-distance grid skipped: no guard factors meet Ts <= {guard_max_ts:.1f}s.')
        else:
            print('??  Hybrid ISI-distance grid skipped: no valid guard factors available.')
        return
    grid_task = None
    if pm:
        desc = f'{mode_label} ISI-distance grid ({total_points} pts)'
        try:
            grid_task = pm.task(total=total_points, description=desc, parent=parent_key, kind='stage')
        except Exception:
            grid_task = None
    try:
        for d, g, ts_hint in combos:
            seed_results = []
            for seed in seeds[:3]:  # Use first 3 seeds for speed
                try:
                    cfg = deepcopy(cfg_base)
                    cfg['pipeline']['distance_um'] = d
                    cfg['pipeline']['guard_factor'] = g
                    cfg['pipeline']['sequence_length'] = 200  # Faster for grid generation
                    cfg['pipeline']['enable_isi'] = True
                    cfg['pipeline']['random_seed'] = int(seed)  # Set seed in config
                    Ts = ts_hint
                    cfg['pipeline']['symbol_period_s'] = Ts
                    min_win = _enforce_min_window(cfg, Ts)
                    cfg['pipeline']['time_window_s'] = max(cfg['pipeline'].get('time_window_s', 0.0), min_win)
                    cfg.setdefault('detection', {})
                    cfg['detection']['decision_window_s'] = min_win  # FIXED: use min_win for consistency
                    try:
                        cal_seeds = list(range(4))  # Fast calibration with 4 seeds
                        th = calibrate_thresholds_cached(cfg, cal_seeds)
                        for k, v in th.items():
                            cfg['pipeline'][k] = v
                    except Exception:
                        pass  # Fallback to default thresholds if calibration fails
                    res = run_single_instance(cfg, seed, attach_isi_meta=True)
                    if res is not None:
                        ser_val = float(res.get('ser', res.get('SER', 1.0)))
                        mosk_err = int(res.get('subsymbol_errors', {}).get('mosk', 0))
                        csk_err = int(res.get('subsymbol_errors', {}).get('csk', 0))
                        L = int(cfg['pipeline']['sequence_length'])
                        mosk_ser_val = (mosk_err / L) if L > 0 else float('nan')
                        csk_ser_val = (csk_err / L) if L > 0 else float('nan')
                        ts_val = float(res.get('symbol_period_s', Ts))
                        exposures = max(L - mosk_err, 0)
                        den = max(exposures, 1)
                        csk_ser_cond_seed = (csk_err / den)
                        mosk_exposure_frac_seed = (exposures / L) if L > 0 else float('nan')
                        csk_ser_eff_seed = mosk_exposure_frac_seed * csk_ser_cond_seed
                        seed_results.append({
                            'ser': ser_val
                            , 'mosk_ser': mosk_ser_val
                            , 'csk_ser': csk_ser_val
                            , 'csk_ser_cond': csk_ser_cond_seed
                            , 'mosk_exposure_frac': mosk_exposure_frac_seed
                            , 'csk_ser_eff': csk_ser_eff_seed
                            , 'symbol_period_s': ts_val
                        })
                except Exception as e:
                    print(f'??  Grid point failed (d={d}, g={g}, seed={seed}): {e}')
                    continue
            if seed_results:
                median_ser = np.median([r['ser'] for r in seed_results])
                median_ts = np.median([r['symbol_period_s'] for r in seed_results])
                rows.append({
                    'distance_um': d
                    , 'guard_factor': g
                    , 'symbol_period_s': median_ts
                    , 'ser': median_ser
                    , 'mosk_ser': float(np.nanmedian([r['mosk_ser'] for r in seed_results]))
                    , 'csk_ser': float(np.nanmedian([r['csk_ser'] for r in seed_results]))
                    , 'csk_ser_cond': float(np.nanmedian([r['csk_ser_cond'] for r in seed_results]))
                    , 'mosk_exposure_frac': float(np.nanmedian([r['mosk_exposure_frac'] for r in seed_results]))
                    , 'csk_ser_eff': float(np.nanmedian([r['csk_ser_eff'] for r in seed_results]))
                    , 'use_ctrl': use_ctrl
                })
            if grid_task:
                try:
                    grid_task.update(1, description=f'{mode_label} d={d:.0f}um g={g:.1f}')
                except Exception:
                    pass
    finally:
        if grid_task is not None:
            try:
                grid_task.close()
            except Exception:
                pass
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
        # Sanity check: verify consistency between csk_ser and csk_ser_eff
        if {'csk_ser', 'csk_ser_eff'}.issubset(new_df.columns):
            diff = (new_df['csk_ser'] - new_df['csk_ser_eff']).abs().max()
            if pd.notna(diff) and diff > 5e-3:
                print(f"⚠️  median(csk_ser) and median(exposure*cond) differ by up to {diff:.3f}")
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
    # Avoid double‑installing tee logging (it is already set in main())
    if not args.no_log:
        root = logging.getLogger()
        if not any(hasattr(h, "baseFilename") for h in root.handlers):
            setup_tee_logging(Path(args.logdir), prefix="run_final_analysis", fsync=args.fsync_logs)
    else:
        print("[log] File logging disabled by --no-log")
    cfg = preprocess_config_full(yaml.safe_load(open(project_root / "config" / "default.yaml", encoding='utf-8')))
    
    # NEW: Apply CLI overrides
    cfg = apply_cli_overrides(cfg, args)
    
    # Adaptive CTRL configuration
    cfg['_ctrl_auto'] = args.ctrl_auto
    cfg['_ctrl_auto_rho_min_abs'] = args.ctrl_rho_min_abs
    cfg['_ctrl_auto_min_gain_db'] = args.ctrl_snr_min_gain_db
    
    # Apply CTRL ablation control
    use_ctrl = args.use_ctrl
    cfg['pipeline']['use_control_channel'] = use_ctrl
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
    suffix = f"_{args.variant}" if getattr(args, 'variant', '') else ''

    cfg['pipeline']['enable_isi'] = not args.disable_isi
    cfg['pipeline']['modulation'] = mode
    cfg['pipeline']['sequence_length'] = args.sequence_length
    mode_label = _canonical_mode_name(mode)
    distance_overrides = getattr(args, 'distances_by_mode', {})
    lod_distance_grid = resolve_mode_distance_grid(mode, cfg, distance_overrides)
    if not lod_distance_grid:
        lod_distance_grid = list(DEFAULT_MODE_DISTANCES.get(mode_label, DEFAULT_MODE_DISTANCES['MoSK']))
    else:
        lod_distance_grid = list(lod_distance_grid)
    cfg['pipeline']['distances_um'] = list(lod_distance_grid)
    lod_cfg = cfg.setdefault('lod_distances_um', {})
    if isinstance(lod_cfg, dict):
        lod_cfg[mode_label] = list(lod_distance_grid)
    override_label: Optional[str] = None
    if distance_overrides:
        if mode_label in distance_overrides:
            override_label = mode_label
        elif 'ALL' in distance_overrides:
            override_label = 'ALL'
    if override_label:
        print(f"?? Using CLI distance grid for {mode} (source={override_label}): {lod_distance_grid}")
    else:
        print(f"?? Using distance grid for {mode}: {lod_distance_grid}")
    profile = str(getattr(args, 'channel_profile', 'tri')).lower()
    cfg['pipeline']['channel_profile'] = profile
    if profile == 'single':
        cfg['pipeline']['csk_dual_channel'] = False
    elif profile == 'dual':
        cfg['pipeline']['csk_dual_channel'] = cfg['pipeline'].get('csk_dual_channel', True)
    cfg['pipeline']['use_control_channel'] = bool(args.use_ctrl)
    if profile in ('single', 'dual'):
        cfg['pipeline']['use_control_channel'] = False
    print(f"CTRL subtraction: {'ON' if cfg['pipeline']['use_control_channel'] else 'OFF'}")
    cfg['verbose'] = args.verbose
    # Stage 13: pass adaptive-CI config via cfg (so workers see it)
    cfg['_stage13_target_ci'] = float(args.target_ci)
    cfg['_stage13_min_ci_seeds'] = int(args.min_ci_seeds)
    cfg['_stage13_lod_delta'] = float(args.lod_screen_delta)
    cfg['_watchdog_secs'] = int(args.watchdog_secs)
    cfg['_nonlod_watchdog_secs'] = int(args.nonlod_watchdog_secs)
    cfg['_guard_max_ts'] = float(args.guard_max_ts)
    guard_cap_cfg = (cfg.get('analysis', {}) or {}).get('guard_samples_cap', DEFAULT_GUARD_SAMPLES_CAP)
    try:
        guard_cap_val = float(guard_cap_cfg)
    except (TypeError, ValueError):
        guard_cap_val = float(DEFAULT_GUARD_SAMPLES_CAP)
    cfg['_guard_total_samples_cap'] = guard_cap_val
    cfg['_analytic_lod_bracket'] = getattr(args, 'analytic_lod_bracket', False)
    # Apply CLI optimization parameters to config
    cfg['_cal_eps_rel'] = args.cal_eps_rel
    cfg['_cal_patience'] = args.cal_patience  
    cfg['_cal_min_seeds'] = args.cal_min_seeds
    cfg['_cal_min_samples_per_class'] = args.cal_min_samples
    cfg['_min_decision_points'] = args.min_decision_points
    
    # NEW: Pass LoD validation sequence length override
    if getattr(args, 'lod_validate_seq_len', None) is not None:
        cfg['_lod_validate_seq_len'] = int(args.lod_validate_seq_len)
    # make LoD skip/limit flags visible to workers via cfg
    if getattr(args, "max_lod_validation_seeds", None) is not None:
        cfg["max_lod_validation_seeds"] = int(args.max_lod_validation_seeds)
    if getattr(args, "max_symbol_duration_s", None) is not None:
        cfg["max_symbol_duration_s"] = float(args.max_symbol_duration_s)
    if getattr(args, "max_ts_for_lod", None) is not None:
        cfg["max_ts_for_lod"] = float(args.max_ts_for_lod)
    # Map --allow-ts-exceed to config for consistency
    if getattr(args, "allow_ts_exceed", False):
        cfg.setdefault("analysis", {})["allow_ts_exceed"] = True

    if mode.startswith("CSK"):
        cfg['pipeline']['csk_levels'] = 4
        cfg['pipeline'].setdefault('csk_target_channel', 'DA')
        cfg['pipeline']['csk_level_scheme'] = args.csk_level_scheme
        profile = cfg['pipeline'].get('channel_profile', 'tri')
        if profile == 'single':
            cfg['pipeline']['csk_dual_channel'] = False
            cfg['pipeline']['csk_target_channel'] = cfg['pipeline'].get('csk_target_channel', 'DA')
        elif profile == 'dual':
            cfg['pipeline']['csk_dual_channel'] = cfg['pipeline'].get('csk_dual_channel', True)
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
    # 🛠 CLI override for Nm grid (preserves YAML default behavior)
    nm_values: List[Union[float, int]]
    if args.nm_grid.strip():
        try:
            nm_values = [int(float(x.strip())) for x in args.nm_grid.split(',') if x.strip()]
            print(f"📋 Using CLI Nm grid: {nm_values}")
        except ValueError as e:
            print(f"⚠️  Invalid --nm-grid format: {args.nm_grid}. Error: {e}")
            print("   Using YAML default instead.")
            nm_values = list(cfg.get('Nm_range', [200,500,1000,1600,2500,4000,6300,10000,16000,25000,40000,63000]))
    else:
        nm_values = list(cfg.get('Nm_range', [200,500,1000,1600,2500,4000,6300,10000,16000,25000,40000,63000]))
        print(f"📋 Using YAML Nm_range: {nm_values}")
    guard_values = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
    ser_jobs = len(nm_values) * args.num_seeds
    lod_seed_cap = 10
    lod_jobs = len(lod_distance_grid) * (lod_seed_cap * 8 + lod_seed_cap + 5)  # initial estimate only
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
    ser_csv = data_dir / f"ser_vs_nm_{mode.lower()}{suffix}.csv"

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
        parent_key=mode_key if hierarchy_supported else None,  # 🛠️ CHANGE: parent_key -> mode_key
        recalibrate=args.recalibrate  # 🛠️ ADD THIS LINE
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

    # --- Auto-refine near target SER (adds a few Nm points between the bracket) ---
    if args.ser_refine:
        try:
            # Always read the latest CSV on disk so resume/de-dupe is consistent
            df_ser_all = pd.read_csv(ser_csv) if ser_csv.exists() else df_ser_nm

            # Propose midpoints between the first bracket that crosses the target
            refine_candidates = _auto_refine_nm_points_from_df(
                df_ser_all,
                target=float(args.ser_target),
                extra_points=int(args.ser_refine_points)
            )

            # Filter out any Nm that are already present for THIS CTRL state
            if refine_candidates:
                desired_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
                done = load_completed_values(ser_csv, 'pipeline_Nm_per_symbol', desired_ctrl)
                refine_candidates = [n for n in refine_candidates if canonical_value_key(n) not in done]

            if refine_candidates:
                print(f"🔎 SER auto-refine around {args.ser_target:.2%}: {refine_candidates}")

                # Run only those extra Nm points and append to the same CSV (resume-safe)
                ser_refine_key = ("sweep", mode, "SER_refine")
                df_refined = run_sweep(
                    cfg, seeds,
                    'pipeline.Nm_per_symbol',
                    [float(n) for n in refine_candidates],
                    f"SER refine near {args.ser_target:.2%} ({mode})",
                    progress_mode=args.progress,
                    persist_csv=ser_csv,
                    resume=args.resume,
                    debug_calibration=args.debug_calibration,
                    pm=pm,
                    sweep_key=ser_refine_key if (args.progress == "gui") else None,
                    parent_key=mode_key if (args.progress == "gui") else None,
                    recalibrate=args.recalibrate
                )

                # Re‑de‑dupe the CSV so plots read a clean file
                if ser_csv.exists():
                    existing = pd.read_csv(ser_csv)
                    nm_key = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in existing.columns else 'pipeline.Nm_per_symbol'
                    combined = existing if df_refined.empty else pd.concat([existing, df_refined], ignore_index=True)
                    if 'use_ctrl' in combined.columns:
                        combined = combined.drop_duplicates(subset=[nm_key, 'use_ctrl'], keep='last').sort_values([nm_key, 'use_ctrl'])
                    else:
                        combined = combined.drop_duplicates(subset=[nm_key], keep='last').sort_values([nm_key])
                    _atomic_write_csv(ser_csv, combined)
                    print(f"✅ SER auto-refine appended; CSV updated: {ser_csv}")
            else:
                print(f"ℹ️  SER auto-refine: no bracket found or all refine Nm already present.")
        except Exception as e:
            print(f"⚠️  SER auto-refine failed: {e}")

    # ---------- 1′) HDS grid (Hybrid only): Nm × distance with component errors ----------
    if mode == "Hybrid":
        print("\n1′. Building Hybrid HDS grid (Nm × distance)…")
        grid_csv = data_dir / "hybrid_hds_grid.csv"
        
        # Use the distances configured for LoD (or fallback to a small set)
        distances = list(lod_distance_grid)
        if not distances:
            # Fallback to a representative subset for the grid
            distances = [25, 50, 100, 150, 200]
        
        # Use a subset of Nm values for the grid (to keep computation manageable)
        grid_nm_values: List[Union[float, int]] = [500, 1000, 1600, 2500, 4000, 6300, 10000] # subset of nm_values
        
        rows = []
        for d in distances:
            cfg_d = deepcopy(cfg)
            cfg_d['pipeline']['distance_um'] = int(d)
            
            # Recompute symbol period if your model depends on distance
            try:
                Ts = calculate_dynamic_symbol_period(int(d), cfg_d)
                cfg_d['pipeline']['symbol_period_s'] = Ts
                dt = float(cfg_d['sim']['dt_s'])
                min_pts = int(cfg_d.get('_min_decision_points', 4))
                min_win = _enforce_min_window(cfg_d, Ts)
                cfg_d['pipeline']['time_window_s'] = max(cfg_d['pipeline'].get('time_window_s', 0.0), min_win)
                cfg_d['detection']['decision_window_s'] = min_win
            except Exception:
                pass
            
            # Run SER sweep for this distance
            df_d = run_sweep(
                cfg_d, seeds,
                'pipeline.Nm_per_symbol',
                grid_nm_values,
                f"HDS grid Hybrid (d={d} um)",
                progress_mode=args.progress,
                persist_csv=None,  # Don't persist intermediate results
                resume=args.resume,
                debug_calibration=args.debug_calibration,
                cache_tag=f"d{int(d)}um"   # <<< NEW: distance-scoped cache tag
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
    d_run = [int(x) for x in lod_distance_grid]
    lod_csv = data_dir / f"lod_vs_distance_{mode.lower()}{suffix}.csv"
    pm.set_status(mode=mode, sweep="LoD vs distance")
    # Use the same worker count chosen at mode start (honors --extreme-mode/--max-workers)
    pool = global_pool.get_pool(max_workers=maxw)
    use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))

    estimated_per_distance = args.num_seeds * 8 + args.num_seeds + 5

    # --- NEW: find fully-completed distances for this CTRL state ---
    done_distances: set[int] = set()
    failed_distances: set[int] = set()

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

            if 'lod_nm' in df_prev.columns and 'distance_um' in df_prev.columns:
                for _, row in df_prev.iterrows():
                    dist = int(row['distance_um'])
                    lod_nm = row.get('lod_nm', np.nan)
                    
                    # Check if this distance succeeded or failed
                    if pd.notna(lod_nm) and float(lod_nm) > 0:
                        # Also check SER if available
                        ser_ok = True
                        if 'ser_at_lod' in df_prev.columns:
                            ser = row.get('ser_at_lod', np.nan)
                            if pd.notna(ser):
                                ser_ok = float(ser) <= 0.01  # or your target
                        
                        if ser_ok:
                            done_distances.add(dist)
                        else:
                            failed_distances.add(dist)
                    else:
                        # NaN or <= 0 means failed
                        failed_distances.add(dist)

                if done_distances:
                    print(f"↩️  Resume: {len(done_distances)} LoD distance(s) already complete: "
                      f"{sorted(done_distances)} um")
                if failed_distances:
                    print(f"🔄  Resume: {len(failed_distances)} LoD distance(s) need retry: "
                      f"{sorted(failed_distances)} um")
                if failed_distances and args.lod_skip_retry:
                    print(f"🔄  --lod-skip-retry: accepting failures for {sorted(failed_distances)} um.")
                    done_distances.update(failed_distances)
                    failed_distances.clear()
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

    # Submit distance jobs with continuous top-up for maximum worker utilization
    pending: Set[Future] = set()
    fut2dist: Dict[Future, int] = {}
    fut2wid: Dict[Future, int] = {}
    next_idx = 0
    last_lod_guess: Optional[int] = None
    default_batch = maxw  # fill the pool by default
    batch_size = max(1, int(getattr(args, "lod_distance_concurrency", default_batch)))
    lod_results: List[Dict[str, Any]] = []
    tmo = float(getattr(args, "lod_distance_timeout_s", 7200.0))

    # Ensure we only process distances that have progress queues
    d_run_work_with_queues = [d for d in d_run_work if d in progress_queues]

    def _submit_one(dist: int, wid_hint: int) -> None:
        """Submit one distance job to the pool."""
        wid = wid_hint % max(1, maxw)
        if hasattr(pm, "worker_update"):
            pm.worker_update(wid, f"LoD | d={dist} um")
        q = progress_queues[dist]
        seeds_for_lod = _choose_seeds_for_distance(float(dist), seeds, args.lod_num_seeds)
        args.full_seeds = seeds
        fut = pool.submit(
            process_distance_for_lod, float(dist), cfg, seeds_for_lod, 0.01,
            args.debug_calibration, q, args.resume, args, warm_lod_guess=last_lod_guess
        )
        pending.add(fut)
        fut2dist[fut] = dist
        fut2wid[fut] = wid

    # Prime the pool with initial batch
    while next_idx < len(d_run_work_with_queues) and len(pending) < batch_size:
        _submit_one(int(d_run_work_with_queues[next_idx]), next_idx)
        next_idx += 1

    # Drain with continuous top-up
    while pending:
        try:
            done_fut = next(as_completed(pending, timeout=tmo if (tmo and tmo > 0) else None))
        except TimeoutError:
            print(f"⚠️  LoD timeout in top-up scheduler (timeout={tmo}s), continuing...")
            continue

        pending.remove(done_fut)
        dist = fut2dist.pop(done_fut)
        wid = fut2wid.pop(done_fut, 0)

        # === Reuse existing result-processing body ===
        res = {}  # Initialize to prevent unbound variable
        try:
            res = done_fut.result(timeout=1.0)  # Already completed, so short timeout
            # Store actual progress for accurate parent counting  
            res['actual_progress'] = res.get('actual_progress', estimated_per_distance)
            
            # NEW: Update warm-start guess if we got a valid LoD (feeds next submissions)
            if res and not pd.isna(res.get('lod_nm', np.nan)):
                last_lod_guess = int(res['lod_nm'])
        except TimeoutError:
            print(f"⚠️  LoD timeout at {dist}μm (mode={mode}, use_ctrl={use_ctrl}, timeout={tmo}s), skipping")
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
                # Update worker bar total to match actual work done
                pm.update_total(key=("worker", wid), total=actual_total,
                                label=f"Worker {wid:02d}", kind="worker", parent=None)
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
            print(f"  [{len(lod_results)}/{len(d_run_work_with_queues)}] {dist}μm done: LoD={res['lod_nm']:.0f} molecules")
        else:
            print(f"  ⚠️  [{len(lod_results)}/{len(d_run_work_with_queues)}] {dist}μm failed")
        
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
                        original_lod_estimate = len(d_run_work_with_queues) * estimated_per_distance
                        actual_lod_total = new_lod_total
                        lod_diff = actual_lod_total - original_lod_estimate
                        
                        # Update overall total if there's a significant difference
                        if abs(lod_diff) > 0:
                            new_overall_total = ser_jobs + actual_lod_total + isi_jobs
                            pm.update_total(key=overall_key, total=new_overall_total,
                                            label=f"Overall ({mode})", kind="overall")
            
            if bar:
                remaining = max(0, actual_total - int(getattr(bar, "completed", 0)))
                if remaining > 0:
                    bar.update(remaining)
                bar.close()

        # Top up with next distance if available
        if next_idx < len(d_run_work_with_queues):
            _submit_one(int(d_run_work_with_queues[next_idx]), next_idx)
            next_idx += 1

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
    failed_distances = set(sorted(
        int(r['distance_um'])
        for r in lod_results
        if (isinstance(r, dict) and 'distance_um' in r and 
            (('lod_nm' not in r) or pd.isna(r.get('lod_nm'))))
    ))

    if failed_distances:
        print(f"⚠️  No LoD found at distances: {failed_distances} um")

    # Build DataFrame of newly computed LoD rows (may be empty if resume skipped everything)
    df_lod_new = pd.DataFrame(real_lod_results) if real_lod_results else pd.DataFrame()

    if lod_csv.exists():
        prior = pd.read_csv(lod_csv)
        subset = ['distance_um'] + (['use_ctrl'] if 'use_ctrl' in set(prior.columns) | set(df_lod_new.columns) else [])
        combined = prior if df_lod_new.empty else pd.concat([prior, df_lod_new], ignore_index=True)
        
        # De-dupe LoD CSV preferring valid rows over NaNs (resume-safe)
        if 'lod_nm' in combined.columns:
            combined['__is_valid__'] = pd.to_numeric(combined['lod_nm'], errors='coerce').gt(0) & np.isfinite(pd.to_numeric(combined['lod_nm'], errors='coerce'))
            keys = ['distance_um'] + (['use_ctrl'] if 'use_ctrl' in combined.columns else [])
            # within each (distance, ctrl) group: prefer last valid; else last row
            sorted_combined = combined.sort_index()  # resume appends at higher index
            latest_per_group = sorted_combined.groupby(keys, as_index=False, group_keys=False).tail(1)
            valid_latest = (
                sorted_combined[sorted_combined['__is_valid__']]
                .groupby(keys, as_index=False, group_keys=False)
                .tail(1)
            )
            combined = (
                pd.concat([latest_per_group, valid_latest])
                .drop_duplicates(subset=keys, keep='last')
                .drop(columns=['__is_valid__'])
                .sort_values(keys)
            )
        else:
            combined = combined.drop_duplicates(subset=subset, keep='last').sort_values(subset)
        
        _atomic_write_csv(lod_csv, combined)
        df_lod = combined
    elif not df_lod_new.empty:
        _atomic_write_csv(lod_csv, df_lod_new)
        df_lod = df_lod_new
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

    # Around line 3889 in run_final_analysis.py
    if not df_lod.empty:
        cols_to_show = [c for c in ['distance_um', 'lod_nm', 'ser_at_lod', 'use_ctrl'] if c in df_lod.columns]
        print(f"\nLoD vs Distance (head) for {mode}:")
        print(df_lod[cols_to_show].head().to_string(index=False))

    try:
        _run_device_fom_sweeps(
            cfg,
            seeds,
            mode,
            data_dir,
            suffix,
            df_lod,
            args.resume,
            args,
            pm=pm,
            mode_key=mode_key if hierarchy_supported else None,
        )
    except Exception as device_exc:
        print(f"??  Device FoM sweep skipped: {device_exc}")
    # ---------- 3) ISI trade-off (guard-factor sweep) ----------
    # BEFORE the sweep
    do_isi = False
    if args.isi_sweep == "always":
        do_isi = True
    elif args.isi_sweep == "auto":
        do_isi = bool(cfg['pipeline'].get('enable_isi', False))
    else:  # "never"
        do_isi = False

    if do_isi:
        print("\n3. Running ISI trade-off sweep (guard factor)…")
        
        # --- pick an anchor for ISI sweep (after LoD) ---
        d_ref = None
        nm_ref = None
        lod_csv = data_dir / f"lod_vs_distance_{mode.lower()}{suffix}.csv"
        if lod_csv.exists():
            df_lod_all = pd.read_csv(lod_csv)
            # choose CTRL-matching rows
            if 'use_ctrl' in df_lod_all.columns:
                df_lod_all = df_lod_all[df_lod_all['use_ctrl'] == bool(cfg['pipeline'].get('use_control_channel', True))]
            # guard against non-numeric distances or LoDs
            if {'distance_um', 'lod_nm'}.issubset(df_lod_all.columns):
                dist_numeric = pd.to_numeric(df_lod_all['distance_um'], errors='coerce')
                lod_numeric = pd.to_numeric(df_lod_all['lod_nm'], errors='coerce')
                valid_mask = dist_numeric.notna() & lod_numeric.notna()
                if valid_mask.any():
                    dist_numeric = dist_numeric[valid_mask]
                    lod_numeric = lod_numeric[valid_mask]
                    # pick median distance and choose the closest available LoD entry
                    d_ref = int(round(np.median(dist_numeric)))
                    nearest_idx = (dist_numeric - d_ref).abs().idxmin()
                    nm_ref = float(lod_numeric.loc[nearest_idx])

        # Fallback if LoD CSV missing
        if d_ref is None:
            d_ref = int(cfg['pipeline'].get('distance_um', 50))
        if nm_ref is None:
            nm_ref = float(cfg['pipeline'].get('Nm_per_symbol', 2000.0))  # conservative default

        # bake into cfg for the ISI sweep
        cfg['pipeline']['distance_um'] = d_ref
        cfg['pipeline']['Nm_per_symbol'] = nm_ref

        # recompute Ts and window for that distance
        Ts_ref = calculate_dynamic_symbol_period(d_ref, cfg)
        min_win = _enforce_min_window(cfg, Ts_ref)
        cfg['pipeline']['symbol_period_s'] = Ts_ref
        cfg['pipeline']['time_window_s'] = max(cfg['pipeline'].get('time_window_s', 0.0), min_win)
        cfg.setdefault('detection', {})['decision_window_s'] = min_win
        
        # ensure ISI ON during the sweep
        cfg['pipeline']['enable_isi'] = True

        guard_values_candidates = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
        guard_pairs = []
        for g in guard_values_candidates:
            cfg_probe = deepcopy(cfg)
            cfg_probe['pipeline']['guard_factor'] = g
            ts_est = calculate_dynamic_symbol_period(d_ref, cfg_probe)
            guard_pairs.append((g, ts_est))

        guard_max_ts = float(cfg.get('_guard_max_ts', 0.0))
        if guard_max_ts > 0.0:
            allowed_ts_pairs = [(g, ts) for g, ts in guard_pairs if ts <= guard_max_ts]
            skipped_ts_pairs = [(g, ts) for g, ts in guard_pairs if ts > guard_max_ts]
            guard_pairs = allowed_ts_pairs
            if skipped_ts_pairs:
                preview = ', '.join(f'g={g:.1f}(Ts={ts:.1f}s)' for g, ts in skipped_ts_pairs[:5])
                more = '...' if len(skipped_ts_pairs) > 5 else ''
                print(f"??  ISI trade-off: skipping guard factors {preview}{more} at d={d_ref:.0f}um (Ts limit {guard_max_ts:.1f}s).")
        else:
            warn_pairs = [(g, ts) for g, ts in guard_pairs if ts > 900.0]
            if warn_pairs:
                g_warn, ts_warn = warn_pairs[0]
                print(f"??  ISI trade-off: guard {g_warn:.1f} gives Ts={ts_warn:.1f}s at d={d_ref:.0f}um; consider --guard-max-ts to cap runtime.")

        allowed_pairs, skipped_samples, samples_cap = _calculate_guard_sampling_load(cfg, guard_pairs)
        if skipped_samples and samples_cap > 0:
            preview = ', '.join(
                f'g={g:.1f}(Ts={ts:.1f}s, ~{total/1e6:.1f}M samples)'
                for g, ts, total in skipped_samples[:5]
            )
            more = '...' if len(skipped_samples) > 5 else ''
            cap_desc = f"{samples_cap/1e6:.1f}M"
            print(f"??  ISI trade-off: skipping guard factors {preview}{more} at d={d_ref:.0f}um (>~{cap_desc} samples per seed).")

        guard_values = [g for g, _, _ in allowed_pairs]
        if not guard_values:
            print("??  ISI trade-off skipped: no guard factors within runtime limits.")
            isi_jobs = 0
        else:
            isi_jobs = len(guard_values) * len(seeds)

        isi_csv = data_dir / f"isi_tradeoff_{mode.lower()}{suffix}.csv"
        pm.set_status(mode=mode, sweep="ISI trade-off")
        isi_key = ("sweep", mode, "ISI_tradeoff")
        isi_bar = None
        if hierarchy_supported:
            isi_bar = pm.task(total=isi_jobs, description="ISI trade-off (guard)",
                            parent=mode_key, key=isi_key, kind="sweep")
        if guard_values:
            df_isi = run_sweep(
                cfg, seeds,
                'pipeline.guard_factor',
                guard_values,
                f"ISI trade-off ({mode})",
                progress_mode=args.progress,
                persist_csv=isi_csv,
                resume=args.resume,
                debug_calibration=args.debug_calibration,
                pm=pm,
                sweep_key=isi_key if hierarchy_supported else None,
                parent_key=mode_key if hierarchy_supported else None,
                recalibrate=args.recalibrate
            )
        else:
            df_isi = pd.DataFrame()
        if isi_bar:
            isi_bar.close()
        # De-duplicate by (guard_factor, use_ctrl)
        if isi_csv.exists():
            existing = pd.read_csv(isi_csv)
            gf_key = 'guard_factor' if 'guard_factor' in existing.columns else 'pipeline.guard_factor'
            if gf_key in existing.columns:
                combined = existing if df_isi.empty else pd.concat([existing, df_isi], ignore_index=True)
                subset_cols: List[str] = []
                if 'distance_um' in combined.columns:
                    subset_cols.append('distance_um')
                subset_cols.append(gf_key)
                if 'use_ctrl' in combined.columns:
                    subset_cols.append('use_ctrl')
                combined = combined.drop_duplicates(subset=subset_cols, keep='last').sort_values(by=subset_cols)
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
                    seeds=seeds,
                    pm=pm,
                    parent_key=mode_key if hierarchy_supported else None
                )
            else:
                print(f"✓ ISI grid exists: {isi_grid_csv}")
    else:
        print(f"\n3. ISI trade-off sweep skipped by --isi-sweep={args.isi_sweep}.")

    try:
        _run_guard_frontier(
            cfg,
            seeds,
            mode,
            data_dir,
            suffix,
            df_lod,
            [float(d) for d in lod_distance_grid],
            [float(g) for g in guard_values],
            args,
            pm=pm,
            mode_key=mode_key if hierarchy_supported else None,
        )
    except Exception as guard_exc:
        print(f"??  Guard frontier sweep skipped: {guard_exc}")
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

    This diagnostic validates that the simulator uses the correct correlation
    coefficient for each modulation/control configuration and that the reported
    differential noise matches the analytic expectation.
    """
    print("Running Cross-Channel Noise Correlation Diagnostics...")

    rho_cases: List[Dict[str, Union[float, str]]] = [
        {"rho": 0.0, "description": "Independent channels"},
        {"rho": 0.3, "description": "Moderate correlation"},
        {"rho": 0.7, "description": "Strong correlation"},
    ]

    mode_scenarios: List[Dict[str, Any]] = [
        {"mode": "MoSK", "ctrl_options": [False]},
        {"mode": "CSK", "ctrl_options": [False, True]},
        {"mode": "Hybrid", "ctrl_options": [False, True]},
    ]

    with open(project_root / "config" / "default.yaml", encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)
    base_config = preprocess_config_full(base_config)

    tolerance = 5e-13
    failures: List[str] = []

    pipeline_logger = logging.getLogger('src.pipeline')
    previous_level = pipeline_logger.level
    level_changed = False
    if pipeline_logger.getEffectiveLevel() < logging.ERROR:
        pipeline_logger.setLevel(logging.ERROR)
        level_changed = True

    def _expected_sigma_diff_for(cfg: Dict[str, Any], sigma_da: float, sigma_sero: float) -> float:
        pipeline_cfg = cfg.get('pipeline', {})
        noise_cfg = cfg.get('noise', {})
        mod = str(pipeline_cfg.get('modulation', '')).upper()
        use_ctrl = bool(pipeline_cfg.get('use_control_channel', True))
        rho_pre = float(noise_cfg.get('rho_corr', noise_cfg.get('rho_correlated', 0.0)))
        rho_post = float(noise_cfg.get('rho_between_channels_after_ctrl', 0.0))
        rho_post = float(pipeline_cfg.get('rho_cc_measured', rho_post))
        if mod == 'MOSK' or not use_ctrl:
            rho_target = rho_pre
        else:
            rho_target = rho_post
        rho_target = max(-1.0, min(1.0, rho_target))
        var_diff = sigma_da * sigma_da + sigma_sero * sigma_sero - 2.0 * rho_target * sigma_da * sigma_sero
        return math.sqrt(max(var_diff, 0.0))

    try:
        for scenario in mode_scenarios:
            mode = str(scenario['mode'])
            ctrl_options = scenario['ctrl_options']
            for use_ctrl in ctrl_options:
                scenario_label = f"{mode} (CTRL {'on' if use_ctrl else 'off'})"
                print(f"  Scenario: {scenario_label}")
    
                for case in rho_cases:
                    rho_val = float(case['rho'])
                    test_config = deepcopy(base_config)
                    pipeline_cfg = test_config.setdefault('pipeline', {})
                    pipeline_cfg.pop('rho_cc_measured', None)
                    pipeline_cfg['modulation'] = mode
                    pipeline_cfg['sequence_length'] = 10
                    pipeline_cfg['Nm_per_symbol'] = 1e4
                    pipeline_cfg['distance_um'] = 100
                    pipeline_cfg['use_control_channel'] = bool(use_ctrl) if mode != 'MoSK' else False
                    pipeline_cfg['channel_profile'] = 'tri' if pipeline_cfg['use_control_channel'] else 'dual'
                    pipeline_cfg['show_progress'] = False
                    pipeline_cfg['random_seed'] = pipeline_cfg.get('random_seed', 2025)
    
                    test_config['disable_progress'] = True
    
                    noise_cfg = test_config.setdefault('noise', {})
                    noise_cfg['rho_between_channels_after_ctrl'] = rho_val
                    noise_cfg['rho_corr'] = rho_val
                    noise_cfg['rho_correlated'] = rho_val
    
                    distance_um = float(pipeline_cfg['distance_um'])
                    Ts = calculate_dynamic_symbol_period(distance_um, test_config)
                    pipeline_cfg['symbol_period_s'] = Ts
                    pipeline_cfg['time_window_s'] = Ts
    
                    sim_cfg = test_config.setdefault('sim', {})
                    sim_cfg['time_window_s'] = Ts
    
                    det_cfg = test_config.setdefault('detection', {})
                    det_cfg.setdefault('decision_window_policy', 'fraction_of_ts')
                    det_cfg.setdefault('decision_window_fraction', 0.9)
    
                    dt = float(sim_cfg.get('dt_s', 0.01))
                    detection_window_s = _resolve_decision_window(test_config, Ts, dt)
    
                    sigma_da, sigma_sero = calculate_proper_noise_sigma(test_config, detection_window_s)
                    expected_sigma_diff = _expected_sigma_diff_for(test_config, sigma_da, sigma_sero)
                    sigma_independent = math.sqrt(max(sigma_da * sigma_da + sigma_sero * sigma_sero, 0.0))
    
                    result = run_sequence(deepcopy(test_config))
                    actual_sigma_diff = float(result.get('noise_sigma_I_diff', 0.0))
                    delta = abs(actual_sigma_diff - expected_sigma_diff)
    
                    if delta <= tolerance:
                        print(f"    OK rho={rho_val:.1f}: sigma_diff={actual_sigma_diff:.2e} (analytic={expected_sigma_diff:.2e})")
                    else:
                        print(f"    !! rho={rho_val:.1f}: sigma_diff={actual_sigma_diff:.2e} vs analytic {expected_sigma_diff:.2e} (delta={delta:.2e})")
                        failures.append(f"{scenario_label} rho={rho_val:.1f}")
    
                    if rho_val > 0.0 and sigma_independent > 0.0:
                        noise_reduction = (sigma_independent - actual_sigma_diff) / sigma_independent * 100.0
                        print(f"       Noise reduction vs independent: {noise_reduction:.1f}%")
    
    finally:
        if level_changed:
            pipeline_logger.setLevel(previous_level)
    if failures:
        print("\nCross-Channel Noise Correlation Diagnostics found mismatches:")
        for item in failures:
            print(f"   - {item}")
    else:
        print("\nCross-Channel Noise Correlation Diagnostics completed successfully!\n")

# ENHANCEMENT: Export canonical formatter for use by other modules
__all__ = ["canonical_value_key"]

if __name__ == "__main__":
    # Windows multiprocessing support
    if platform.system() == "Windows":
        mp.freeze_support()
    
    # Handle integration test option
    if len(sys.argv) > 1 and sys.argv[1] == "--test-noise-correlation":
        integration_test_noise_correlation()
    else:
        main()
