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
from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict, cast, Callable, Set, Iterable, Sequence, Mapping
import gc
import os
import stat
import platform
import time
import typing
import hashlib
import threading
import queue as pyqueue
import signal
import subprocess
import logging
from statistics import NormalDist

_SYNTHETIC_NOISE_WARNING_SHOWN = False

# NOTE: LoD debug instrumentation scaffolding (remove once diagnostics conclude)
LOD_DEBUG_ENV_KEY = "MCVD_LOD_DEBUG"
def _env_lod_debug_enabled() -> bool:
    raw = os.environ.get(LOD_DEBUG_ENV_KEY, "")
    return raw.lower() in ("1", "true", "yes", "on")

LOD_DEBUG_ENABLED: bool = _env_lod_debug_enabled()
_LOD_DEBUG_PATH: Optional[Path] = None
_LOD_DEBUG_LOCK = threading.Lock()


def _warn_synthetic_noise(reason: str) -> None:
    """Emit a one-time warning that analytic (synthetic) noise is being used."""
    global _SYNTHETIC_NOISE_WARNING_SHOWN
    if _SYNTHETIC_NOISE_WARNING_SHOWN:
        return
    _SYNTHETIC_NOISE_WARNING_SHOWN = True
    print(f"⚠️  Synthetic noise model in effect: {reason}")


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
from src.constants import get_nt_params
from src.config_utils import preprocess_config

# Progress UI
from analysis.ui_progress import ProgressManager
from analysis.log_utils import setup_tee_logging


# NOTE: LoD debug instrumentation helper – remove when diagnostics complete
def _lod_debug_log(event: Dict[str, Any]) -> None:
    global _LOD_DEBUG_PATH
    if not LOD_DEBUG_ENABLED:
        return
    try:
        # Lazily initialize per-process debug sink
        if _LOD_DEBUG_PATH is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            debug_dir = project_root / "results" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            _LOD_DEBUG_PATH = debug_dir / f"lod_debug_{timestamp}_pid{os.getpid()}.jsonl"
        payload = dict(event)
        payload.setdefault("timestamp", time.time())
        payload.setdefault("pid", os.getpid())
        payload_line = json.dumps(_json_safe(payload))
        with _LOD_DEBUG_LOCK:
            with open(_LOD_DEBUG_PATH, "a", encoding="utf-8") as fh:
                fh.write(payload_line + "\n")
            try:
                print(f"[LOD-DEBUG] {payload_line}", flush=True)
            except Exception:
                pass
    except Exception:
        pass

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

DEFAULT_NM_RANGES: Dict[str, List[int]] = {
    "MoSK": [200, 350, 650, 1100, 1750, 2250, 2500, 2900, 3300, 3600],
    "Hybrid": [500, 950, 1600, 2800, 4400, 5600, 6500, 7200, 8400, 10000],
    "CSK": [1000, 1800, 3000, 5400, 8400, 10800, 12000, 13800, 16200, 19200],
}

def _normalize_nm_values(values: Iterable[Any]) -> List[int]:
    """Normalize an iterable of Nm values into a sorted list of positive integers."""
    result: List[int] = []
    seen: Set[int] = set()
    for value in values:
        if value is None:
            continue
        try:
            nm = int(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Nm value '{value}'") from exc
        if nm <= 0:
            raise ValueError(f"Nm value must be positive, got {nm}")
        if nm not in seen:
            seen.add(nm)
            result.append(nm)
    if not result:
        raise ValueError("No Nm values supplied")
    return result


def _parse_nm_grid_spec(spec: str) -> List[int]:
    """Parse a comma/semicolon separated Nm grid specification from CLI flags."""
    cleaned = (spec or "").strip()
    if not cleaned:
        raise ValueError("Empty Nm grid specification")
    tokens = [token.strip() for token in cleaned.replace(';', ',').split(',') if token.strip()]
    return _normalize_nm_values(tokens)


def _get_cfg_nm_grid(cfg_nm: Any, mode: str) -> Optional[List[int]]:
    """Return Nm grid for the given mode from config, handling dict or legacy list."""
    if cfg_nm is None:
        return None
    canonical = _canonical_mode_name(mode)
    if isinstance(cfg_nm, dict):
        for key, values in cfg_nm.items():
            if _canonical_mode_name(str(key)) == canonical:
                try:
                    return _normalize_nm_values(values)
                except ValueError:
                    return None
        return None
    if isinstance(cfg_nm, (list, tuple)):
        try:
            return _normalize_nm_values(cfg_nm)
        except ValueError:
            return None
    return None

DEFAULT_GUARD_SAMPLES_CAP: float = 4.0e7  # per-seed limit on total time samples (sequence_length * Ts/dt)

DIAGNOSTIC_THRESHOLD_KEYS: Set[str] = {
    "q_values",
    "raw_stats",
    "zscore_stats",
    "whitened_stats",
    "aux_q",
    "channel_integrals",
    "_calibration_trace",
}


def _should_apply_threshold_key(key: Any) -> bool:
    """Return True when a threshold entry should be copied into the runtime pipeline."""
    if not isinstance(key, str):
        return True
    if key.startswith("__"):
        return False
    return key not in DIAGNOSTIC_THRESHOLD_KEYS


def _apply_thresholds_into_cfg(cfg: Dict[str, Any], thresholds: Dict[str, Any]) -> None:
    """
    Copy calibration thresholds into cfg in a routing-safe way:
    - route noise.* keys into cfg['noise']
    - route recognised threshold keys into cfg['pipeline']
    - ignore diagnostic payloads (handled by _should_apply_threshold_key)
    """
    pipe = cfg.setdefault('pipeline', {})
    noise = cfg.setdefault('noise', {})
    for key, value in thresholds.items():
        if isinstance(key, str) and key.startswith('noise.'):
            noise[key.split('.', 1)[1]] = value
        elif _should_apply_threshold_key(key):
            pipe[key] = value

    # Honor calibration metadata so runtime uses the same decision window
    detection = cfg.setdefault('detection', {})
    dw_used: Optional[float] = None

    def _safe_float(value: Any) -> Optional[float]:
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    meta = thresholds.get('__meta__')
    if isinstance(meta, dict):
        raw_dw = meta.get('decision_window_used', meta.get('decision_window_s'))
        dw_used = _safe_float(raw_dw)

    if dw_used is None:
        fallback = thresholds.get('decision_window_used')
        dw_used = _safe_float(fallback)

    if dw_used is None or not (math.isfinite(dw_used) and dw_used > 0.0):
        Ts_val_raw = pipe.get('symbol_period_s')
        Ts_val = _safe_float(Ts_val_raw)
        if Ts_val is not None and math.isfinite(Ts_val) and Ts_val > 0.0:
            try:
                dw_used = _enforce_min_window(cfg, Ts_val)
            except Exception:
                dw_used = None

    if dw_used is not None and math.isfinite(dw_used) and dw_used > 0.0:
        detection['decision_window_s'] = float(dw_used)
        detection['decision_window_policy'] = 'fixed'
        pipe['time_window_s'] = max(float(pipe.get('time_window_s', 0.0)), float(dw_used))


def _sanitize_frozen_noise_payload(
    payload: Any,
    *,
    keep_window: bool = True,
    keep_components: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Return a copy of the frozen-noise payload with sigma/ρ overrides removed.

    LoD searches rely on live signal+noise statistics. This helper strips the
    keys that would otherwise clamp the detector to the Nm=0 snapshot while
    optionally preserving deterministic windowing metadata.
    """
    if not isinstance(payload, dict):
        return None

    sanitized = deepcopy(payload)
    for key in (
        'sigma_da',
        'sigma_sero',
        'sigma_diff',
        'sigma_diff_mosk',
        'rho_for_diff',
        'rho_cc',
    ):
        sanitized.pop(key, None)

    if not keep_window:
        sanitized.pop('detection_window_s', None)
    if not keep_components:
        sanitized.pop('noise_components', None)
    else:
        components = sanitized.get('noise_components')
        if isinstance(components, dict):
            sanitized['noise_components'] = dict(components)

    return sanitized if sanitized else None


def _infer_csk_threshold_orientation(thresholds: Iterable[Any], default_qeff: float) -> bool:
    """
    Infer the monotonic orientation of CSK decision thresholds.
    Falls back to the effective charge sign when thresholds are unavailable.
    """
    seq: List[float] = []
    for val in thresholds or []:
        try:
            seq.append(float(val))
        except (TypeError, ValueError):
            continue
    if len(seq) >= 2:
        return bool(seq[-1] >= seq[0])
    return bool(default_qeff >= 0.0)


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
        print("?? Intel 13th/14th gen hybrid CPU detected! P-core optimization available.")

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
    global LOD_DEBUG_ENABLED
    try:
        if _env_lod_debug_enabled():
            LOD_DEBUG_ENABLED = True
    except Exception:
        pass
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

_CACHE_STRIP_TOPLEVEL_KEYS: Set[str] = {
    'disable_progress',
    'progress',
    'verbose',
    '_resume_active',
    '_lod_eval_trace',
    'lod_trace',
    'actual_progress',
    'skipped_reason',
    'full_seeds',
    '_warm_lod_guess',
    '_warm_bracket_min',
    '_warm_bracket_max',
    '_sanitized_distance_freeze_cache',
    '_sanitized_freeze_cache',
}

_CACHE_STRIP_PIPELINE_KEYS: Set[str] = {
    'random_seed',
    'show_progress',
    'progress_label',
    'lod_trace',
    '_warm_lod_guess',
    '_warm_bracket_min',
    '_warm_bracket_max',
    '_lod_eval_trace',
    '_sanitized_distance_freeze_cache',
    '_sanitized_freeze_cache',
}

_CACHE_STRIP_DETECTION_KEYS: Set[str] = {
    'decision_window_policy_runtime',
    'decision_window_runtime_s',
}


def _stable_cache_token(value: Any) -> str:
    """Return a deterministic string token for cache hashing."""
    if isinstance(value, dict):
        items = (f"{k}:{_stable_cache_token(value[k])}" for k in sorted(value))
        return "{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_cache_token(v) for v in value) + "]"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return f"{value:.12g}"
    if isinstance(value, (int, bool)):
        return str(int(value)) if isinstance(value, bool) else str(value)
    if value is None:
        return "None"
    return str(value)


def _prepare_cfg_for_threshold_cache(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Strip run-only flags so logically equivalent configs share cache keys."""
    for key in list(_CACHE_STRIP_TOPLEVEL_KEYS):
        cfg.pop(key, None)

    pipeline = cfg.get('pipeline')
    if isinstance(pipeline, dict):
        for key in list(_CACHE_STRIP_PIPELINE_KEYS):
            pipeline.pop(key, None)

    detection = cfg.get('detection')
    if isinstance(detection, dict):
        for key in list(_CACHE_STRIP_DETECTION_KEYS):
            detection.pop(key, None)

    return cfg


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
    token = "::".join(_stable_cache_token(p) for p in key_params)
    return hashlib.sha1(token.encode('utf-8')).hexdigest()

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
            if 'csk_thresholds_increasing' in cfg['pipeline']:
                del cfg['pipeline']['csk_thresholds_increasing']
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
    cfg = deepcopy(cfg)
    pipeline_clean = cfg.setdefault('pipeline', {})
    pipeline_clean.pop('_frozen_noise', None)
    cfg.pop('_prefer_distance_freeze', None)

    mode = cfg['pipeline']['modulation']
    detector_mode = str(cfg['pipeline'].get('detector_mode', 'zscore')).lower()
    if detector_mode not in ('zscore', 'raw', 'whitened'):
        detector_mode = 'zscore'
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

    collect_trace = bool(cfg.get('analysis', {}).get('store_calibration_stats', False))
    calibration_trace: Dict[str, Dict[str, Any]] = {} if collect_trace else {}

    def _accumulate_trace(mode_key: str, symbol_key: Union[int, str], result: Optional[Dict[str, Any]]) -> None:
        if not collect_trace or not result:
            return
        symbol_map = calibration_trace.setdefault(mode_key, {})
        entry = symbol_map.setdefault(str(symbol_key), {})
        detector_mode_local = str(result.get('detector_mode', cfg['pipeline'].get('detector_mode', 'zscore'))).lower()
        entry.setdefault('detector_mode', detector_mode_local)

        def _extend_entry(key: str, values: Iterable[Any]) -> None:
            existing = entry.setdefault(key, [])
            for val in values:
                try:
                    fv = float(val)
                except Exception:
                    continue
                if np.isfinite(fv):
                    existing.append(fv)

        for key in ('q_values', 'raw_stats', 'whitened_stats', 'zscore_stats'):
            if key in result:
                _extend_entry(key, result[key])

        if 'aux_q' in result:
            aux_list = entry.setdefault('aux_q', [])
            aux_list.extend([[float(a), float(b)] for a, b in result['aux_q']])

        if 'channel_integrals' in result:
            ci_dest = entry.setdefault('channel_integrals', {})
            channel_integrals = result['channel_integrals']
            for ci_key, pairs in channel_integrals.items():
                dest = ci_dest.setdefault(ci_key, [])
                dest.extend([[float(a), float(b)] for a, b in pairs])

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

            cached_mode = str(meta.get('detector_mode', 'zscore')).lower()
            if cached_mode not in ('zscore', 'raw', 'whitened'):
                cached_mode = 'zscore'
            if cached_mode != detector_mode:
                if verbose: print("  ↪︎ cache invalid: detector mode mismatch")
                raise RuntimeError("cache: detector_mode mismatch")

            # NT pair + q_eff signs
            if meta.get('nt_pair_fp') != _fingerprint_nt_pair(cfg):
                if verbose: print("  ↪︎ cache invalid: NT pair fingerprint mismatch")
                raise RuntimeError("cache: nt_pair_fp mismatch")
            s_da, s_se = _qeff_signs(cfg)
            if tuple(meta.get('q_eff_signs', (s_da, s_se))) != (s_da, s_se):
                if verbose: print("  ↪︎ cache invalid: q_eff sign mismatch")
                raise RuntimeError("cache: q_eff_signs mismatch")

            # Mode-specific checks
            tgt: str = ""
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
            result = {k: v for k, v in cached.items() if k not in ("_metadata", "__meta__")}
            if meta:
                result['__meta__'] = meta
            if mode.startswith("CSK") and tgt:
                key = f"csk_thresholds_{tgt.lower()}"
                thresh_vals = result.get(key)
                if isinstance(thresh_vals, list) and 'csk_thresholds_increasing' not in result:
                    nt_cfg = cfg.get('neurotransmitters', {})
                    qeff = float((nt_cfg.get(tgt, {}) or {}).get('q_eff_e', 0.0))
                    result['csk_thresholds_increasing'] = _infer_csk_threshold_orientation(thresh_vals, qeff)
            return result
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

    thresholds: Dict[str, Any] = {}

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
                _accumulate_trace('MoSK', 0, r_da)
            if r_se and 'q_values' in r_se:
                mosk_stats['sero'].extend(_clean(r_se['q_values']))
                _accumulate_trace('MoSK', 1, r_se)

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

        # Persist MoSK detector metadata so decoding matches calibration
        if detector_mode == 'raw':
            mosk_statistic_label = 'sign_aware_diff_raw'
            mosk_stat_units = 'raw_charge'
            mosk_normalization = 'none'
        elif detector_mode == 'whitened':
            mosk_statistic_label = 'sign_aware_diff_whitened'
            mosk_stat_units = 'whitened_sigma_diff'
            mosk_normalization = 'sigma_diff'
        else:
            mosk_statistic_label = 'sign_aware_diff'
            mosk_stat_units = 'normalized_sigma_diff'
            mosk_normalization = 'sigma_diff'

        thresholds['mosk_statistic'] = mosk_statistic_label
        thresholds['mosk_stat_units'] = mosk_stat_units
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
            'normalization': mosk_normalization
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
                    _accumulate_trace('CSK', level, r)

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
        nt_cfg = cfg.get('neurotransmitters', {})
        qeff_target = float((nt_cfg.get(target_channel, {}) or {}).get('q_eff_e', 0.0))
        thresholds['csk_thresholds_increasing'] = _infer_csk_threshold_orientation(final_tau or [], qeff_target)

        # provenance for resume
        cfg['_threshold_cache_meta'] = {
            'M': M,
            'target_channel': target_channel,
            'combiner': str(cfg['pipeline'].get('csk_combiner', 'zscore'))
        }

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
                    _accumulate_trace('Hybrid', sym, r)

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

    thresholds['detector_mode'] = detector_mode

    # Record calibration metadata for downstream consumers
    try:
        qeff_signs = _qeff_signs(cfg)
    except Exception:
        qeff_signs = (0, 0)
    meta_payload: Dict[str, Any] = {
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
        "nt_pair_label": f"{cfg['neurotransmitters']['DA'].get('name','DA')}–{cfg['neurotransmitters']['SERO'].get('name','SERO')}"
                          if 'neurotransmitters' in cfg else "",
        "nt_pair_fp": _fingerprint_nt_pair(cfg),
        "q_eff_signs": qeff_signs,
        "version": "2.0",
        "detector_mode": detector_mode,
        "timestamp": time.time()
    }
    thresholds['__meta__'] = meta_payload

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

            payload["__meta__"] = meta_payload

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

    if collect_trace and calibration_trace:
        thresholds['_calibration_trace'] = calibration_trace

    return thresholds

def calibrate_thresholds_cached(cfg: Dict[str, Any], seeds: List[int], recalibrate: bool = False) -> Dict[str, Union[float, List[float], str]]:
    """
    Memory + disk cached calibration. Persist JSON so multiple processes/runs reuse it.
    """
    cfg_clean = deepcopy(cfg)
    pipeline_clean = cfg_clean.setdefault('pipeline', {})
    pipeline_clean.pop('_frozen_noise', None)
    cfg_clean.pop('_prefer_distance_freeze', None)
    cfg_clean = _prepare_cfg_for_threshold_cache(cfg_clean)
    cfg = cfg_clean

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
        cached_result = calibration_cache[cache_key]
        if '__meta__' not in cached_result:
            threshold_file = _thresholds_filename(cfg)
            if threshold_file.exists():
                try:
                    with open(threshold_file, 'r', encoding='utf-8') as fh:
                        cached_payload = json.load(fh)
                    meta_disk = cached_payload.get('__meta__')
                    if meta_disk:
                        cached_copy = dict(cached_result)
                        cached_copy['__meta__'] = meta_disk
                        calibration_cache[cache_key] = cached_copy
                        cached_result = cached_copy
                except Exception:
                    pass
        if cfg['pipeline']['modulation'].startswith('CSK') and \
           'csk_thresholds_increasing' not in cached_result:
            cached_copy = dict(cached_result)
            target_channel = str(cfg['pipeline'].get('csk_target_channel', 'DA')).upper()
            key = f'csk_thresholds_{target_channel.lower()}'
            nt_cfg = cfg.get('neurotransmitters', {})
            qeff_target = float((nt_cfg.get(target_channel, {}) or {}).get('q_eff_e', 0.0))
            thresh_vals = cached_copy.get(key)
            if not isinstance(thresh_vals, list):
                thresh_vals = []
            cached_copy['csk_thresholds_increasing'] = _infer_csk_threshold_orientation(
                thresh_vals,
                qeff_target,
            )
            calibration_cache[cache_key] = cached_copy
            cached_result = cached_copy
        return cached_result
    
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

    if getattr(args, 'store_calibration_stats', False):
        cfg.setdefault('analysis', {})['store_calibration_stats'] = True
    
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

    if getattr(args, 'detector_mode', None) is not None:
        cfg.setdefault('pipeline', {})['detector_mode'] = str(args.detector_mode).lower()

    if getattr(args, 'freeze_calibration', False):
        cfg.setdefault('analysis', {})['freeze_calibration'] = True

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
    Also emits a JSON sidecar with summary metadata for reproducibility.
    """
    unique_tag = f".tmp.{os.getpid()}_{threading.get_ident()}_{int(time.time()*1e9)}"
    tmp = csv_path.with_suffix(csv_path.suffix + unique_tag)
    df.to_csv(tmp, index=False)

    def _replace() -> None:
        try:
            os.replace(tmp, csv_path)
            metadata = _extract_csv_metadata(df)
            _write_csv_metadata(csv_path, metadata)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    _with_file_lock(csv_path, _replace)


_STAGE_CSV_BASENAMES = {
    "ser": "ser_vs_nm_{mode}",
    "lod": "lod_vs_distance_{mode}",
    "dist": "ser_snr_vs_distance_{mode}",
    "isi": "isi_tradeoff_{mode}",
}


def _ablation_token(use_ctrl: bool) -> str:
    return "ctrl" if use_ctrl else "noctrl"


def _stage_csv_paths(stage: str,
                     data_dir: Path,
                     mode: str,
                     suffix: str,
                     use_ctrl: bool) -> Tuple[Path, Path, Path]:
    template = _STAGE_CSV_BASENAMES[stage]
    base = template.format(mode=mode.lower())
    canonical = data_dir / f"{base}{suffix}.csv"
    token = _ablation_token(use_ctrl)
    branch = data_dir / f"{base}__{token}{suffix}.csv"
    other_token = "noctrl" if token == "ctrl" else "ctrl"
    other_branch = data_dir / f"{base}__{other_token}{suffix}.csv"
    return canonical, branch, other_branch


def _dedupe_ser_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    nm_key = None
    for cand in ("pipeline_Nm_per_symbol", "pipeline.Nm_per_symbol"):
        if cand in df.columns:
            nm_key = cand
            break
    if nm_key is None:
        return df
    subset = [nm_key]
    if 'use_ctrl' in df.columns:
        subset.append('use_ctrl')
    return df.drop_duplicates(subset=subset, keep='last').sort_values(subset)


def _dedupe_lod_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'distance_um' not in df.columns:
        return df
    df = df.copy()
    keys = ['distance_um']
    if 'use_ctrl' in df.columns:
        keys.append('use_ctrl')
    if 'lod_nm' in df.columns:
        df['__is_valid__'] = pd.to_numeric(df['lod_nm'], errors='coerce').gt(0) & np.isfinite(pd.to_numeric(df['lod_nm'], errors='coerce'))
        ordered = df.sort_index()
        latest = ordered.groupby(keys, as_index=False, group_keys=False).tail(1)
        valid = ordered[ordered['__is_valid__']].groupby(keys, as_index=False, group_keys=False).tail(1)
        result = pd.concat([latest, valid]).drop_duplicates(subset=keys, keep='last').drop(columns=['__is_valid__'])
    else:
        result = df.drop_duplicates(subset=keys, keep='last')
    return result.sort_values(keys)


def _dedupe_distance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'distance_um' not in df.columns:
        return df
    keys = ['distance_um']
    if 'use_ctrl' in df.columns:
        keys.append('use_ctrl')
    return df.drop_duplicates(subset=keys, keep='last').sort_values(keys)


def _dedupe_isi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys: List[str] = []
    if 'distance_um' in df.columns:
        keys.append('distance_um')
    if 'pipeline.guard_factor' in df.columns:
        keys.append('pipeline.guard_factor')
    elif 'guard_factor' in df.columns:
        keys.append('guard_factor')
    if 'use_ctrl' in df.columns:
        keys.append('use_ctrl')
    if not keys:
        return df
    return df.drop_duplicates(subset=keys, keep='last').sort_values(keys)


def _row_use_ctrl(row: Dict[str, Any], default: Optional[bool] = None) -> Optional[bool]:
    if 'use_ctrl' in row:
        val = row['use_ctrl']
        if pd.isna(val):
            return default
        if isinstance(val, bool):
            return val
        txt = str(val).strip().lower()
        if txt in {"true", "1", "t", "yes"}:
            return True
        if txt in {"false", "0", "f", "no"}:
            return False
        try:
            return bool(int(float(txt)))
        except (TypeError, ValueError):
            return default
    return default


def _lod_row_is_valid(row: Dict[str, Any]) -> bool:
    value = row.get('lod_nm')
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(val) and val > 0.0


def _extract_numeric(row: Dict[str, Any], *candidates: str) -> Optional[float]:
    for key in candidates:
        if key in row:
            val = row[key]
            if pd.isna(val):
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


class _StageBranchAccumulator:
    """Streaming accumulator for per-ablation branch CSV updates."""

    def __init__(self,
                 stage: str,
                 default_use_ctrl: Optional[bool] = None,
                 filter_use_ctrl: Optional[bool] = None) -> None:
        self.stage = stage
        self.default_use_ctrl = default_use_ctrl
        self.filter_use_ctrl = filter_use_ctrl
        self._records: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._seq = 0

    def add_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        for row in df.to_dict(orient="records"):
            row_dict = {str(k): v for k, v in row.items()}
            self.add_row(row_dict)

    def add_row(self, row: Dict[str, Any]) -> None:
        local = dict(row)
        use_ctrl = _row_use_ctrl(local, self.default_use_ctrl)
        if self.filter_use_ctrl is not None and use_ctrl is not None and use_ctrl != self.filter_use_ctrl:
            return
        if use_ctrl is None and self.default_use_ctrl is not None:
            use_ctrl = self.default_use_ctrl
        if use_ctrl is not None:
            local['use_ctrl'] = bool(use_ctrl)
        key = self._make_key(local, use_ctrl)
        if key is None:
            self._seq += 1
            return
        status = self._status(local)
        existing = self._records.get(key)
        if existing is None or self._should_replace(existing['__status__'], status):
            local['__status__'] = status
            self._records[key] = local
        self._seq += 1

    def finalize(self, dedupe_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        rows = []
        for record in self._records.values():
            rec = dict(record)
            rec.pop('__status__', None)
            rows.append(rec)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return dedupe_fn(df)

    def _make_key(self, row: Dict[str, Any], use_ctrl: Optional[bool]) -> Optional[Tuple[Any, ...]]:
        if self.stage == "ser":
            nm_val = _extract_numeric(row, "pipeline_Nm_per_symbol", "pipeline.Nm_per_symbol")
            if nm_val is None:
                return None
            return (canonical_value_key(nm_val), use_ctrl)
        if self.stage == "lod":
            dist_val = _extract_numeric(row, "distance_um")
            if dist_val is None:
                return None
            return (int(round(dist_val)), use_ctrl)
        if self.stage == "isi":
            gf_val = _extract_numeric(row, "pipeline.guard_factor", "guard_factor")
            if gf_val is None:
                return None
            dist_val = _extract_numeric(row, "distance_um")
            dist_key = canonical_value_key(dist_val) if dist_val is not None else None
            return (dist_key, canonical_value_key(gf_val), use_ctrl)
        return None

    def _status(self, row: Dict[str, Any]) -> Tuple[int, int]:
        if self.stage == "lod":
            priority = 1 if _lod_row_is_valid(row) else 0
        else:
            priority = 0
        return (priority, self._seq)

    @staticmethod
    def _should_replace(existing: Tuple[int, int], new: Tuple[int, int]) -> bool:
        if new[0] > existing[0]:
            return True
        if new[0] == existing[0] and new[1] >= existing[1]:
            return True
        return False


def _load_stage_csv(stage: str,
                    csv_path: Path,
                    dedupe_fn: Callable[[pd.DataFrame], pd.DataFrame],
                    use_ctrl: Optional[bool] = None,
                    filter_use_ctrl: Optional[bool] = None) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    acc = _StageBranchAccumulator(stage, default_use_ctrl=use_ctrl, filter_use_ctrl=filter_use_ctrl)
    try:
        for chunk in pd.read_csv(csv_path, chunksize=50_000):
            acc.add_dataframe(chunk)
    except Exception:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
        acc.add_dataframe(df)
    return acc.finalize(dedupe_fn)


def _update_branch_csv(stage: str,
                       branch_csv: Path,
                       new_frames: Iterable[pd.DataFrame],
                       use_ctrl: bool,
                       dedupe_fn: Callable[[pd.DataFrame], pd.DataFrame],
                       existing_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    flush_staged_rows(branch_csv)
    acc = _StageBranchAccumulator(stage, default_use_ctrl=use_ctrl)
    if existing_df is not None and not existing_df.empty:
        acc.add_dataframe(existing_df)
    elif branch_csv.exists():
        existing_df = _load_stage_csv(stage, branch_csv, dedupe_fn, use_ctrl=use_ctrl)
        if not existing_df.empty:
            acc.add_dataframe(existing_df)
    for frame in new_frames:
        if frame is not None and not frame.empty:
            acc.add_dataframe(frame)
    combined = acc.finalize(dedupe_fn)
    if not combined.empty:
        _atomic_write_csv(branch_csv, combined)
    return combined


def _discover_stage_suffixes(stage: str, mode: str, data_dir: Path) -> Set[str]:
    template = _STAGE_CSV_BASENAMES.get(stage)
    if template is None:
        return {""}
    stem = template.format(mode=mode.lower())
    suffixes: Set[str] = set()
    for path in data_dir.glob(f"{stem}*.csv"):
        if not path.is_file():
            continue
        stem_full = path.stem
        base_stem = stem_full.split("__", 1)[0]
        if not base_stem.startswith(stem):
            continue
        suffix = base_stem[len(stem):]
        suffixes.add(suffix)
    return suffixes or {""}


def merge_ablation_csvs_for_modes(modes: Sequence[str],
                                  stages: Optional[Sequence[str]] = None,
                                  data_dir: Optional[Path] = None) -> None:
    target_dir = data_dir or (project_root / "results" / "data")
    if not target_dir.exists():
        print(f"⚠️  Results data directory missing: {target_dir}")
        return
    dedupe_map: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "ser": _dedupe_ser_dataframe,
        "lod": _dedupe_lod_dataframe,
        "dist": _dedupe_distance_dataframe,
        "isi": _dedupe_isi_dataframe,
    }
    selected_stages = list(stages) if stages else list(dedupe_map.keys())
    for stage in selected_stages:
        if stage not in dedupe_map:
            print(f"⚠️  Unknown stage '{stage}' requested; skipping.")
            continue
        dedupe_fn = dedupe_map[stage]
        for mode in modes:
            if not mode:
                continue
            suffixes = _discover_stage_suffixes(stage, mode, target_dir)
            for suffix in sorted(suffixes):
                canonical, branch_ctrl, branch_other = _stage_csv_paths(stage, target_dir, mode, suffix, True)
                _, branch_noctrl, _ = _stage_csv_paths(stage, target_dir, mode, suffix, False)
                _ensure_ablation_branch(stage, canonical, branch_ctrl, True, True, dedupe_fn)
                _ensure_ablation_branch(stage, canonical, branch_noctrl, False, True, dedupe_fn)
                _merge_branch_csv(canonical, [branch_ctrl, branch_noctrl], dedupe_fn)
                merged = _load_stage_csv(stage, canonical, dedupe_fn) if canonical.exists() else pd.DataFrame()
                suffix_label = suffix if suffix else ""
                if not merged.empty:
                    print(f"   ↪︎ Merged {stage.upper()} ({mode}{suffix_label}) — {len(merged)} rows")
                else:
                    print(f"   ↪︎ No rows to merge for {stage.upper()} ({mode}{suffix_label})")


def _ensure_ablation_branch(stage: str,
                            canonical_csv: Path,
                            branch_csv: Path,
                            use_ctrl: bool,
                            resume: bool,
                            dedupe_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """
    Guarantee the per-ablation branch file exists when resuming, seeding it from the
    canonical CSV if necessary. Returns the current branch dataframe (deduped).
    """
    if branch_csv.exists():
        branch_df = _load_stage_csv(stage, branch_csv, dedupe_fn, use_ctrl=use_ctrl)
        if not branch_df.empty:
            return branch_df
    if resume and canonical_csv.exists():
        branch_df = _load_stage_csv(stage, canonical_csv, dedupe_fn,
                                    use_ctrl=use_ctrl, filter_use_ctrl=use_ctrl)
        if not branch_df.empty:
            _atomic_write_csv(branch_csv, branch_df)
            return branch_df
    return pd.DataFrame()


def _merge_branch_csv(canonical_csv: Path,
                      branch_paths: List[Path],
                      dedupe_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    for path in branch_paths + [canonical_csv]:
        flush_staged_rows(path)
    frames: List[pd.DataFrame] = []
    for path in branch_paths + [canonical_csv]:
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except Exception as exc:
                print(f"⚠️  Warning: could not read {path.name} during merge ({exc})")
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined = dedupe_fn(combined)
    _atomic_write_csv(canonical_csv, combined)


def _atomic_write_json(json_path: Path, payload: Dict[str, Any]) -> None:
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(1, 4):
        try:
            os.replace(tmp, json_path)
            break
        except PermissionError:
            if attempt == 3:
                try:
                    tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                raise
            try:
                if json_path.exists():
                    os.chmod(json_path, stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass
            time.sleep(0.1 * attempt)


def _noise_fingerprint(mode: str,
                       suffix: str,
                       nm_values: Sequence[Union[int, float]],
                       lod_distance_grid: Sequence[Union[int, float]],
                       noise_seeds: Sequence[int],
                       noise_seq_len: int,
                       cfg: Dict[str, Any]) -> str:
    pipeline_cfg = cfg.get('pipeline', {})
    payload = {
        "version": "v1",
        "mode": mode,
        "suffix": suffix,
        "modulation": pipeline_cfg.get('modulation', mode),
        "detector_mode": pipeline_cfg.get('detector_mode', 'zscore'),
        "enable_isi": bool(pipeline_cfg.get('enable_isi', True)),
        "nm_values": [float(v) for v in nm_values],
        "distance_grid": [float(v) for v in lod_distance_grid],
        "noise_seq_len": int(noise_seq_len),
        "noise_seed_count": len(noise_seeds),
        "noise_seed_hash": hashlib.sha256(
            ",".join(str(int(s)) for s in noise_seeds).encode('utf-8')
        ).hexdigest(),
        "dt_s": float(cfg.get("sim", {}).get("dt_s", 0.01)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()


def _load_noise_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_csv_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a reproducibility-friendly metadata summary for a results DataFrame.
    """
    meta: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    numeric_candidates = [
        "symbol_period_s",
        "decision_window_s",
        "guard_factor",
        "Nm_per_symbol",
        "distance_um",
        "noise_sigma_single",
        "noise_sigma_I_diff",
        "noise_sigma_diff_charge",
        "I_dc_used_A",
        "gm_S",
        "C_tot_F",
        "rho_pre_ctrl",
        "rho_cc",
    ]

    medians: Dict[str, float] = {}
    for col in numeric_candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if not series.empty:
                medians[col] = float(series.median())
    if medians:
        meta["medians"] = medians

    categorical_candidates = [
        "mode",
        "detector_mode",
        "burst_shape",
        "use_ctrl",
    ]
    categories: Dict[str, List[str]] = {}
    for col in categorical_candidates:
        if col in df.columns:
            unique_vals = df[col].dropna().unique().tolist()
            if unique_vals:
                categories[col] = [str(v) for v in unique_vals]
    if categories:
        meta["categories"] = categories

    return meta


def _write_csv_metadata(csv_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Persist metadata JSON next to the CSV (same basename + `.meta.json`).
    """
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    metadata = dict(metadata)
    metadata["generated_at"] = time.time()
    metadata.setdefault("source_csv", str(csv_path))
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_meta, meta_path)

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

def _staging_dir_for(csv_path: Path) -> Path:
    staging = csv_path.parent / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging

def _shard_path(csv_path: Path) -> Path:
    pid = os.getpid()
    tid = threading.get_ident()
    staging = _staging_dir_for(csv_path)
    return staging / f"{csv_path.stem}__{pid}_{tid}.csv"

def flush_staged_rows(csv_path: Path) -> None:
    staging = csv_path.parent / ".staging"
    if not staging.exists():
        return
    pattern = f"{csv_path.stem}__*.csv"
    for shard in staging.glob(pattern):
        try:
            shard_df = pd.read_csv(shard)
        except Exception:
            shard.unlink(missing_ok=True)
            continue
        if shard_df.empty:
            shard.unlink(missing_ok=True)
            continue

        def _append_shard() -> None:
            if csv_path.exists():
                try:
                    existing = pd.read_csv(csv_path)
                    combined = pd.concat([existing, shard_df], ignore_index=True)
                except Exception:
                    combined = shard_df
            else:
                combined = shard_df
            _atomic_write_csv(csv_path, combined)

        _with_file_lock(csv_path, _append_shard)
        shard.unlink(missing_ok=True)

def append_row_atomic(csv_path: Path, row: Dict[str, Any], columns: Optional[List[str]] = None) -> None:
    """
    Stage rows to per-worker shard files; use ``flush_staged_rows`` to merge them later.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if row.get('mode', '').startswith('CSK') and row.get('csk_store_combiner_meta', True):
        if 'combiner' not in row:
            row['combiner'] = row.get('csk_selected_combiner', row.get('csk_combiner', 'zscore'))
        if 'sigma_da' not in row:
            row['sigma_da'] = row.get('noise_sigma_da', np.nan)
        if 'sigma_sero' not in row:
            row['sigma_sero'] = row.get('noise_sigma_sero', np.nan)
        if 'rho_cc' not in row:
            row['rho_cc'] = row.get('rho_cc', 0.0)
        if 'leakage_frac' not in row:
            combiner = row.get('combiner', 'zscore')
            row['leakage_frac'] = row.get('csk_leakage_frac', 0.0) if combiner == 'leakage' else np.nan
        row.setdefault('csk_levels', row.get('M', 4))
        row.setdefault('csk_target_channel', 'DA')

    new_row = pd.DataFrame([row])
    if columns is not None:
        new_row = new_row.reindex(columns=columns)

    shard = _shard_path(csv_path)
    header = not shard.exists()
    new_row.to_csv(shard, mode='a', header=header, index=False)

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

def canonical_value_key(v):
    """Convert a numeric parameter value to a standard string key for CSV/cache lookups."""
    if isinstance(v, (int, float)):
        return f"{float(v):.10g}"
    return str(v)


def _value_key(v):
    """
    Backward-compatibility shim forwarding to the canonical value key helper.
    """
    return canonical_value_key(v)


def _cached_sanitized_freeze_payload(cfg_root: Dict[str, Any],
                                     param_name: str,
                                     param_value: Union[float, int],
                                     payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a cached sanitized noise payload for the given parameter value."""
    if not isinstance(payload, dict) or not payload:
        return None
    cache_root = cfg_root.setdefault('_sanitized_freeze_cache', {})
    param_cache = cache_root.setdefault(param_name, {})
    key = _value_key(param_value)
    if key in param_cache:
        cached = param_cache[key]
        return deepcopy(cached) if cached is not None else None
    sanitized = _sanitize_frozen_noise_payload(payload)
    param_cache[key] = deepcopy(sanitized) if sanitized else None
    return deepcopy(sanitized) if sanitized else None


def _get_distance_freeze_payload(cfg_root: Dict[str, Any],
                                 distance_um: float) -> Optional[Dict[str, Any]]:
    """Fetch a sanitized distance-based freeze payload with caching."""
    noise_distance_map = cfg_root.get('_noise_freeze_distance_map', {})
    if not isinstance(noise_distance_map, dict):
        return None
    payload = noise_distance_map.get(_value_key(float(distance_um)))
    return _cached_sanitized_freeze_payload(cfg_root, "pipeline.distance_um", float(distance_um), payload)


def _sanitize_thresholds_for_resume(th: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in th.items():
        if key in ('_calibration_trace', '__meta__'):
            continue
        clean[key] = value
    return clean


def _float_close(a: float, b: float, rel: float = 1e-8, abs_tol: float = 1e-12) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b), 1.0))


def _derive_hds_nm_values(lod_entry: Optional[Dict[str, Any]],
                          lod_state: Optional[Dict[str, Any]],
                          default_nm_values: Sequence[Union[int, float]],
                          nm_min_default: int,
                          nm_max_default: int) -> List[int]:
    candidates: Set[int] = {max(50, int(float(v))) for v in default_nm_values if float(v) > 0}
    nm_min = nm_min_default
    nm_max = nm_max_default

    if lod_state:
        try:
            nm_min_state = int(float(lod_state.get("nm_min", nm_min_default)))
            nm_min = max(nm_min, nm_min_state)
        except Exception:
            pass
        try:
            nm_max_state = int(float(lod_state.get("nm_max", nm_max_default)))
            nm_max = min(nm_max, nm_max_state)
        except Exception:
            pass
        tested = lod_state.get("tested", {})
        if isinstance(tested, dict):
            for key in tested.keys():
                try:
                    candidates.add(max(50, int(float(key))))
                except Exception:
                    continue

    lod_nm_val = None
    if lod_entry and math.isfinite(float(lod_entry.get("lod_nm", float("nan")))):
        lod_nm_val = max(50, int(float(lod_entry["lod_nm"])))
        rel_steps = [0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
        for ratio in rel_steps:
            nm = int(max(50, round(lod_nm_val * ratio)))
            candidates.add(nm)

    filtered = [nm for nm in candidates if nm_min <= nm <= nm_max]
    if not filtered:
        filtered = [nm for nm in candidates if nm_min_default <= nm <= nm_max_default]
    filtered_sorted = sorted(set(filtered))

    if len(filtered_sorted) > 18:
        step = max(1, len(filtered_sorted) // 18)
        filtered_sorted = filtered_sorted[::step]

    return filtered_sorted


def _replay_hds_from_lod_cache(dist_um: float,
                               cfg_base: Dict[str, Any],
                               nm_values: Sequence[int],
                               seeds: Sequence[int],
                               cache_tag: Optional[str]) -> Optional[pd.DataFrame]:
    if not nm_values or not seeds:
        return None

    mode = cfg_base.get('pipeline', {}).get('modulation', '')
    if mode != 'Hybrid':
        return None

    use_ctrl = bool(cfg_base.get('pipeline', {}).get('use_control_channel', True))
    Ts_expected = float(cfg_base.get('pipeline', {}).get('symbol_period_s', float('nan')))
    window_expected = float(cfg_base.get('detection', {}).get('decision_window_s', float('nan')))
    baseline_expected = str(cfg_base.get('pipeline', {}).get('calibration_baseline_id', '') or '')
    seq_len_expected = int(cfg_base.get('pipeline', {}).get('sequence_length', 0))

    cache_candidates: List[Optional[str]] = []
    if cache_tag:
        base_tag = cache_tag[:-4] if cache_tag.endswith('_hds') else cache_tag
        cache_candidates.append(base_tag)
        if base_tag != cache_tag:
            cache_candidates.append(cache_tag)
    cache_candidates.append(f"d{int(dist_um)}um")
    if None not in cache_candidates:
        cache_candidates.append(None)

    rows: List[Dict[str, Any]] = []

    for nm in nm_values:
        seed_payloads: Optional[List[Dict[str, Any]]] = None
        for tag in cache_candidates:
            candidate_payloads: List[Dict[str, Any]] = []
            for seed in seeds:
                payload = read_seed_cache(mode, "lod_search", nm, seed, use_ctrl, cache_tag=tag)
                if payload is None:
                    candidate_payloads = []
                    break

                Ts_cache = payload.get('symbol_period_s')
                if math.isfinite(Ts_expected) and Ts_cache is not None and math.isfinite(float(Ts_cache)):
                    if not _float_close(float(Ts_cache), Ts_expected):
                        candidate_payloads = []
                        break

                window_cache = payload.get('decision_window_s')
                if math.isfinite(window_expected) and window_cache is not None and math.isfinite(float(window_cache)):
                    if not _float_close(float(window_cache), window_expected):
                        candidate_payloads = []
                        break

                baseline_cache = payload.get('calibration_baseline_id')
                if baseline_expected:
                    if str(baseline_cache or '') != baseline_expected:
                        candidate_payloads = []
                        break

                candidate_payloads.append(payload)

            if candidate_payloads:
                seed_payloads = candidate_payloads
                break

        if seed_payloads is None:
            return None

        seq_lengths = [int(p.get('sequence_length', 0)) for p in seed_payloads]
        total_symbols = sum(seq_lengths)
        expected_seq_len = seq_len_expected or (seq_lengths[0] if seq_lengths else 0)
        min_symbols_required = max(expected_seq_len * len(seeds), len(seeds) * 100)
        if total_symbols < max(min_symbols_required, expected_seq_len):
            return None

        total_errors = sum(int(p.get('errors', 0)) for p in seed_payloads)
        ser = total_errors / total_symbols if total_symbols > 0 else float('nan')

        mosk_errors = sum(int(p.get('subsymbol_errors', {}).get('mosk', 0)) for p in seed_payloads)
        csk_errors = sum(int(p.get('subsymbol_errors', {}).get('csk', 0)) for p in seed_payloads)
        mosk_ser = mosk_errors / total_symbols if total_symbols > 0 else float('nan')
        csk_ser = csk_errors / total_symbols if total_symbols > 0 else float('nan')

        mosk_correct_total = sum(int(p.get('n_mosk_correct', 0)) for p in seed_payloads)
        if mosk_correct_total <= 0 and total_symbols > 0:
            mosk_correct_total = max(total_symbols - mosk_errors, 0)

        conditional_csk_ser = csk_errors / mosk_correct_total if mosk_correct_total > 0 else 0.0
        mosk_exposure_frac = mosk_correct_total / total_symbols if total_symbols > 0 else 0.0

        mosk_error_pct = (mosk_errors / total_symbols * 100.0) if total_symbols > 0 else float('nan')
        csk_error_pct = (csk_errors / total_symbols * 100.0) if total_symbols > 0 else float('nan')
        hybrid_error_pct = ((mosk_errors + csk_errors) / total_symbols * 100.0) if total_symbols > 0 else float('nan')

        bits_per_symbol_csk_branch = float(math.log2(max(cfg_base.get('pipeline', {}).get('csk_levels', 4), 2)))
        if total_symbols > 0:
            bits_mosk_realized = (total_symbols - mosk_errors) / total_symbols
            bits_csk_realized = ((mosk_correct_total - csk_errors) / total_symbols) * bits_per_symbol_csk_branch
        else:
            bits_mosk_realized = float('nan')
            bits_csk_realized = float('nan')
        bits_total = bits_mosk_realized + bits_csk_realized if (math.isfinite(bits_mosk_realized) and math.isfinite(bits_csk_realized)) else float('nan')

        ci_low, ci_high = _wilson_interval(total_errors, total_symbols, z=1.96) if total_symbols > 0 else (float('nan'), float('nan'))

        detector_mode = str(next((p.get('detector_mode') for p in seed_payloads if p.get('detector_mode')), cfg_base.get('pipeline', {}).get('detector_mode', 'zscore'))).lower()
        baseline_ids = {str(p.get('calibration_baseline_id', '')).strip() for p in seed_payloads if p.get('calibration_baseline_id')}
        if baseline_expected:
            baseline_ids.add(baseline_expected)

        first_payload = seed_payloads[0]
        ctrl_auto = bool(first_payload.get('ctrl_auto_applied', cfg_base.get('_ctrl_auto', False)))
        rho_cc_measured = first_payload.get('rho_cc_measured')
        burst_shape = cfg_base.get('pipeline', {}).get('burst_shape', '')
        t_release_ms = first_payload.get('T_release_ms', float('nan'))

        def _payload_float(payload: Dict[str, Any], key: str, default: float = float('nan')) -> float:
            try:
                val = payload.get(key, default)
                return float(val)
            except (TypeError, ValueError):
                return float('nan')

        snr_db = _payload_float(first_payload, 'snr_db')
        snr_db_csk_min = _payload_float(first_payload, 'snr_db_csk_min')
        snr_db_mosk = _payload_float(first_payload, 'snr_db_mosk')
        snr_db_amp = _payload_float(first_payload, 'snr_db_amp')
        snr_semantics = first_payload.get('snr_semantics')
        if not isinstance(snr_semantics, str) or not snr_semantics.strip():
            if mode in ("MoSK", "Hybrid"):
                snr_semantics = "MoSK contrast statistic (sign-aware DA vs SERO)"
            else:
                snr_semantics = "CSK Q-statistic (dual-channel combiner)"

        row: Dict[str, Any] = {
            'pipeline.Nm_per_symbol': float(nm),
            'Nm_per_symbol': float(nm),
            'ser': float(ser),
            'ser_ci_low': float(ci_low),
            'ser_ci_high': float(ci_high),
            'num_runs': len(seed_payloads),
            'symbols_evaluated': int(total_symbols),
            'sequence_length': int(expected_seq_len or (seq_lengths[0] if seq_lengths else 0)),
            'distance_um': float(dist_um),
            'use_ctrl': use_ctrl,
            'mode': mode,
            'detector_mode': detector_mode,
            'snr_db': float(snr_db),
            'snr_db_csk_min': float(snr_db_csk_min),
            'snr_db_mosk': float(snr_db_mosk),
            'snr_db_amp': float(snr_db_amp),
            'snr_semantics': snr_semantics,
            'symbol_period_s': float(Ts_expected),
            'decision_window_s': float(window_expected),
            'mosk_ser': float(mosk_ser),
            'csk_ser': float(csk_ser),
            'csk_ser_cond': float(conditional_csk_ser),
            'conditional_csk_ser': float(conditional_csk_ser),
            'csk_ser_eff': float(conditional_csk_ser * mosk_exposure_frac),
            'mosk_exposure_frac': float(mosk_exposure_frac),
            'mosk_correct_total': int(mosk_correct_total),
            'csk_exposure_rate': float(mosk_exposure_frac),
            'conditional_csk_error_given_exposure': float(conditional_csk_ser),
            'mosk_error_pct': float(mosk_error_pct),
            'csk_error_pct': float(csk_error_pct),
            'hybrid_total_error_pct': float(hybrid_error_pct),
            'hybrid_bits_per_symbol_mosk': float(bits_mosk_realized),
            'hybrid_bits_per_symbol_csk': float(bits_csk_realized),
            'hybrid_bits_per_symbol_total': float(bits_total),
            'rho_cc': float(cfg_base.get('noise', {}).get('rho_between_channels_after_ctrl', cfg_base.get('pipeline', {}).get('rho_cc', 0.0))),
            'rho_cc_measured': float(rho_cc_measured) if rho_cc_measured is not None else float('nan'),
            'ctrl_auto_applied': ctrl_auto,
            'calibration_frozen': any(bool(p.get('calibration_frozen', False)) for p in seed_payloads),
            'calibration_baseline_id': next(iter(baseline_ids)) if baseline_ids else '',
            'calibration_baseline_param': '',
            'calibration_baseline_value': float('nan'),
            'burst_shape': burst_shape,
            'T_release_ms': float(t_release_ms) if t_release_ms is not None else float('nan'),
            'source': 'lod_cache_replay',
            'data_source': 'lod_cache_replay',
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def _compute_hds_distance(dist_um: float,
                          cfg_base: Dict[str, Any],
                          nm_values: Sequence[int],
                          seeds: Sequence[int],
                          progress_mode: str,
                          resume: bool,
                          debug_calibration: bool,
                          cache_tag: str) -> Optional[pd.DataFrame]:
    cfg_d = deepcopy(cfg_base)
    cfg_d['pipeline']['distance_um'] = float(dist_um)

    try:
        Ts = calculate_dynamic_symbol_period(float(dist_um), cfg_d)
        cfg_d['pipeline']['symbol_period_s'] = Ts
        min_win = _enforce_min_window(cfg_d, Ts)
        cfg_d['pipeline']['time_window_s'] = max(cfg_d['pipeline'].get('time_window_s', 0.0), min_win)
        cfg_d.setdefault('detection', {})
        cfg_d['detection']['decision_window_s'] = min_win
    except Exception:
        pass

    freeze_payload = _get_distance_freeze_payload(cfg_base, float(dist_um))
    if freeze_payload:
        cfg_d['pipeline']['_frozen_noise'] = deepcopy(freeze_payload)
        cfg_d['_prefer_distance_freeze'] = True
    else:
        cfg_d['pipeline'].pop('_frozen_noise', None)
        cfg_d.pop('_prefer_distance_freeze', None)

    if not nm_values:
        return None

    # Attempt cache replay from LoD seed results first
    replay_df = _replay_hds_from_lod_cache(dist_um, cfg_d, nm_values, seeds, cache_tag)
    if replay_df is not None:
        return replay_df

    df_d = run_sweep(
        cfg_d,
        list(seeds),
        'pipeline.Nm_per_symbol',
        [float(n) for n in nm_values],
        f"HDS grid Hybrid (d={int(dist_um)} um)",
        progress_mode=progress_mode,
        persist_csv=None,
        resume=resume,
        debug_calibration=debug_calibration,
        cache_tag=cache_tag,
    )

    if df_d.empty:
        return None

    df_d = df_d.copy()
    df_d['distance_um'] = float(dist_um)
    return df_d

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


def _select_distance_nm_anchor(cfg: Dict[str, Any],
                               df_lod: Optional[pd.DataFrame],
                               use_ctrl: bool,
                               override_nm: Optional[float] = None) -> float:
    """
    Resolve the Nm value used for the SER/SNR vs distance sweep.
    """
    if override_nm is not None and math.isfinite(override_nm) and override_nm > 0:
        return float(override_nm)

    baseline_nm = float(cfg.get('pipeline', {}).get('Nm_per_symbol', 2000.0))
    if not math.isfinite(baseline_nm) or baseline_nm <= 0:
        baseline_nm = 2000.0

    if df_lod is None or df_lod.empty or 'lod_nm' not in df_lod.columns:
        return baseline_nm

    df = df_lod
    if 'use_ctrl' in df.columns:
        df = df[df['use_ctrl'] == bool(use_ctrl)]
    if df.empty:
        return baseline_nm

    nm_series = pd.to_numeric(df['lod_nm'], errors='coerce').dropna()
    if nm_series.empty:
        return baseline_nm

    nm_anchor = float(np.nanmedian(nm_series))
    if not math.isfinite(nm_anchor) or nm_anchor <= 0:
        return baseline_nm
    return nm_anchor


def _run_distance_metric_sweep(cfg: Dict[str, Any],
                               seeds: Sequence[int],
                               mode: str,
                               distances: Sequence[Union[float, int]],
                               data_dir: Path,
                               suffix: str,
                               df_lod: Optional[pd.DataFrame],
                               args: argparse.Namespace,
                               pm: ProgressManager,
                               use_ctrl: bool,
                               hierarchy_supported: bool,
                               mode_key: Optional[Tuple[str, str]],
                               dist_key: Optional[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    Execute a distance sweep at a fixed Nm to capture both SER and SNR metrics.
    """
    distance_values = [float(d) for d in distances]
    if not distance_values:
        print("?? Distance sweep skipped: no distances available.")
        return pd.DataFrame()

    pm.set_status(mode=mode, sweep="SER/SNR vs distance")

    dist_csv, dist_csv_branch, dist_csv_other = _stage_csv_paths("dist", data_dir, mode, suffix, use_ctrl)
    existing_dist_branch = _ensure_ablation_branch(
        "dist",
        dist_csv,
        dist_csv_branch,
        use_ctrl,
        bool(args.resume),
        _dedupe_distance_dataframe,
    )

    nm_anchor = _select_distance_nm_anchor(cfg, df_lod, use_ctrl, getattr(args, "distance_sweep_nm", None))
    if not math.isfinite(nm_anchor) or nm_anchor <= 0:
        nm_anchor = 2000.0
    nm_anchor = max(50.0, float(nm_anchor))
    nm_anchor_int = int(round(nm_anchor))

    print(f"\n2.5. Running SER/SNR vs distance sweep (Nm={nm_anchor_int:,d})")

    prev_nm = cfg['pipeline'].get('Nm_per_symbol')
    cfg['pipeline']['Nm_per_symbol'] = nm_anchor_int

    try:
        df_distance = run_sweep(
            cfg,
            list(seeds),
            'pipeline.distance_um',
            distance_values,
            f"SER/SNR vs distance ({mode})",
            progress_mode=args.progress,
            persist_csv=dist_csv_branch,
            resume=args.resume,
            debug_calibration=args.debug_calibration,
            pm=pm,
            sweep_key=dist_key if hierarchy_supported else None,
            parent_key=mode_key if hierarchy_supported else None,
            recalibrate=args.recalibrate,
        )
    finally:
        cfg['pipeline']['Nm_per_symbol'] = prev_nm

    dist_branch_frames = [df_distance] if df_distance is not None and not df_distance.empty else []
    dist_branch_combined = _update_branch_csv(
        "dist",
        dist_csv_branch,
        dist_branch_frames,
        use_ctrl,
        _dedupe_distance_dataframe,
        existing_dist_branch,
    )

    if not args.ablation_parallel:
        _merge_branch_csv(dist_csv, [dist_csv_branch, dist_csv_other], _dedupe_distance_dataframe)

    results_path = dist_csv_branch if args.ablation_parallel else dist_csv
    if args.ablation_parallel:
        print(f"? Distance sweep results saved to {results_path} (branch; canonical merge deferred)")
    else:
        print(f"? Distance sweep results saved to {results_path}")

    if dist_branch_combined is not None and not dist_branch_combined.empty:
        cols_to_show = [c for c in ['distance_um', 'ser', 'snr_db', 'use_ctrl'] if c in dist_branch_combined.columns]
        if cols_to_show:
            print(f"\nSER/SNR vs Distance (head) for {mode}:")
            print(dist_branch_combined[cols_to_show].head().to_string(index=False))

    return dist_branch_combined if dist_branch_combined is not None else (df_distance if df_distance is not None else pd.DataFrame())

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


def _wilson_interval(k: int, n: int, z: float) -> Tuple[float, float]:
    """Wilson score interval for Bernoulli proportion with normal quantile z."""
    if n <= 0:
        return (0.0, 1.0)
    p_hat = k / n
    z2_over_n = (z * z) / n
    denom = 1.0 + z2_over_n
    center = (p_hat + z2_over_n / 2.0) / denom
    rad = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2_over_n / (4.0 * n))) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))

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
            data.setdefault("__value_key", canonical_value_key(value))
            return data
        except Exception:
            return None
    return None

def write_seed_cache(mode: str, sweep: str, value: Union[float, int], seed: int, payload: Dict[str, Any],
                     use_ctrl: Optional[bool] = None, cache_tag: Optional[str] = None) -> None:
    cache_path = seed_cache_path(mode, sweep, value, seed, use_ctrl, cache_tag)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    # Tag the payload with its seed for later de-duplication in aggregations
    payload = dict(payload)
    payload.setdefault("__seed", int(seed))
    payload.setdefault("__value_key", canonical_value_key(value))
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
    parser.add_argument(
        "--run-stages",
        type=str,
        default="all",
        help="Compatibility shim for legacy orchestrators; stage gating is handled upstream.",
    )
    parser.add_argument(
        "--merge-ablation-csvs",
        action="store_true",
        help="Merge per-ablation CSV branches into canonical outputs and exit.")
    parser.add_argument(
        "--merge-stages",
        nargs="+",
        choices=list(_STAGE_CSV_BASENAMES.keys()),
        default=None,
        help="Limit --merge-ablation-csvs to specific stages (ser,lod,dist,isi).")
    parser.add_argument(
        "--merge-data-dir",
        type=str,
        default=None,
        help="Override the data directory for --merge-ablation-csvs (expects results/data path).")
    parser.add_argument(
        "--ablation-parallel",
        action="store_true",
        help="Hint from orchestrator: defer canonical CSV merges while both ablations run in parallel.",
    )
    parser.add_argument("--skip-noise-sweep", action="store_true",
                        help="Skip zero-signal noise-only sweeps (use analytic noise instead).")
    parser.add_argument("--noise-only-seeds", type=int, default=8,
                        help="Number of seeds for noise-only sweeps (default: 8).")
    parser.add_argument("--noise-only-seq-len", type=int, default=200,
                        help="Sequence length for each noise-only run (default: 200).")
    parser.add_argument("--force-noise-resample", action="store_true",
                        help="Force rerunning zero-signal noise sweeps even if resume cache matches.")
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
    parser.add_argument("--lod-debug", action="store_true",
                        help="Enable verbose diagnostics for LoD sweeps (writes results/debug/*.jsonl)")
    parser.add_argument("--channel-profile", choices=["tri", "dual", "single"], default="tri",
                        help="Physical channel setup: tri (DA+SERO+CTRL), dual (DA+SERO), single (DA only).")
    parser.add_argument("--csk-level-scheme", choices=["uniform", "zero-based"], default="uniform",
                       help="CSK level mapping scheme")
    parser.add_argument("--resume", action="store_true", help="Resume: skip finished values and append results as we go")
    parser.add_argument("--with-ctrl", dest="use_ctrl", action="store_true", help="Use CTRL differential subtraction")
    parser.add_argument("--no-ctrl", dest="use_ctrl", action="store_false", help="Disable CTRL subtraction (ablation)")
    parser.add_argument("--progress", choices=["tqdm", "rich", "gui", "none"], default="tqdm",
                    help="Progress UI backend")
    parser.add_argument("--detector-mode", choices=["zscore", "raw", "whitened"], default="zscore",
                        help="Detector statistic normalisation (default: zscore).")
    parser.add_argument("--freeze-calibration", action="store_true",
                        help="Reuse baseline thresholds/sigmas during parameter sweeps.")
    parser.add_argument("--store-calibration-stats", action="store_true",
                        help="Persist per-symbol calibration statistics (raw/zscore/whitened, integrals) into the threshold cache.")
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
    parser.add_argument("--lod-analytic-noise", action="store_true",
                        help="Force analytic noise during LoD sweeps (ignore cached/frozen noise).")
    parser.add_argument("--analytic-noise-all", action="store_true",
                        help="Force analytic noise for both zero-signal and LoD stages (skip zero-signal measurements).")
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
    parser.add_argument("--distance-sweep", choices=["always", "auto", "never"], default="always",
                    help="Generate SER/SNR vs distance sweep: always (default), auto (only when distances available), or never.")
    parser.add_argument("--distance-sweep-nm", type=float, default=None,
                    help="Override Nm_per_symbol used when sweeping distance (default: median LoD Nm or config baseline).")
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

    cli_nm_overrides: Dict[str, List[int]] = {}
    if getattr(args, "nm_grid", "").strip():
        try:
            cli_nm_overrides["__global__"] = _parse_nm_grid_spec(args.nm_grid)
        except ValueError as exc:
            print(f"⚠️  Invalid --nm-grid format: {args.nm_grid}. Error: {exc}")
            print("   Ignoring global Nm grid override.")
    for attr, mode_name in (("nm_grid_mosk", "MoSK"),
                            ("nm_grid_csk", "CSK"),
                            ("nm_grid_hybrid", "Hybrid")):
        spec = getattr(args, attr, "")
        if spec and spec.strip():
            try:
                cli_nm_overrides[_canonical_mode_name(mode_name)] = _parse_nm_grid_spec(spec)
            except ValueError as exc:
                flag_name = attr.replace("_", "-")
                print(f"⚠️  Invalid --{flag_name} format: {spec}. Error: {exc}")
                print(f"   Ignoring Nm grid override for {mode_name}.")
    args.cli_nm_overrides = cli_nm_overrides
    
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
    """Calibrate decision statistics for the requested detector mode."""
    try:
        from src.pipeline import _csk_dual_channel_Q

        cal_cfg = deepcopy(cfg)
        cal_cfg['pipeline']['sequence_length'] = num_symbols
        cal_cfg['disable_progress'] = True
        if mode == 'MoSK':
            cal_cfg['pipeline']['use_control_channel'] = False
        detector_mode = str(cal_cfg['pipeline'].get('detector_mode', cfg.get('pipeline', {}).get('detector_mode', 'zscore'))).lower()
        if detector_mode not in ('zscore', 'raw', 'whitened'):
            detector_mode = 'zscore'
        cal_cfg['pipeline']['detector_mode'] = detector_mode

        tx_symbols = [symbol] * num_symbols
        q_da_values: List[float] = []
        q_sero_values: List[float] = []
        q_da_tail_single: List[float] = []
        q_sero_tail_single: List[float] = []
        q_da_tail_diff: List[float] = []
        q_sero_tail_diff: List[float] = []
        q_da_full_single: List[float] = []
        q_sero_full_single: List[float] = []
        q_da_full_diff: List[float] = []
        q_sero_full_diff: List[float] = []
        stats_map: Dict[str, List[float]] = {key: [] for key in ('zscore', 'raw', 'whitened')}
        dt = cal_cfg['sim']['dt_s']
        detection = cal_cfg.setdefault('detection', {})
        detection_window_s = detection.get('decision_window_s', cal_cfg['pipeline']['symbol_period_s'])
        anchor_token = str(detection.get('decision_window_anchor', 'start')).lower()
        tail_mode = anchor_token in ('tail', 'end')
        sigma_da, sigma_sero = calculate_proper_noise_sigma(cal_cfg, detection_window_s)

        target_channel = cal_cfg['pipeline'].get('csk_target_channel', 'DA')
        combiner_cfg = cal_cfg['pipeline'].get('csk_combiner', 'zscore')
        combiner_selected = cal_cfg['pipeline'].get('csk_selected_combiner', combiner_cfg)
        use_dual = bool(cal_cfg['pipeline'].get('csk_dual_channel', True))
        leakage = float(cal_cfg['pipeline'].get('csk_leakage_frac', 0.0))

        rho_pre = float(cal_cfg.get('noise', {}).get('rho_corr', cal_cfg.get('noise', {}).get('rho_correlated', 0.9)))
        use_ctrl = bool(cal_cfg['pipeline'].get('use_control_channel', True))

        noise_cfg_cal = cal_cfg.get('noise', {})
        if 'rho_between_channels_after_ctrl' not in noise_cfg_cal:
            rho_post_default = 0.5 if use_ctrl else 0.0
        else:
            rho_post_default = noise_cfg_cal['rho_between_channels_after_ctrl']
        rho_post = float(noise_cfg_cal.get('rho_between_channels_after_ctrl', rho_post_default))
        rho_cc = rho_post if (use_ctrl and mode != 'MoSK') else rho_pre
        rho_cc = max(-1.0, min(1.0, rho_cc))

        rng = np.random.default_rng(cal_cfg['pipeline'].get('random_seed', 0))

        rho_cc_measured = np.nan
        if use_ctrl and (mode.startswith('CSK') or mode == 'Hybrid'):
            try:
                base_noise_samples = 20
                nm_value = cfg['pipeline'].get('Nm_per_symbol', 1e6)
                if nm_value < 1000:
                    noise_samples = 100
                elif nm_value < 10000:
                    noise_samples = 50
                else:
                    noise_samples = base_noise_samples

                q_da_noise, q_sero_noise = [], []
                cal_cfg_noise = deepcopy(cal_cfg)
                cal_cfg_noise['pipeline']['Nm_per_symbol'] = 1e-6

                n_total_samples_noise = int(cal_cfg_noise['pipeline']['symbol_period_s'] / dt)
                n_detect_samples_noise = min(int(detection_window_s / dt), n_total_samples_noise)
                start_noise = max(n_total_samples_noise - n_detect_samples_noise, 0) if tail_mode else 0
                end_noise = n_total_samples_noise if tail_mode else n_detect_samples_noise
                if end_noise - start_noise > 1:
                    for _ in range(noise_samples):
                        ig_n, ia_n, ic_n, _, _ = _single_symbol_currents(0, [], cal_cfg_noise, rng)
                        sig_da_n = ig_n - ic_n
                        sig_sero_n = ia_n - ic_n
                        q_da_n = float(np.trapezoid(sig_da_n[start_noise:end_noise], dx=dt))
                        q_sero_n = float(np.trapezoid(sig_sero_n[start_noise:end_noise], dx=dt))
                        q_da_noise.append(q_da_n)
                        q_sero_noise.append(q_sero_n)

                if len(q_da_noise) >= 3:
                    rho_cc_measured = float(np.corrcoef(q_da_noise, q_sero_noise)[0, 1])
                    if not np.isfinite(rho_cc_measured):
                        rho_cc_measured = np.nan
            except Exception:
                rho_cc_measured = np.nan

        for s_tx in tx_symbols:
            ig, ia, ic, Nm_actual, _ = _single_symbol_currents(s_tx, [], cal_cfg, rng)
            n_total_samples = len(ig)
            n_detect_samples = min(int(detection_window_s / dt), n_total_samples)
            if n_detect_samples <= 1:
                continue

            start_idx = max(n_total_samples - n_detect_samples, 0) if tail_mode else 0
            end_idx = n_total_samples if tail_mode else n_detect_samples
            window_len = max(end_idx - start_idx, 0)
            if window_len <= 1:
                continue

            sig_da_full = ig[start_idx:end_idx]
            sig_sero_full = ia[start_idx:end_idx]
            sig_ctrl_full = ic[start_idx:end_idx] if ic.size else np.zeros(window_len)
            if sig_ctrl_full.shape[0] < window_len:
                pad_width = window_len - sig_ctrl_full.shape[0]
                sig_ctrl_full = np.pad(sig_ctrl_full, (0, pad_width), mode='constant')

            tail_fraction = 1.0 if (tail_mode or mode == 'MoSK') else float(
                cal_cfg['pipeline'].get(
                    'csk_tail_fraction',
                    detection.get('tail_fraction', cal_cfg['pipeline'].get('hybrid_tail_fraction', 1.0))
                )
            )
            tail_fraction = min(max(tail_fraction, 0.1), 1.0)
            tail_start = int((1.0 - tail_fraction) * window_len)
            tail_start = min(max(tail_start, 0), max(window_len - 1, 0))

            use_ctrl = bool(cal_cfg['pipeline'].get('use_control_channel', True))
            subtract_for_q = (mode != 'MoSK') and use_ctrl

            sig_da_single_tail = sig_da_full[tail_start:]
            sig_sero_single_tail = sig_sero_full[tail_start:]
            q_da_tail_single_val = float(np.trapezoid(sig_da_single_tail, dx=dt))
            q_sero_tail_single_val = float(np.trapezoid(sig_sero_single_tail, dx=dt))

            if use_ctrl and len(sig_ctrl_full):
                sig_ctrl_tail = sig_ctrl_full[tail_start:]
                sig_da_diff_tail = (sig_da_single_tail - sig_ctrl_tail)
                sig_sero_diff_tail = (sig_sero_single_tail - sig_ctrl_tail)
                sig_da_diff_full = sig_da_full - sig_ctrl_full
                sig_sero_diff_full = sig_sero_full - sig_ctrl_full
            else:
                sig_da_diff_tail = sig_da_single_tail
                sig_sero_diff_tail = sig_sero_single_tail
                sig_da_diff_full = sig_da_full
                sig_sero_diff_full = sig_sero_full

            q_da_tail_diff_val = float(np.trapezoid(sig_da_diff_tail, dx=dt))
            q_sero_tail_diff_val = float(np.trapezoid(sig_sero_diff_tail, dx=dt))
            q_da_full_single_val = float(np.trapezoid(sig_da_full, dx=dt))
            q_sero_full_single_val = float(np.trapezoid(sig_sero_full, dx=dt))
            q_da_full_diff_val = float(np.trapezoid(sig_da_diff_full, dx=dt))
            q_sero_full_diff_val = float(np.trapezoid(sig_sero_diff_full, dx=dt))

            q_da_values.append(q_da_tail_diff_val if subtract_for_q else q_da_tail_single_val)
            q_sero_values.append(q_sero_tail_diff_val if subtract_for_q else q_sero_tail_single_val)

            q_da_tail_single.append(q_da_tail_single_val)
            q_sero_tail_single.append(q_sero_tail_single_val)
            q_da_tail_diff.append(q_da_tail_diff_val)
            q_sero_tail_diff.append(q_sero_tail_diff_val)
            q_da_full_single.append(q_da_full_single_val)
            q_sero_full_single.append(q_sero_full_single_val)
            q_da_full_diff.append(q_da_full_diff_val)
            q_sero_full_diff.append(q_sero_full_diff_val)

            q_da = q_da_values[-1]
            q_sero = q_sero_values[-1]

            if mode == 'MoSK':
                q_eff_da = get_nt_params(cfg, 'DA')['q_eff_e']
                q_eff_sero = get_nt_params(cfg, 'SERO')['q_eff_e']
                sign_da = 1.0 if q_eff_da >= 0 else -1.0
                sign_sero = 1.0 if q_eff_sero >= 0 else -1.0
                sigma_diff = math.sqrt(sigma_da * sigma_da + sigma_sero * sigma_sero - 2.0 * rho_pre * sigma_da * sigma_sero)
                sigma_diff = max(sigma_diff, 1e-15)
                raw_stat = (sign_da * q_da) - (sign_sero * q_sero)
                zscore_stat = raw_stat / sigma_diff
                stats_map['raw'].append(float(raw_stat))
                stats_map['zscore'].append(float(zscore_stat))
                stats_map['whitened'].append(float(zscore_stat))
            elif mode.startswith('CSK'):
                primary_charge = q_da if str(target_channel).upper() == 'DA' else q_sero
                stats_map['raw'].append(float(primary_charge))
                stat_zscore = _csk_dual_channel_Q(
                    q_da=q_da,
                    q_sero=q_sero,
                    sigma_da=sigma_da,
                    sigma_sero=sigma_sero,
                    rho_cc=rho_cc,
                    combiner=combiner_selected,
                    leakage_frac=leakage,
                    target=target_channel,
                    cfg=cal_cfg
                )
                stats_map['zscore'].append(float(stat_zscore))
                stat_whitened = _csk_dual_channel_Q(
                    q_da=q_da,
                    q_sero=q_sero,
                    sigma_da=sigma_da,
                    sigma_sero=sigma_sero,
                    rho_cc=rho_cc,
                    combiner='whitened',
                    leakage_frac=leakage,
                    target=target_channel,
                    cfg=cal_cfg
                )
                stats_map['whitened'].append(float(stat_whitened))
            elif mode == 'Hybrid':
                mol_type = symbol >> 1
                target = 'DA' if mol_type == 0 else 'SERO'
                combiner_hybrid = str(cal_cfg['pipeline'].get('hybrid_combiner', combiner_cfg))
                leakage_hybrid = float(cal_cfg['pipeline'].get('hybrid_leakage_frac', leakage))
                raw_stat = q_da if target == 'DA' else q_sero
                stats_map['raw'].append(float(raw_stat))
                stat_zscore = _csk_dual_channel_Q(
                    q_da=q_da,
                    q_sero=q_sero,
                    sigma_da=sigma_da,
                    sigma_sero=sigma_sero,
                    rho_cc=rho_cc,
                    combiner=combiner_hybrid,
                    leakage_frac=leakage_hybrid,
                    target=target,
                    cfg=cal_cfg
                )
                stats_map['zscore'].append(float(stat_zscore))
                stat_whitened = _csk_dual_channel_Q(
                    q_da=q_da,
                    q_sero=q_sero,
                    sigma_da=sigma_da,
                    sigma_sero=sigma_sero,
                    rho_cc=rho_cc,
                    combiner='whitened',
                    leakage_frac=leakage_hybrid,
                    target=target,
                    cfg=cal_cfg
                )
                stats_map['whitened'].append(float(stat_whitened))

        aux_q = [(float(a), float(b)) for a, b in zip(q_da_values, q_sero_values)]
        channel_integrals = {
            'tail_single': [(float(a), float(b)) for a, b in zip(q_da_tail_single, q_sero_tail_single)],
            'tail_diff': [(float(a), float(b)) for a, b in zip(q_da_tail_diff, q_sero_tail_diff)],
            'full_single': [(float(a), float(b)) for a, b in zip(q_da_full_single, q_sero_full_single)],
            'full_diff': [(float(a), float(b)) for a, b in zip(q_da_full_diff, q_sero_full_diff)],
        }
        chosen_stats = stats_map.get(detector_mode, stats_map['zscore'])
        result: Dict[str, Any] = {
            'q_values': chosen_stats,
            'zscore_stats': stats_map['zscore'],
            'raw_stats': stats_map['raw'],
            'whitened_stats': stats_map['whitened'],
            'aux_q': aux_q,
            'detector_mode': detector_mode,
            'channel_integrals': channel_integrals,
        }

        if mode.startswith('CSK') and cal_cfg['pipeline'].get('csk_store_combiner_meta', True):
            result.update({
                'combiner': combiner_selected,
                'rho_cc': rho_cc,
                'rho_cc_measured': rho_cc_measured,
                'leakage_frac': leakage if combiner_selected == 'leakage' else np.nan
            })

        if mode == 'Hybrid':
            result['rho_cc_measured'] = float(rho_cc_measured) if np.isfinite(rho_cc_measured) else float('nan')

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
    detect_cfg = cal.setdefault('detection', {})
    win = float(detect_cfg.get('decision_window_s', cal['pipeline'].get('symbol_period_s', dt)))
    anchor = str(detect_cfg.get('decision_window_anchor', 'start')).lower()
    tail_mode = anchor in ('tail', 'end')
    n = max(1, int(win / dt))
    qd, qs = [], []
    rng = np.random.default_rng(cal['pipeline'].get('random_seed', 0))
    
    for _ in range(20):
        ig, ia, ic, _, _ = _single_symbol_currents(0, [], cal, rng)
        sig_da = ig - ic
        sig_se = ia - ic
        n_int = min(n, sig_da.shape[0], sig_se.shape[0])
        if n_int <= 1:
            continue
        start = max(sig_da.shape[0] - n_int, 0) if tail_mode else 0
        end = start + n_int
        qd.append(float(np.trapezoid(sig_da[start:end], dx=dt)))
        qs.append(float(np.trapezoid(sig_se[start:end], dx=dt)))
    
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

def _apply_sweep_param(cfg_obj: Dict[str, Any], param_name: str, param_value: Union[float, int]) -> None:
    """
    Apply a sweep parameter value to the working configuration, keeping all
    secondary quantities (Ts, decision window, ISI memory) consistent with
    the updated setting.
    """
    if '.' in param_name:
        keys = param_name.split('.')
        target = cfg_obj
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = param_value
    else:
        cfg_obj[param_name] = param_value

    detection = cfg_obj.setdefault('detection', {})

    if param_name == 'pipeline.distance_um':
        Ts = calculate_dynamic_symbol_period(float(param_value), cfg_obj)
        min_win = _enforce_min_window(cfg_obj, Ts)
        cfg_obj['pipeline']['symbol_period_s'] = Ts
        cfg_obj['pipeline']['time_window_s'] = max(cfg_obj['pipeline'].get('time_window_s', 0.0), min_win)
        detection['decision_window_s'] = min_win
        if cfg_obj['pipeline'].get('enable_isi', False):
            D_da = cfg_obj['neurotransmitters']['DA']['D_m2_s']
            lambda_da = cfg_obj['neurotransmitters']['DA']['lambda']
            D_eff = D_da / (lambda_da ** 2)
            time_95 = 3.0 * ((float(param_value) * 1e-6)**2) / D_eff
            guard_factor = cfg_obj['pipeline'].get('guard_factor', 0.1)
            isi_memory = math.ceil((1 + guard_factor) * time_95 / Ts)
            cfg_obj['pipeline']['isi_memory_symbols'] = isi_memory
    elif param_name == 'pipeline.guard_factor':
        dist = float(cfg_obj['pipeline']['distance_um'])
        Ts = calculate_dynamic_symbol_period(dist, cfg_obj)
        min_win = _enforce_min_window(cfg_obj, Ts)
        cfg_obj['pipeline']['symbol_period_s'] = Ts
        cfg_obj['pipeline']['time_window_s'] = max(cfg_obj['pipeline'].get('time_window_s', 0.0), min_win)
        detection['decision_window_s'] = min_win
        if cfg_obj['pipeline'].get('enable_isi', False):
            D_da = cfg_obj['neurotransmitters']['DA']['D_m2_s']
            lambda_da = cfg_obj['neurotransmitters']['DA']['lambda']
            D_eff = D_da / (lambda_da ** 2)
            time_95 = 3.0 * ((dist * 1e-6)**2) / D_eff
            guard_factor = float(param_value)
            isi_memory = math.ceil((1 + guard_factor) * time_95 / Ts)
            cfg_obj['pipeline']['isi_memory_symbols'] = isi_memory
    elif param_name == 'pipeline.Nm_per_symbol':
        Ts = float(cfg_obj['pipeline'].get('symbol_period_s',
                 calculate_dynamic_symbol_period(float(cfg_obj['pipeline'].get('distance_um', 50.0)), cfg_obj)))
        min_win = _enforce_min_window(cfg_obj, Ts)
        cfg_obj['pipeline']['time_window_s'] = max(cfg_obj['pipeline'].get('time_window_s', 0.0), min_win)
        detection['decision_window_s'] = min_win
    elif param_name in ['oect.gm_S', 'oect.C_tot_F']:
        Ts = float(cfg_obj['pipeline'].get('symbol_period_s',
                 calculate_dynamic_symbol_period(float(cfg_obj['pipeline'].get('distance_um', 50.0)), cfg_obj)))
        min_win = _enforce_min_window(cfg_obj, Ts)
        cfg_obj['pipeline']['symbol_period_s'] = Ts
        cfg_obj['pipeline']['time_window_s'] = max(cfg_obj['pipeline'].get('time_window_s', 0.0), min_win)
        detection['decision_window_s'] = min_win


def _get_param_value(cfg_obj: Dict[str, Any], param_name: str) -> Any:
    if '.' in param_name:
        keys = param_name.split('.')
        target = cfg_obj
        for key in keys[:-1]:
            target = target.get(key, {})
        return target.get(keys[-1])
    return cfg_obj.get(param_name)


def _build_freeze_snapshot(cfg_base: Dict[str, Any],
                           param_name: str,
                           baseline_value: Union[float, int, None],
                           cal_seeds: List[int],
                           recalibrate: bool) -> Optional[Dict[str, Any]]:
    if baseline_value is None:
        return None

    cfg_snap = deepcopy(cfg_base)
    _apply_sweep_param(cfg_snap, param_name, baseline_value)

    detector_mode = str(cfg_snap['pipeline'].get('detector_mode', 'zscore')).lower()

    thresholds: Dict[str, Union[float, List[float], str]] = {}
    if cal_seeds:
        thresholds = calibrate_thresholds_cached(cfg_snap, cal_seeds, recalibrate)

    Ts = float(cfg_snap['pipeline']['symbol_period_s'])
    dt = float(cfg_snap['sim']['dt_s'])
    detection_window_s = _resolve_decision_window(cfg_snap, Ts, dt)
    noise_components: Dict[str, float] = {}
    sigma_da, sigma_sero = calculate_proper_noise_sigma(cfg_snap, detection_window_s, components_out=noise_components)

    noise_cfg = cfg_snap.get('noise', {})
    use_ctrl = bool(cfg_snap['pipeline'].get('use_control_channel', True))
    mod = cfg_snap['pipeline']['modulation']

    rho_pre = float(noise_cfg.get('rho_corr', noise_cfg.get('rho_correlated', 0.9)))
    rho_post_default = 0.0
    rho_post = float(noise_cfg.get('rho_between_channels_after_ctrl', rho_post_default))
    rho_post = float(cfg_snap['pipeline'].get('rho_cc_measured', rho_post))

    if mod == 'MoSK':
        rho_for_diff = rho_pre
    else:
        rho_for_diff = rho_post if use_ctrl else rho_pre
    rho_for_diff = max(-1.0, min(1.0, rho_for_diff))

    sigma_diff = math.sqrt(max(
        sigma_da * sigma_da + sigma_sero * sigma_sero - 2.0 * rho_for_diff * sigma_da * sigma_sero,
        0.0
    ))

    cfg_mosk = deepcopy(cfg_snap)
    cfg_mosk['pipeline']['modulation'] = 'MoSK'
    sigma_da_mosk, sigma_sero_mosk = calculate_proper_noise_sigma(cfg_mosk, detection_window_s)
    sigma_diff_mosk = math.sqrt(max(
        sigma_da_mosk * sigma_da_mosk + sigma_sero_mosk * sigma_sero_mosk - 2.0 * rho_pre * sigma_da_mosk * sigma_sero_mosk,
        0.0
    ))

    noise_payload = {
        'sigma_da': float(sigma_da),
        'sigma_sero': float(sigma_sero),
        'sigma_diff': float(sigma_diff),
        'sigma_diff_mosk': float(sigma_diff_mosk),
        'rho_for_diff': float(rho_for_diff),
        'rho_cc': float(rho_post if use_ctrl else rho_pre),
        'detection_window_s': float(detection_window_s),
        'noise_components': dict(noise_components),
        'detector_mode': detector_mode,
    }

    fingerprint = json.dumps({
        "thresholds": _json_safe(thresholds),
        "noise": _json_safe(noise_payload),
        "detector_mode": detector_mode,
        "param": param_name,
    }, sort_keys=True, allow_nan=True, default=str)
    baseline_id = hashlib.sha1(fingerprint.encode('utf-8')).hexdigest()[:12]

    snapshot = {
        'thresholds': thresholds,
        'noise': noise_payload,
        'detector_mode': detector_mode,
        'baseline_id': baseline_id,
        'sweep_param': param_name,
        'baseline_value': baseline_value,
    }
    return snapshot


def _median_nan(values: Iterable[Any]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float('nan')
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float('nan')
    return float(np.median(finite))


def _measure_noise_snapshot(cfg_base: Dict[str, Any],
                            param_name: str,
                            param_value: Union[float, int],
                            seeds: List[int],
                            noise_seq_len: int) -> Optional[Dict[str, Any]]:
    cfg_noise = deepcopy(cfg_base)
    _apply_sweep_param(cfg_noise, param_name, param_value)

    pipeline_cfg = cfg_noise.setdefault('pipeline', {})
    pipeline_cfg['Nm_per_symbol'] = 0
    pipeline_cfg['_noise_only_run'] = True
    pipeline_cfg['_collect_noise_components'] = True
    pipeline_cfg['sequence_length'] = max(int(noise_seq_len), 8)
    cfg_noise['disable_progress'] = True
    cfg_noise['verbose'] = False

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        pipeline_cfg['random_seed'] = int(seed)
        res = run_single_instance(cfg_noise, seed, attach_isi_meta=False)
        if res is not None:
            results.append(res)

    if not results:
        return None

    pipeline_cfg.pop('_collect_noise_components', None)

    def _collect(field: str) -> List[float]:
        collected: List[float] = []
        for r in results:
            try:
                collected.append(float(r.get(field, float('nan'))))
            except (TypeError, ValueError):
                collected.append(float('nan'))
        return collected

    med_sigma_da = _median_nan(_collect('noise_sigma_da'))
    med_sigma_sero = _median_nan(_collect('noise_sigma_sero'))
    med_sigma_diff = _median_nan(_collect('noise_sigma_I_diff'))
    med_sigma_diff_charge = _median_nan(_collect('noise_sigma_diff_charge'))
    med_sigma_single = _median_nan(_collect('noise_sigma_single'))
    med_rho = _median_nan(_collect('noise_rho_measured'))
    med_sample_size = _median_nan(_collect('noise_sigma_sample_size'))
    med_window = _median_nan(_collect('detection_window_s'))
    med_sigma_diff_current = _median_nan(_collect('noise_sigma_diff_current_measured'))
    med_sigma_thermal = _median_nan(_collect('noise_sigma_thermal'))
    med_sigma_flicker = _median_nan(_collect('noise_sigma_flicker'))
    med_sigma_drift = _median_nan(_collect('noise_sigma_drift'))
    med_fraction_thermal = _median_nan(_collect('noise_thermal_fraction'))
    med_fraction_flicker = _median_nan(_collect('noise_flicker_fraction'))
    med_fraction_drift = _median_nan(_collect('noise_drift_fraction'))
    med_sigma_thermal_single = _median_nan(_collect('noise_sigma_thermal_single_measured'))
    med_sigma_flicker_single = _median_nan(_collect('noise_sigma_flicker_single_measured'))
    med_sigma_drift_single = _median_nan(_collect('noise_sigma_drift_single_measured'))
    med_ctrl_reduction = _median_nan(_collect('noise_ctrl_reduction_fraction_mean'))

    noise_payload = {
        'sigma_da': med_sigma_da,
        'sigma_sero': med_sigma_sero,
        'sigma_diff': med_sigma_diff,
        'sigma_diff_mosk': med_sigma_diff,
        'rho_for_diff': med_rho,
        'rho_cc': med_rho,
        'detection_window_s': med_window,
        'noise_components': {
            'sigma_diff_charge': med_sigma_diff_charge,
            'single_ended_sigma': med_sigma_single,
            'thermal_sigma': med_sigma_thermal,
            'flicker_sigma': med_sigma_flicker,
            'drift_sigma': med_sigma_drift,
            'thermal_fraction': med_fraction_thermal,
            'flicker_fraction': med_fraction_flicker,
            'drift_fraction': med_fraction_drift,
            'thermal_sigma_single': med_sigma_thermal_single,
            'flicker_sigma_single': med_sigma_flicker_single,
            'drift_sigma_single': med_sigma_drift_single,
            'ctrl_reduction_fraction_mean': med_ctrl_reduction,
        },
        'detector_mode': str(pipeline_cfg.get('detector_mode', 'zscore')).lower(),
        'sample_size': med_sample_size,
        'source': 'noise_only_sweep',
        'ctrl_reduction_fraction_mean': med_ctrl_reduction,
    }

    median_map = {
        'noise_sigma_da': med_sigma_da,
        'noise_sigma_sero': med_sigma_sero,
        'noise_sigma_I_diff': med_sigma_diff,
        'noise_sigma_diff_charge': med_sigma_diff_charge,
        'noise_sigma_single': med_sigma_single,
        'noise_rho_measured': med_rho,
        'noise_sigma_sample_size': med_sample_size,
        'noise_sigma_diff_current_measured': med_sigma_diff_current,
        'detection_window_s': med_window,
        'noise_sigma_thermal': med_sigma_thermal,
        'noise_sigma_flicker': med_sigma_flicker,
        'noise_sigma_drift': med_sigma_drift,
        'noise_thermal_fraction': med_fraction_thermal,
        'noise_flicker_fraction': med_fraction_flicker,
        'noise_drift_fraction': med_fraction_drift,
        'noise_sigma_thermal_single_measured': med_sigma_thermal_single,
        'noise_sigma_flicker_single_measured': med_sigma_flicker_single,
        'noise_sigma_drift_single_measured': med_sigma_drift_single,
        'noise_ctrl_reduction_fraction_mean': med_ctrl_reduction,
    }

    return {
        'noise': noise_payload,
        'medians': median_map,
        'seed_count': len(results),
    }


def _build_noise_freeze_map(cfg_base: Dict[str, Any],
                            param_name: str,
                            values: Iterable[Union[float, int]],
                            seeds: List[int],
                            noise_seq_len: int,
                            pm: Optional[ProgressManager] = None,
                            progress_key: Optional[Any] = None,
                            description: str = "Noise-only sweep") -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    values_list = list(values)
    noise_map: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    if not values_list or not seeds:
        return noise_map, rows

    task = None
    if pm is not None:
        try:
            task = pm.task(total=len(values_list), description=description,
                           parent=progress_key, kind='sweep')
        except Exception:
            task = None

    for value in values_list:
        snapshot = _measure_noise_snapshot(cfg_base, param_name, value, seeds, noise_seq_len)
        key = _value_key(value)
        if snapshot:
            noise_map[key] = snapshot['noise']
            med = snapshot['medians']
            snr_db = float('nan')
            snr_db_csk_min = float('nan')
            snr_db_mosk = float('nan')
            snr_db_amp = float('nan')
            snr_semantics = "Noise-only measurement"
            snr_csk_method = ''
            snr_csk_boundary = ''
            snr_amp_method = ''
            snr_amp_worst = ''
            csk_summary_export: Dict[int, Dict[str, float]] = {}
            hybrid_amp_summary_export: Dict[str, Dict[str, float]] = {}
            hybrid_amp_detail: Dict[str, float] = {}
            row: Dict[str, Any] = {
                'sweep_param': param_name,
                'value': float(value),
                'value_key': key,
                'noise_sigma_da': med.get('noise_sigma_da', float('nan')),
                'noise_sigma_sero': med.get('noise_sigma_sero', float('nan')),
                'noise_sigma_I_diff': med.get('noise_sigma_I_diff', float('nan')),
                'noise_sigma_diff_charge': med.get('noise_sigma_diff_charge', float('nan')),
                'noise_sigma_single': med.get('noise_sigma_single', float('nan')),
                'noise_rho_measured': med.get('noise_rho_measured', float('nan')),
                'noise_sigma_sample_size': med.get('noise_sigma_sample_size', float('nan')),
                'noise_sigma_diff_current_measured': med.get('noise_sigma_diff_current_measured', float('nan')),
                'detection_window_s': med.get('detection_window_s', float('nan')),
                'noise_sigma_thermal': med.get('noise_sigma_thermal', float('nan')),
                'noise_sigma_flicker': med.get('noise_sigma_flicker', float('nan')),
                'noise_sigma_drift': med.get('noise_sigma_drift', float('nan')),
                'noise_thermal_fraction': med.get('noise_thermal_fraction', float('nan')),
                'noise_flicker_fraction': med.get('noise_flicker_fraction', float('nan')),
                'noise_drift_fraction': med.get('noise_drift_fraction', float('nan')),
                'noise_sigma_thermal_single_measured': med.get('noise_sigma_thermal_single_measured', float('nan')),
                'noise_sigma_flicker_single_measured': med.get('noise_sigma_flicker_single_measured', float('nan')),
                'noise_sigma_drift_single_measured': med.get('noise_sigma_drift_single_measured', float('nan')),
                'noise_ctrl_reduction_fraction_mean': med.get('noise_ctrl_reduction_fraction_mean', float('nan')),
                'seed_count': snapshot.get('seed_count', len(seeds)),
            }
            row['snr_db'] = snr_db
            row['snr_db_csk_min'] = snr_db_csk_min
            row['snr_db_mosk'] = snr_db_mosk
            row['snr_db_amp'] = snr_db_amp
            row['snr_semantics'] = snr_semantics
            row['snr_csk_method'] = snr_csk_method
            row['snr_csk_boundary_min'] = snr_csk_boundary
            row['snr_amp_method'] = snr_amp_method
            row['snr_amp_worst_molecule'] = snr_amp_worst
            row['csk_level_stats_summary'] = csk_summary_export
            row['hybrid_amp_stats_summary'] = hybrid_amp_summary_export
            row['hybrid_amp_detail_db'] = hybrid_amp_detail
            rows.append(row)
        if task:
            task.update(1)

    if task:
        task.close()

    return noise_map, rows


def run_zero_signal_noise_analysis(
    cfg: Dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    nm_values: Sequence[Union[int, float]],
    lod_distance_grid: Sequence[Union[int, float]],
    seeds: Sequence[int],
    data_dir: Path,
    suffix: str,
    pm: ProgressManager,
    hierarchy_supported: bool,
    mode_key: Optional[Any],
) -> None:
    if getattr(args, "skip_noise_sweep", False):
        _warn_synthetic_noise("zero-signal noise sweep was skipped (--skip-noise-sweep).")
        return

    noise_seed_count = min(int(max(args.noise_only_seeds, 1)), len(seeds))
    if noise_seed_count <= 0:
        print("??  Skipping zero-signal noise sweep (no seeds available).")
        _warn_synthetic_noise("zero-signal noise sweep could not run (no seeds available).")
        return

    noise_seeds = [int(s) for s in seeds[:noise_seed_count]]
    noise_seq_len = int(max(args.noise_only_seq_len, 8))
    cfg.setdefault('_noise_freeze_map', {})

    noise_csv = data_dir / f"noise_only_{mode.lower()}{suffix}.csv"
    noise_meta_path = data_dir / f"noise_only_{mode.lower()}{suffix}.meta.json"
    fingerprint = _noise_fingerprint(
        mode=mode,
        suffix=suffix,
        nm_values=nm_values,
        lod_distance_grid=lod_distance_grid,
        noise_seeds=noise_seeds,
        noise_seq_len=noise_seq_len,
        cfg=cfg,
    )

    if (getattr(args, "resume", False)
            and not getattr(args, "recalibrate", False)
            and not getattr(args, "force_noise_resample", False)):
        cached_meta = _load_noise_meta(noise_meta_path)
        if cached_meta and cached_meta.get("fingerprint") == fingerprint:
            saved_map = cached_meta.get("freeze_map")
            if isinstance(saved_map, dict) and saved_map:
                cfg['_noise_freeze_map'].update(deepcopy(saved_map))
                saved_dist_map = cached_meta.get("freeze_distance_map")
                if isinstance(saved_dist_map, dict) and saved_dist_map:
                    cfg['_noise_freeze_distance_map'] = deepcopy(saved_dist_map)
                print(f"↩️  Resume: zero-signal noise sweep already complete ({mode}).")
                return

    print("? Measuring zero-signal noise floor...")

    cfg_noise_ready = deepcopy(cfg)
    cfg_noise_ready.setdefault('pipeline', {})['_suppress_threshold_warnings'] = True
    try:
        warm_seeds = list(range(max(1, min(len(noise_seeds), 6))))
        warm_thresholds = calibrate_thresholds(
            cfg_noise_ready,
            warm_seeds,
            recalibrate=args.recalibrate,
            save_to_file=False,
            verbose=False,
        )
        for key, value in warm_thresholds.items():
            if isinstance(key, str) and key.startswith("noise."):
                cfg_noise_ready.setdefault("noise", {})[key.split(".", 1)[1]] = value
            elif _should_apply_threshold_key(key):
                cfg_noise_ready.setdefault('pipeline', {})[key] = value
    except Exception as exc:
        print(f"??  Pre-noise calibration skipped: {exc}")

    noise_rows: List[Dict[str, Any]] = []

    noise_map_nm, rows_nm = _build_noise_freeze_map(
        cfg_noise_ready,
        "pipeline.Nm_per_symbol",
        nm_values,
        noise_seeds,
        noise_seq_len=noise_seq_len,
        pm=pm if hierarchy_supported else None,
        progress_key=mode_key if hierarchy_supported else None,
        description="Noise-only (Nm)"
    )
    if noise_map_nm:
        cfg['_noise_freeze_map'].setdefault("pipeline.Nm_per_symbol", {}).update(noise_map_nm)
        noise_rows.extend([{**row, 'mode': mode, 'stage': 'Nm'} for row in rows_nm])
    else:
        _warn_synthetic_noise("noise-only sweep for Nm failed to produce measurements.")

    noise_map_dist, rows_dist = _build_noise_freeze_map(
        cfg_noise_ready,
        "pipeline.distance_um",
        lod_distance_grid,
        noise_seeds,
        noise_seq_len=noise_seq_len,
        pm=pm if hierarchy_supported else None,
        progress_key=mode_key if hierarchy_supported else None,
        description="Noise-only (distance)"
    )
    if noise_map_dist:
        cfg['_noise_freeze_map'].setdefault("pipeline.distance_um", {}).update(noise_map_dist)
        cfg['_noise_freeze_distance_map'] = noise_map_dist
        noise_rows.extend([{**row, 'mode': mode, 'stage': 'distance'} for row in rows_dist])
    else:
        _warn_synthetic_noise("noise-only sweep for distance failed to produce measurements.")

    if not noise_rows:
        return

    noise_df = pd.DataFrame(noise_rows)
    if noise_csv.exists():
        try:
            existing_noise = pd.read_csv(noise_csv)
            combined_noise = pd.concat([existing_noise, noise_df], ignore_index=True)
        except Exception as exc:
            print(f"??  Could not read existing noise CSV ({exc}); overwriting.")
            combined_noise = noise_df
    else:
        combined_noise = noise_df
    noise_subset_cols = [col for col in ['mode', 'stage', 'sweep_param', 'value_key'] if col in combined_noise.columns]
    if noise_subset_cols:
        combined_noise = combined_noise.drop_duplicates(subset=noise_subset_cols, keep='last')
    _atomic_write_csv(noise_csv, combined_noise)
    print(f"? Noise-only sweep results saved to {noise_csv}")

    meta_payload = {
        "version": "v1",
        "fingerprint": fingerprint,
        "noise_seed_count": len(noise_seeds),
        "noise_seq_len": noise_seq_len,
        "noise_seeds": noise_seeds,
        "freeze_map": deepcopy(cfg.get('_noise_freeze_map', {})),
        "freeze_distance_map": deepcopy(cfg.get('_noise_freeze_distance_map', {})),
        "timestamp": time.time(),
    }
    _atomic_write_json(noise_meta_path, meta_payload)

def run_param_seed_combo(cfg_base: Dict[str, Any], param_name: str,
                         param_value: Union[float, int], seed: int,
                         debug_calibration: bool = False,
                         thresholds_override: Optional[Dict[str, Union[float, List[float], str]]] = None,
                         sweep_name: str = "ser_vs_nm", cache_tag: Optional[str] = None,
                         recalibrate: bool = False,
                         freeze_snapshot: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Worker for parameter sweep with window match; accepts optional precomputed thresholds."""
    try:
        cfg_run = deepcopy(cfg_base)
        cfg_run['disable_progress'] = True
        cfg_run['verbose'] = False
        pipeline = cfg_run.setdefault('pipeline', {})
        force_analytic_noise = bool(cfg_run.get('_force_analytic_noise', False))

        preexisting_freeze = pipeline.get('_frozen_noise')
        have_distance_freeze = isinstance(preexisting_freeze, dict) and bool(preexisting_freeze)
        prefer_distance_freeze = bool(cfg_run.get('_prefer_distance_freeze', False))

        _apply_sweep_param(cfg_run, param_name, param_value)

        expect_frozen_noise = False
        payload: Optional[Dict[str, Any]] = None
        if not force_analytic_noise:
            noise_freeze_map = cfg_run.get('_noise_freeze_map')
            if isinstance(noise_freeze_map, dict):
                param_noise_map = noise_freeze_map.get(param_name) or {}
                candidate = param_noise_map.get(_value_key(param_value))
                if isinstance(candidate, dict) and candidate:
                    payload = candidate

        if pipeline.get('_freeze_calibration_active'):
            expect_frozen_noise = True

        if freeze_snapshot:
            thresholds_freeze = freeze_snapshot.get('thresholds', {})
            if thresholds_override:
                merged = dict(thresholds_freeze)
                merged.update(thresholds_override)
                thresholds_override = merged
            else:
                thresholds_override = dict(thresholds_freeze)
            freeze_payload = freeze_snapshot.get('noise', {})
            pipeline['_freeze_calibration_active'] = True
            pipeline['_freeze_baseline_id'] = freeze_snapshot.get('baseline_id', '')
            if not force_analytic_noise and isinstance(freeze_payload, dict) and freeze_payload:
                pipeline['_frozen_noise'] = freeze_payload
                expect_frozen_noise = True
                payload = None

        if force_analytic_noise:
            payload = None
            pipeline.pop('_frozen_noise', None)
            expect_frozen_noise = False
        elif param_name == 'pipeline.Nm_per_symbol' and prefer_distance_freeze and have_distance_freeze:
            pass
        else:
            if isinstance(payload, dict) and payload:
                sanitized_payload = _cached_sanitized_freeze_payload(cfg_base, param_name, param_value, payload)
                if sanitized_payload:
                    pipeline['_frozen_noise'] = sanitized_payload
                    expect_frozen_noise = True
                else:
                    pipeline.pop('_frozen_noise', None)
                    expect_frozen_noise = False
            else:
                if not isinstance(pipeline.get('_frozen_noise'), dict):
                    expect_frozen_noise = False

        # Thresholds: use override if supplied, else cached calibration
        if thresholds_override is not None:
            _apply_thresholds_into_cfg(cfg_run, thresholds_override)
        elif cfg_run['pipeline']['modulation'] in ['MoSK', 'CSK', 'Hybrid'] and \
             param_name in ['pipeline.Nm_per_symbol', 'pipeline.distance_um', 'pipeline.guard_factor', 'oect.gm_S', 'oect.C_tot_F']:
            cal_seeds = list(range(10))
            thresholds = calibrate_thresholds_cached(cfg_run, cal_seeds, recalibrate)
            _apply_thresholds_into_cfg(cfg_run, thresholds)
            if debug_calibration and cfg_run['pipeline']['modulation'] == 'CSK':
                target_ch = cfg_run['pipeline'].get('csk_target_channel', 'DA').lower()
                key = f'csk_thresholds_{target_ch}'
                if key in cfg_run['pipeline']:
                    print(f"[DEBUG] CSK Thresholds @ {param_value}: {cfg_run['pipeline'][key]}")

        if expect_frozen_noise and not isinstance(cfg_run['pipeline'].get('_frozen_noise'), dict):
            _warn_synthetic_noise(
                f"no measured noise payload for {param_name}={param_value}; falling back to analytic sigmas."
            )

        # Run the instance and attach per-run ISI metrics
        result = run_single_instance(cfg_run, seed, attach_isi_meta=True)
        if result is not None:
            # Tag the in-memory result so mixed cached+fresh paths dedupe correctly
            try:
                result["__seed"] = int(seed)
            except Exception:
                pass
            if freeze_snapshot:
                result['calibration_frozen'] = True
                result['calibration_baseline_id'] = freeze_snapshot.get('baseline_id', '')
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

    if persist_csv is not None:
        flush_staged_rows(persist_csv)

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

    freeze_enabled = bool(cfg.get('analysis', {}).get('freeze_calibration', False))
    freeze_snapshot: Optional[Dict[str, Any]] = None
    if freeze_enabled and values_to_run:
        baseline_value = _get_param_value(cfg, sweep_param)
        snapshot_seeds = cal_seeds if cal_seeds else list(range(10))
        freeze_snapshot = _build_freeze_snapshot(cfg, sweep_param, baseline_value, snapshot_seeds, recalibrate)
        if freeze_snapshot:
            for v in values_to_run:
                thresholds_map[v] = dict(freeze_snapshot.get('thresholds', {}))
        else:
            print(f"?? Freeze calibration requested but baseline snapshot unavailable for {sweep_param}; proceeding without freeze.")
            freeze_snapshot = None
            freeze_enabled = False

    if not freeze_enabled:
        for v in values_to_run:
            cfg_v = deepcopy(cfg)
            _apply_sweep_param(cfg_v, sweep_param, v)
            thresholds_map[v] = calibrate_thresholds_cached(cfg_v, cal_seeds, recalibrate)

    job_bar: Optional[Any] = None
    try:
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
            thresholds_override = dict(thresholds_map.get(v, {}))
            
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
                        sweep_name=sweep_folder, cache_tag=cache_tag, recalibrate=recalibrate,
                        freeze_snapshot=freeze_snapshot
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
                                                    sweep_name=sweep_folder, cache_tag=retry_tag, recalibrate=recalibrate,
                                                    freeze_snapshot=freeze_snapshot)
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
            if total_symbols > 0:
                z = 1.96
                p_hat = total_errors / total_symbols
                den = 1.0 + (z * z) / total_symbols
                center = (p_hat + (z * z) / (2 * total_symbols)) / den
                half = z * math.sqrt((p_hat * (1.0 - p_hat) / total_symbols) + (z * z) / (4 * total_symbols * total_symbols)) / den
                ci_low = max(0.0, center - half)
                ci_high = min(1.0, center + half)
            else:
                ci_low = float('nan')
                ci_high = float('nan')
    
            # pooled decision stats for SNR proxy
            def _coerce_finite(values: Iterable[Any]) -> List[float]:
                clean: List[float] = []
                for val in values:
                    try:
                        f = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(f):
                        clean.append(f)
                return clean

            def _merge_stat_summary(
                dest: Dict[Any, Dict[str, float]],
                key: Any,
                payload: Mapping[str, Any],
            ) -> None:
                try:
                    count = float(payload.get("count", 0.0))
                    mean = float(payload.get("mean", float("nan")))
                    var = float(payload.get("var", float("nan")))
                except (TypeError, ValueError):
                    return
                if not (math.isfinite(count) and count > 0):
                    return
                if not math.isfinite(mean):
                    return
                if not math.isfinite(var) or var < 0:
                    var = 0.0
                m2 = var * max(count - 1.0, 0.0)
                entry = dest.get(key)
                if entry is None:
                    dest[key] = {"count": count, "mean": mean, "m2": m2}
                    return
                prev_count = entry.get("count", 0.0)
                prev_mean = entry.get("mean", 0.0)
                prev_m2 = entry.get("m2", 0.0)
                total = prev_count + count
                if total <= 0:
                    return
                delta = mean - prev_mean
                entry["m2"] = prev_m2 + m2 + delta * delta * prev_count * count / total
                entry["mean"] = (prev_mean * prev_count + mean * count) / total
                entry["count"] = total

            def _summary_variance(entry: Mapping[str, Any]) -> float:
                count = float(entry.get("count", 0.0))
                if count <= 1:
                    return 0.0
                if "m2" in entry:
                    m2 = float(entry.get("m2", 0.0))
                    if not math.isfinite(m2) or m2 < 0:
                        return 0.0
                    return m2 / (count - 1.0)
                var = float(entry.get("var", 0.0))
                if not math.isfinite(var) or var < 0:
                    return 0.0
                return var

            def _export_summary(entry: Mapping[str, Any]) -> Dict[str, float]:
                return {
                    "count": float(entry.get("count", 0.0)),
                    "mean": float(entry.get("mean", float("nan"))),
                    "var": _summary_variance(entry),
                }

            def _paired_snr_db(samples_a: Sequence[float], samples_b: Sequence[float]) -> float:
                if not samples_a or not samples_b:
                    return float('nan')
                arr_a = np.asarray(samples_a, dtype=float)
                arr_b = np.asarray(samples_b, dtype=float)
                if arr_a.size < 2 or arr_b.size < 2:
                    return float('nan')
                mean_a = float(np.mean(arr_a))
                mean_b = float(np.mean(arr_b))
                diff = abs(mean_b - mean_a)
                var_a = float(np.var(arr_a, ddof=1))
                var_b = float(np.var(arr_b, ddof=1))
                denom = (arr_a.size + arr_b.size - 2)
                if denom > 0 and (var_a > 0 or var_b > 0):
                    pooled_var = (((arr_a.size - 1) * var_a) + ((arr_b.size - 1) * var_b)) / denom
                else:
                    pooled_var = (var_a + var_b) / 2.0
                if pooled_var <= 0 or not math.isfinite(pooled_var):
                    return float('nan')
                if diff == 0:
                    return float('-inf')
                snr_linear = (diff / math.sqrt(pooled_var)) ** 2
                if snr_linear <= 0 or not math.isfinite(snr_linear):
                    return float('nan')
                return 10.0 * math.log10(snr_linear)

            def _paired_snr_db_summary(stat_a: Mapping[str, Any], stat_b: Mapping[str, Any]) -> float:
                count_a = float(stat_a.get("count", 0.0))
                count_b = float(stat_b.get("count", 0.0))
                if count_a <= 0 or count_b <= 0:
                    return float('nan')
                mean_a = float(stat_a.get("mean", float('nan')))
                mean_b = float(stat_b.get("mean", float('nan')))
                if not (math.isfinite(mean_a) and math.isfinite(mean_b)):
                    return float('nan')
                diff = abs(mean_b - mean_a)
                if diff == 0:
                    return float('-inf')
                var_a = _summary_variance(stat_a)
                var_b = _summary_variance(stat_b)
                pooled_num = 0.0
                pooled_den = 0.0
                if count_a > 1 and math.isfinite(var_a):
                    pooled_num += (count_a - 1.0) * var_a
                    pooled_den += count_a - 1.0
                if count_b > 1 and math.isfinite(var_b):
                    pooled_num += (count_b - 1.0) * var_b
                    pooled_den += count_b - 1.0
                if pooled_den > 0:
                    pooled_var = pooled_num / pooled_den
                else:
                    finite_terms = [v for v in (var_a, var_b) if math.isfinite(v) and v >= 0.0]
                    if not finite_terms:
                        return float('nan')
                    pooled_var = float(np.mean(finite_terms))
                if pooled_var <= 0:
                    return float('nan')
                snr_linear = (diff ** 2) / pooled_var
                if snr_linear <= 0 or not math.isfinite(snr_linear):
                    return float('nan')
                return 10.0 * math.log10(snr_linear)

            def _min_adjacent_snr(level_map: Dict[int, List[float]]) -> Tuple[float, Optional[Tuple[int, int]]]:
                if not level_map:
                    return float('nan'), None
                keys = sorted(level_map.keys())
                best = float('nan')
                best_pair: Optional[Tuple[int, int]] = None
                for idx_level in range(len(keys) - 1):
                    a_key = keys[idx_level]
                    b_key = keys[idx_level + 1]
                    snr_val = _paired_snr_db(level_map.get(a_key, []), level_map.get(b_key, []))
                    if math.isnan(snr_val):
                        continue
                    if math.isnan(best) or snr_val < best:
                        best = snr_val
                        best_pair = (a_key, b_key)
                return best, best_pair

            def _min_adjacent_snr_summary(summary_map: Dict[int, Dict[str, float]]) -> Tuple[float, Optional[Tuple[int, int]]]:
                if not summary_map:
                    return float('nan'), None
                keys = sorted(summary_map.keys())
                best = float('nan')
                best_pair: Optional[Tuple[int, int]] = None
                for idx_level in range(len(keys) - 1):
                    a_key = keys[idx_level]
                    b_key = keys[idx_level + 1]
                    snr_val = _paired_snr_db_summary(summary_map[a_key], summary_map[b_key])
                    if math.isnan(snr_val):
                        continue
                    if math.isnan(best) or snr_val < best:
                        best = snr_val
                        best_pair = (a_key, b_key)
                return best, best_pair

            all_a: List[float] = []
            all_b: List[float] = []
            charge_da_all: List[float] = []
            charge_sero_all: List[float] = []
            current_da_all: List[float] = []
            current_sero_all: List[float] = []
            csk_level_samples: Dict[int, List[float]] = {}
            csk_level_summary: Dict[int, Dict[str, float]] = {}
            hybrid_amp_samples: Dict[int, List[float]] = {}
            hybrid_amp_summary: Dict[Tuple[str, int], Dict[str, float]] = {}
            for r in results:
                all_a.extend(cast(List[float], r.get('stats_da', [])))
                all_b.extend(cast(List[float], r.get('stats_sero', [])))
                charge_da_all.extend(cast(List[float], r.get('stats_charge_da', [])))
                charge_sero_all.extend(cast(List[float], r.get('stats_charge_sero', [])))
                current_da_all.extend(cast(List[float], r.get('stats_current_da', [])))
                current_sero_all.extend(cast(List[float], r.get('stats_current_sero', [])))

                level_summary = r.get('csk_level_stats')
                if isinstance(level_summary, Mapping):
                    for key_raw, payload in level_summary.items():
                        if not isinstance(payload, Mapping):
                            continue
                        try:
                            level_idx = int(key_raw)
                        except (TypeError, ValueError):
                            continue
                        _merge_stat_summary(csk_level_summary, level_idx, payload)

                levels = r.get('stats_csk_levels', [])
                if isinstance(levels, list):
                    for idx_level, samples in enumerate(levels):
                        if not isinstance(samples, list):
                            continue
                        filtered = _coerce_finite(samples)
                        if filtered:
                            csk_level_samples.setdefault(idx_level, []).extend(filtered)

                amp_summary = r.get('hybrid_amp_stats')
                if isinstance(amp_summary, Mapping):
                    for key_raw, payload in amp_summary.items():
                        if not isinstance(payload, Mapping):
                            continue
                        try:
                            mol_label, amp_token = str(key_raw).split('_', 1)
                            amp_idx = int(amp_token)
                        except (ValueError, AttributeError):
                            continue
                        mol_norm = mol_label.upper()
                        _merge_stat_summary(hybrid_amp_summary, (mol_norm, amp_idx), payload)

                amp_stats = r.get('stats_hybrid_amp', [])
                if isinstance(amp_stats, list):
                    for bit_idx, samples in enumerate(amp_stats[:2]):
                        if not isinstance(samples, list):
                            continue
                        filtered = _coerce_finite(samples)
                        if filtered:
                            hybrid_amp_samples.setdefault(bit_idx, []).extend(filtered)

            snr_lin = calculate_snr_from_stats(all_a, all_b) if all_a and all_b else 0.0
            snr_legacy_db = (10.0 * float(np.log10(snr_lin))) if snr_lin > 0 else float('nan')
            snr_db = snr_legacy_db
            mode_upper = mode_name.upper()
            snr_db_csk_min = float('nan')
            snr_csk_method = ''
            snr_csk_boundary: str = ''
            csk_summary_export = {level: _export_summary(summary) for level, summary in csk_level_summary.items()}
            snr_db_mosk = snr_legacy_db if mode_upper in ("MOSK", "HYBRID") else float('nan')
            snr_db_amp = float('nan')
            snr_amp_method = ''
            snr_amp_worst = ''
            hybrid_amp_detail: Dict[str, float] = {}
            hybrid_amp_summary_export = {
                f"{mol}_{amp}": _export_summary(summary)
                for (mol, amp), summary in hybrid_amp_summary.items()
            }

            if csk_level_summary:
                snr_db_csk_min, boundary = _min_adjacent_snr_summary(csk_level_summary)
                if not math.isnan(snr_db_csk_min):
                    snr_csk_method = 'summary'
                    if boundary:
                        snr_csk_boundary = f"{boundary[0]}-{boundary[1]}"
            if math.isnan(snr_db_csk_min):
                fallback_csk, fallback_pair = _min_adjacent_snr(csk_level_samples)
                if not math.isnan(fallback_csk):
                    snr_db_csk_min = fallback_csk
                    snr_csk_method = 'samples' if snr_csk_method == '' else snr_csk_method
                    if not snr_csk_boundary and fallback_pair:
                        snr_csk_boundary = f"{fallback_pair[0]}-{fallback_pair[1]}"

            if hybrid_amp_summary:
                grouped: Dict[str, Dict[int, Dict[str, float]]] = {}
                for (mol, amp_idx), summary in hybrid_amp_summary.items():
                    grouped.setdefault(mol, {})[amp_idx] = summary
                worst_val = float('nan')
                worst_mol = ''
                for mol_label, entries in grouped.items():
                    if 0 in entries and 1 in entries:
                        snr_val = _paired_snr_db_summary(entries[0], entries[1])
                        if not math.isnan(snr_val):
                            hybrid_amp_detail[f"{mol_label}_0-1_db"] = snr_val
                            if math.isnan(worst_val) or snr_val < worst_val:
                                worst_val = snr_val
                                worst_mol = mol_label
                if not math.isnan(worst_val):
                    snr_db_amp = worst_val
                    snr_amp_method = 'summary'
                    snr_amp_worst = worst_mol
            if math.isnan(snr_db_amp):
                bit0_samples = hybrid_amp_samples.get(0, [])
                bit1_samples = hybrid_amp_samples.get(1, [])
                if bit0_samples and bit1_samples:
                    snr_db_amp = _paired_snr_db(bit0_samples, bit1_samples)
                    if not math.isnan(snr_db_amp) and not snr_amp_method:
                        snr_amp_method = 'samples'

            if mode_upper == "CSK":
                if not math.isnan(snr_db_csk_min):
                    snr_db = snr_db_csk_min
            elif mode_upper == "HYBRID":
                candidates = [val for val in (snr_db_mosk, snr_db_amp) if not math.isnan(val)]
                if candidates:
                    snr_db = min(candidates)
            mean_charge_da = float(np.nanmean(charge_da_all)) if charge_da_all else float('nan')
            mean_charge_sero = float(np.nanmean(charge_sero_all)) if charge_sero_all else float('nan')
            mean_current_da = float(np.nanmean(current_da_all)) if current_da_all else float('nan')
            mean_current_sero = float(np.nanmean(current_sero_all)) if current_sero_all else float('nan')
            delta_q_diff = float(mean_charge_da - mean_charge_sero) if (
                np.isfinite(mean_charge_da) and np.isfinite(mean_charge_sero)
            ) else float('nan')
            delta_i_diff_raw = float(mean_current_da - mean_current_sero) if (
                np.isfinite(mean_current_da) and np.isfinite(mean_current_sero)
            ) else float('nan')
    
            # ISI context
            isi_enabled = any(bool(r.get('isi_enabled', False)) for r in results)
            isi_memory_symbols = int(np.nanmedian([float(r.get('isi_memory_symbols', np.nan)) for r in results if r is not None])) if isi_enabled else 0
            symbol_period_s = float(np.nanmedian([float(r.get('symbol_period_s', np.nan)) for r in results]))
            decision_window_s = float(np.nanmedian([float(r.get('decision_window_s', np.nan)) for r in results]))
            isi_overlap_mean = float(np.nanmean([float(r.get('isi_overlap_ratio', 0.0)) for r in results]))
    
            # Stage 14: aggregate noise sigmas across seeds
            def _to_float(value: Any) -> float:
                if value is None:
                    return float('nan')
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float('nan')

            def _prefer_measured(record: Dict[str, Any], base_key: str) -> float:
                measured_key = f"{base_key}_measured"
                val_f = _to_float(record.get(measured_key))
                if math.isfinite(val_f):
                    return val_f
                return _to_float(record.get(base_key))

            ns_da = [_prefer_measured(r, 'noise_sigma_da') for r in results]
            ns_sero = [_prefer_measured(r, 'noise_sigma_sero') for r in results]
            ns_diff = [_prefer_measured(r, 'noise_sigma_I_diff') for r in results]
            ns_diff_charge = [_prefer_measured(r, 'noise_sigma_diff_charge') for r in results]
            ns_thermal = [_prefer_measured(r, 'noise_sigma_thermal') for r in results]
            ns_flicker = [_prefer_measured(r, 'noise_sigma_flicker') for r in results]
            ns_drift = [_prefer_measured(r, 'noise_sigma_drift') for r in results]
            ns_single = [_prefer_measured(r, 'noise_sigma_single') for r in results]
            thermal_fracs = [_prefer_measured(r, 'noise_thermal_fraction') for r in results]
            flicker_fracs = [_prefer_measured(r, 'noise_flicker_fraction') for r in results]
            drift_fracs = [_prefer_measured(r, 'noise_drift_fraction') for r in results]
            ns_thermal_single = [_to_float(r.get('noise_sigma_thermal_single_measured')) for r in results]
            ns_flicker_single = [_to_float(r.get('noise_sigma_flicker_single_measured')) for r in results]
            ns_drift_single = [_to_float(r.get('noise_sigma_drift_single_measured')) for r in results]
            ctrl_reduction_vals = [_to_float(r.get('noise_ctrl_reduction_fraction_mean')) for r in results]
            rho_pre_vals = [float(r.get('rho_pre_ctrl', float('nan'))) for r in results]
            rho_noise_vals = [float(r.get('noise_rho_measured', float('nan'))) for r in results]
            sigma_sample_sizes = [_to_float(r.get('noise_sigma_sample_size')) for r in results]
            burst_shapes = [str(r.get('burst_shape', '')) for r in results if r.get('burst_shape') not in (None, '')]
            trelease_vals = [float(r.get('T_release_ms', float('nan'))) for r in results]
            arr_da = np.asarray(ns_da, dtype=float)
            arr_rho_pre = np.asarray(rho_pre_vals, dtype=float)
            arr_rho_noise = np.asarray(rho_noise_vals, dtype=float)
            arr_sero = np.asarray(ns_sero, dtype=float)
            arr_diff = np.asarray(ns_diff, dtype=float)
            arr_diff_charge = np.asarray(ns_diff_charge, dtype=float)
            arr_thermal = np.asarray(ns_thermal, dtype=float)
            arr_flicker = np.asarray(ns_flicker, dtype=float)
            arr_drift = np.asarray(ns_drift, dtype=float)
            arr_single = np.asarray(ns_single, dtype=float)
            arr_thermal_frac = np.asarray(thermal_fracs, dtype=float)
            arr_flicker_frac = np.asarray(flicker_fracs, dtype=float)
            arr_drift_frac = np.asarray(drift_fracs, dtype=float)
            arr_thermal_single = np.asarray(ns_thermal_single, dtype=float)
            arr_flicker_single = np.asarray(ns_flicker_single, dtype=float)
            arr_drift_single = np.asarray(ns_drift_single, dtype=float)
            arr_ctrl_reduction = np.asarray(ctrl_reduction_vals, dtype=float)
            arr_sigma_samples = np.asarray(sigma_sample_sizes, dtype=float)
            med_sigma_da = float(np.nanmedian(arr_da)) if np.isfinite(arr_da).any() else float('nan')
            med_sigma_sero = float(np.nanmedian(arr_sero)) if np.isfinite(arr_sero).any() else float('nan')
            med_sigma_diff = float(np.nanmedian(arr_diff)) if np.isfinite(arr_diff).any() else float('nan')
            med_sigma_diff_charge = float(np.nanmedian(arr_diff_charge)) if np.isfinite(arr_diff_charge).any() else float('nan')
            med_sigma_single = float(np.nanmedian(arr_single)) if np.isfinite(arr_single).any() else float('nan')
            med_sigma_thermal = float(np.nanmedian(arr_thermal)) if np.isfinite(arr_thermal).any() else float('nan')
            med_sigma_flicker = float(np.nanmedian(arr_flicker)) if np.isfinite(arr_flicker).any() else float('nan')
            med_sigma_drift = float(np.nanmedian(arr_drift)) if np.isfinite(arr_drift).any() else float('nan')
            med_thermal_frac = float(np.nanmedian(arr_thermal_frac)) if np.isfinite(arr_thermal_frac).any() else float('nan')
            med_flicker_frac = float(np.nanmedian(arr_flicker_frac)) if np.isfinite(arr_flicker_frac).any() else float('nan')
            med_drift_frac = float(np.nanmedian(arr_drift_frac)) if np.isfinite(arr_drift_frac).any() else float('nan')
            med_sigma_thermal_single = float(np.nanmedian(arr_thermal_single)) if np.isfinite(arr_thermal_single).any() else float('nan')
            med_sigma_flicker_single = float(np.nanmedian(arr_flicker_single)) if np.isfinite(arr_flicker_single).any() else float('nan')
            med_sigma_drift_single = float(np.nanmedian(arr_drift_single)) if np.isfinite(arr_drift_single).any() else float('nan')
            med_ctrl_reduction = float(np.nanmedian(arr_ctrl_reduction)) if np.isfinite(arr_ctrl_reduction).any() else float('nan')
            med_rho_pre = float(np.nanmedian(arr_rho_pre)) if np.isfinite(arr_rho_pre).any() else float('nan')
            med_rho_noise = float(np.nanmedian(arr_rho_noise)) if np.isfinite(arr_rho_noise).any() else float('nan')
            med_sigma_sample_size = float(np.nanmedian(arr_sigma_samples)) if np.isfinite(arr_sigma_samples).any() else float('nan')

            sigma_diff_current_vals: List[float] = []
            for sigma_charge_val, res in zip(ns_diff_charge, results):
                det_win = float(res.get('detection_window_s', decision_window_s))
                if det_win > 0 and math.isfinite(sigma_charge_val):
                    sigma_diff_current_vals.append(float(sigma_charge_val) / det_win)
                else:
                    sigma_diff_current_vals.append(float('nan'))
            arr_diff_current = np.asarray(sigma_diff_current_vals, dtype=float)
            med_sigma_diff_current = float(np.nanmedian(arr_diff_current)) if np.isfinite(arr_diff_current).any() else float('nan')
    
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
            delta_Q_diff = float(delta_q_diff) if np.isfinite(delta_q_diff) else float('nan')
            delta_over_sigma_Q = float(delta_Q_diff / med_sigma_diff_charge) if (
                np.isfinite(delta_Q_diff) and np.isfinite(med_sigma_diff_charge) and med_sigma_diff_charge > 0
            ) else float('nan')
            delta_I_diff = float(delta_i_diff_raw) if np.isfinite(delta_i_diff_raw) else float('nan')
            delta_over_sigma_I = float(delta_I_diff / med_sigma_diff_current) if (
                np.isfinite(delta_I_diff) and np.isfinite(med_sigma_diff_current) and med_sigma_diff_current > 0
            ) else float('nan')
    
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
            med_t_release = _finite_median(np.asarray(trelease_vals, dtype=float))
            burst_shape_val = burst_shapes[0] if burst_shapes else ''
    
            mode_name = cfg['pipeline']['modulation']
            mode_upper = mode_name.upper()
            if mode_upper == "MOSK":
                snr_semantics = "MoSK contrast statistic (sign-aware DA vs SERO)"
            elif mode_upper == "CSK":
                if math.isnan(snr_db_csk_min):
                    snr_semantics = "CSK half-constellation SNR (insufficient boundary samples)"
                else:
                    method_label = snr_csk_method or "adjacent levels"
                    snr_semantics = f"CSK min-boundary SNR ({method_label})"
            elif mode_upper == "HYBRID":
                parts: List[str] = []
                if not math.isnan(snr_db_mosk):
                    parts.append("MoSK")
                if not math.isnan(snr_db_amp):
                    parts.append("amplitude")
                if parts:
                    parts_label = " & ".join(parts)
                    suffix_parts: List[str] = []
                    if snr_amp_method:
                        suffix_parts.append(f"amp={snr_amp_method}")
                    if snr_csk_method and not math.isnan(snr_db_csk_min):
                        suffix_parts.append(f"gate={snr_csk_method}")
                    suffix = f" [{', '.join(suffix_parts)}]" if suffix_parts else ""
                    snr_semantics = f"Hybrid min({parts_label}) SNR{suffix}"
                else:
                    snr_semantics = "Hybrid MoSK SNR (amplitude statistic unavailable)"
            else:
                snr_semantics = "SNR statistic (dual-channel combiner)"
    
            # Extract rho_cc_measured safely before creating the dictionary
            rho_cc_raw = thresholds_map.get(v, {}).get('rho_cc_measured', float('nan'))
            
            current_distance = float(cfg['pipeline'].get('distance_um', float('nan')))
            current_nm = float(cfg['pipeline'].get('Nm_per_symbol', float('nan')))
            if sweep_param == 'pipeline.distance_um':
                current_distance = float(v)
            if sweep_param == 'pipeline.Nm_per_symbol':
                current_nm = float(v)
            detector_mode = str(cfg['pipeline'].get('detector_mode', 'zscore')).lower()
            for r in results:
                if r.get('detector_mode'):
                    detector_mode = str(r.get('detector_mode')).lower()
                    break

            row: Dict[str, Any] = {
                sweep_param: v,
                'ser': ser,
                'snr_db': snr_db,
                'snr_semantics': snr_semantics,
                'snr_db_csk_min': snr_db_csk_min,
                'snr_csk_method': snr_csk_method,
                'snr_csk_boundary_min': snr_csk_boundary,
                'snr_db_mosk': snr_db_mosk,
                'snr_db_amp': snr_db_amp,
                'snr_amp_method': snr_amp_method,
                'snr_amp_worst_molecule': snr_amp_worst,
                'csk_level_stats_summary': csk_summary_export,
                'hybrid_amp_stats_summary': hybrid_amp_summary_export,
                'hybrid_amp_detail_db': hybrid_amp_detail,
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
                'noise_sigma_diff_charge': med_sigma_diff_charge,
                'noise_sigma_single': med_sigma_single,
                'noise_sigma_thermal': med_sigma_thermal,
                'noise_sigma_flicker': med_sigma_flicker,
                'noise_sigma_drift': med_sigma_drift,
                'noise_thermal_fraction': med_thermal_frac,
                'noise_flicker_fraction': med_flicker_frac,
                'noise_drift_fraction': med_drift_frac,
                'noise_sigma_thermal_single_measured': med_sigma_thermal_single,
                'noise_sigma_flicker_single_measured': med_sigma_flicker_single,
                'noise_sigma_drift_single_measured': med_sigma_drift_single,
                'noise_ctrl_reduction_fraction_mean': med_ctrl_reduction,
                'noise_rho_measured': med_rho_noise,
                'noise_sigma_sample_size': med_sigma_sample_size,
                'rho_pre_ctrl': med_rho_pre,
                'I_dc_used_A': med_I_dc,
                'V_g_bias_V_used': med_V_g_bias,
                'gm_S': med_gm,
                'C_tot_F': med_c_tot,
                'delta_over_sigma': med_delta_stat,
                'delta_Q_diff': delta_Q_diff,
                'sigma_Q_diff': med_sigma_diff_charge,
                'delta_over_sigma_Q': delta_over_sigma_Q,
                'delta_I_diff': delta_I_diff,
                'sigma_I_diff': med_sigma_diff_current,
                'delta_over_sigma_I': delta_over_sigma_I,
                'distance_um': current_distance,
                'Nm_per_symbol': current_nm,
                'rho_cc': float(cfg.get('noise', {}).get('rho_between_channels_after_ctrl', 0.0)),
                'use_ctrl': bool(thresholds_map.get(v, {}).get('use_control_channel',
                                cfg['pipeline'].get('use_control_channel', True))),
                'ctrl_auto_applied': bool(thresholds_map.get(v, {}).get('ctrl_auto_applied', False)),
                'rho_cc_measured': float(rho_cc_raw) if isinstance(rho_cc_raw, (int, float)) else float('nan'),
                'mode': mode_name,
                'detector_mode': detector_mode,
                'ser_ci_low': float(ci_low),
                'ser_ci_high': float(ci_high),
                'burst_shape': burst_shape_val,
                'T_release_ms': med_t_release,
            }
            freeze_flags = [bool(r.get('calibration_frozen', False)) for r in results]
            baseline_ids = {str(r.get('calibration_baseline_id', '')).strip() for r in results if r.get('calibration_baseline_id')}
            if freeze_snapshot and freeze_snapshot.get('baseline_id'):
                baseline_ids.add(str(freeze_snapshot['baseline_id']))
            row['calibration_frozen'] = bool(freeze_snapshot) or any(freeze_flags)
            row['calibration_baseline_id'] = next(iter(baseline_ids)) if baseline_ids else ''
            if freeze_snapshot:
                row['calibration_baseline_param'] = freeze_snapshot.get('sweep_param', '')
                row['calibration_baseline_value'] = freeze_snapshot.get('baseline_value', float('nan'))
            else:
                row['calibration_baseline_param'] = ''
                row['calibration_baseline_value'] = float('nan')
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
    if persist_csv is not None:
        flush_staged_rows(persist_csv)
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
                _apply_thresholds_into_cfg(cfg_p, th)

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

        mode = str(cfg_test['pipeline'].get('modulation', 'MoSK')).upper()
        th_override: Dict[str, Any] = {}
        if mode in ('CSK', 'HYBRID'):
            try:
                cfg_cal = deepcopy(cfg_test)
                th_override = calibrate_thresholds_cached(cfg_cal, list(range(4)))
            except Exception:
                th_override = {}

        debug_entry: Optional[Dict[str, Any]] = None
        if LOD_DEBUG_ENABLED:
            # NOTE: LoD debug instrumentation (remove once diagnostics conclude)
            meta = th_override.get('__meta__') if isinstance(th_override, dict) else None
            def _maybe_float(val: Any) -> Optional[float]:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
            debug_entry = {
                "phase": "lod_quick_ser_start",
                "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
                "nm": int(nm),
                "mode": mode,
                "sequence_length": int(cfg_test['pipeline'].get('sequence_length', 0)),
                "decision_window_s": float(cfg_test.get('detection', {}).get('decision_window_s', float('nan'))),
                "decision_window_policy": str(cfg_test.get('detection', {}).get('decision_window_policy', 'unknown')),
                "threshold_meta": {
                    "decision_window_used": _maybe_float(meta.get('decision_window_used')),
                    "Ts": _maybe_float(meta.get('Ts')),
                    "lod_nm": _maybe_float(meta.get('Nm')),
                } if isinstance(meta, dict) else None,
                "quick_seeds": [int(s) for s in quick_seeds],
            }

        results = []
        seed_logs: List[Dict[str, Any]] = []
        for seed in quick_seeds:
            res = run_param_seed_combo(cfg_test, 'pipeline.Nm_per_symbol', nm, seed,
                                       sweep_name="bracket_validation",
                                       cache_tag=cache_tag,
                                       thresholds_override=th_override if th_override else None)
            if res:
                results.append(res)
                if debug_entry is not None:
                    seed_logs.append({
                        "seed": int(seed),
                        "errors": int(res.get('errors', 0)),
                        "sequence_length": int(cfg_test['pipeline']['sequence_length']),
                        "ser": float(res.get('ser', res.get('SER', float('nan')))),
                        "decision_window_s": float(res.get('decision_window_s', float('nan'))),
                        "symbol_period_s": float(res.get('symbol_period_s', float('nan'))),
                    })

        if not results:
            if debug_entry is not None:
                debug_entry.update({
                    "phase": "lod_quick_ser_result",
                    "result_ser": float('nan'),
                    "seed_results": seed_logs,
                    "status": "no_results",
                })
                _lod_debug_log(debug_entry)
            return 1.0  # Assume failure if no results

        total_symbols = len(results) * cfg_test['pipeline']['sequence_length']
        total_errors = sum(int(r.get('errors', 0)) for r in results)
        ser_val = total_errors / total_symbols if total_symbols > 0 else 1.0

        if debug_entry is not None:
            debug_entry.update({
                "phase": "lod_quick_ser_result",
                "result_ser": float(ser_val),
                "total_errors": int(total_errors),
                "total_symbols": int(total_symbols),
                "seed_results": seed_logs,
                "status": "ok",
            })
            _lod_debug_log(debug_entry)

        return ser_val
    
    # Check lower bound
    ser_min = _quick_ser(nm_min)
    if LOD_DEBUG_ENABLED:
        _lod_debug_log({
            "phase": "lod_quick_ser_lower_bound",
            "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
            "nm": int(nm_min),
            "ser": float(ser_min),
            "target_ser": float(target_ser),
        })
    if ser_min <= target_ser:
        # Lower bound too good, push it down
        while nm_min > 50 and ser_min <= target_ser:
            nm_min = max(50, int(nm_min / 2))
            ser_min = _quick_ser(nm_min)
    
    # Check upper bound
    ser_max = _quick_ser(nm_max)
    if LOD_DEBUG_ENABLED:
        _lod_debug_log({
            "phase": "lod_quick_ser_upper_bound",
            "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
            "nm": int(nm_max),
            "ser": float(ser_max),
            "target_ser": float(target_ser),
            "nm_ceiling": int(nm_ceiling),
        })
    if ser_max > target_ser:
        # Upper bound not good enough, try to grow it
        while nm_max < nm_ceiling and ser_max > target_ser:
            nm_max = min(nm_ceiling, nm_max * 2)
            ser_max = _quick_ser(nm_max)
        
        # If we hit ceiling and still can't achieve target
        if nm_max >= nm_ceiling and ser_max > target_ser:
            if LOD_DEBUG_ENABLED:
                _lod_debug_log({
                    "phase": "lod_quick_ser_ceiling",
                    "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
                    "nm_min": int(nm_min),
                    "nm_max": int(nm_max),
                    "ser_min": float(ser_min),
                    "ser_max": float(ser_max),
                    "target_ser": float(target_ser),
                })
            return nm_min, nm_max, "nm_ceiling_exhausted"

    # Final validation
    if ser_min <= target_ser or ser_max > target_ser:
        # Still invalid bracket
        if LOD_DEBUG_ENABLED:
            _lod_debug_log({
                "phase": "lod_quick_ser_bracket_failure",
                "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
                "nm_min": int(nm_min),
                "nm_max": int(nm_max),
                "ser_min": float(ser_min),
                "ser_max": float(ser_max),
                "target_ser": float(target_ser),
            })
        return nm_min, nm_max, "bracket_validation_failed"

    return nm_min, nm_max, None


def _default_bracket_from_guess(guess_nm: float,
                                distance_um: float,
                                nm_min: int,
                                nm_max: int) -> Tuple[int, int]:
    """Generate a conservative bracket around an Nm guess for the next LoD search."""
    if not math.isfinite(guess_nm) or guess_nm <= 0:
        return (nm_min, nm_max)
    if distance_um >= 175:
        lower_mult, upper_mult = 0.45, 1.85
    elif distance_um >= 125:
        lower_mult, upper_mult = 0.5, 1.7
    elif distance_um >= 75:
        lower_mult, upper_mult = 0.55, 1.55
    else:
        lower_mult, upper_mult = 0.6, 1.45
    lower = max(nm_min, max(50, int(guess_nm * lower_mult)))
    upper = min(nm_max, max(lower + 1, int(guess_nm * upper_mult)))
    return (lower, upper)


def _predict_lod_from_history(distance_um: float,
                              history_points: Sequence[Dict[str, Any]],
                              target_ser: float,
                              nm_min: int,
                              nm_max: int) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
    """
    Fit a simple log-linear model using historical (distance, Nm, SER) samples
    to predict the Nm required to hit the target SER at the requested distance.
    Returns (nm_guess, (nm_lower, nm_upper)) or (None, None) when insufficient data.
    """
    candidates: List[Tuple[float, float, float, float]] = []
    for entry in history_points:
        try:
            d_val = float(entry.get("distance_um", float("nan")))
            nm_val = float(entry.get("nm", 0.0))
            ser_val = float(entry.get("ser", float("nan")))
            n_seen = float(entry.get("n_seen", entry.get("samples", 0.0)))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(d_val) and math.isfinite(nm_val) and math.isfinite(ser_val)):
            continue
        if d_val <= 0 or nm_val <= 0:
            continue
        if ser_val <= 0.0:
            ser_val = 1e-6
        if ser_val >= 0.6:
            continue  # skip very loose points that provide little slope info
        candidates.append((d_val, nm_val, ser_val, max(n_seen, 1.0)))

    if len(candidates) < 3:
        return (None, None)

    # Prepare weighted least squares to favour samples with more observations
    X_rows: List[List[float]] = []
    y_vals: List[float] = []
    for d_val, nm_val, ser_val, n_seen in candidates:
        weight = math.sqrt(max(n_seen, 1.0))
        log_nm = math.log(nm_val)
        log_ser = math.log(max(ser_val, 1e-6))
        X_rows.append([weight, weight * log_nm, weight * d_val])
        y_vals.append(weight * log_ser)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)

    try:
        beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return (None, None)

    if rank < 3:
        return (None, None)

    a, b, c = (float(beta[0]), float(beta[1]), float(beta[2]))
    if not math.isfinite(b) or abs(b) < 1e-4 or b >= 0:
        return (None, None)

    log_target = math.log(max(target_ser, 1e-6))
    log_nm = (log_target - a - c * distance_um) / b
    if not math.isfinite(log_nm):
        return (None, None)
    nm_guess = math.exp(log_nm)
    nm_guess = max(float(nm_min), min(float(nm_max), nm_guess))

    # Estimate residual spread in log-space
    if residuals.size > 0 and len(candidates) > 3:
        dof = max(len(candidates) - 3, 1)
        sigma = math.sqrt(max(residuals[0] / dof, 1e-10))
    else:
        fitted = X @ beta
        residual_vec = y - fitted
        sigma = math.sqrt(max(float(np.mean(residual_vec ** 2)), 1e-10))

    delta_log_nm = sigma / max(abs(b), 1e-4)
    safety = max(delta_log_nm * 1.75, math.log(1.4))
    lower = int(max(nm_min, max(50.0, math.exp(log_nm - safety))))
    upper = int(min(nm_max, max(lower + 1, math.exp(log_nm + safety))))

    # Ensure monotonic consistency with nearest neighbours
    below = [nm for d, nm, ser, _ in candidates if d <= distance_um and ser <= target_ser * 1.5]
    above = [nm for d, nm, ser, _ in candidates if d >= distance_um and ser >= target_ser * 0.67]
    if below:
        lower = max(lower, int(0.75 * max(below)))
    if above:
        upper = min(upper, int(1.25 * min(above)))
        upper = max(upper, lower + 1)

    return (nm_guess, (lower, upper))


def _extend_history_with_lod_rows(history: List[Dict[str, Any]],
                                  df: Optional[pd.DataFrame],
                                  target_ser: float = 0.01) -> None:
    """Populate history list with (distance, Nm, SER) samples derived from existing CSV rows."""
    if df is None or df.empty:
        return
    for _, row in df.iterrows():
        try:
            dist = float(row.get('distance_um', float('nan')))
            nm_val = float(row.get('lod_nm', float('nan')))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(dist) and math.isfinite(nm_val)):
            continue
        if nm_val <= 0 or dist <= 0:
            continue
        ser_val = row.get('ser_at_lod', target_ser)
        try:
            ser_val_f = float(ser_val)
        except (TypeError, ValueError):
            ser_val_f = target_ser
        if not math.isfinite(ser_val_f) or ser_val_f <= 0:
            ser_val_f = target_ser
        samples = row.get('symbols_evaluated')
        try:
            n_seen = float(samples) if samples is not None else 0.0
        except (TypeError, ValueError):
            n_seen = 0.0
        history.append({
            "distance_um": dist,
            "nm": nm_val,
            "ser": max(ser_val_f, 1e-6),
            "n_seen": max(n_seen, 1.0),
            "source": "lod_csv",
        })


def _normalize_lod_trace_entries(distance_um: float,
                                 entries: Sequence[Dict[str, Any]],
                                 default_target: float = 0.01) -> List[Dict[str, Any]]:
    """Normalize raw trace entries into a consistent schema for persistence/regression."""
    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            nm_val = float(entry.get("nm", float("nan")))
            ser_val = float(entry.get("ser", float("nan")))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(nm_val) and math.isfinite(ser_val)):
            continue
        if nm_val <= 0:
            continue
        try:
            n_seen = float(entry.get("n_seen", entry.get("samples", 0.0)))
        except (TypeError, ValueError):
            n_seen = 0.0
        try:
            k_err = float(entry.get("k_err", 0.0))
        except (TypeError, ValueError):
            k_err = 0.0
        timestamp = entry.get("timestamp", time.time())
        try:
            timestamp_f = float(timestamp)
        except (TypeError, ValueError):
            timestamp_f = time.time()
        normalized.append({
            "distance_um": float(distance_um),
            "nm": nm_val,
            "ser": max(ser_val, 1e-6),
            "n_seen": max(n_seen, 0.0),
            "k_err": max(k_err, 0.0),
            "phase": str(entry.get("phase", "unknown")),
            "status": str(entry.get("status", "")),
            "timestamp": timestamp_f,
            "target_ser": float(entry.get("target_ser", default_target)),
        })
    return normalized


def _lod_history_cache_path(mode: str, use_ctrl: bool, suffix: str = "") -> Path:
    ctrl_seg = "wctrl" if use_ctrl else "noctrl"
    base = project_root / "results" / "cache" / mode.lower() / "lod_history"
    base.mkdir(parents=True, exist_ok=True)
    suffix_clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in suffix)
    return base / f"lod_trace_{ctrl_seg}{suffix_clean}.jsonl"


def _load_lod_history_cache(mode: str, use_ctrl: bool, suffix: str = "") -> List[Dict[str, Any]]:
    path = _lod_history_cache_path(mode, use_ctrl, suffix)
    if not path.exists():
        return []
    points: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    points.append(obj)
    except Exception:
        return []
    return points


def _append_lod_history_cache(mode: str,
                              use_ctrl: bool,
                              suffix: str,
                              entries: Sequence[Dict[str, Any]]) -> None:
    if not entries:
        return
    path = _lod_history_cache_path(mode, use_ctrl, suffix)
    try:
        with path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(_json_safe(entry)) + "\n")
    except Exception:
        pass
def find_lod_for_ser(cfg_base: Dict[str, Any], seeds: List[int],
                     target_ser: float = 0.01,
                     debug_calibration: bool = False,
                     progress_cb: Optional[Any] = None,
                     resume: bool = False,
                     cache_tag: Optional[str] = None) -> Tuple[Union[int, float], float, int, Dict[str, Any]]:
    nm_min = int(cfg_base['pipeline'].get('lod_nm_min', 50))
    # NEW: configurable ceiling with fallback for backward compatibility
    nm_ceiling = int(cfg_base.get('pipeline', {}).get('lod_nm_max', 
                    cfg_base.get('lod_max_nm', 1000000)))
    nm_max = nm_ceiling
    nm_max_default = nm_ceiling

    lod_delta = float(cfg_base.get("_stage13_lod_delta", 1e-4))
    ci_margin = float(cfg_base.get('_lod_ci_margin', 0.2))
    ci_margin = min(max(ci_margin, 0.0), 0.9)
    wilson_delta = max(min(lod_delta, 0.2), 1e-10)
    try:
        wilson_z = float(NormalDist().inv_cdf(1.0 - wilson_delta / 2.0))
        if not math.isfinite(wilson_z) or wilson_z <= 0:
            wilson_z = 3.0
    except Exception:
        wilson_z = 3.0

    cfg_base['_lod_eval_trace'] = []
    eval_trace: List[Dict[str, Any]] = cfg_base['_lod_eval_trace']

    def _record_trace_entry(nm_value: int,
                            phase: str,
                            k_val: int,
                            n_val: int,
                            ser_val: float,
                            seeds_completed: int,
                            extra: Optional[Dict[str, Any]] = None) -> None:
        entry: Dict[str, Any] = {
            "nm": int(nm_value),
            "phase": phase,
            "k_err": int(k_val),
            "n_seen": int(n_val),
            "ser": float(ser_val),
            "seeds_completed": int(seeds_completed),
            "timestamp": time.time(),
            "distance_um": float(cfg_base['pipeline'].get('distance_um', 0.0)),
            "target_ser": float(target_ser),
        }
        if extra:
            entry.update(extra)
        entry.setdefault("status", "pass" if ser_val <= target_ser else "fail")
        eval_trace.append(entry)

    metrics: Dict[str, Any] = {
        "calibration_s": 0.0,
        "simulation_s": 0.0,
        "downstep_s": 0.0,
        "overhead_s": 0.0,
        "seeds_simulated": 0,
        "iterations": 0,
    }

    def _final_metrics() -> Dict[str, Any]:
        metrics["total_s"] = (
            float(metrics["calibration_s"])
            + float(metrics["simulation_s"])
            + float(metrics["overhead_s"])
        )
        return metrics

    warm_lower = int(cfg_base.get('_warm_bracket_min', 0))
    warm_upper = int(cfg_base.get('_warm_bracket_max', 0))
    if warm_lower > 0:
        nm_min = max(nm_min, warm_lower)
    if warm_upper > 0:
        nm_max = min(nm_max, warm_upper)
    if nm_min > nm_max:
        nm_min, nm_max = max(50, nm_min), max(51, nm_min + 1)

    if LOD_DEBUG_ENABLED:
        # NOTE: LoD debug instrumentation (remove once diagnostics conclude)
        _lod_debug_log({
            "phase": "lod_find_start",
            "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
            "nm_min_initial": int(nm_min),
            "nm_max_initial": int(nm_max),
            "nm_ceiling": int(nm_ceiling),
            "target_ser": float(target_ser),
            "seeds": [int(s) for s in seeds],
            "resume": bool(resume),
        })

    # NEW: Try analytic bracketing if enabled (experimental feature)
    analytic_bracket_cache = None  # Cache for analytic bracket result
    if cfg_base.get('_analytic_lod_bracket', False):
        analytic_bracket_cache = _analytic_lod_bracket(cfg_base, seeds, target_ser)
        nm_min_analytic, nm_max_analytic = analytic_bracket_cache
        if nm_min_analytic > 0 and nm_max_analytic > nm_min_analytic:
            nm_min = max(nm_min, nm_min_analytic)
            nm_max = min(nm_max, nm_max_analytic)
            print(f"    📊 Using analytic bracket: [{nm_min} - {nm_max}]")
            if LOD_DEBUG_ENABLED:
                _lod_debug_log({
                    "phase": "lod_find_analytic_bracket",
                    "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
                    "nm_min": int(nm_min),
                    "nm_max": int(nm_max),
                    "analytic_bounds": [int(nm_min_analytic), int(nm_max_analytic)],
                })

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
                if LOD_DEBUG_ENABLED:
                    _lod_debug_log({
                        "phase": "lod_find_warm_intersect",
                        "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
                        "warm_guess": int(warm),
                        "nm_min": int(nm_min),
                        "nm_max": int(nm_max),
                        "upper_from_warm": int(upper_from_warm),
                        "upper_from_analytic": int(upper_from_analytic),
                    })
        else:
            # Pure warm-start without analytic constraints
            nm_max = min(nm_max_default, int(mult * warm))
            print(f"    🔥 Warm-start bracket: [{nm_min} - {nm_max}]")
            if LOD_DEBUG_ENABLED:
                _lod_debug_log({
                    "phase": "lod_find_warm_only",
                    "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
                    "warm_guess": int(warm),
                    "nm_min": int(nm_min),
                    "nm_max": int(nm_max),
                })
    
    lod_nm: float = float('nan')
    best_ser: float = 1.0
    best_nm: Optional[int] = None  # NEW: Track the Nm that gave best_ser
    dist_um = cfg_base['pipeline'].get('distance_um', 0)
    mode_name = cfg_base['pipeline']['modulation']
    use_ctrl = bool(cfg_base['pipeline'].get('use_control_channel', True))
    
    # Track actual progress increments
    progress_count = 0
    
    # Persisted thresholds across resumes
    th_cache: Dict[int, Dict[str, Union[float, List[float], str]]] = {}
    
    # Load prior state if resuming
    state = _lod_state_load(mode_name, float(dist_um), use_ctrl) if resume else None

    if state:
        if LOD_DEBUG_ENABLED:
            _lod_debug_log({
                "phase": "lod_find_resume_state",
                "distance_um": float(dist_um),
                "state_keys": list(state.keys()),
            })
        # 1) Fast exit when a previous run already marked 'done' (robust to NaN)
        nm_min_state = state.get("nm_min")
        nm_max_state = state.get("nm_max")
        if (state.get("done") and 
            nm_min_state is not None and nm_max_state is not None and
            all(isinstance(x, (int, float)) and math.isfinite(x) for x in (nm_min_state, nm_max_state)) and
            int(nm_min_state) == int(nm_max_state) and int(nm_min_state) > 0):
            lod_nm = int(nm_min_state)
            print(f"    ✔ Resume: LoD already found in previous run → {lod_nm}")
            if LOD_DEBUG_ENABLED:
                _lod_debug_log({
                    "phase": "lod_find_resume_done",
                    "distance_um": float(dist_um),
                    "lod_nm": int(lod_nm),
                })
            return lod_nm, target_ser, 0, _final_metrics()

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
            _lod_state_save(mode_name, float(dist_um), use_ctrl, {"tested": {}, "th_cache": {}})
            nm_min = cfg_base['pipeline'].get('lod_nm_min', 50)
            nm_max = nm_ceiling
        else:
            print(f"    ↩️  Resuming LoD search @ {dist_um}μm: range {nm_min}-{nm_max}")
        
        stored_th = state.get("th_cache")
        if isinstance(stored_th, dict):
            for key, payload in stored_th.items():
                try:
                    nm_key = int(float(key))
                except Exception:
                    continue
                if isinstance(payload, dict):
                    th_cache[nm_key] = payload
    
    # Extract CTRL state for debug logging
    ctrl_str = "CTRL" if use_ctrl else "NoCtrl"

    def _get_th(nm: int):
        if nm in th_cache:
            return th_cache[nm]
        cfg_tmp = deepcopy(cfg_base)
        cfg_tmp['pipeline']['Nm_per_symbol'] = nm
        cfg_tmp.setdefault('pipeline', {}).pop('_frozen_noise', None)
        cfg_tmp.pop('_prefer_distance_freeze', None)
        t0 = time.perf_counter()
        th = calibrate_thresholds_cached(cfg_tmp, list(range(6)))  # faster with fewer seeds
        metrics["calibration_s"] += time.perf_counter() - t0
        th_cache[nm] = th
        return th
    
    # NEW: Validate and fix bracket before proceeding to bisection
    tag = f"d{int(dist_um)}um"
    nm_min, nm_max, skip_reason = _validate_and_fix_bracket(
        cfg_base, nm_min, nm_max, nm_ceiling, target_ser, seeds, cache_tag=tag
    )

    seq_len_full = int(cfg_base['pipeline'].get('sequence_length', 1000))
    seq_len_min = int(cfg_base.get('_lod_search_seq_min', max(50, seq_len_full // 5)))
    seq_len_mid = int(cfg_base.get('_lod_search_seq_mid', max(seq_len_min, seq_len_full // 2)))

    if skip_reason:
        print(f"    [{dist_um}μm|{ctrl_str}] Skipping LoD search: {skip_reason}")
        # Clear any misleading state when ceiling is exhausted
        if skip_reason == "nm_ceiling_exhausted":
            _lod_state_save(mode_name, float(dist_um), use_ctrl, {"tested": {}, "th_cache": {}})
            print(f"    🧹 Cleared LoD state for future runs with higher ceiling")
        if LOD_DEBUG_ENABLED:
            _lod_debug_log({
                "phase": "lod_find_skip",
                "distance_um": float(dist_um),
                "skip_reason": skip_reason,
                "nm_min": int(nm_min),
                "nm_max": int(nm_max),
                "target_ser": float(target_ser),
            })
        return float('nan'), 1.0, 0, _final_metrics()

    for iteration in range(20):
        if CANCEL.is_set():
            break
        if nm_min > nm_max:
            break
        nm_mid = int((nm_min + nm_max) / 2)
        if nm_mid == 0 or nm_mid > nm_max:
            break

        print(f"    [{dist_um}μm|{ctrl_str}] Testing Nm={nm_mid} (iteration {iteration+1}/20, range: {nm_min}-{nm_max})")
        iter_start = time.perf_counter()
        cal_before = metrics["calibration_s"]
        sim_before = metrics["simulation_s"]
        bracket_span = max(1, nm_max - nm_min)
        seq_len_dynamic = seq_len_full
        close_to_target = best_ser <= target_ser * 1.5
        if not close_to_target:
            if iteration < 2 and bracket_span > nm_mid:
                seq_len_dynamic = seq_len_min
            elif bracket_span > max(nm_mid // 2, 25):
                seq_len_dynamic = seq_len_mid
        if close_to_target or bracket_span <= max(10, nm_mid // 3):
            seq_len_dynamic = seq_len_full
        seq_len_dynamic = max(seq_len_min, min(seq_len_full, seq_len_dynamic))

        cfg_test = deepcopy(cfg_base)
        cfg_test['pipeline']['Nm_per_symbol'] = nm_mid
        cfg_test['pipeline']['sequence_length'] = int(seq_len_dynamic)

        # Apply cached thresholds for the current test point
        _apply_thresholds_into_cfg(cfg_test, _get_th(nm_mid))

        # --- Gather cached seed results first (if any) ---
        results: List[Dict[str, Any]] = []
        k_err = 0
        n_seen = 0
        seq_len = int(cfg_test['pipeline']['sequence_length'])
        total_planned = len(seeds) * seq_len
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
        
        delta_conf = lod_delta
        total_planned = len(seeds) * seq_len if seq_len > 0 else 0
        
        # loop remaining (non-cached) seeds with screening
        for i, seed in enumerate(seeds):
            # skip if already cached
            if any(sd == seed for sd, _ in cached):
                continue
            
            # reuse our generic worker to also persist the per-seed result
            t_sim_start = time.perf_counter()
            res = run_param_seed_combo(cfg_test, 'pipeline.Nm_per_symbol', nm_mid, seed,
                                       debug_calibration=False,
                                       thresholds_override=_get_th(nm_mid),
                                       sweep_name="lod_search",
                                       cache_tag=cache_tag)
            metrics["simulation_s"] += time.perf_counter() - t_sim_start
            if res is not None:
                results.append(res)
                k_err += int(res.get('errors', 0))
                n_seen += seq_len
                metrics["seeds_simulated"] += 1
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
                    },
                    "th_cache": {
                        str(k): _sanitize_thresholds_for_resume(th_cache[k])
                        for k in th_cache
                    },
                }
                _lod_state_save(mode_name, float(dist_um), use_ctrl, checkpoint)
                
                # Deterministic screen: if even worst/best remaining cannot cross target
                decide_below, decide_above = _deterministic_screen(
                    k_err, n_seen, total_planned, target_ser
                )
                if decide_below or decide_above:
                    break
                if n_seen > 0:
                    # Variance-aware Wilson interval
                    wilson_low, wilson_high = _wilson_interval(k_err, n_seen, wilson_z)
                    margin_low = max(0.0, target_ser * (1.0 - ci_margin))
                    margin_high = target_ser * (1.0 + ci_margin)
                    if wilson_high <= margin_low or wilson_low >= margin_high:
                        break
                    # Hoeffding bounds: if CI fully below/above target
                    hoeff_low, hoeff_high = _hoeffding_bounds(k_err, n_seen, delta=delta_conf)
                    if hoeff_high < target_ser or hoeff_low > target_ser:
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

        trace_extra: Dict[str, Any] = {"sequence_length": int(seq_len_dynamic)}
        if n_seen > 0:
            w_low, w_high = _wilson_interval(k_err, n_seen, wilson_z)
            h_low, h_high = _hoeffding_bounds(k_err, n_seen, delta=delta_conf)
            trace_extra.update({
                "wilson_low": float(w_low),
                "wilson_high": float(w_high),
                "hoeffding_low": float(h_low),
                "hoeffding_high": float(h_high),
            })
        _record_trace_entry(nm_mid, "binary_search", k_err, n_seen, ser, len(results), trace_extra)

        if LOD_DEBUG_ENABLED:
            _lod_debug_log({
                "phase": "lod_iteration_result",
                "distance_um": float(dist_um),
                "iteration": int(iteration),
                "nm_mid": int(nm_mid),
                "nm_min": int(nm_min),
                "nm_max": int(nm_max),
                "ser": float(ser),
                "target_ser": float(target_ser),
                "k_err": int(k_err),
                "n_seen": int(n_seen),
                "results_cached": len(cached),
                "results_collected": len(results),
            })

        print(f"      [{dist_um}μm|{ctrl_str}] Nm={nm_mid}: SER={ser:.4f} {'✓ PASS' if ser <= target_ser else '✗ FAIL'}")

        # Track best attempt regardless of whether it passes
        if ser < best_ser:
            best_ser = ser
            best_nm = nm_mid

        if ser <= target_ser:
            lod_nm = nm_mid
            if LOD_DEBUG_ENABLED:
                _lod_debug_log({
                    "phase": "lod_iteration_pass",
                    "distance_um": float(dist_um),
                    "iteration": int(iteration),
                    "nm_mid": int(nm_mid),
                    "ser": float(ser),
                    "nm_min": int(nm_min),
                    "nm_max": int(nm_max),
                })
            
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
                _apply_thresholds_into_cfg(cfg_probe, _get_th(nm_probe))
                probe_seq_len = max(seq_len_min, int(max(50, seq_len_dynamic * 0.5)))
                cfg_probe['pipeline']['sequence_length'] = probe_seq_len
                
                for probe_seed in probe_seeds:
                    # Check cache first
                    cached_probe = read_seed_cache(mode_name, "lod_search", nm_probe, probe_seed, use_ctrl, cache_tag=cache_tag)
                    if cached_probe:
                        probe_k_err += int(cached_probe.get('errors', 0))
                        probe_n_seen += probe_seq_len
                    else:
                        # Run minimal simulation
                        t_probe = time.perf_counter()
                        res_probe = run_param_seed_combo(cfg_probe, 'pipeline.Nm_per_symbol', nm_probe, probe_seed,
                                                        debug_calibration=False, thresholds_override=_get_th(nm_probe),
                                                        sweep_name="lod_search", cache_tag=cache_tag)
                        elapsed_probe = time.perf_counter() - t_probe
                        metrics["simulation_s"] += elapsed_probe
                        metrics["downstep_s"] += elapsed_probe
                        if res_probe:
                            probe_k_err += int(res_probe.get('errors', 0))
                            probe_n_seen += probe_seq_len
                            metrics["seeds_simulated"] += 1
                    
                    # Early deterministic screen after each seed
                    total_planned = len(probe_seeds) * probe_seq_len
                    decide_below, decide_above = _deterministic_screen(probe_k_err, probe_n_seen, total_planned, target_ser)
                    if decide_below or decide_above:
                        probe_ser = probe_k_err / probe_n_seen if probe_n_seen > 0 else 1.0
                        trace_flag = "success" if decide_below else "fail"
                        trace_extra_probe = {"status": trace_flag, "sequence_length": int(probe_seq_len)}
                        if probe_n_seen > 0:
                            w_low_p, w_high_p = _wilson_interval(probe_k_err, probe_n_seen, wilson_z)
                            h_low_p, h_high_p = _hoeffding_bounds(probe_k_err, probe_n_seen, delta=delta_conf)
                            trace_extra_probe.update({
                                "wilson_low": float(w_low_p),
                                "wilson_high": float(w_high_p),
                                "hoeffding_low": float(h_low_p),
                                "hoeffding_high": float(h_high_p),
                            })
                        _record_trace_entry(nm_probe, "downstep_probe", probe_k_err, probe_n_seen,
                                            probe_ser, len(probe_seeds), trace_extra_probe)

                        if decide_below:
                            # Probe passes! Skip bisection iterations
                            if probe_ser < best_ser:
                                best_ser = probe_ser
                                best_nm = nm_probe
                            print(f"      [{dist_um}μm|{ctrl_str}] ✓ Down-step probe SUCCESS → skip to Nm={nm_probe}")
                            lod_nm = nm_probe
                            nm_max = nm_probe - 1
                            if LOD_DEBUG_ENABLED:
                                _lod_debug_log({
                                    "phase": "lod_downstep_success",
                                    "distance_um": float(dist_um),
                                    "nm_probe": int(nm_probe),
                                    "probe_ser": float(probe_ser),
                                    "probe_k_err": int(probe_k_err),
                                    "probe_n_seen": int(probe_n_seen),
                                })
                        else:
                            # Probe fails decisively, stick with original nm_mid
                            print(f"      [{dist_um}μm|{ctrl_str}] ✗ Down-step probe FAIL → continue bisection")
                            if LOD_DEBUG_ENABLED:
                                _lod_debug_log({
                                    "phase": "lod_downstep_fail",
                                    "distance_um": float(dist_um),
                                    "nm_probe": int(nm_probe),
                                    "probe_ser": float(probe_ser),
                                    "probe_k_err": int(probe_k_err),
                                    "probe_n_seen": int(probe_n_seen),
                                })
                        progress_count += len(probe_seeds)  # Count probe work
                        break
                else:
                    # All probe seeds completed, check final SER
                    if probe_n_seen > 0:
                        probe_ser = probe_k_err / probe_n_seen
                        trace_extra_probe = {"status": "success" if probe_ser <= target_ser else "fail",
                                             "sequence_length": int(probe_seq_len)}
                        w_low_p, w_high_p = _wilson_interval(probe_k_err, probe_n_seen, wilson_z)
                        h_low_p, h_high_p = _hoeffding_bounds(probe_k_err, probe_n_seen, delta=delta_conf)
                        trace_extra_probe.update({
                            "wilson_low": float(w_low_p),
                            "wilson_high": float(w_high_p),
                            "hoeffding_low": float(h_low_p),
                            "hoeffding_high": float(h_high_p),
                        })
                        _record_trace_entry(nm_probe, "downstep_probe", probe_k_err, probe_n_seen,
                                            probe_ser, len(probe_seeds), trace_extra_probe)
                        if probe_ser <= target_ser:
                            print(f"      [{dist_um}μm|{ctrl_str}] ✓ Down-step probe SUCCESS (SER={probe_ser:.4f}) → skip to Nm={nm_probe}")
                            lod_nm = nm_probe
                            nm_max = nm_probe - 1
                            if LOD_DEBUG_ENABLED:
                                _lod_debug_log({
                                    "phase": "lod_downstep_success",
                                    "distance_um": float(dist_um),
                                    "nm_probe": int(nm_probe),
                                    "probe_ser": float(probe_ser),
                                    "probe_k_err": int(probe_k_err),
                                    "probe_n_seen": int(probe_n_seen),
                                })
                        else:
                            print(f"      [{dist_um}μm|{ctrl_str}] ✗ Down-step probe FAIL (SER={probe_ser:.4f}) → continue bisection")
                            if LOD_DEBUG_ENABLED:
                                _lod_debug_log({
                                    "phase": "lod_downstep_fail",
                                    "distance_um": float(dist_um),
                                    "nm_probe": int(nm_probe),
                                    "probe_ser": float(probe_ser),
                                    "probe_k_err": int(probe_k_err),
                                    "probe_n_seen": int(probe_n_seen),
                                })
                        progress_count += len(probe_seeds)
            
            nm_max = nm_mid - 1  # Standard bisection update
        else:
            nm_min = nm_mid + 1
            if LOD_DEBUG_ENABLED:
                _lod_debug_log({
                    "phase": "lod_iteration_fail",
                    "distance_um": float(dist_um),
                    "iteration": int(iteration),
                    "nm_mid": int(nm_mid),
                    "ser": float(ser),
                    "nm_min": int(nm_min),
                    "nm_max": int(nm_max),
                })
            
        iter_elapsed = time.perf_counter() - iter_start
        used_cal = metrics["calibration_s"] - cal_before
        used_sim = metrics["simulation_s"] - sim_before
        metrics["overhead_s"] += max(0.0, iter_elapsed - max(0.0, used_cal) - max(0.0, used_sim))
        metrics["iterations"] += 1
        # Count each binary search iteration
        progress_count += 1
        
        # persist bounds after each iteration
        _lod_state_save(mode_name, float(dist_um), use_ctrl,
                        {"nm_min": nm_min, "nm_max": nm_max, "iteration": iteration,
                         "last_nm": nm_mid,
                         "th_cache": {
                             str(k): _sanitize_thresholds_for_resume(th_cache[k])
                             for k in th_cache
                         }})
        if LOD_DEBUG_ENABLED:
            _lod_debug_log({
                "phase": "lod_iteration_bounds",
                "distance_um": float(dist_um),
                "iteration": int(iteration),
                "nm_min": int(nm_min),
                "nm_max": int(nm_max),
                "lod_nm_current": float(lod_nm),
                "best_ser": float(best_ser),
                "best_nm": int(best_nm) if best_nm is not None else None,
            })

    # OPTIMIZATION 1: Cap LoD validation retries
    # Only run a single-point validation when the bracket collapsed to one point
    if math.isnan(lod_nm) and nm_min <= nm_ceiling and nm_min == nm_max:
        print(f"    [{dist_um}μm|{ctrl_str}] Final validation at Nm={nm_min}")
        cfg_final = deepcopy(cfg_base)
        cfg_final['pipeline']['Nm_per_symbol'] = nm_min
        if LOD_DEBUG_ENABLED:
            _lod_debug_log({
                "phase": "lod_final_validation_start",
                "distance_um": float(dist_um),
                "nm": int(nm_min),
                "nm_ceiling": int(nm_ceiling),
                "seeds": [int(s) for s in seeds],
            })
        
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
        _apply_thresholds_into_cfg(cfg_final, thresholds)

        # NEW: Cap validation seeds for performance
        max_validation_seeds = cfg_base.get('max_lod_validation_seeds', len(seeds))
        validation_seeds = seeds[:max_validation_seeds] if max_validation_seeds < len(seeds) else seeds
        if len(validation_seeds) < len(seeds):
            print(f"    [{dist_um}μm|{ctrl_str}] Validation capped at {max_validation_seeds}/{len(seeds)} seeds")

        results2: List[Dict[str, Any]] = []
        validation_start = time.perf_counter()
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

        validation_time = time.perf_counter() - validation_start

        if results2:
            total_symbols = len(results2) * cfg_final['pipeline']['sequence_length']
            total_errors = sum(cast(int, r['errors']) for r in results2)
            final_ser = total_errors / total_symbols if total_symbols > 0 else 1.0
            _record_trace_entry(
                int(lod_nm),
                "validation_summary",
                int(total_errors),
                int(total_symbols),
                float(final_ser),
                len(results2),
                {
                    "status": "pass" if final_ser <= target_ser else "fail",
                    "validation_time_s": float(validation_time),
                },
            )
            if final_ser <= target_ser:
                # Track actual progress - binary search iterations + final check seeds + overhead
                actual_progress = 20 + len(seeds) + 5  # max 20 iterations + validation seeds + overhead
                if LOD_DEBUG_ENABLED:
                    _lod_debug_log({
                        "phase": "lod_final_validation_success",
                        "distance_um": float(dist_um),
                        "nm": int(nm_min),
                        "final_ser": float(final_ser),
                        "validation_seeds": len(results2),
                    })
                return nm_min, final_ser, actual_progress, _final_metrics()

    # Return best attempt if no solution found, otherwise return found solution
    final_lod_nm = lod_nm if not math.isnan(lod_nm) else float('nan')
    final_ser = best_ser  # Always return the best SER seen (either successful or closest attempt)
    
    if LOD_DEBUG_ENABLED:
        _lod_debug_log({
            "phase": "lod_find_result",
            "distance_um": float(cfg_base['pipeline'].get('distance_um', float('nan'))),
            "lod_nm": float(final_lod_nm),
            "best_ser": float(final_ser),
            "progress_count": int(progress_count),
            "nm_min_final": int(nm_min),
            "nm_max_final": int(nm_max),
        })
    
    # Return actual count instead of constant
    return final_lod_nm, final_ser, progress_count, _final_metrics()

def _validate_lod_point_with_full_seeds(cfg_base: Dict[str, Any],
                                        lod_nm: int,
                                        full_seeds: List[int],
                                        frozen_distance_payload: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float, float]:
    """
    Run one pass at the chosen LoD using the FULL seeds + FULL sequence_length
    to report paper-grade SER and data-rate with 95% CI.
    Returns: (ser_at_lod, data_rate_bps, ci_low, ci_high)
    """
    cfg = deepcopy(cfg_base)
    distance_um = float(cfg['pipeline'].get('distance_um', cfg_base.get('pipeline', {}).get('distance_um', 0.0)))
    if frozen_distance_payload is None:
        frozen_distance_payload = _get_distance_freeze_payload(cfg_base, distance_um)
    # Ensure symbol period and decision window are consistent for this distance
    Ts = calculate_dynamic_symbol_period(float(cfg['pipeline']['distance_um']), cfg)
    cfg['pipeline']['symbol_period_s'] = Ts
    dt = float(cfg['sim']['dt_s'])
    min_pts = int(cfg.get('_min_decision_points', 4))
    min_win = _enforce_min_window(cfg, Ts)
    cfg['pipeline']['time_window_s'] = max(cfg['pipeline'].get('time_window_s', 0.0), min_win)
    cfg.setdefault('detection', {})
    cfg['detection']['decision_window_s'] = min_win

    frozen_payload = deepcopy(frozen_distance_payload) if frozen_distance_payload else None
    if frozen_payload is None:
        frozen_payload = _get_distance_freeze_payload(cfg_base, distance_um)
    if frozen_payload:
        cfg['pipeline']['_frozen_noise'] = frozen_payload
        cfg['_prefer_distance_freeze'] = True
    else:
        cfg['pipeline'].pop('_frozen_noise', None)
        cfg.pop('_prefer_distance_freeze', None)
    
    cfg['pipeline']['Nm_per_symbol'] = int(lod_nm)

    # NEW: Apply LoD validation sequence length override if specified
    lod_validate_seq_len = cfg_base.get('_lod_validate_seq_len', None)
    if lod_validate_seq_len:
        cfg['pipeline']['sequence_length'] = int(lod_validate_seq_len)
        print(f"    📏 LoD validation using shorter sequences: {lod_validate_seq_len} symbols/seed")

    # Apply thresholds at this exact operating point
    th = calibrate_thresholds_cached(cfg, list(range(10)))
    _apply_thresholds_into_cfg(cfg, th)

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
                             warm_lod_guess: Optional[int] = None,
                             warm_bracket: Optional[Tuple[int, int]] = None,
                             force_analytic_bracket: Optional[bool] = None) -> Dict[str, Any]:
    """
    Process a single distance for LoD calculation.
    Returns dict with lod_nm, ser_at_lod, and data_rate_bps.
    """
    actual_progress = 0  # Initialize before try block
    skipped_reason = None  # ✅ Initialize skipped_reason variable
    cfg = deepcopy(cfg_base)
    distance_freeze_payload = _get_distance_freeze_payload(cfg_base, float(dist_um))
    lod_trace: List[Dict[str, Any]] = []
    validation_time = 0.0
    sigma_time = 0.0
    
    # ✅ FIX: Set distance before LoD search and rebuild window consistently
    cfg['pipeline']['distance_um'] = float(dist_um)  # Bake distance into cfg
    
    # Recompute Ts and enforce a consistent minimum decision window
    Ts_dyn = calculate_dynamic_symbol_period(float(dist_um), cfg)
    cfg['pipeline']['symbol_period_s'] = Ts_dyn
    min_win = _enforce_min_window(cfg, Ts_dyn)
    cfg['pipeline']['time_window_s'] = max(cfg['pipeline'].get('time_window_s', 0.0), min_win)
    cfg.setdefault('detection', {})
    cfg['detection']['decision_window_s'] = min_win

    if LOD_DEBUG_ENABLED:
        # NOTE: LoD debug instrumentation (remove once diagnostics conclude)
        _lod_debug_log({
            "phase": "lod_distance_start",
            "distance_um": float(dist_um),
            "target_ser": float(target_ser),
            "seeds": [int(s) for s in seeds],
            "resume": bool(resume),
            "warm_lod_guess": int(warm_lod_guess) if warm_lod_guess else None,
            "analytic_lod_bracket": bool(cfg.get('_analytic_lod_bracket', False)),
            "lod_nm_min": int(cfg_base['pipeline'].get('lod_nm_min', 50)),
            "lod_nm_max": int(cfg_base.get('pipeline', {}).get('lod_nm_max', cfg_base.get('lod_max_nm', 1000000))),
            "decision_window_s": float(cfg['detection']['decision_window_s']),
            "decision_window_policy": str(cfg.get('detection', {}).get('decision_window_policy', 'unknown')),
            "sequence_length": int(cfg['pipeline'].get('sequence_length', 0)),
            "progress_mode": getattr(args, "progress", None) if args else None,
        })

    if distance_freeze_payload:
        cfg['pipeline']['_frozen_noise'] = deepcopy(distance_freeze_payload)
        cfg['_prefer_distance_freeze'] = True
    else:
        cfg['pipeline'].pop('_frozen_noise', None)
        cfg.pop('_prefer_distance_freeze', None)

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
            if LOD_DEBUG_ENABLED:
                _lod_debug_log({
                    "phase": "lod_distance_skip",
                    "distance_um": float(dist_um),
                    "reason": "ts_explosion",
                    "Ts_dyn": float(Ts_dyn),
                    "max_symbol_duration_s": float(max_symbol_duration_s),
                })
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
                'skipped_reason': f'Ts_explosion_{Ts_dyn:.1f}s',
                'lod_trace': [],
            }
    # Continue with existing logic for args.max_ts_for_lod (keep this too)
    cap_cli = getattr(args, "max_ts_for_lod", None) if args else None
    if (not allow_ts) and (cap_cli is not None) and (float(cap_cli) > 0) and (Ts_dyn > float(cap_cli)):
        if warn_only:
            print(f"⚠️  WARNING: distance {dist_um}μm has long symbol period {Ts_dyn:.1f}s (exceeds CLI limit {cap_cli}s), continuing anyway")
            # Continue with LoD analysis instead of returning NaN
        else:
            if LOD_DEBUG_ENABLED:
                _lod_debug_log({
                    "phase": "lod_distance_skip",
                    "distance_um": float(dist_um),
                    "reason": "ts_cli_limit",
                    "Ts_dyn": float(Ts_dyn),
                    "cli_limit": float(cap_cli),
                })
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
                'actual_progress': 0, 'skipped_reason': f'Ts>{cap_cli}s',
                'lod_trace': [],
            }
    
    # LoD search can use shorter sequences to bracket quickly
    if args and getattr(args, "lod_seq_len", None):
        cfg['pipeline']['sequence_length'] = args.lod_seq_len
    
    # NEW: Propagate warm-start guess
    if warm_lod_guess and warm_lod_guess > 0:
        cfg['_warm_lod_guess'] = int(warm_lod_guess)
    else:
        cfg.pop('_warm_lod_guess', None)

    if warm_bracket:
        lb, ub = warm_bracket
        cfg['_warm_bracket_min'] = int(lb)
        cfg['_warm_bracket_max'] = int(ub)
    else:
        cfg.pop('_warm_bracket_min', None)
        cfg.pop('_warm_bracket_max', None)
    
    analytic_flag = force_analytic_bracket
    if analytic_flag is None:
        analytic_flag = bool(getattr(args, "analytic_lod_bracket", False)) if args else False
    if analytic_flag:
        cfg['_analytic_lod_bracket'] = True
    else:
        cfg.pop('_analytic_lod_bracket', None)
    
    # NEW: Set LoD max from args (use consistent key)
    cfg['pipeline']['lod_nm_max'] = int(getattr(args, "lod_max_nm", 1000000))
    
    search_metrics: Dict[str, Any] = {
        "calibration_s": 0.0,
        "simulation_s": 0.0,
        "downstep_s": 0.0,
        "overhead_s": 0.0,
        "total_s": 0.0,
        "seeds_simulated": 0,
        "iterations": 0,
    }
    search_wall = 0.0
    search_start = time.perf_counter()

    try:
        cache_tag = f"d{int(dist_um)}um"
        lod_nm, ser_at_lod, actual_progress, search_metrics = find_lod_for_ser(
            cfg, seeds, target_ser, debug_calibration, progress_cb,
            resume=resume, cache_tag=cache_tag
        )
        search_wall = time.perf_counter() - search_start
        raw_trace = cfg.pop('_lod_eval_trace', [])
        if isinstance(raw_trace, list):
            lod_trace = [dict(entry) for entry in raw_trace if isinstance(entry, dict)]
        else:
            lod_trace = []
        
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
            _apply_thresholds_into_cfg(cfg, th)
            
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
        search_wall = time.perf_counter() - search_start
        print(f"Error processing distance {dist_um}: {e}")
        lod_nm = float('nan')
        ser_at_lod = float('nan')
        data_rate_bps = float('nan')
        data_rate_ci_low = float('nan')
        data_rate_ci_high = float('nan')
        skipped_reason = 'lod_search_failed'  # ✅ Set in exception block
        search_metrics = {
            "calibration_s": 0.0,
            "simulation_s": 0.0,
            "downstep_s": 0.0,
            "overhead_s": 0.0,
            "total_s": 0.0,
            "seeds_simulated": 0,
            "iterations": 0,
        }
    finally:
        cfg.pop('_warm_lod_guess', None)
        cfg.pop('_warm_bracket_min', None)
        cfg.pop('_warm_bracket_max', None)
        cfg.pop('_analytic_lod_bracket', None)
    
    # NEW: Re-validate final LoD with full seeds for publication-grade statistics
    full_seeds = getattr(args, "full_seeds", seeds)  # default to reduced set if absent
    if not np.isnan(lod_nm) and lod_nm > 0:
        # Build a cfg with the chosen distance baked in (so Ts is recomputed inside helper)
        cfg_base_with_distance = deepcopy(cfg_base)
        cfg_base_with_distance['pipeline']['distance_um'] = dist_um
        ser_at_lod, data_rate_bps, data_rate_ci_low, data_rate_ci_high = \
            _validate_lod_point_with_full_seeds(cfg_base_with_distance, int(lod_nm), full_seeds, distance_freeze_payload)
        
        # NEW: Enforce 1% target after validation - adjust upward if needed
        target_ser = 0.01
        if ser_at_lod > target_ser:
            nm_try = int(max(lod_nm, 1))
            for _ in range(6):   # hard cap safety
                nm_try = int(math.ceil(nm_try * 1.25))
                ser2, rate2, lo2, hi2 = _validate_lod_point_with_full_seeds(cfg_base_with_distance, nm_try, full_seeds, distance_freeze_payload)
                if ser2 <= target_ser:
                    lod_nm, ser_at_lod = nm_try, ser2
                    data_rate_bps, data_rate_ci_low, data_rate_ci_high = rate2, lo2, hi2
                    seq_len_val = int(cfg_base_with_distance['pipeline'].get('sequence_length', cfg['pipeline']['sequence_length']))
                    n_seen_adjust = seq_len_val * len(full_seeds)
                    lod_trace.append({
                        "phase": "validation_adjust",
                        "nm": int(nm_try),
                        "ser": float(ser2),
                        "distance_um": float(dist_um),
                        "n_seen": int(n_seen_adjust),
                        "k_err": int(round(ser2 * n_seen_adjust)),
                        "status": "pass",
                        "timestamp": time.time(),
                    })
                    break
    
    # NEW: Optional noise_sigma persistence aggregation
    lod_found = (
        isinstance(lod_nm, (int, float))
        and math.isfinite(float(lod_nm))
        and float(lod_nm) > 0
        and math.isfinite(ser_at_lod)
        and ser_at_lod <= target_ser
    )
    if lod_found:
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

        if distance_freeze_payload:
            cfg_lod['pipeline']['_frozen_noise'] = deepcopy(distance_freeze_payload)
            cfg_lod['_prefer_distance_freeze'] = True
        else:
            cfg_lod['pipeline'].pop('_frozen_noise', None)
            cfg_lod.pop('_prefer_distance_freeze', None)

        cfg_lod['pipeline']['Nm_per_symbol'] = lod_nm

        # Apply thresholds at this exact (distance, Ts, Nm)
        th_lod = calibrate_thresholds_cached(cfg_lod, list(range(4)))
        _apply_thresholds_into_cfg(cfg_lod, th_lod)
        
        def _as_float(val: Any) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return float('nan')

        cfg_lod['pipeline']['_collect_noise_components'] = True

        sigma_start = time.perf_counter()
        sigma_values: List[float] = []
        sigma_values_measured: List[float] = []
        sigma_thermal_values: List[float] = []
        sigma_thermal_measured: List[float] = []
        sigma_flicker_values: List[float] = []
        sigma_flicker_measured: List[float] = []
        sigma_drift_values: List[float] = []
        sigma_drift_measured: List[float] = []
        thermal_fraction_values: List[float] = []
        thermal_fraction_measured: List[float] = []
        i_dc_values: List[float] = []
        v_g_bias_values: List[float] = []
        gm_values: List[float] = []
        c_tot_values: List[float] = []
        for seed in seeds[:5]:  # Limited seeds for efficiency
            result = run_param_seed_combo(
                cfg_lod,
                'pipeline.Nm_per_symbol',
                lod_nm,
                seed,
                debug_calibration=debug_calibration,
                sweep_name="lod_validation",
                cache_tag="lod_sigma",
                thresholds_override=th_lod,
            )
            if result is None:
                continue
            result_dict = cast(Dict[str, Any], result)

            sigma_meas = _as_float(result_dict.get('noise_sigma_I_diff_measured'))
            if math.isfinite(sigma_meas):
                sigma_values_measured.append(sigma_meas)
                sigma_values.append(sigma_meas)
            else:
                sigma_val = _as_float(result_dict.get('noise_sigma_I_diff'))
                if math.isfinite(sigma_val):
                    sigma_values.append(sigma_val)

            sigma_thermal_meas = _as_float(result_dict.get('noise_sigma_thermal_measured'))
            if math.isfinite(sigma_thermal_meas):
                sigma_thermal_measured.append(sigma_thermal_meas)
                sigma_thermal_values.append(sigma_thermal_meas)
            else:
                sigma_thermal_val = _as_float(result_dict.get('noise_sigma_thermal'))
                if math.isfinite(sigma_thermal_val):
                    sigma_thermal_values.append(sigma_thermal_val)

            sigma_flicker_meas = _as_float(result_dict.get('noise_sigma_flicker_measured'))
            if math.isfinite(sigma_flicker_meas):
                sigma_flicker_measured.append(sigma_flicker_meas)
                sigma_flicker_values.append(sigma_flicker_meas)
            else:
                sigma_flicker_val = _as_float(result_dict.get('noise_sigma_flicker'))
                if math.isfinite(sigma_flicker_val):
                    sigma_flicker_values.append(sigma_flicker_val)

            sigma_drift_meas = _as_float(result_dict.get('noise_sigma_drift_measured'))
            if math.isfinite(sigma_drift_meas):
                sigma_drift_measured.append(sigma_drift_meas)
                sigma_drift_values.append(sigma_drift_meas)
            else:
                sigma_drift_val = _as_float(result_dict.get('noise_sigma_drift'))
                if math.isfinite(sigma_drift_val):
                    sigma_drift_values.append(sigma_drift_val)

            thermal_frac_meas = _as_float(result_dict.get('noise_thermal_fraction_measured'))
            if math.isfinite(thermal_frac_meas):
                thermal_fraction_measured.append(thermal_frac_meas)
                thermal_fraction_values.append(thermal_frac_meas)
            else:
                thermal_frac_val = _as_float(result_dict.get('noise_thermal_fraction'))
                if math.isfinite(thermal_frac_val):
                    thermal_fraction_values.append(thermal_frac_val)

            if 'I_dc_used_A' in result_dict:
                i_dc_values.append(result_dict['I_dc_used_A'])
            if 'V_g_bias_V_used' in result_dict:
                v_g_bias_values.append(result_dict['V_g_bias_V_used'])
            if 'gm_S' in result_dict:
                gm_values.append(result_dict['gm_S'])
            if 'C_tot_F' in result_dict:
                c_tot_values.append(result_dict['C_tot_F'])
            # Progress callback for completed noise sigma seed
            if progress_cb is not None:
                try:
                    progress_cb.put(1)
                except Exception:
                    pass
        
        sigma_time = time.perf_counter() - sigma_start

        cfg_lod['pipeline'].pop('_collect_noise_components', None)

        def _finite_list_median(values):
            arr = np.asarray(values, dtype=float)
            finite = arr[np.isfinite(arr)]
            return float(np.median(finite)) if finite.size else float('nan')

        lod_sigma_median = _finite_list_median(sigma_values)
        lod_sigma_measured = _finite_list_median(sigma_values_measured)
        lod_sigma_thermal = _finite_list_median(sigma_thermal_values)
        lod_sigma_thermal_measured = _finite_list_median(sigma_thermal_measured)
        lod_sigma_flicker = _finite_list_median(sigma_flicker_values)
        lod_sigma_flicker_measured = _finite_list_median(sigma_flicker_measured)
        lod_sigma_drift = _finite_list_median(sigma_drift_values)
        lod_sigma_drift_measured = _finite_list_median(sigma_drift_measured)
        lod_thermal_fraction = _finite_list_median(thermal_fraction_values)
        lod_thermal_fraction_measured = _finite_list_median(thermal_fraction_measured)
        lod_I_dc = _finite_list_median(i_dc_values)
        lod_V_g_bias = _finite_list_median(v_g_bias_values)
        lod_gm = _finite_list_median(gm_values)
        lod_c_tot = _finite_list_median(c_tot_values)
    else:
        lod_sigma_median = float('nan')
        lod_sigma_measured = float('nan')
        lod_sigma_thermal = float('nan')
        lod_sigma_thermal_measured = float('nan')
        lod_sigma_flicker = float('nan')
        lod_sigma_flicker_measured = float('nan')
        lod_sigma_drift = float('nan')
        lod_sigma_drift_measured = float('nan')
        lod_thermal_fraction = float('nan')
        lod_thermal_fraction_measured = float('nan')
        lod_I_dc = float('nan')
        lod_V_g_bias = float('nan')
        lod_gm = float('nan')
        lod_c_tot = float('nan')

    search_total = float(search_metrics.get("total_s", float(search_metrics.get("calibration_s", 0.0))
                                           + float(search_metrics.get("simulation_s", 0.0))
                                           + float(search_metrics.get("overhead_s", 0.0))))

    lod_trace.append({
        "phase": "timing_summary",
        "distance_um": float(dist_um),
        "search_wall_s": float(search_wall),
        "search_calibration_s": float(search_metrics.get("calibration_s", 0.0)),
        "search_simulation_s": float(search_metrics.get("simulation_s", 0.0)),
        "search_downstep_s": float(search_metrics.get("downstep_s", 0.0)),
        "search_overhead_s": float(search_metrics.get("overhead_s", 0.0)),
        "search_total_s": float(search_total),
        "search_iterations": int(search_metrics.get("iterations", 0)),
        "search_seeds": int(search_metrics.get("seeds_simulated", 0)),
        "validation_s": float(validation_time),
        "sigma_s": float(sigma_time),
    })

    # mark LoD state as done ONLY if valid result
    try:
        if isinstance(lod_nm, (int, float)) and math.isfinite(lod_nm) and lod_nm > 0:
            current_state = _lod_state_load(cfg['pipeline']['modulation'], float(dist_um),
                                            bool(cfg['pipeline'].get('use_control_channel', True))) or {}
            current_state.update({"done": True, "nm_min": int(lod_nm), "nm_max": int(lod_nm)})
            _lod_state_save(cfg['pipeline']['modulation'], float(dist_um),
                            bool(cfg['pipeline'].get('use_control_channel', True)), current_state)
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

    if LOD_DEBUG_ENABLED:
        _lod_debug_log({
            "phase": "lod_distance_result",
            "distance_um": float(dist_um),
            "lod_nm": float(lod_nm),
            "ser_at_lod": float(ser_at_lod),
            "skipped_reason": skipped_reason,
            "actual_progress": int(actual_progress),
        })

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
        'noise_sigma_I_diff_measured': lod_sigma_measured,
        'noise_sigma_thermal_measured': lod_sigma_thermal_measured,
        'noise_sigma_flicker_measured': lod_sigma_flicker_measured,
        'noise_sigma_drift_measured': lod_sigma_drift_measured,
        'noise_thermal_fraction_measured': lod_thermal_fraction_measured,
        'noise_sigma_I_diff_is_measured': bool(math.isfinite(lod_sigma_measured)),
        'noise_sigma_thermal_is_measured': bool(math.isfinite(lod_sigma_thermal_measured)),
        'noise_sigma_flicker_is_measured': bool(math.isfinite(lod_sigma_flicker_measured)),
        'noise_sigma_drift_is_measured': bool(math.isfinite(lod_sigma_drift_measured)),
        'noise_thermal_fraction_is_measured': bool(math.isfinite(lod_thermal_fraction_measured)),
        'I_dc_used_A': float(lod_I_dc),
        'V_g_bias_V_used': float(lod_V_g_bias),
        'gm_S': float(lod_gm),
        'C_tot_F': float(lod_c_tot),
        'actual_progress': int(actual_progress),
        'skipped_reason': skipped_reason,  # ✅ Use the variable instead of None
        'lod_trace': lod_trace,
        'lod_time_search_wall_s': float(search_wall),
        'lod_time_calibration_s': float(search_metrics.get('calibration_s', 0.0)),
        'lod_time_simulation_s': float(search_metrics.get('simulation_s', 0.0)),
        'lod_time_downstep_s': float(search_metrics.get('downstep_s', 0.0)),
        'lod_time_overhead_s': float(search_metrics.get('overhead_s', 0.0)),
        'lod_time_total_s': float(search_total),
        'lod_time_validation_s': float(validation_time),
        'lod_time_sigma_s': float(sigma_time),
        'lod_time_search_iterations': int(search_metrics.get('iterations', 0)),
        'lod_seeds_simulated': int(search_metrics.get('seeds_simulated', 0)),
        'lod_found': bool(lod_found),
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
        _apply_thresholds_into_cfg(cfg_pair, thresholds)
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
    distances_recomputed: set = set()
    nm_lookup: Dict[float, float] = {}

    try:
        for dist in distance_candidates:
            dist_float = float(dist)
            nm_target = _nm_for_distance(dist_float)
            nm_lookup[dist_float] = nm_target
    
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
        guard_col = 'guard_factor' if 'guard_factor' in combined_tradeoff.columns else 'pipeline.guard_factor'
        combined_tradeoff[guard_col] = pd.to_numeric(combined_tradeoff[guard_col], errors='coerce')
        combined_tradeoff['symbol_period_s'] = pd.to_numeric(combined_tradeoff['symbol_period_s'], errors='coerce')
        combined_tradeoff['ser'] = pd.to_numeric(combined_tradeoff['ser'], errors='coerce')
        subset_cols = ['distance_um', guard_col, 'symbol_period_s', 'ser']
        if 'use_ctrl' in combined_tradeoff.columns:
            subset_cols.append('use_ctrl')
            combined_tradeoff = combined_tradeoff[combined_tradeoff['use_ctrl'] == desired_ctrl]
        combined_tradeoff = combined_tradeoff.dropna(subset=subset_cols)
        combined_tradeoff = combined_tradeoff.drop_duplicates(subset=subset_cols, keep='last').sort_values(['distance_um', guard_col])
        _atomic_write_csv(tradeoff_csv, combined_tradeoff)
        print(f"?? Guard trade-off data saved to {tradeoff_csv} ({len(combined_tradeoff)} rows)")
    elif not frames_tradeoff:
        print(f"?? Guard trade-off not updated (no new data) at {tradeoff_csv}")

    def _compute_frontier(df_source: pd.DataFrame) -> pd.DataFrame:
        if df_source.empty:
            return pd.DataFrame()
        df_front = df_source.copy()
        guard_name = 'guard_factor' if 'guard_factor' in df_front.columns else 'pipeline.guard_factor'
        df_front['distance_um'] = pd.to_numeric(df_front['distance_um'], errors='coerce')
        df_front[guard_name] = pd.to_numeric(df_front[guard_name], errors='coerce')
        df_front['symbol_period_s'] = pd.to_numeric(df_front['symbol_period_s'], errors='coerce')
        df_front['ser'] = pd.to_numeric(df_front['ser'], errors='coerce')
        df_front = df_front.dropna(subset=['distance_um', guard_name, 'symbol_period_s', 'ser'])
        if df_front.empty:
            return pd.DataFrame()
        df_front['IRT'] = (bits_per_symbol / df_front['symbol_period_s']) * (1.0 - df_front['ser'])
        rows: List[Dict[str, Any]] = []
        for dist_val, group in df_front.groupby('distance_um'):
            if group.empty:
                continue
            idx = group['IRT'].idxmax()
            if idx is None or idx not in group.index:
                continue
            row = group.loc[idx]
            dist_val_f = _coerce_float(dist_val)
            guard_val = _coerce_float(row.get(guard_name))
            irt_val = _coerce_float(row.get('IRT'))
            ser_val = _coerce_float(row.get('ser'))
            symbol_period_val = _coerce_float(row.get('symbol_period_s'))
            nm_raw = row.get('Nm_per_symbol')
            nm_default = _coerce_float(
                nm_lookup.get(dist_val_f, cfg_base['pipeline'].get('Nm_per_symbol', 1e4))
            )
            nm_val = _coerce_float(nm_raw, nm_default)
            if not math.isfinite(nm_val):
                nm_val = nm_default
            rows.append({
                'distance_um': dist_val_f,
                'best_guard_factor': guard_val,
                'max_irt_bps': irt_val,
                'ser_at_best_guard': ser_val,
                'symbol_period_s': symbol_period_val,
                'Nm_per_symbol': nm_val,
                'use_ctrl': desired_ctrl,
                'mode': mode,
                'bits_per_symbol': bits_per_symbol,
            })
        if not rows:
            return pd.DataFrame()
        df_frontier = pd.DataFrame(rows)
        df_frontier.sort_values(by=['distance_um'], inplace=True)
        return df_frontier

    if not combined_tradeoff.empty:
        combined_frontier = _compute_frontier(combined_tradeoff)
    else:
        combined_frontier = existing_frontier.copy()

    if not combined_frontier.empty:
        combined_frontier['distance_um'] = pd.to_numeric(combined_frontier['distance_um'], errors='coerce')
        subset_cols = ['distance_um']
        if 'use_ctrl' in combined_frontier.columns:
            combined_frontier['use_ctrl'] = combined_frontier['use_ctrl'].astype(bool)
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
                        _apply_thresholds_into_cfg(cfg, th)
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
    canonical_mode = _canonical_mode_name(mode)
    overrides: Dict[str, List[int]] = getattr(args, 'cli_nm_overrides', {})
    if canonical_mode in overrides:
        raw_nm_values = overrides[canonical_mode]
        print(f"CLI Nm grid for {canonical_mode}: {raw_nm_values}")
    elif "__global__" in overrides:
        raw_nm_values = overrides["__global__"]
        print(f"CLI Nm grid override (global): {raw_nm_values}")
    else:
        cfg_nm = _get_cfg_nm_grid(cfg.get('Nm_range'), canonical_mode)
        if cfg_nm is not None:
            raw_nm_values = cfg_nm
            print(f"CFG Nm_range[{canonical_mode}]: {raw_nm_values}")
        else:
            raw_nm_values = DEFAULT_NM_RANGES.get(canonical_mode, DEFAULT_NM_RANGES['MoSK'])
            print(f"CFG default Nm_range[{canonical_mode}]: {raw_nm_values}")
    nm_values: List[Union[float, int]] = [cast(Union[float, int], int(v)) for v in raw_nm_values]

    guard_values = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
    ser_jobs = len(nm_values) * args.num_seeds
    lod_seed_cap = 10
    lod_jobs = len(lod_distance_grid) * (lod_seed_cap * 8 + lod_seed_cap + 5)  # initial estimate only

    distance_policy = getattr(args, "distance_sweep", "always")
    distances_available = bool(lod_distance_grid)
    do_distance_metrics = False
    if distance_policy == "always":
        do_distance_metrics = distances_available
    elif distance_policy == "auto":
        do_distance_metrics = distances_available
    else:  # "never"
        do_distance_metrics = False
    dist_jobs = (len(lod_distance_grid) * args.num_seeds) if do_distance_metrics else 0

    do_isi = False
    if args.isi_sweep == "always":
        do_isi = not args.disable_isi
    elif args.isi_sweep == "auto":
        do_isi = bool(cfg['pipeline'].get('enable_isi', False))
    else:
        do_isi = False
    isi_jobs = (len(guard_values) * args.num_seeds) if do_isi else 0

    manual_total = 2  # SER + LoD
    if do_distance_metrics:
        manual_total += 1
    if do_isi:
        manual_total += 1

    # Hierarchy
    # Create overall progress bar first
    overall_key = ("overall", mode)
    # Create hierarchy only for GUI backend (avoids duplicate bars in rich/tqdm)
    hierarchy_supported = (args.progress == "gui")
    
    # Initialize variables to prevent unbound issues
    overall_manual = None
    mode_bar = None
    overall = None
    df_distance_metrics = pd.DataFrame()
    
    # Initialize variables with proper types
    mode_key: Optional[Tuple[str, str]] = None
    ser_key: Optional[Tuple[str, str, str]] = None
    lod_key: Optional[Tuple[str, str, str]] = None
    dist_key: Optional[Tuple[str, str, str]] = None
    isi_key: Optional[Tuple[str, str, str]] = None
    
    if hierarchy_supported:
        overall_key = ("overall", mode)
        overall = pm.task(total=ser_jobs + lod_jobs + dist_jobs + isi_jobs,
                         description=f"Overall ({mode})",
                         key=overall_key, kind="overall")
        
        mode_key = ("mode", mode)
        mode_bar = pm.task(total=ser_jobs + lod_jobs + dist_jobs + isi_jobs,
                          description=f"{mode} Mode",
                          parent=overall_key, key=mode_key, kind="mode")

        ser_key = ("sweep", mode, "SER_vs_Nm")
        lod_key = ("sweep", mode, "LoD_vs_distance")
        dist_key = ("sweep", mode, "SER_SNR_vs_distance")
        isi_key = ("sweep", mode, "ISI_vs_guard")

    # Always create the ser_bar with appropriate parent

    if hierarchy_supported:
        ser_bar = pm.task(total=ser_jobs, description="SER vs Nm",
                          parent=mode_key, key=("sweep", mode, "SER_vs_Nm"), kind="sweep")
    else:
        ser_bar = None

    if hierarchy_supported and do_distance_metrics:
        dist_bar = pm.task(total=dist_jobs, description="SER/SNR vs distance",
                           parent=mode_key, key=dist_key, kind="sweep")
    else:
        dist_bar = None

    use_ctrl_flag = bool(cfg['pipeline'].get('use_control_channel', True))
    ser_csv, ser_csv_branch, ser_csv_other = _stage_csv_paths("ser", data_dir, mode, suffix, use_ctrl_flag)
    ser_results_path = ser_csv if not args.ablation_parallel else ser_csv_branch
    existing_ser_branch = _ensure_ablation_branch(
        "ser", ser_csv, ser_csv_branch, use_ctrl_flag, bool(args.resume), _dedupe_ser_dataframe
    )

    # ---------- 1) SER vs Nm ----------
    print("\n1. Running SER vs. Nm sweep...")

    # initial calibration (kept; thresholds hoisted per Nm in run_sweep)
    if mode in ['CSK', 'Hybrid']:
        print(f"\n📊 Initial calibration for {mode} mode...")
        cal_seeds = list(range(10))
        # store to disk so subsequent processes reuse quickly
        initial_thresholds = calibrate_thresholds(cfg, cal_seeds, recalibrate=False, save_to_file=True, verbose=args.debug_calibration)
        print("✅ Calibration complete")
        _apply_thresholds_into_cfg(cfg, initial_thresholds)

    df_ser_nm = run_sweep(
        cfg, seeds,
        'pipeline.Nm_per_symbol', nm_values,
        f"SER vs Nm ({mode})",
        progress_mode=args.progress,
        persist_csv=ser_csv_branch,
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
    ser_branch_frames = [df_ser_nm] if not df_ser_nm.empty else []
    branch_combined = _update_branch_csv(
        "ser",
        ser_csv_branch,
        ser_branch_frames,
        use_ctrl_flag,
        _dedupe_ser_dataframe,
        existing_ser_branch
    )
    if not args.ablation_parallel:
        _merge_branch_csv(ser_csv, [ser_csv_branch, ser_csv_other], _dedupe_ser_dataframe)

    if args.ablation_parallel:
        print(f"✅ SER vs Nm results saved to {ser_results_path} (branch; canonical merge deferred)")
    else:
        print(f"✅ SER vs Nm results saved to {ser_results_path}")
    
    # Manual parent update for non-GUI backends  
    if not hierarchy_supported:
        # For rich/tqdm, create simple overall progress tracker
        if overall_manual is None:
            overall_manual = pm.task(total=manual_total, description=f"{mode} Progress")
        overall_manual.update(1, description=f"{mode} - SER vs Nm completed")

    # --- Auto-refine near target SER (adds a few Nm points between the bracket) ---
    if args.ser_refine:
        try:
            # Always read the latest CSV on disk so resume/de-dupe is consistent
            df_ser_all = pd.read_csv(ser_results_path) if ser_results_path.exists() else df_ser_nm

            # Propose midpoints between the first bracket that crosses the target
            refine_candidates = _auto_refine_nm_points_from_df(
                df_ser_all,
                target=float(args.ser_target),
                extra_points=int(args.ser_refine_points)
            )

            # Filter out any Nm that are already present for THIS CTRL state
            if refine_candidates:
                desired_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
                done = load_completed_values(ser_results_path, 'pipeline_Nm_per_symbol', desired_ctrl)
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
                    persist_csv=ser_csv_branch,
                    resume=args.resume,
                    debug_calibration=args.debug_calibration,
                    pm=pm,
                    sweep_key=ser_refine_key if (args.progress == "gui") else None,
                    parent_key=mode_key if (args.progress == "gui") else None,
                    recalibrate=args.recalibrate
                )

                # Re-de-dupe the CSV so plots read a clean file
                _update_branch_csv(
                    "ser",
                    ser_csv_branch,
                    [df_refined],
                    use_ctrl_flag,
                    _dedupe_ser_dataframe
                )
                if not args.ablation_parallel:
                    _merge_branch_csv(ser_csv, [ser_csv_branch, ser_csv_other], _dedupe_ser_dataframe)
                    print(f"✅ SER auto-refine appended; CSV updated: {ser_csv}")
                else:
                    print(f"✅ SER auto-refine appended; branch updated: {ser_csv_branch.name}")
            else:
                print(f"ℹ️  SER auto-refine: no bracket found or all refine Nm already present.")
        except Exception as e:
            print(f"⚠️  SER auto-refine failed: {e}")

    if not df_ser_nm.empty:
        nm_col_print = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in df_ser_nm.columns else 'pipeline.Nm_per_symbol'
        cols_to_show = [c for c in [nm_col_print, 'ser', 'snr_db', 'use_ctrl'] if c in df_ser_nm.columns]
        print(f"\nSER vs Nm Results (head) for {mode}:")
        print(df_ser_nm[cols_to_show].head().to_string(index=False))

    # After the standard CSK SER vs Nm sweep finishes and nm_values are known:
    if mode == "CSK" and (args.nt_pairs or ""):
        run_csk_nt_pair_sweeps(args, cfg, seeds, nm_values)

    # Standalone zero-signal noise characterization (optional stage)
    force_lod_analytic = args.lod_analytic_noise or args.analytic_noise_all

    if args.analytic_noise_all:
        print("??  Skipping zero-signal noise sweep: analytic noise forced.")
    else:
        run_zero_signal_noise_analysis(
            cfg=cfg,
            mode=mode,
            args=args,
            nm_values=nm_values,
            lod_distance_grid=lod_distance_grid,
            seeds=seeds,
            data_dir=data_dir,
            suffix=suffix,
            pm=pm,
            hierarchy_supported=hierarchy_supported,
            mode_key=mode_key,
        )

    # ---------- 2) LoD vs Distance ----------
    print('\n2. Building LoD vs distance curve.')
    d_run = [int(x) for x in lod_distance_grid]
    lod_csv, lod_csv_branch, lod_csv_other = _stage_csv_paths("lod", data_dir, mode, suffix, use_ctrl)
    lod_results_path = lod_csv if not args.ablation_parallel else lod_csv_branch
    if force_lod_analytic:
        cfg['_force_analytic_noise'] = True
    flush_staged_rows(lod_csv_branch)
    pm.set_status(mode=mode, sweep="LoD vs distance")
    # Use the same worker count chosen at mode start (honors --extreme-mode/--max-workers)
    pool = global_pool.get_pool(max_workers=maxw)
    use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
    existing_lod_branch = _ensure_ablation_branch(
        "lod", lod_csv, lod_csv_branch, use_ctrl, bool(args.resume), _dedupe_lod_dataframe
    )

    estimated_per_distance = args.num_seeds * 8 + args.num_seeds + 5

    # --- NEW: find fully-completed distances for this CTRL state ---
    done_distances: set[int] = set()
    failed_distances: set[int] = set()

    df_prev: Optional[pd.DataFrame] = None
    if args.resume:
        df_prev = existing_lod_branch.copy()
        if df_prev.empty and lod_csv_branch.exists():
            try:
                df_prev = pd.read_csv(lod_csv_branch)
            except Exception as e:
                print(f"⚠️  Could not read per-ablation LoD CSV ({e}); will fall back to canonical if available.")
                df_prev = pd.DataFrame()
        if df_prev.empty and lod_csv.exists():
            try:
                df_prev = pd.read_csv(lod_csv)
                if 'use_ctrl' in df_prev.columns:
                    df_prev = df_prev[df_prev['use_ctrl'] == use_ctrl]
                df_prev = _dedupe_lod_dataframe(cast(pd.DataFrame, df_prev))
            except Exception as e:
                print(f"⚠️  Could not read existing LoD CSV ({e}); will recompute all distances")
                df_prev = pd.DataFrame()

        if not df_prev.empty and {'lod_nm', 'distance_um'}.issubset(df_prev.columns):
            for _, row in df_prev.iterrows():
                dist = int(row['distance_um'])
                lod_nm = row.get('lod_nm', np.nan)

                # Check if this distance succeeded or failed
                if pd.notna(lod_nm) and float(lod_nm) > 0:
                    ser_ok = True
                    if 'ser_at_lod' in df_prev.columns:
                        ser = row.get('ser_at_lod', np.nan)
                        if pd.notna(ser):
                            ser_ok = float(ser) <= 0.01
                    if ser_ok:
                        done_distances.add(dist)
                    else:
                        failed_distances.add(dist)
                else:
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
        elif args.resume and not df_prev.empty:
            print("ℹ️  Existing LoD CSV has no 'lod_nm' column; will recompute all distances.")

    lod_history_points: List[Dict[str, Any]] = []
    _extend_history_with_lod_rows(lod_history_points, existing_lod_branch)
    _extend_history_with_lod_rows(lod_history_points, df_prev)
    lod_history_points.extend(_load_lod_history_cache(mode, use_ctrl, suffix))

    # Worklist excludes done distances
    d_run_work = [int(d) for d in d_run if int(d) not in done_distances]

    lod_jobs = len(d_run) * estimated_per_distance
    lod_key = ("sweep", mode, "LoD")
    if hierarchy_supported:
        lod_bar = pm.task(total=lod_jobs, description="LoD vs distance",
                          parent=mode_key, key=lod_key, kind="sweep")
        # Keep the overall headline consistent with the true LoD total
        if hasattr(pm, "update_total") and overall is not None:
            pm.update_total(key=("overall", mode), total=ser_jobs + lod_jobs + dist_jobs + isi_jobs,
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
        dist_progress_key = ("dist", mode, "LoD", float(d_um))
        if hierarchy_supported:
            dist_bar = pm.task(total=estimated_per_distance,
                            description=f"LoD @ {float(d_um):.0f}μm",
                            parent=lod_key, key=dist_progress_key, kind="dist")
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

    target_ser = 0.01

    def _submit_one(dist: int, wid_hint: int) -> None:
        """Submit one distance job to the pool."""
        wid = wid_hint % max(1, maxw)
        if hasattr(pm, "worker_update"):
            pm.worker_update(wid, f"LoD | d={dist} um")
        q = progress_queues[dist]
        seeds_for_lod = _choose_seeds_for_distance(float(dist), seeds, args.lod_num_seeds)
        args.full_seeds = seeds

        nm_min_cfg = int(cfg['pipeline'].get('lod_nm_min', 50))
        nm_max_cfg = int(cfg.get('pipeline', {}).get('lod_nm_max', cfg.get('lod_max_nm', 1000000)))
        hist_guess, hist_bracket = _predict_lod_from_history(float(dist), lod_history_points,
                                                             target_ser, nm_min_cfg, nm_max_cfg)

        warm_guess: Optional[int] = None
        warm_bracket: Optional[Tuple[int, int]] = None
        if hist_guess is not None:
            warm_guess = int(max(nm_min_cfg, min(nm_max_cfg, round(hist_guess))))
            bracket = hist_bracket or _default_bracket_from_guess(hist_guess, float(dist), nm_min_cfg, nm_max_cfg)
            warm_bracket = (int(bracket[0]), int(bracket[1]))
            print(f"    [warm] {dist}μm history fit → Nm≈{warm_guess} bracket {warm_bracket[0]}-{warm_bracket[1]}")
        elif last_lod_guess:
            warm_guess = int(last_lod_guess)
            warm_bracket = _default_bracket_from_guess(warm_guess, float(dist), nm_min_cfg, nm_max_cfg)

        if warm_bracket is not None:
            lb, ub = warm_bracket
            lb = max(nm_min_cfg, int(lb))
            ub = min(nm_max_cfg, int(ub))
            if lb >= ub:
                lb, ub = _default_bracket_from_guess(warm_guess or nm_min_cfg, float(dist), nm_min_cfg, nm_max_cfg)
            warm_bracket = (int(lb), int(ub))

        enable_analytic = bool(getattr(args, "analytic_lod_bracket", False))
        if not enable_analytic and hist_bracket and canonical_mode in ("MoSK", "CSK"):
            enable_analytic = True

        fut = pool.submit(
            process_distance_for_lod, float(dist), cfg, seeds_for_lod, target_ser,
            args.debug_calibration, q, args.resume, args,
            warm_lod_guess=warm_guess,
            warm_bracket=warm_bracket,
            force_analytic_bracket=enable_analytic
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
            trace_entries = res.get('lod_trace', [])
            normalized_trace = _normalize_lod_trace_entries(
                float(res.get('distance_um', dist)),
                trace_entries if isinstance(trace_entries, list) else [],
                target_ser)
            if normalized_trace:
                lod_history_points.extend(normalized_trace)
                _append_lod_history_cache(mode, use_ctrl, suffix, normalized_trace)
            res.pop('lod_trace', None)
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
            append_row_atomic(lod_csv_branch, res, list(res.keys()))
        
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

    flush_staged_rows(lod_csv_branch)
    lod_branch_combined = _update_branch_csv(
        "lod",
        lod_csv_branch,
        [df_lod_new],
        use_ctrl,
        _dedupe_lod_dataframe,
        existing_lod_branch
    )
    if not args.ablation_parallel:
        _merge_branch_csv(lod_csv, [lod_csv_branch, lod_csv_other], _dedupe_lod_dataframe)
    cfg.pop('_force_analytic_noise', None)

    try:
        df_lod_merged = pd.read_csv(lod_results_path) if lod_results_path.exists() else pd.DataFrame()
    except Exception:
        df_lod_merged = pd.DataFrame()

    df_lod = lod_branch_combined if not lod_branch_combined.empty else pd.DataFrame()

    merged_points = len(df_lod_merged) if not df_lod_merged.empty else 0
    if not df_lod.empty:
        if merged_points:
            if args.ablation_parallel:
                merged_note = f"; branch total {merged_points}"
            else:
                merged_note = f"; merged total {merged_points}"
        else:
            merged_note = ""
        target_label = "branch" if args.ablation_parallel else "canonical"
        print(f"\n✅ LoD vs distance saved to {lod_results_path} ({len(df_lod)} points{merged_note}; {target_label} {lod_results_path.name})")
    else:
        print(f"\n⚠️  No valid LoD data to save for use_ctrl={use_ctrl}")
    
    # Manual parent update for non-GUI backends
    if not hierarchy_supported:
        if overall_manual is None:
            overall_manual = pm.task(total=manual_total, description=f"{mode} Progress")
        overall_manual.update(1, description=f"{mode} - LoD vs Distance completed")

    # Around line 3889 in run_final_analysis.py
    if not df_lod.empty:
        cols_to_show = [c for c in ['distance_um', 'lod_nm', 'ser_at_lod', 'use_ctrl'] if c in df_lod.columns]
        print(f"\nLoD vs Distance (head) for {mode}:")
        print(df_lod[cols_to_show].head().to_string(index=False))

    if do_distance_metrics:
        df_distance_metrics = _run_distance_metric_sweep(
            cfg,
            seeds,
            mode,
            lod_distance_grid,
            data_dir,
            suffix,
            df_lod,
            args,
            pm,
            use_ctrl_flag,
            hierarchy_supported,
            mode_key,
            dist_key,
        )
        if dist_bar:
            dist_bar.close()
        if not hierarchy_supported:
            if overall_manual is None:
                overall_manual = pm.task(total=manual_total, description=f"{mode} Progress")
            overall_manual.update(1, description=f"{mode} - SER/SNR vs Distance completed")
    else:
        if distance_policy == "never":
            print("\n2.5. SER/SNR vs distance sweep skipped (--distance-sweep=never).")
        elif not lod_distance_grid:
            print("\n2.5. SER/SNR vs distance sweep skipped: no distance grid available.")

    if mode == "Hybrid":
        print("\n2′. Updating Hybrid HDS grid (Nm × distance)…")
        grid_csv = data_dir / "hybrid_hds_grid.csv"

        hds_distances = [int(float(d)) for d in lod_distance_grid] or [25, 50, 100, 150, 200]
        nm_min_default = int(cfg['pipeline'].get('lod_nm_min', 50))
        nm_max_default = int(cfg.get('pipeline', {}).get('lod_nm_max', cfg.get('lod_max_nm', 1000000)))
        default_nm_list: Sequence[Union[int, float]] = nm_values if nm_values else [500, 1000, 1600, 2500, 4000, 6300, 10000]

        cfg_hds_base = deepcopy(cfg)
        lod_by_distance: Dict[int, Dict[str, Any]] = {}
        for entry in lod_results:
            if not isinstance(entry, dict):
                continue
            dist_val = entry.get('distance_um')
            if dist_val is None:
                continue
            try:
                dist_int = int(float(dist_val))
            except Exception:
                continue
            lod_by_distance[dist_int] = entry

        hds_pending: Set[Future] = set()
        hds_futures: Dict[Future, int] = {}
        hds_rows: List[pd.DataFrame] = []

        for dist in sorted(set(hds_distances)):
            lod_state = _lod_state_load(mode, float(dist), bool(cfg['pipeline'].get('use_control_channel', True)))
            lod_entry = lod_by_distance.get(dist)
            nm_candidates = _derive_hds_nm_values(lod_entry, lod_state, default_nm_list, nm_min_default, nm_max_default)
            if not nm_candidates:
                continue

            future = pool.submit(
                _compute_hds_distance,
                float(dist),
                cfg_hds_base,
                nm_candidates,
                seeds,
                args.progress,
                args.resume,
                args.debug_calibration,
                # Use the same per-distance cache tag as LoD so Hybrid seeds fall back cleanly
                # into the warmed state without fragmenting cache directories.
                f"d{dist}um"
            )
            hds_pending.add(future)
            hds_futures[future] = dist

        for done_future in as_completed(list(hds_pending)):
            dist = hds_futures.pop(done_future, -1)
            if dist == -1:
                continue
            try:
                df_result = done_future.result()
            except Exception as hds_exc:
                print(f"⚠️  HDS grid failed at distance {dist}: {hds_exc}")
                continue
            if isinstance(df_result, pd.DataFrame) and not df_result.empty:
                hds_rows.append(df_result)
        hds_pending.clear()

        if hds_rows:
            grid = pd.concat(hds_rows, ignore_index=True)
            nm_key = 'pipeline_Nm_per_symbol' if 'pipeline_Nm_per_symbol' in grid.columns else 'pipeline.Nm_per_symbol'
            subset_cols = ['distance_um', nm_key]
            if 'use_ctrl' in grid.columns:
                subset_cols.append('use_ctrl')
            grid = grid.drop_duplicates(subset=subset_cols, keep='last').sort_values(subset_cols)
            _atomic_write_csv(grid_csv, grid)
            print(f"✅ HDS grid saved to {grid_csv} ({len(grid)} rows)")
        else:
            print("⚠️ HDS grid: no rows produced (skipping update).")

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
    if do_isi:
        print("\n3. Running ISI trade-off sweep (guard factor)…")
        
        # --- pick an anchor for ISI sweep (after LoD) ---
        d_ref = None
        nm_ref = None
        lod_canonical_path = data_dir / f"lod_vs_distance_{mode.lower()}{suffix}.csv"
        lod_anchor_path = lod_results_path if lod_results_path.exists() else lod_canonical_path
        if not lod_anchor_path.exists() and lod_canonical_path.exists():
            lod_anchor_path = lod_canonical_path
        if lod_anchor_path.exists():
            df_lod_all = pd.read_csv(lod_anchor_path)
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

        ctrl_flag = bool(cfg['pipeline'].get('use_control_channel', True))
        isi_csv, isi_csv_branch, isi_csv_other = _stage_csv_paths("isi", data_dir, mode, suffix, ctrl_flag)
        isi_results_path = isi_csv if not args.ablation_parallel else isi_csv_branch
        existing_isi_branch = _ensure_ablation_branch(
            "isi", isi_csv, isi_csv_branch, ctrl_flag, bool(args.resume), _dedupe_isi_dataframe
        )
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
                persist_csv=isi_csv_branch,
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
        isi_branch_combined = _update_branch_csv(
            "isi",
            isi_csv_branch,
            [df_isi],
            ctrl_flag,
            _dedupe_isi_dataframe,
            existing_isi_branch
        )

        if not args.ablation_parallel:
            _merge_branch_csv(isi_csv, [isi_csv_branch, isi_csv_other], _dedupe_isi_dataframe)

        try:
            df_isi_merged = pd.read_csv(isi_results_path) if isi_results_path.exists() else pd.DataFrame()
        except Exception:
            df_isi_merged = pd.DataFrame()

        if not isi_branch_combined.empty:
            if not df_isi_merged.empty:
                merged_note = f"; merged total {len(df_isi_merged)}" if not args.ablation_parallel else f"; branch total {len(df_isi_merged)}"
            else:
                merged_note = ""
            target_label = "branch" if args.ablation_parallel else "canonical"
            print(f"✅ ISI trade-off saved to {isi_results_path} ({len(isi_branch_combined)} points{merged_note}; {target_label} {isi_results_path.name})")
        elif not df_isi_merged.empty:
            status_label = "branch" if args.ablation_parallel else "merged"
            print(f"↩️  ISI trade-off already up-to-date for use_ctrl={ctrl_flag} ({status_label} total {len(df_isi_merged)})")
        else:
            print(f"⚠️  No ISI trade-off data to save for use_ctrl={ctrl_flag}")
        
        # Manual parent update for non-GUI backends
        if not hierarchy_supported:
            if overall_manual is None:
                overall_manual = pm.task(total=manual_total, description=f"{mode} Progress")
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
    global LOD_DEBUG_ENABLED
    if getattr(args, "lod_debug", False):
        os.environ[LOD_DEBUG_ENV_KEY] = "1"
        LOD_DEBUG_ENABLED = True
        # NOTE: LoD debug instrumentation (remove once diagnostics conclude)
        _lod_debug_log({"phase": "lod_debug_enabled"})
    else:
        os.environ.pop(LOD_DEBUG_ENV_KEY, None)
        LOD_DEBUG_ENABLED = False
    
    if args.merge_ablation_csvs:
        if args.mode and args.mode != "ALL":
            merge_modes = [args.mode]
        elif args.modes:
            merge_modes = ["MoSK", "CSK", "Hybrid"] if str(args.modes).lower() == "all" else [args.modes]
        else:
            merge_modes = ["MoSK", "CSK", "Hybrid"]
        print("🔗 Merging per-ablation CSV branches...")
        custom_dir = Path(args.merge_data_dir) if args.merge_data_dir else None
        merge_ablation_csvs_for_modes(merge_modes, args.merge_stages, custom_dir)
        return
    
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
