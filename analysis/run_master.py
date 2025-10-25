# analysis/run_master.py
"""
Master driver (Stage 11) — single command to reproduce all main-text results.

This script standardizes the end-to-end workflow:
  A) Simulate all modes (MoSK/CSK/Hybrid) with crash-safe resume
     • Includes ablation runs (with-CTRL and/or without-CTRL) via --ablation
     • Pass-through performance flags to analysis/run_final_analysis.py
  B) Generate comparative figures (Fig. 7/10/11)
  C) Plot ISI trade-off
  D) Plot Hybrid multidimensional benchmarks (Stage 10)
  E) Build Table I and Table II
  F) Optionally: build Supplementary figures (gated)

Usage (full suite, GUI, resume, ablation both):
    python analysis/run_master.py --modes all --resume --progress gui --ablation both

Usage (IEEE publication preset):
    python analysis/run_master.py --preset ieee --progress rich --resume
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, no_type_check
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

# Disable BLAS/OpenMP oversubscription for optimal process-level parallelism
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") 
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Import TclError for GUI exception handling
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tkinter import TclError
else:
    try:
        from tkinter import TclError
    except ImportError:
        # Fallback for environments without tkinter
        class TclError(Exception):
            """Fallback TclError for environments without tkinter"""
            pass

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.ui_progress import ProgressManager
from analysis.log_utils import setup_tee_logging
from analysis.maintenance_suite import (
    AVAILABLE_LIST_CATEGORIES,
    STAGE_REGISTRY,
    execute_maintenance_flags,
)

# Maintenance stage numbering shared with run_final_analysis:
#   Stage 1 - SER vs Nm sweeps
#   Stage 2 - LoD vs distance search + SER/SNR sweep
#   Stage 3 - Device FoM parameter sweeps
#   Stage 4 - Guard-frontier / ISI trade-off
#   Stage 5 - Main figures
#   Stage 6 - Supplementary figures
#   Stage 7 - Tables & summaries

try:
    from argparse import BooleanOptionalAction  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - fallback for Python < 3.9
    class BooleanOptionalAction(argparse.Action):  # type: ignore[no-redef]
        def __init__(self, option_strings, dest, default=None, **kwargs):
            if default is None:
                default = True
            super().__init__(option_strings=option_strings, dest=dest, nargs=0, default=default, **kwargs)
            self._negative = {opt for opt in option_strings if opt.startswith("--no-")}

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string in self._negative:
                setattr(namespace, self.dest, False)
            else:
                setattr(namespace, self.dest, True)

OECT_FIG_RELATIVE = [
    Path("results/figures/notebook_replicas/oect_differential_psd.png"),
    Path("results/figures/notebook_replicas/oect_noise_breakdown.png"),
]


def _log_oect_figures(prefix: str = "[oect]") -> None:
    """Emit a short status line for each expected OECT PSD figure."""
    print(f"{prefix} Expected PSD figures:")
    for rel_path in OECT_FIG_RELATIVE:
        full_path = project_root / rel_path
        status = "available" if full_path.exists() else "missing"
        print(f"{prefix}   - {rel_path} ({status})")


def _extend_nm_grid_flags(cmd: List[str], args: argparse.Namespace) -> None:
    """Append Nm grid overrides to the child command if provided."""
    if getattr(args, "nm_grid", ""):
        cmd.extend(["--nm-grid", args.nm_grid])
    for attr, flag in (
        ("nm_grid_mosk", "--nm-grid-mosk"),
        ("nm_grid_csk", "--nm-grid-csk"),
        ("nm_grid_hybrid", "--nm-grid-hybrid"),
    ):
        value = getattr(args, attr, "")
        if value:
            cmd.extend([flag, value])


DEFAULT_CONFIG_PATH = project_root / "config" / "default.yaml"


@lru_cache(maxsize=1)
def _load_default_config() -> Dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as cfg_file:
        return yaml.safe_load(cfg_file)


def _format_cli_number(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num.is_integer():
        return str(int(num))
    return str(num)


def _distance_cli_specs_from_config() -> List[str]:
    cfg = _load_default_config()
    dist_cfg = cfg.get("lod_distances_um", {}) or {}
    mode_order = ("MoSK", "CSK", "Hybrid")
    specs: List[str] = []
    for mode in mode_order:
        values = dist_cfg.get(mode)
        if not values:
            continue
        formatted = ",".join(_format_cli_number(v) for v in values)
        if formatted:
            specs.append(f"{mode}={formatted}")
    return specs


_STAGE_ALIAS_MAP = {
    "ser": 1,
    "ser_vs_nm": 1,
    "nm": 1,
    "lod": 2,
    "lod_vs_distance": 2,
    "distance": 2,
    "dist": 2,
    "ser_snr_vs_distance": 2,
    "device": 3,
    "device_fom": 3,
    "devicefom": 3,
    "guard": 4,
    "isi": 4,
    "guard_frontier": 4,
    "isi_tradeoff": 4,
    "fig": 5,
    "figure": 5,
    "figures": 5,
    "main": 5,
    "plots": 5,
    "supp": 6,
    "supplementary": 6,
    "appendix": 6,
    "table": 7,
    "tables": 7,
}


def _parse_stage_selection(spec: str) -> Set[int]:
    tokens = [token.strip() for token in str(spec or "").replace(";", ",").split(",") if token.strip()]
    if not tokens:
        return {1, 2, 3, 4, 5, 6, 7}
    stages: Set[int] = set()
    for token in tokens:
        lower = token.lower()
        if lower in ("all", "*"):
            return {1, 2, 3, 4, 5, 6, 7}
        if lower in _STAGE_ALIAS_MAP:
            stages.add(_STAGE_ALIAS_MAP[lower])
            continue
        try:
            stages.add(int(float(token)))
        except ValueError as exc:
            raise ValueError(f"Unrecognized stage token '{token}'") from exc
    return stages

STATE_FILE = project_root / "results" / "cache" / "run_master_state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = project_root / "results"
GUARD_FRONTIER_MODES = ["mosk", "csk", "hybrid"]

"""# Results Data README

This folder contains CSVs produced by `analysis/run_final_analysis.py`.

## Common columns
- `ser`: Symbol Error Rate (across seeds at that sweep point)
- `snr_db`: SNR proxy (semantics depend on mode; see below)
- `snr_semantics`: Human-readable note about how SNR is computed
- `symbols_evaluated`: Total symbols aggregated across seeds
- `use_ctrl`: True = CTRL subtraction enabled, False = not enabled
- `mode`: MoSK | CSK | Hybrid

## SNR semantics by mode
- **MoSK** and **Hybrid (MoSK bit)**: SNR from the **MoSK contrast statistic** (sign-aware DA vs SERO), using **pre-CTRL** correlation in the denominator.
- **CSK**: SNR from the **Q-statistic** used by the dual-channel combiner.

## OECT PSD figures
- `results/figures/notebook_replicas/oect_differential_psd.png`
- `results/figures/notebook_replicas/oect_noise_breakdown.png`

Notes:
- Hybrid amplitude decisions use CTRL subtraction when enabled; MoSK decisions do not.
- LoD CSVs include `data_rate_bps` and `symbol_period_s` at the LoD operating point.

"""

def _safe_close_progress(pm: ProgressManager, overall, sub_dict: Dict[str, Any], sub_key: Optional[str] = None) -> None:
    """Safely close progress managers with idempotent protection."""
    try:
        if sub_key and sub_key in sub_dict:
            sub_dict[sub_key].close()
    except Exception:
        pass
    try:
        overall.close()
    except Exception:
        pass
    try:
        pm.stop()
    except Exception:
        pass

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(data), indent=2), encoding='utf-8')
    tmp.replace(path)


def _parse_guard_variant(mode: str, path: Path) -> Optional[str]:
    stem = path.stem
    prefix = f"guard_frontier_{mode}"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix):]
    if suffix.startswith("_"):
        return suffix[1:]
    return "" if not suffix else suffix


def _discover_guard_frontier_variants() -> List[str]:
    data_dir = RESULTS_DIR / "data"
    if not data_dir.exists():
        return []
    variants: Set[str] = set()
    for mode in GUARD_FRONTIER_MODES:
        prefix = f"guard_frontier_{mode}"
        for path in data_dir.glob(f"{prefix}*.csv"):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            variant = _parse_guard_variant(mode, path)
            if variant is not None:
                variants.add(variant)
    # Sort with default (empty) first, then alphabetically
    return sorted(variants, key=lambda v: (v != "", v))


def _guard_frontier_ctrl_states(variant: str) -> Set[bool]:
    data_dir = RESULTS_DIR / "data"
    states: Set[bool] = set()
    suffix = f"_{variant}" if variant else ""
    required_cols = {"best_guard_factor", "max_irt_bps"}
    for mode in GUARD_FRONTIER_MODES:
        path = data_dir / f"guard_frontier_{mode}{suffix}.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                header = {col.strip() for col in reader.fieldnames if col}
                if not required_cols.issubset(header):
                    continue
                if 'use_ctrl' not in header:
                    states.add(True)
                    continue
                for row in reader:
                    val = (row or {}).get('use_ctrl')
                    if val is None:
                        continue
                    token = str(val).strip().lower()
                    if token in ("true", "1", "t", "yes"):
                        states.add(True)
                    elif token in ("false", "0", "f", "no"):
                        states.add(False)
        except Exception:
            continue
    return states


def _build_guard_frontier_commands(plot_script: Path) -> List[List[str]]:
    commands: List[List[str]] = []
    if not plot_script.exists():
        return commands
    for variant in _discover_guard_frontier_variants():
        ctrl_states = _guard_frontier_ctrl_states(variant)
        if not ctrl_states:
            continue
        for ctrl_state in sorted(ctrl_states, reverse=True):
            cmd = [sys.executable, "-u", str(plot_script)]
            if variant:
                cmd.extend(["--variant", variant])
            cmd.extend(["--ctrl", "on" if ctrl_state else "off"])
            commands.append(cmd)
    return commands

def _get_run_fingerprint(args: argparse.Namespace) -> str:
    """Generate a fingerprint for the current run configuration."""
    key_params = [
        args.modes,
        args.num_seeds,
        args.sequence_length,
        args.ablation,
        args.parallel_modes,
        getattr(args, 'preset', None),
        getattr(args, 'nm_grid', ''),
        '|'.join(getattr(args, 'distances', []) or []),
        getattr(args, 'nt_pairs', ''),
        args.target_ci,
        args.min_ci_seeds,
        args.lod_screen_delta,
        getattr(args, 'lod_num_seeds', None),
        getattr(args, 'lod_seq_len', None),
        args.baseline_isi,
        args.supplementary,
        # Add version/tag for future compatibility
        "v1.0"
    ]
    return str(hash(tuple(str(p) for p in key_params)))

def _load_state(args: argparse.Namespace) -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            # Check fingerprint compatibility
            current_fp = _get_run_fingerprint(args)
            stored_fp = state.get('_fingerprint')
            if stored_fp != current_fp:
                print(f"⚠️  Configuration changed - invalidating resume state")
                print(f"   Previous: {stored_fp}")
                print(f"   Current:  {current_fp}")
                return {"_fingerprint": current_fp}
            return state
        except Exception:
            return {"_fingerprint": _get_run_fingerprint(args)}
    return {"_fingerprint": _get_run_fingerprint(args)}

def _mark_done(state: Dict[str, Any], step: str) -> None:
    state[step] = {"done": True, "ts": time.time()}
    _atomic_write_json(STATE_FILE, state)

def _run(cmd: List[str]) -> int:
    """Run a command streaming output; return return-code. Propagate Ctrl+C."""
    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = 0x00000200  # CREATE_NEW_PROCESS_GROUP
    else:
        preexec_fn = getattr(os, 'setsid', None)
    proc = subprocess.Popen(cmd, cwd=project_root,
                            creationflags=creationflags, preexec_fn=preexec_fn)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\n^C received — stopping child process...", flush=True)
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                killpg = getattr(os, 'killpg', None)
                if killpg:
                    killpg(proc.pid, signal.SIGINT)
                else:
                    proc.terminate()
        except Exception:
            proc.terminate()
        return proc.wait()

def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWUSR)
        func(path)
    except Exception:
        pass

def _safe_rmtree(path: Path) -> None:
    root = (project_root / "results").resolve()
    target = path.resolve()
    if not str(target).startswith(str(root)):
        raise RuntimeError(f"Refusing to delete outside results/: {target}")
    if target.exists():
        shutil.rmtree(target, onerror=_on_rm_error)

def _reset_state(mode: str = "all") -> None:
    mode = (mode or "all").lower()
    if mode not in ("all", "cache"):
        raise ValueError(f"Unknown --reset mode: {mode}")
    if mode == "all":
        _safe_rmtree(RESULTS_DIR)
    else:
        _safe_rmtree(RESULTS_DIR / "cache")
        try:
            STATE_FILE.unlink(missing_ok=True)
        except TypeError:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
        for maybe_cached in ("simulations", "tmp", "intermediate"):
            _safe_rmtree(RESULTS_DIR / maybe_cached)
    (RESULTS_DIR / "cache").mkdir(parents=True, exist_ok=True)

def _build_run_final_cmd(args: argparse.Namespace, use_ctrl: bool) -> List[str]:
    """Assemble the run_final_analysis.py command line with pass-through flags."""
    # GUI limitation: only force fallback on macOS, where Tk must run in the main thread
    child_progress = args.progress
    if (args.progress == "gui" and platform.system() == "Darwin" and 
        ((args.parallel_modes and args.parallel_modes > 1) or args.ablation_parallel)):
        child_progress = "rich"
    cmd = [
        sys.executable, "-u", "analysis/run_final_analysis.py",
        "--mode", "ALL" if args.modes.lower() == "all" else args.modes,
        "--num-seeds", str(args.num_seeds),
        "--sequence-length", str(args.sequence_length),
        "--progress", child_progress,
        "--target-ci", str(args.target_ci),
        "--min-ci-seeds", str(args.min_ci_seeds),
        "--lod-screen-delta", str(args.lod_screen_delta),
    ]
    # Ablation flag
    cmd.append("--with-ctrl" if use_ctrl else "--no-ctrl")
    if getattr(args, 'channel_profile', 'tri') != 'tri':
        cmd.extend(['--channel-profile', args.channel_profile])
    if getattr(args, 'ablation_parallel', False):
        cmd.append("--ablation-parallel")
    if args.resume and not args.reset:
        cmd.append("--resume")
    if args.recalibrate:
        cmd.append("--recalibrate")
    # ISI baseline control
    if args.baseline_isi == "off":
        cmd.append("--disable-isi")
    # Performance flags
    if args.max_workers is not None:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if args.extreme_mode:
        cmd.append("--extreme-mode")
    elif args.beast_mode:
        cmd.append("--beast-mode")
    # NT-pairs (forward for CSK versatility)
    if args.nt_pairs:
        cmd.extend(["--nt-pairs", args.nt_pairs])
    # Nm grid override (forward sweep parameters)
    _extend_nm_grid_flags(cmd, args)
    stage_spec = getattr(args, "_run_stage_spec_data", "")
    if stage_spec:
        cmd.extend(["--run-stages", stage_spec])
    # Forward new optimization flags
    if args.distances:
        for dist_spec in args.distances:
            cmd.extend(["--distances", dist_spec])
    if args.lod_num_seeds is not None:
        cmd.extend(["--lod-num-seeds", str(args.lod_num_seeds)])
    if args.lod_seq_len is not None:
        cmd.extend(["--lod-seq-len", str(args.lod_seq_len)])
    if getattr(args, 'guard_samples_cap', None) is not None:
        cmd.extend(["--guard-samples-cap", str(args.guard_samples_cap)])
    if getattr(args, 'lod_skip_retry', False):
        cmd.append("--lod-skip-retry")
    if getattr(args, 'guard_samples_cap', None) is not None:
        cmd.extend(["--guard-samples-cap", str(args.guard_samples_cap)])
    if getattr(args, 'force_noise_resample', False):
        cmd.append("--force-noise-resample")
    if getattr(args, 'lod_validate_seq_len', None) is not None:
        cmd.extend(["--lod-validate-seq-len", str(args.lod_validate_seq_len)])
    if getattr(args, 'analytic_lod_bracket', False):
        cmd.append("--analytic-lod-bracket")
    if getattr(args, 'lod_debug', False):
        cmd.append("--lod-debug")
    # pass through optional LoD skips/limits:
    if getattr(args, 'max_ts_for_lod', None) is not None:
        cmd.extend(["--max-ts-for-lod", str(args.max_ts_for_lod)])
    if getattr(args, 'max_lod_validation_seeds', None) is not None:
        cmd.extend(["--max-lod-validation-seeds", str(args.max_lod_validation_seeds)])
    if getattr(args, 'max_symbol_duration_s', None) is not None:
        cmd.extend(["--max-symbol-duration-s", str(args.max_symbol_duration_s)])
    # Forward optimization parameters if specified
    if hasattr(args, 'cal_eps_rel') and args.cal_eps_rel != 0.01:
        cmd.extend(["--cal-eps-rel", str(args.cal_eps_rel)])
    if hasattr(args, 'cal_patience') and args.cal_patience != 2:
        cmd.extend(["--cal-patience", str(args.cal_patience)])
    if hasattr(args, 'cal_min_seeds') and args.cal_min_seeds != 4:
        cmd.extend(["--cal-min-seeds", str(args.cal_min_seeds)])
    if hasattr(args, 'cal_min_samples') and args.cal_min_samples != 50:
        cmd.extend(["--cal-min-samples", str(args.cal_min_samples)])
    if hasattr(args, 'min_decision_points') and args.min_decision_points != 4:
        cmd.extend(["--min-decision-points", str(args.min_decision_points)])

    # NEW: Forward decision window and ISI optimization flags
    if args.decision_window_policy is not None:
        cmd.extend(["--decision-window-policy", args.decision_window_policy])
    if args.decision_window_frac is not None:
        cmd.extend(["--decision-window-frac", str(args.decision_window_frac)])
    if args.allow_ts_exceed:
        cmd.append("--allow-ts-exceed")
    if args.ts_cap_s is not None:
        cmd.extend(["--ts-cap-s", str(args.ts_cap_s)])
    if args.isi_memory_cap is not None:
        cmd.extend(["--isi-memory-cap", str(args.isi_memory_cap)])
    if args.guard_factor is not None:
        cmd.extend(["--guard-factor", str(args.guard_factor)])
    if getattr(args, 'guard_samples_cap', None) is not None:
        cmd.extend(["--guard-samples-cap", str(args.guard_samples_cap)])
    if getattr(args, 'lod_distance_timeout_s', None) is not None:
        cmd.extend(["--lod-distance-timeout-s", str(args.lod_distance_timeout_s)])
    if args.ts_warn_only:
        cmd.append("--ts-warn-only")
    if hasattr(args, 'watchdog_secs') and args.watchdog_secs != 1800:
        cmd.extend(["--watchdog-secs", str(args.watchdog_secs)])
    if hasattr(args, 'nonlod_watchdog_secs') and args.nonlod_watchdog_secs != 3600:
        cmd.extend(["--nonlod-watchdog-secs", str(args.nonlod_watchdog_secs)])
    if hasattr(args, 'guard_max_ts') and args.guard_max_ts != 0.0:
        cmd.extend(["--guard-max-ts", str(args.guard_max_ts)])
    if getattr(args, 'lod_max_nm', None) is not None:
        cmd.extend(["--lod-max-nm", str(args.lod_max_nm)])
    if getattr(args, 'lod_distance_concurrency', None) is not None:
        cmd.extend(["--lod-distance-concurrency", str(args.lod_distance_concurrency)])
    if getattr(args, 'force_noise_resample', False):
        cmd.append("--force-noise-resample")
    if getattr(args, 'lod_skip_retry', False):
        cmd.append("--lod-skip-retry")

    # NEW: Forward SER auto-refine flags
    if args.ser_refine:
        cmd.append("--ser-refine")
    if args.ser_target != 0.01:
        cmd.extend(["--ser-target", str(args.ser_target)])
    if args.ser_refine_points != 4:
        cmd.extend(["--ser-refine-points", str(args.ser_refine_points)])

    # Forward adaptive CTRL arguments
    if getattr(args, 'ctrl_auto', False):
        cmd.append("--ctrl-auto")
    if hasattr(args, 'ctrl_rho_min_abs') and args.ctrl_rho_min_abs != 0.10:
        cmd.extend(["--ctrl-rho-min-abs", str(args.ctrl_rho_min_abs)])
    if hasattr(args, 'ctrl_snr_min_gain_db') and args.ctrl_snr_min_gain_db != 0.0:
        cmd.extend(["--ctrl-snr-min-gain-db", str(args.ctrl_snr_min_gain_db)])

    if getattr(args, 'store_calibration_stats', False):
        cmd.append("--store-calibration-stats")

    # Pass logging controls through to child...
    return cmd

def _build_run_final_cmd_for_mode(args: argparse.Namespace, mode: str, use_ctrl: bool, 
                                   concurrent_modes: int = 1) -> List[str]:
    """Assemble the run_final_analysis.py command line for a single mode.
    
    Args:
        args: Master arguments
        mode: Mode to run (MoSK/CSK/Hybrid)
        use_ctrl: Whether to use CTRL
        concurrent_modes: Number of modes running concurrently (for worker allocation)
    """
    # GUI limitation: only force fallback on macOS, where Tk must run in the main thread
    child_progress = args.progress
    if args.progress == "gui" and platform.system() == "Darwin" and args.ablation_parallel:
        child_progress = "rich"
        
    cmd = [
        sys.executable, "-u", "analysis/run_final_analysis.py",
        "--mode", mode,
        "--num-seeds", str(args.num_seeds),
        "--sequence-length", str(args.sequence_length),
        "--progress", child_progress,
        "--target-ci", str(args.target_ci),
        "--min-ci-seeds", str(args.min_ci_seeds),
        "--lod-screen-delta", str(args.lod_screen_delta),
    ]
    # Ablation flag
    cmd.append("--with-ctrl" if use_ctrl else "--no-ctrl")
    if getattr(args, 'channel_profile', 'tri') != 'tri':
        cmd.extend(['--channel-profile', args.channel_profile])
    # Resume / recalibrate
    if args.resume and not args.reset:
        cmd.append("--resume")
    if args.recalibrate:
        cmd.append("--recalibrate")
    # ISI baseline control
    if args.baseline_isi == "off":
        cmd.append("--disable-isi")
    # Performance flags with auto-splitting for concurrent modes
    child_workers = _child_max_workers(args, concurrent_modes)
    if child_workers is not None:
        cmd.extend(["--max-workers", str(child_workers)])
    if args.extreme_mode:
        cmd.append("--extreme-mode")
    elif args.beast_mode:
        cmd.append("--beast-mode")
    # NT-pairs (forward for CSK versatility)
    if args.nt_pairs:
        cmd.extend(["--nt-pairs", args.nt_pairs])
    # Nm grid override (forward sweep parameters)
    _extend_nm_grid_flags(cmd, args)
    stage_spec = getattr(args, "_run_stage_spec_data", "")
    if stage_spec:
        cmd.extend(["--run-stages", stage_spec])
    # Forward new optimization flags
    if args.distances:
        for dist_spec in args.distances:
            cmd.extend(["--distances", dist_spec])
    if args.lod_num_seeds is not None:
        cmd.extend(["--lod-num-seeds", str(args.lod_num_seeds)])
    if args.lod_seq_len is not None:
        cmd.extend(["--lod-seq-len", str(args.lod_seq_len)])
    if getattr(args, 'lod_validate_seq_len', None) is not None:
        cmd.extend(["--lod-validate-seq-len", str(args.lod_validate_seq_len)])
    if getattr(args, 'analytic_lod_bracket', False):
        cmd.append("--analytic-lod-bracket")
    if getattr(args, 'lod_debug', False):
        cmd.append("--lod-debug")
    # pass through optional LoD skips/limits:
    if getattr(args, 'max_ts_for_lod', None) is not None:
        cmd.extend(["--max-ts-for-lod", str(args.max_ts_for_lod)])
    if getattr(args, 'max_lod_validation_seeds', None) is not None:
        cmd.extend(["--max-lod-validation-seeds", str(args.max_lod_validation_seeds)])
    if getattr(args, 'max_symbol_duration_s', None) is not None:
        cmd.extend(["--max-symbol-duration-s", str(args.max_symbol_duration_s)])
    # Forward optimization parameters if specified
    if hasattr(args, 'cal_eps_rel') and args.cal_eps_rel != 0.01:
        cmd.extend(["--cal-eps-rel", str(args.cal_eps_rel)])
    if hasattr(args, 'cal_patience') and args.cal_patience != 2:
        cmd.extend(["--cal-patience", str(args.cal_patience)])
    if hasattr(args, 'cal_min_seeds') and args.cal_min_seeds != 4:
        cmd.extend(["--cal-min-seeds", str(args.cal_min_seeds)])
    if hasattr(args, 'cal_min_samples') and args.cal_min_samples != 50:
        cmd.extend(["--cal-min-samples", str(args.cal_min_samples)])
    if hasattr(args, 'min_decision_points') and args.min_decision_points != 4:
        cmd.extend(["--min-decision-points", str(args.min_decision_points)])

    # NEW: Forward decision window and ISI optimization flags
    if args.decision_window_policy is not None:
        cmd.extend(["--decision-window-policy", args.decision_window_policy])
    if args.decision_window_frac is not None:
        cmd.extend(["--decision-window-frac", str(args.decision_window_frac)])
    if args.allow_ts_exceed:
        cmd.append("--allow-ts-exceed")
    if args.ts_cap_s is not None:
        cmd.extend(["--ts-cap-s", str(args.ts_cap_s)])
    if args.isi_memory_cap is not None:
        cmd.extend(["--isi-memory-cap", str(args.isi_memory_cap)])
    if args.guard_factor is not None:
        cmd.extend(["--guard-factor", str(args.guard_factor)])
    if getattr(args, 'lod_distance_timeout_s', None) is not None:
        cmd.extend(["--lod-distance-timeout-s", str(args.lod_distance_timeout_s)])
    if args.ts_warn_only:
        cmd.append("--ts-warn-only")
    if hasattr(args, 'watchdog_secs') and args.watchdog_secs != 1800:
        cmd.extend(["--watchdog-secs", str(args.watchdog_secs)])
    if hasattr(args, 'nonlod_watchdog_secs') and args.nonlod_watchdog_secs != 3600:
        cmd.extend(["--nonlod-watchdog-secs", str(args.nonlod_watchdog_secs)])
    if hasattr(args, 'guard_max_ts') and args.guard_max_ts != 0.0:
        cmd.extend(["--guard-max-ts", str(args.guard_max_ts)])
    if getattr(args, 'lod_max_nm', None) is not None:
        cmd.extend(["--lod-max-nm", str(args.lod_max_nm)])
    if getattr(args, 'lod_distance_concurrency', None) is not None:
        cmd.extend(["--lod-distance-concurrency", str(args.lod_distance_concurrency)])

    # NEW: Forward SER auto-refine flags
    if args.ser_refine:
        cmd.append("--ser-refine")
    if args.ser_target != 0.01:
        cmd.extend(["--ser-target", str(args.ser_target)])
    if args.ser_refine_points != 4:
        cmd.extend(["--ser-refine-points", str(args.ser_refine_points)])

    # Forward adaptive CTRL arguments
    if getattr(args, 'ctrl_auto', False):
        cmd.append("--ctrl-auto")
    if hasattr(args, 'ctrl_rho_min_abs') and args.ctrl_rho_min_abs != 0.10:
        cmd.extend(["--ctrl-rho-min-abs", str(args.ctrl_rho_min_abs)])
    if hasattr(args, 'ctrl_snr_min_gain_db') and args.ctrl_snr_min_gain_db != 0.0:
        cmd.extend(["--ctrl-snr-min-gain-db", str(args.ctrl_snr_min_gain_db)])

    # Pass logging controls through to child...
    return cmd

def _clone_args_for_mode(args: argparse.Namespace, mode: str, use_ctrl: bool, shared_workers: Optional[int] = None) -> List[str]:
    """Create command line arguments for a specific mode with shared pool support."""
    cmd = [
        sys.executable, "-u", "analysis/run_final_analysis.py",
        "--mode", mode,
        "--num-seeds", str(args.num_seeds),
        "--sequence-length", str(args.sequence_length),
        "--progress", args.progress,
        "--target-ci", str(args.target_ci),
        "--min-ci-seeds", str(args.min_ci_seeds),
        "--lod-screen-delta", str(args.lod_screen_delta),
    ]
    
    # Ablation flag
    cmd.append("--with-ctrl" if use_ctrl else "--no-ctrl")
    
    # Resume / recalibrate
    if args.resume and not args.reset:
        cmd.append("--resume")
    if args.recalibrate:
        cmd.append("--recalibrate")
    
    # ISI baseline control
    if args.baseline_isi == "off":
        cmd.append("--disable-isi")
    
    # Worker allocation (shared pool or individual)
    if shared_workers is not None:
        cmd.extend(["--max-workers", str(shared_workers)])
    elif args.max_workers is not None:
        cmd.extend(["--max-workers", str(args.max_workers)])
    
    # Performance flags
    if args.extreme_mode:
        cmd.append("--extreme-mode")
    elif args.beast_mode:
        cmd.append("--beast-mode")
    
    # Pass through all other relevant arguments...
    if args.nt_pairs:
        cmd.extend(["--nt-pairs", args.nt_pairs])
    _extend_nm_grid_flags(cmd, args)
    if args.distances:
        for dist_spec in args.distances:
            cmd.extend(["--distances", dist_spec])
    if args.lod_num_seeds is not None:
        cmd.extend(["--lod-num-seeds", str(args.lod_num_seeds)])
    if args.lod_seq_len is not None:
        cmd.extend(["--lod-seq-len", str(args.lod_seq_len)])
    if getattr(args, 'lod_debug', False):
        cmd.append("--lod-debug")
    if getattr(args, 'force_noise_resample', False):
        cmd.append("--force-noise-resample")
    
    # Forward adaptive CTRL arguments
    if getattr(args, 'ctrl_auto', False):
        cmd.append("--ctrl-auto")
    if hasattr(args, 'ctrl_rho_min_abs') and args.ctrl_rho_min_abs != 0.10:
        cmd.extend(["--ctrl-rho-min-abs", str(args.ctrl_rho_min_abs)])
    if hasattr(args, 'ctrl_snr_min_gain_db') and args.ctrl_snr_min_gain_db != 0.0:
        cmd.extend(["--ctrl-snr-min-gain-db", str(args.ctrl_snr_min_gain_db)])

    if getattr(args, 'store_calibration_stats', False):
        cmd.append("--store-calibration-stats")

    return cmd

def _child_max_workers(args: argparse.Namespace, num_modes: int = 1) -> Optional[int]:
    """
    Calculate optimal worker allocation when running modes concurrently.
    Returns None to use default behavior, or specific worker count for load balancing.
    """
    if num_modes <= 1:
        return args.max_workers  # Single mode - pass through user's setting (None or specified)
    
    # For concurrent modes, calculate split
    if args.max_workers is not None:
        # User specified - split their value
        workers_per_mode = max(1, args.max_workers // num_modes)
    else:
        # Auto-detect and split
        import psutil
        total = psutil.cpu_count(logical=True) or os.cpu_count() or 16
        workers_per_mode = max(1, (total - 2) // num_modes)  # Reserve some for OS
    
    return workers_per_mode

@no_type_check
def main() -> None:  # pyright: ignore[reportGeneralTypeIssues]
    p = argparse.ArgumentParser(description="Master pipeline for tri-channel OECT paper")
    p.add_argument("--progress", choices=["gui", "rich", "tqdm", "none"], default="rich")
    p.add_argument("--resume", action="store_true", help="Resume completed steps")
    p.add_argument("-preset", "--preset", choices=["ieee", "verify", "production"], help="Apply preset configurations (ieee: publication-grade, verify: fast sanity, production: long-run batch)")
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--sequence-length", type=int, default=1000)
    p.add_argument("--recalibrate", action="store_true", help="Force recalibration (ignore JSON cache)")
    p.add_argument("--supplementary", action="store_true", help="Also generate supplementary figures")
    p.add_argument("--supplementary-include-synthetic", action="store_true",
                   help="Include synthetic supplementary panels (S3/S4/S6).")
    p.add_argument("--supplementary-allow-fallback", action="store_true",
                   help="Allow illustrative supplementary fallbacks when caches are missing.")
    p.add_argument("--baseline-isi", choices=["off", "on"], default="off",
                help="ISI state for baseline SER/LoD sweeps; ISI trade-off always runs ON.")
    p.add_argument("--shared-pool", action="store_true", dest="shared_pool",
                help="Run modes in shared process pool for maximum utilization")
    # Modes
    p.add_argument("--modes", choices=["MoSK", "CSK", "Hybrid", "all"], default="all")
    p.add_argument("--parallel-modes", type=int, default=1,
                   help="Run modes concurrently within each ablation run (e.g., 3 for all three)")
    # Quick-stage execution (Stage references mirror run_final_analysis)
    #   Stage 1 - SER vs Nm
#   Stage 2 - LoD vs distance + SER/SNR sweep
    #   Stage 3 - Device FoM sweeps
    #   Stage 4 - Guard frontier / ISI trade-off
    #   Stage 5 - Main figures
    #   Stage 6 - Supplementary figures
    #   Stage 7 - Tables & summaries
    p.add_argument(
        "--run-stages",
        type=str,
        default="all",
        help="Comma-separated stage list (or 'all') to execute via run_final_analysis stages (default: all stages).",
    )
    # CTRL ablation controller
    p.add_argument("--ablation", choices=["both", "on", "off"], default="both",
                   help="Run with CTRL (on), without CTRL (off), or both (default)")
    p.add_argument("--ablation-parallel", action="store_true",
                   help="Launch CTRL-on and CTRL-off runs concurrently (use with care)")
    # Device/plots options
    p.add_argument("--realistic-onsi", action="store_true",
                   help="Use cached simulation noise for ONSI calculation in hybrid benchmarks")
    p.add_argument("--nt-pairs", type=str, default="",
                    help="Comma-separated NT pairs for CSK sweeps, e.g. DA-5HT,DA-DA")
    p.add_argument("--csk-baselines", action="store_true",
                   help="Generate CSK single/dual baseline sweeps and comparison plots.")
    p.add_argument("--channel-suite", action="store_true",
                   help="Run physical channel baselines (single-channel DA and dual-channel DA+SERO).")
    p.add_argument("--validate-theory", action="store_true",
                   help="Run analytic BER/SEP and diffusion validation figure scripts.")
    p.add_argument("--gain-step", action=BooleanOptionalAction, default=True,
                   help="Run the ΔI/ΔQ gain plot step (use --no-gain-step to skip).")
    p.add_argument("--organoid-study", action=BooleanOptionalAction, default=True,
                   help="Run organoid sensitivity sweeps (alpha_H, bias, ionic strength).")
    p.add_argument("--nm-grid", type=str, default="",
                   help="Global Nm values for SER sweeps (pass-through to run_final_analysis).")
    p.add_argument("--nm-grid-mosk", type=str, default="",
                   help="Nm grid override for MoSK only.")
    p.add_argument("--nm-grid-csk", type=str, default="",
                   help="Nm grid override for CSK only.")
    p.add_argument("--nm-grid-hybrid", type=str, default="",
                   help="Nm grid override for Hybrid only.")
    p.add_argument("--extreme-mode", action="store_true", help="Pass through to run_final_analysis (max P-core threads)")
    p.add_argument("--beast-mode", action="store_true", help="Pass through to run_final_analysis (P-cores minus margin)")
    p.add_argument("--max-workers", type=int, default=None, help="Override worker count in run_final_analysis")
    p.add_argument("--store-calibration-stats", action="store_true",
                   help="Forward to analysis runner to persist full calibration statistics.")
    
    # NEW: Decision window and ISI optimization toggles
    p.add_argument("--decision-window-policy", choices=["fixed", "fraction_of_Ts", "full_Ts"], default=None,
                   help="Override decision window policy (default: use YAML config)")
    p.add_argument("--decision-window-frac", type=float, default=None,
                   help="Decision window fraction for fraction_of_Ts policy (0.1-1.0)")
    p.add_argument("--allow-ts-exceed", action="store_true", 
                   help="Allow Ts to exceed limits during LoD sweeps")
    p.add_argument("--ts-cap-s", type=float, default=None,
                   help="Symbol period cap in seconds (0 = no cap)")
    p.add_argument("--isi-memory-cap", type=int, default=None,
                   help="ISI memory cap in symbols (0 = no cap)")
    p.add_argument("--guard-factor", type=float, default=None,
                   help="Override guard factor for ISI calculations")
    p.add_argument("--guard-samples-cap", type=float, default=None,
                   help="Per-seed sample cap for guard sweeps (0 disables cap)")
    p.add_argument("--lod-skip-retry", action="store_true",
                   help="On resume, do not retry LoD distances that previously failed (keep NaN).")
    p.add_argument("--force-noise-resample", action="store_true",
                   help="Force rerunning zero-signal noise sweeps even if resume cache exists.")
    
    # ------ SER auto-refine near target SER ------
    p.add_argument("--ser-refine", action="store_true",
                   help="After coarse SER vs Nm sweep, auto-run a few Nm points that bracket the target SER.")
    p.add_argument("--ser-target", type=float, default=0.01,
                   help="Target SER for auto-refine (default: 0.01).")
    p.add_argument("--ser-refine-points", type=int, default=4,
                   help="How many log-spaced Nm points to add between the bracketing Nm pair (default: 4).")
    
    # Stage-13 tuning pass-through
    p.add_argument("--target-ci", type=float, default=0.0,
                   help="Stop adding seeds once Wilson 95%% CI half-width <= target; 0 disables (pass-through)")
    p.add_argument("--min-ci-seeds", type=int, default=6,
                   help="Minimum seeds before CI stopping can trigger (pass-through)")
    p.add_argument("--lod-screen-delta", type=float, default=1e-4,
                   help="Hoeffding screening significance for LoD binary search (pass-through)")
    p.add_argument(
        "--distances",
        action="append",
        default=None,
        metavar="MODE=LIST",
        help=("Comma-separated distance grid in um for LoD (pass-through). Example: --distances MoSK=25,35,45 --distances CSK=15,25"),
    )
    p.add_argument("--lod-num-seeds", type=str, default=None,
                help=("LoD seed schedule. N | min,max | rules like "
                        "'<=100:6,<=150:8,>150:10' (pass-through)"))
    p.add_argument("--lod-seq-len", type=int, default=None,
                help="Override sequence_length during LoD search only (pass-through)")
    p.add_argument("--lod-validate-seq-len", type=int, default=None,
                help="Override sequence_length during final LoD validation only (pass-through)")
    p.add_argument("--analytic-lod-bracket", action="store_true",
                help="Use Gaussian SER approximation for tighter LoD bracketing (pass-through)")
    p.add_argument("--lod-debug", action="store_true",
                help="Enable verbose LoD diagnostics (writes debug logs)")
    p.add_argument("--max-lod-validation-seeds", type=int, default=None,
                help="Cap #seeds for final LoD validation (pass-through)")
    p.add_argument("--max-symbol-duration-s", type=float, default=None,
                help="Skip LoD when dynamic Ts exceeds this (seconds; pass-through)")
    p.add_argument("--max-ts-for-lod", type=float, default=None,
                help="Optional Ts cutoff to skip LoD at a distance (pass-through)")
    p.add_argument("--lod-distance-timeout-s", type=float, default=7200.0,
                help="Per-distance time budget during LoD analysis. <=0 disables timeout (pass-through)")
    p.add_argument("--ts-warn-only", action="store_true",
                help="Issue warnings for long Ts instead of skipping (overrides all Ts limits; pass-through)")
    p.add_argument("--watchdog-secs", type=int, default=1800,
                   help="Soft timeout for seed completion before retry hint (default: 1800s/30min; pass-through)")
    p.add_argument("--nonlod-watchdog-secs", type=int, default=3600,
                   help="Timeout for non-LoD sweeps (pass-through).")
    p.add_argument("--guard-max-ts", type=float, default=0.0,
                   help="Cap guard-factor sweeps when symbol period exceeds this many seconds (0 disables; pass-through).")
    p.add_argument("--lod-max-nm", type=int, default=1000000,
                help="Upper bound for Nm during LoD search (default: 1000000; pass-through)")
    # Optimization tuning (pass-through to run_final_analysis.py)
    p.add_argument("--cal-eps-rel", type=float, default=0.01,
                   help="Adaptive calibration convergence threshold (pass-through)")
    p.add_argument("--cal-patience", type=int, default=2,
                   help="Calibration patience before stopping (pass-through)")
    p.add_argument("--cal-min-seeds", type=int, default=4,
                   help="Minimum seeds before early stopping (pass-through)")
    p.add_argument("--cal-min-samples", type=int, default=50,
                   help="Minimum samples per class for stable thresholds (pass-through)")
    p.add_argument("--min-decision-points", type=int, default=4,
                   help="Minimum time points for window guard (pass-through)")
    p.add_argument("--studies", type=str, default="",
                   help="Comma list from {sensitivity,capacity,isi-analytic}. Empty = none.")
    p.add_argument("--lod-distance-concurrency", type=int, default=8,
                help="How many distances to run concurrently in LoD sweep (default: 8).")
    # Reset
    p.add_argument(
        "--reset",
        nargs="?",
        choices=["cache", "all"],
        const="all",
        help="Reset simulator state. 'cache' removes caches/state only; 'all' (default) removes results/*"
    )
    # Logging
    p.add_argument("--logdir", default=str(project_root / "results" / "logs"),
                   help="Directory for log files")
    p.add_argument("--no-log", action="store_true",
                   help="Disable file logging")
    p.add_argument("--fsync-logs", action="store_true",
                   help="Force fsync on each write")
    p.add_argument("--inhibit-sleep", action="store_true",
                   help="Prevent the OS from sleeping while the pipeline runs")

    # Maintenance & cache management
    p.add_argument(
        "--maintenance-dry-run",
        action="store_true",
        help="Preview maintenance actions without deleting files.",
    )
    p.add_argument(
        "--maintenance-log",
        type=str,
        default=None,
        help="Path for maintenance log output (default: results/logs/maintenance.log). Use 'none' or '-' to disable logging.",
    )
    stage_help = ", ".join(f"{sid}:{info.name}" for sid, info in STAGE_REGISTRY.items())
    list_help = ", ".join(AVAILABLE_LIST_CATEGORIES)
    p.add_argument(
        "--list-maintenance",
        nargs="+",
        choices=AVAILABLE_LIST_CATEGORIES,
        metavar="CATEGORY",
        help=f"List maintenance resources (choices: {list_help}).",
    )
    p.add_argument(
        "--list-stages",
        action="store_true",
        help="Show maintenance stage numbering and descriptions.",
    )
    p.add_argument(
        "--maintenance-only",
        action="store_true",
        help="Run maintenance actions then exit without executing the master workflow.",
    )
    p.add_argument(
        "--clear-threshold-cache",
        nargs="*",
        metavar="MODE",
        default=None,
        help="Clear cached threshold JSON files (optionally filtered by MODE).",
    )
    p.add_argument(
        "--clear-lod-state",
        nargs="*",
        metavar="SPEC",
        default=None,
        help="Clear LoD caches/state. SPEC formats: 'all', 'mode', 'mode:distance', 'distance'.",
    )
    p.add_argument(
        "--clear-seed-cache",
        nargs="*",
        metavar="SWEEP",
        default=None,
        help="Clear cached seed payloads (e.g., 'lod_search', 'ser_vs_nm'). Use without values to clear all sweeps.",
    )
    p.add_argument(
        "--clear-distance",
        nargs="+",
        type=float,
        metavar="DIST",
        help="Remove LoD results for the given distance(s) in micrometres.",
    )
    p.add_argument(
        "--clear-nm",
        nargs="+",
        type=float,
        metavar="NM",
        help="Remove SER vs Nm data for the specified molecule counts.",
    )
    p.add_argument(
        "--reset-stage",
        nargs="+",
        metavar="STAGE",
        help=f"Reset entire stages/sweeps by number or alias ({stage_help}).",
    )
    p.add_argument(
        "--nuke-results",
        action="store_true",
        help="Delete the entire results/ directory before running the master workflow.",
    )
    
    # Adaptive CTRL control (pass-through)
    p.add_argument("--ctrl-auto", action="store_true",
                   help="Enable adaptive CTRL on/off (pass-through)")
    p.add_argument("--ctrl-rho-min-abs", type=float, default=0.10,
                   help="Minimum correlation threshold for CTRL (pass-through)")
    p.add_argument("--ctrl-snr-min-gain-db", type=float, default=0.0,
                   help="Minimum SNR gain for CTRL (pass-through)")
    
    p.add_argument("--keep-display-on", action="store_true",
                   help="Also keep the display awake (Windows/macOS)")

    args = p.parse_args()

    if getattr(args, "reset", None):
        if args.reset == "all":
            args.nuke_results = True
        elif args.reset == "cache":
            if args.clear_seed_cache is None:
                args.clear_seed_cache = []
            if args.clear_lod_state is None:
                args.clear_lod_state = []
            if args.clear_threshold_cache is None:
                args.clear_threshold_cache = []
            args.clear_seed_cache.append("all")
            args.clear_lod_state.append("all")
            args.clear_threshold_cache.append("all")
        args.reset = None

    maintenance_performed = execute_maintenance_flags(args, project_root)
    if getattr(args, "maintenance_only", False):
        if not maintenance_performed:
            print("Maintenance-only flag provided but no maintenance actions were executed.")
        return

    try:
        stage_selection = _parse_stage_selection(getattr(args, "run_stages", "all"))
    except ValueError as exc:
        print(f"[error] {exc}")
        sys.exit(2)
    supported_stages = {stage for stage in stage_selection if 1 <= stage <= 7}
    ignored_stages = sorted(stage_selection - supported_stages)
    if ignored_stages:
        ignored_str = ", ".join(str(stage) for stage in ignored_stages)
        print(f"⚠️  Ignoring unsupported stage(s) {ignored_str} in run_master.")
    if not supported_stages:
        print("⚠️  No stages selected (--run-stages). Nothing to do.")
        return

    args._run_stage_set = supported_stages
    args.run_stages = ",".join(str(stage) for stage in sorted(stage_selection))
    data_stage_tokens = [str(stage) for stage in (1, 2, 3, 4) if stage in supported_stages]
    args._run_stage_spec_data = ",".join(data_stage_tokens)

    if 6 in supported_stages:
        args.supplementary = True
    elif args.supplementary and 6 not in supported_stages:
        args.supplementary = False

    if args.distances is None:
        args.distances = []
    else:
        args.distances = [spec for spec in args.distances if spec]

    # Auto-enable sleep inhibition when GUI is requested
    if args.progress == "gui":
        args.inhibit_sleep = True

    # Apply preset configurations
    if args.preset:
        def _set_if_default(field: str, value):
            """Only override if the user left it at the parser default"""
            try:
                if getattr(args, field) == p.get_default(field):
                    setattr(args, field, value)
            except Exception:
                pass

        def _apply_ieee_base(lock_modes: bool = True) -> None:
            # Publication-grade statistical parameters
            _set_if_default("num_seeds", 50)
            _set_if_default("sequence_length", 2000)
            _set_if_default("target_ci", 0.004)       # Updated: align with new default (0.4%)
            _set_if_default("lod_screen_delta", 1e-3)  # Stronger Hoeffding screen
            _set_if_default("nonlod_watchdog_secs", 3600)

            # NEW: LoD search accelerators (search uses fewer resources, validation uses full)
            _set_if_default("lod_num_seeds", "<=100:6,<=150:8,>150:10")  # distance-aware schedule
            _set_if_default("lod_seq_len", 250)     # Updated: align with new default (250 symbols/seed)
            _set_if_default("lod_validate_seq_len", None)  # Updated: use full sequence length for validation
            _set_if_default("analytic_lod_bracket", True)  # enable analytic bracketing

            # NEW: Add these lines at the end of the IEEE preset:
            _set_if_default("max_lod_validation_seeds", 12)    # Cap expensive validation retries
            _set_if_default("max_symbol_duration_s", 180.0)    # Skip when Ts > 3 minutes
            _set_if_default("min_ci_seeds", 8)    # Match run_final_analysis.py default

            # Avoid infeasible tails: skip LoD when Ts is too large (no physics change)
            if not args.distances:
                cfg_specs = _distance_cli_specs_from_config()
                if cfg_specs:
                    args.distances = cfg_specs

            # Force comprehensive coverage
            if lock_modes:
                args.modes = "all"
            args.ablation = "both"
            args.supplementary = True
            _set_if_default("channel_suite", True)
            modes_value = getattr(args, "modes", "") or ""
            if isinstance(modes_value, str) and modes_value.lower() in ("csk", "all"):
                _set_if_default("csk_baselines", True)
            _set_if_default("validate_theory", True)
            _set_if_default("baseline_isi", "off")
            _set_if_default("studies", "sensitivity,capacity,isi-analytic")

            # Optimize performance (but allow manual override)
            if args.max_workers is None and not args.extreme_mode and not args.beast_mode:
                args.extreme_mode = True  # Use max P-core threads (changed from beast_mode)

        if args.preset == "verify":
            # Fast sanity check: minimal computation for quick verification
            _set_if_default("num_seeds", 4)
            _set_if_default("sequence_length", 200)
            _set_if_default("target_ci", 0.02)        # 95% CI half-width ≤ 2%
            _set_if_default("min_ci_seeds", 4)
            _set_if_default("lod_screen_delta", 1e-3) # more aggressive screening
            _set_if_default("parallel_modes", 3)      # interleave MoSK/CSK/Hybrid
            args.ablation = "on"                      # CTRL only (always override)
            args.modes = "all"                        # Test all modes (always override)

            print("🔧 Verify preset applied: fast sanity check configuration")
            print(f"   • Seeds: {args.num_seeds}, Sequences: {args.sequence_length}")
            print(f"   • Target CI: {args.target_ci}, Parallel modes: {args.parallel_modes}")
            print(f"   • CTRL only, All modes, Aggressive screening")

        elif args.preset == "ieee":
            _apply_ieee_base(lock_modes=True)

            print("🏆 IEEE preset applied: publication-grade configuration")
            print(f"   • Seeds: {args.num_seeds}, Sequences: {args.sequence_length}")
            lod_seeds_display = getattr(args, 'lod_num_seeds', '8')
            if isinstance(lod_seeds_display, str) and (',' in lod_seeds_display or ':' in lod_seeds_display):
                lod_seeds_display = f"{lod_seeds_display} (rule)"
            print(f"   • LoD search: {lod_seeds_display} × {getattr(args, 'lod_seq_len', 250)} symbols")
            print(f"   • Target CI: {args.target_ci}, All modes, Both ablations")
            print(f"   • Supplementary: {args.supplementary}, Performance: {'extreme-mode' if args.extreme_mode else 'default'}")

        elif args.preset == "production":
            _apply_ieee_base(lock_modes=False)
            _set_if_default("progress", "rich")
            _set_if_default("allow_ts_exceed", True)
            _set_if_default("lod_distance_timeout_s", 0.0)
            _set_if_default("watchdog_secs", 0)
            _set_if_default("nonlod_watchdog_secs", 3600)
            _set_if_default("ts_warn_only", True)
            _set_if_default("lod_max_nm", 1000000)
        _set_if_default("lod_seq_len", 600)
        _set_if_default("lod_validate_seq_len", 1000)
        _set_if_default("lod_distance_concurrency", 16)
        if not (getattr(args, "lod_analytic_noise", False) or getattr(args, "analytic_noise_all", False)):
            if getattr(args, "max_symbol_duration_s", None) in (None, 180.0):
                args.max_symbol_duration_s = 0.0
            if getattr(args, "guard_samples_cap", None) is None:
                args.guard_samples_cap = 0.0
            args.store_calibration_stats = True
            args.organoid_study = True

            search_len = getattr(args, "lod_seq_len", 250)
            validate_len = getattr(args, "lod_validate_seq_len", None)
            validate_display = validate_len if validate_len is not None else "full"
            print("🚀 Production preset applied: batch run configuration")
            print(f"   • Seeds: {args.num_seeds}, Sequences: {args.sequence_length}, Progress: {args.progress}")
            print(f"   • LoD search: {search_len} → validate {validate_display}, concurrency {args.lod_distance_concurrency}")
            print(f"   • Timeouts: per-distance {args.lod_distance_timeout_s}s, watchdog {args.watchdog_secs}s, non-LoD watchdog {args.nonlod_watchdog_secs}s, max Ts {args.max_symbol_duration_s}s")
            print(f"   • Flags: allow-ts-exceed={args.allow_ts_exceed}, ts-warn-only={args.ts_warn_only}, ser-refine={args.ser_refine}")

    # Initialize master-level logging
    if not args.no_log:
        setup_tee_logging(Path(args.logdir), prefix="run_master", fsync=args.fsync_logs)

    # Handle reset before any state is read or steps run
    if args.reset:
        print(f"⚠  Reset requested: {args.reset}. Deleting saved data under {RESULTS_DIR} ...")
        _reset_state(args.reset)

    class SleepInhibitor:
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
                        print("⚠️  Warning: Failed to restore execution state (Windows).")
            except Exception:
                pass
            try:
                if self._proc:
                    self._proc.terminate()
            except Exception:
                pass

    studies_set = {s.strip().lower() for s in (args.studies or "").split(",") if s.strip()}

    # Resolve ablation plan
    if args.ablation == "on":
        ablation_runs = [True]   # CTRL ON only
    elif args.ablation == "off":
        ablation_runs = [False]  # CTRL OFF only
    else:
        ablation_runs = [True, False]  # BOTH (default)

    if args.modes:
        mode_list = ["MoSK", "CSK", "Hybrid"] if args.modes.lower() == "all" else [args.modes]
    elif args.mode:
        mode_list = ["MoSK", "CSK", "Hybrid"] if args.mode == "ALL" else [args.mode]
    else:
        mode_list = ["MoSK", "CSK", "Hybrid"]

    # Enhanced session metadata for GUI header
    flag_list: List[str] = [
        f"--num-seeds={args.num_seeds}",
        f"--sequence-length={args.sequence_length}",
        f"--ablation={args.ablation}",
        f"--parallel-modes={args.parallel_modes}",
        f"--target-ci={args.target_ci}",
        f"--min-ci-seeds={args.min_ci_seeds}",
        f"--lod-screen-delta={args.lod_screen_delta}",
    ]
    if args.preset:
        flag_list.append(f"--preset={args.preset}")
    if args.nt_pairs:
        flag_list.append(f"--nt-pairs={args.nt_pairs}")
    if args.nm_grid:
        flag_list.append(f"--nm-grid={args.nm_grid}")
    for attr, flag in (
        ("nm_grid_mosk", "--nm-grid-mosk"),
        ("nm_grid_csk", "--nm-grid-csk"),
        ("nm_grid_hybrid", "--nm-grid-hybrid"),
    ):
        value = getattr(args, attr, "")
        if value:
            flag_list.append(f"{flag}={value}")
    if args.csk_baselines:
        flag_list.append("--csk-baselines")
    if args.channel_suite:
        flag_list.append("--channel-suite")
    if args.validate_theory:
        flag_list.append("--validate-theory")
    if args.recalibrate:
        flag_list.append("--recalibrate")
    if args.extreme_mode:
        flag_list.append("--extreme-mode")
    elif args.beast_mode:
        flag_list.append("--beast-mode")
    if args.store_calibration_stats:
        flag_list.append("--store-calibration-stats")
    if args.max_workers is not None:
        flag_list.append(f"--max-workers={args.max_workers}")
    if args.studies:
        flag_list.append(f"--studies={args.studies}")
    flag_list.append("--gain-step" if args.gain_step else "--no-gain-step")
    flag_list.append("--organoid-study" if args.organoid_study else "--no-organoid-study")
    flag_list.append("oect_psd")

    session_meta = {
        "modes": (["MoSK","CSK","Hybrid"] if args.modes.lower()=="all" else [args.modes]),
        "progress": args.progress,
        "resume": bool(args.resume and not args.reset),
        "with_ctrl": None if len(ablation_runs) == 2 else bool(ablation_runs[0]),
        "isi": args.baseline_isi == "on",  # Reflect actual baseline setting
        "flags": flag_list,
    }

    # Dynamic step plan: split simulate into per-ablation steps
    stage_set_active = getattr(args, '_run_stage_set', {1, 2, 3, 4, 5, 6, 7})
    run_data_stages = bool(stage_set_active & {1, 2, 3, 4})
    run_stage5 = 5 in stage_set_active
    run_stage6 = 6 in stage_set_active
    run_stage7 = 7 in stage_set_active

    if not run_data_stages:
        print("[stage] Stage 1-4 disabled via --run-stages; skipping simulation/data sweeps.")
    if not run_stage5:
        print("[stage] Stage 5 disabled via --run-stages; skipping main-figure rebuilds.")
    if not run_stage6:
        print("[stage] Stage 6 disabled via --run-stages; supplementary figures suppressed.")
    if not run_stage7:
        print("[stage] Stage 7 disabled via --run-stages; tables will not be regenerated.")

    steps: List[str] = []
    if run_data_stages:
        steps.extend(['simulate_ctrl_on' if use else 'simulate_ctrl_off' for use in ablation_runs])
        if args.csk_baselines:
            steps.append('csk_baselines')
        if args.channel_suite:
            steps.append('channel_suite')
        if args.validate_theory:
            steps.append('validate_theory')
    if run_stage5:
        steps.append('plots')
        if args.gain_step:
            steps.append('gain')
        steps += ['snr_ts', 'snr_panels', 'device_fom', 'oect_psd', 'guard_frontier', 'isi', 'hybrid', 'onsi_vs_rho', 'nb_replicas']
        if 'sensitivity' in studies_set:
            steps.append('study_sensitivity')
        if 'capacity' in studies_set:
            steps.append('study_capacity')
        if 'isi-analytic' in studies_set:
            steps.append('study_isi_analytic')
        if args.organoid_study:
            steps.append('organoid_sensitivity')
    if run_stage7:
        steps.append('tables')
    if run_stage6 and args.supplementary:
        steps.extend(['supplementary', 'appendix'])

    pm = ProgressManager(mode=args.progress, gui_session_meta=session_meta)
    
    # Add global stop mechanism
    master_cancelled = threading.Event()
    # Track all children to support concurrent kills
    current_processes: "set[subprocess.Popen]" = set()
    process_lock = threading.Lock()
    
    def stop_callback():
        """Called when GUI stop button is pressed."""
        print("\n🛑 GUI Stop button pressed - initiating graceful shutdown...")
        master_cancelled.set()
        
        # Terminate *all* active subprocesses
        with process_lock:
            procs = list(current_processes)
        for proc in procs:
            try:
                print(f"   → Terminating subprocess pid={getattr(proc, 'pid', '?')}")
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    killpg = getattr(os, 'killpg', None)
                    if killpg and hasattr(proc, 'pid'):
                        killpg(proc.pid, signal.SIGTERM)
                    else:
                        proc.terminate()
            except Exception as e:
                print(f"   ⚠️  Error terminating subprocess: {e}")
    
    # Connect stop callback to progress manager
    if hasattr(pm, 'set_stop_callback'):
        pm.set_stop_callback(stop_callback)
    else:
        print("⚠️  Stop button not available in this ProgressManager version")
    
    overall = pm.task(total=len(steps), description="Master Pipeline", key="overall", kind="overall")
    sub = {s: pm.task(total=1, description=s.replace("_"," ").title(),
                      parent="overall", key=("step", s), kind="mode") for s in steps}

    state: Dict[str, Any] = _load_state(args) if args.resume and not args.reset else {"_fingerprint": _get_run_fingerprint(args)}
    t0 = time.time()

    # Initialize ctx to prevent "possibly unbound" error
    ctx = None
    
    try:
        ctx = SleepInhibitor() if args.inhibit_sleep else None
        if ctx and args.keep_display_on:
            setattr(ctx, "_keep_display_on", True)
        if ctx:
            ctx.__enter__()
            
        # Enhanced _run function to track current process
        def _run_tracked(cmd: List[str]) -> int:
            """Run a command with process tracking for stop button."""
            if master_cancelled.is_set():
                print("🛑 Run cancelled before starting")
                return 130  # Standard cancellation exit code
            
            creationflags = 0
            preexec_fn = None
            if os.name == "nt":
                creationflags = 0x00000200  # CREATE_NEW_PROCESS_GROUP
            else:
                preexec_fn = getattr(os, 'setsid', None)
            
            proc = subprocess.Popen(
                cmd, cwd=project_root,
                creationflags=creationflags,
                preexec_fn=preexec_fn
            )
            with process_lock:
                current_processes.add(proc)
            
            try:
                # Poll for completion or cancellation
                while proc.poll() is None:
                    if master_cancelled.is_set():
                        print("🛑 Cancelling due to stop button...")
                        try:
                            if os.name == "nt":
                                proc.send_signal(signal.CTRL_BREAK_EVENT)
                            else:
                                killpg = getattr(os, 'killpg', None)
                                if killpg and hasattr(proc, 'pid'):
                                    killpg(proc.pid, signal.SIGTERM)
                                else:
                                    proc.terminate()
                        except Exception:
                            proc.terminate()
                        return 130
                    time.sleep(0.1)  # Check every 100ms
                
                return proc.returncode
                
            except KeyboardInterrupt:
                print("\n^C received — stopping child process...", flush=True)
                try:
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        killpg = getattr(os, 'killpg', None)
                        if killpg and hasattr(proc, 'pid'):
                            killpg(proc.pid, signal.SIGTERM)
                        else:
                            proc.terminate()
                except Exception:
                    proc.terminate()
                return proc.wait()
            finally:
                with process_lock:
                    current_processes.discard(proc)

        # --- Simulations per ablation state ---
        def _do_one_ablation(use_ctrl: bool) -> int:
            if master_cancelled.is_set():
                return 130
                
            skey = "simulate_ctrl_on" if use_ctrl else "simulate_ctrl_off"
            if args.resume and state.get(skey, {}).get("done"):
                print(f"↩️  Resume: skipping {skey} (already done)")
                return 0
            
            # Determine which modes to run
            modes = list(mode_list)
            if use_ctrl:
                modes = [m for m in modes if m.upper() != "MOSK"]
            if not modes:
                print(f"\n🧪 Simulate ({'CTRL' if use_ctrl else 'NoCTRL'}) — modes: [] (skipped)")
                _mark_done(state, skey)
                return 0
            print(f"\n🧪 Simulate ({'CTRL' if use_ctrl else 'NoCTRL'}) — modes: {modes}")

            # Run per-mode children concurrently (isolated process pools)
            maxp = min(args.parallel_modes or 1, len(modes))
            rcs = []
            
            if maxp > 1 and len(modes) > 1:
                # Concurrent mode execution
                actual_concurrent = min(maxp, len(modes))  # How many will actually run at once
                if getattr(args, 'shared_pool', False):
                    # Shared pool mode: Direct function call for maximum efficiency
                    print(f"  🔀 Shared pool mode: running {len(modes)} mode(s) concurrently...")
                    import sys
                    sys.path.insert(0, str(project_root / "analysis"))
                    import run_final_analysis as rfa    # type: ignore[import-not-found]
                    from analysis.run_final_analysis import parse_arguments as rfa_parse_args
                    
                    # IMPORTANT: In shared pool mode, use FULL worker count, not divided!
                    # All modes share the same pool, so we want maximum workers available
                    if args.extreme_mode:
                        # Let run_final_analysis determine the optimal P-core count
                        shared_workers = None  # Will use default extreme mode detection
                    elif args.beast_mode:
                        shared_workers = None  # Will use default beast mode detection  
                    elif args.max_workers is not None:
                        shared_workers = args.max_workers  # Use user's explicit value
                    else:
                        # Default: use all available minus a small margin
                        import psutil
                        total = psutil.cpu_count(logical=True) or os.cpu_count() or 16
                        shared_workers = max(1, total - 2)
                    
                    with ThreadPoolExecutor(max_workers=len(modes)) as tpool:
                        futs = []
                        for m in modes:
                            # Create a complete namespace with all required attributes for run_final_analysis
                            mode_args = argparse.Namespace()
                            
                            # Copy all existing args
                            for key, value in vars(args).items():
                                setattr(mode_args, key, value)
                            
                            # Mode-specific overrides
                            mode_args.mode = m
                            mode_args.modes = None  # Clear modes to use mode
                            mode_args.parallel_modes = 1  # Single mode per call
                            
                            # Set use_ctrl for this ablation run
                            mode_args.use_ctrl = use_ctrl
                            if use_ctrl:
                                mode_args.with_ctrl = True
                                mode_args.no_ctrl = False
                            else:
                                mode_args.with_ctrl = False
                                mode_args.no_ctrl = True
                            
                            mode_args.disable_isi = (args.baseline_isi == "off")
                            
                            # Worker allocation: DO NOT DIVIDE in shared pool mode!
                            if shared_workers is not None:
                                mode_args.max_workers = shared_workers
                            # Otherwise let run_final_analysis use its own detection
                            
                            # Hardcode run_final_analysis defaults to avoid argv parsing issues
                            rfa_defaults = argparse.Namespace(
                                verbose=False,
                                isi_sweep='always',
                                debug_calibration=False,
                                csk_level_scheme='uniform',
                                min_decision_points=4
                            )
                            
                            # Ensure all required attributes exist with defaults from run_final_analysis
                            # These are the critical attributes that run_final_analysis expects
                            required_attrs = {
                                # Master-level args (use getattr from args)
                                'progress': getattr(args, 'progress', 'tqdm'),
                                'num_seeds': getattr(args, 'num_seeds', 20),
                                'sequence_length': getattr(args, 'sequence_length', 1000),
                                'recalibrate': getattr(args, 'recalibrate', False),
                                'beast_mode': getattr(args, 'beast_mode', False),
                                'extreme_mode': getattr(args, 'extreme_mode', False),
                                'resume': getattr(args, 'resume', False),
                                'watchdog_secs': getattr(args, 'watchdog_secs', 1800),
                                'nonlod_watchdog_secs': getattr(args, 'nonlod_watchdog_secs', 3600),
                                'guard_max_ts': getattr(args, 'guard_max_ts', 0.0),
                                'target_ci': getattr(args, 'target_ci', 0.004),
                                'min_ci_seeds': getattr(args, 'min_ci_seeds', 8),
                                'lod_screen_delta': getattr(args, 'lod_screen_delta', 1e-4),
                                'run_stages': getattr(args, 'run_stages', 'all'),
                                '_run_stage_set': getattr(args, '_run_stage_set', {1, 2, 3, 4}),
                                '_run_stage_spec_data': getattr(args, '_run_stage_spec_data', ''),
                                
                                # run_final_analysis-only args (use hardcoded defaults)
                                'verbose': False,
                                'isi_sweep': 'always', 
                                'debug_calibration': False,
                                'csk_level_scheme': 'uniform',
                                'min_decision_points': 4,
                                'channel_profile': getattr(args, 'channel_profile', 'tri'),
                                
                                # Special handling for disable_isi (computed from baseline_isi)
                                'disable_isi': (args.baseline_isi == "off"),
                                
                                # Continue with master-level args...
                                'nt_pairs': getattr(args, 'nt_pairs', ''),
                                'distances': getattr(args, 'distances', None),
                                'lod_num_seeds': getattr(args, 'lod_num_seeds', None),
                                'lod_seq_len': getattr(args, 'lod_seq_len', None),
                                'lod_validate_seq_len': getattr(args, 'lod_validate_seq_len', None),
                                'logdir': getattr(args, 'logdir', str(project_root / "results" / "logs")),
                                'no_log': getattr(args, 'no_log', False),
                                'fsync_logs': getattr(args, 'fsync_logs', False),
                                'inhibit_sleep': getattr(args, 'inhibit_sleep', False),
                                'keep_display_on': getattr(args, 'keep_display_on', False),
                                'max_ts_for_lod': getattr(args, 'max_ts_for_lod', None),
                                'max_lod_validation_seeds': getattr(args, 'max_lod_validation_seeds', 12),
                                'max_symbol_duration_s': getattr(args, 'max_symbol_duration_s', None),
                                'analytic_lod_bracket': getattr(args, 'analytic_lod_bracket', True),
                                'cal_eps_rel': getattr(args, 'cal_eps_rel', 0.01),
                                'cal_patience': getattr(args, 'cal_patience', 2),
                                'cal_min_seeds': getattr(args, 'cal_min_seeds', 4),
                                'cal_min_samples': getattr(args, 'cal_min_samples', 50),
                                'nm_grid': getattr(args, 'nm_grid', ''),
                                'nm_grid_mosk': getattr(args, 'nm_grid_mosk', ''),
                                'nm_grid_csk': getattr(args, 'nm_grid_csk', ''),
                                'nm_grid_hybrid': getattr(args, 'nm_grid_hybrid', ''),
                                'decision_window_policy': getattr(args, 'decision_window_policy', None),
                                'decision_window_frac': getattr(args, 'decision_window_frac', None),
                                'allow_ts_exceed': getattr(args, 'allow_ts_exceed', False),
                                'ts_cap_s': getattr(args, 'ts_cap_s', None),
                                'force_noise_resample': getattr(args, 'force_noise_resample', False),
                                'isi_memory_cap': getattr(args, 'isi_memory_cap', None),
                                'guard_factor': getattr(args, 'guard_factor', None),
                                'lod_distance_timeout_s': getattr(args, 'lod_distance_timeout_s', 7200.0),
                                'lod_distance_concurrency': getattr(args, 'lod_distance_concurrency', 8),
                                'lod_max_nm': getattr(args, 'lod_max_nm', 1000000),
                                'ts_warn_only': getattr(args, 'ts_warn_only', False),
                                'ser_refine': getattr(args, 'ser_refine', False),
                                'ser_target': getattr(args, 'ser_target', 0.01),
                                'ser_refine_points': getattr(args, 'ser_refine_points', 4)
                            }
                            
                            # Set all required attributes, using existing values if present, defaults otherwise
                            for attr, default_value in required_attrs.items():
                                if not hasattr(mode_args, attr):
                                    setattr(mode_args, attr, default_value)
                            
                            futs.append(tpool.submit(rfa.run_one_mode, mode_args, m))
                        
                        for f in as_completed(futs):
                            if master_cancelled.is_set():
                                # Cancel remaining futures
                                for remaining in futs:
                                    remaining.cancel()
                                return 130
                            try:
                                f.result()
                                rcs.append(0)  # Track successful completion
                            except Exception as e:
                                print(f"💥 Shared pool mode execution failed: {e}")
                                rcs.append(1)  # Track failure
                        # Check if any mode failed
                        if any(rc != 0 for rc in rcs):
                            print(f"✗ One or more modes failed in shared pool mode: {skey}")
                            return 1
                else:
                    # Original behavior: each mode gets split worker allocation
                    with ThreadPoolExecutor(max_workers=maxp) as tpool:
                        futs = []
                        for m in modes:
                            cmd = _build_run_final_cmd_for_mode(args, m, use_ctrl, actual_concurrent)  # <-- CHANGED LINE
                            print(f"  $ {' '.join(cmd)}")
                            futs.append(tpool.submit(_run_tracked, cmd))
                        
                        for f in as_completed(futs):
                            if master_cancelled.is_set():
                                return 130
                            rc = f.result()
                            rcs.append(rc)
                            if rc == 130:
                                _safe_close_progress(pm, overall, sub, skey)
                                import sys
                                print("🛑 Master pipeline stopped by user")
                                sys.exit(130)
                            elif rc != 0:
                                _safe_close_progress(pm, overall, sub, skey)
                                import sys
                                sys.exit(rc)
            else:
                # Sequential mode execution (fallback)
                for m in modes:
                    if master_cancelled.is_set():
                        return 130
                    cmd = _build_run_final_cmd_for_mode(args, m, use_ctrl)  # <-- defaults to concurrent_modes=1
                    print(f"  $ {' '.join(cmd)}")
                    rc = _run_tracked(cmd)
                    rcs.append(rc)
                    if rc == 130:
                        print(f"🛑 Mode cancelled: {skey}")
                        return 130
                    elif rc != 0:
                        print(f"✗ Mode failed with exit code {rc}: {skey}")
                        return rc
            
            # All modes completed successfully
            if all(rc == 0 for rc in rcs):
                _mark_done(state, skey)
                return 0
            else:
                return 130 if any(rc == 130 for rc in rcs) else 1

        if run_data_stages:
            # Run ablations (check cancellation between each)
            if args.ablation_parallel and len(ablation_runs) == 2:
                # Run CTRL ON/OFF concurrently and still honor stop
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futs = {pool.submit(_do_one_ablation, use): use for use in ablation_runs}
                    for fut in as_completed(futs):
                        if master_cancelled.is_set():
                            # Cancel remaining futures
                            for f in futs:
                                f.cancel()
                            break
                        use = futs[fut]
                        rc = fut.result()
                        key = "simulate_ctrl_on" if use else "simulate_ctrl_off"
                        if rc == 130:  # Cancellation
                            _safe_close_progress(pm, overall, sub, key)
                            print("🛑 Master pipeline stopped by user")
                            sys.exit(130)
                        elif rc != 0:
                            _safe_close_progress(pm, overall, sub, key)
                            sys.exit(rc)
                        sub[key].update(1); sub[key].close(); overall.update(1)
            else:
                for use in ablation_runs:
                    if master_cancelled.is_set():
                        break
                    rc = _do_one_ablation(use)
                    key = "simulate_ctrl_on" if use else "simulate_ctrl_off"
                    if rc == 130:  # Cancellation
                        _safe_close_progress(pm, overall, sub, key)
                        print("🛑 Master pipeline stopped by user")
                        sys.exit(130)
                    elif rc != 0:
                        _safe_close_progress(pm, overall, sub, key)
                        sys.exit(rc)
                    sub[key].update(1); sub[key].close(); overall.update(1)

            if len(ablation_runs) == 2 and not master_cancelled.is_set():
                print("\n🔗 Finalizing per-ablation CSV merges…")
                for mode in mode_list:
                    if master_cancelled.is_set():
                        break
                    merge_cmd = [sys.executable, "-u", "analysis/run_final_analysis.py",
                                 "--merge-ablation-csvs", "--mode", mode]
                    print(f"  $ {' '.join(merge_cmd)}")
                    rc = _run_tracked(merge_cmd)
                    if rc == 130:
                        _safe_close_progress(pm, overall, sub)
                        print("🛑 Master pipeline stopped by user")
                        sys.exit(130)
                    elif rc != 0:
                        _safe_close_progress(pm, overall, sub)
                        sys.exit(rc)

            if args.csk_baselines:
                has_csk_mode = any(m.upper() == "CSK" for m in mode_list)
                if not has_csk_mode:
                    print("\n?? CSK baseline sweeps skipped (CSK mode not requested).")
                    _mark_done(state, "csk_baselines")
                    sub["csk_baselines"].update(1); sub["csk_baselines"].close(); overall.update(1)
                else:
                    if master_cancelled.is_set():
                        _safe_close_progress(pm, overall, sub)
                        print("?? Master pipeline stopped by user")
                        sys.exit(130)

                    if not (args.resume and state.get("csk_baselines", {}).get("done")):
                        print("\n?? Running CSK baseline sweeps (single vs dual)...")

                        def _exec_baseline(cmd: List[str]) -> None:
                            print(f"  $ {' '.join(cmd)}")
                            rc = _run_tracked(cmd)
                            if rc == 130:
                                _safe_close_progress(pm, overall, sub, "csk_baselines")
                                print("?? Master pipeline stopped by user")
                                sys.exit(130)
                            elif rc != 0:
                                _safe_close_progress(pm, overall, sub, "csk_baselines")
                                sys.exit(rc)

                        ser_baselines = [
                            (False, ["--variant", "single_DA_noctrl", "--csk-target", "DA", "--csk-dual", "off"]),
                            (False, ["--variant", "single_SERO_noctrl", "--csk-target", "SERO", "--csk-dual", "off"]),
                            (False, ["--variant", "dual_noctrl", "--csk-dual", "on"]),
                        ]
                        for use_ctrl, extra in ser_baselines:
                            cmd = _build_run_final_cmd_for_mode(args, "CSK", use_ctrl, concurrent_modes=1)
                            cmd.extend(extra)
                            _exec_baseline(cmd)

                        lod_baselines = [
                            (True, ["--variant", "single_DA_ctrl", "--csk-target", "DA", "--csk-dual", "off"]),
                            (True, ["--variant", "single_SERO_ctrl", "--csk-target", "SERO", "--csk-dual", "off"]),
                            (True, ["--variant", "dual_ctrl", "--csk-dual", "on"]),
                        ]
                        for use_ctrl, extra in lod_baselines:
                            cmd = _build_run_final_cmd_for_mode(args, "CSK", use_ctrl, concurrent_modes=1)
                            cmd.extend(extra)
                            _exec_baseline(cmd)

                        plot_script = project_root / "analysis" / "plot_csk_single_dual.py"
                        if plot_script.exists():
                            _exec_baseline([sys.executable, "-u", str(plot_script)])
                        else:
                            print("??  CSK baseline plot script not found; skipping plot step")

                        _mark_done(state, "csk_baselines")

                    sub["csk_baselines"].update(1); sub["csk_baselines"].close(); overall.update(1)
            if args.channel_suite:
                if master_cancelled.is_set():
                    _safe_close_progress(pm, overall, sub)
                    print("?? Master pipeline stopped by user")
                    sys.exit(130)

                key = "channel_suite"
                if not (args.resume and state.get(key, {}).get('done')):
                    print("\n?? Running physical channel suite (single & dual channels)...")

                    def _exec_suite(cmd: List[str]) -> None:
                        print(f"  $ {' '.join(cmd)}")
                        rc = _run_tracked(cmd)
                        if rc == 130:
                            _safe_close_progress(pm, overall, sub, key)
                            print("?? Master pipeline stopped by user")
                            sys.exit(130)
                        elif rc != 0:
                            _safe_close_progress(pm, overall, sub, key)
                            sys.exit(rc)

                    modes = ["MoSK", "CSK", "Hybrid"] if args.modes.lower() == "all" else [args.modes]
                    profiles = [("single", "single_physical", False), ("dual", "dual_physical", False)]
                    for profile, variant, use_ctrl in profiles:
                        print(f"   -> {profile.title()} profile")
                        for mode in modes:
                            cmd = _build_run_final_cmd_for_mode(args, mode, use_ctrl, concurrent_modes=1)
                            cmd.extend(["--channel-profile", profile, "--variant", variant])
                            if profile == 'single':
                                if mode.startswith('CSK'):
                                    cmd.extend(["--csk-target", "DA", "--csk-dual", "off"])
                                elif mode == 'Hybrid':
                                    cmd.extend(["--csk-dual", "off"])
                            elif profile == 'dual' and mode.startswith('CSK'):
                                cmd.extend(["--csk-dual", "on"])

                    plot_script = project_root / "analysis" / "plot_channel_profiles.py"
                    if plot_script.exists():
                        _exec_suite([sys.executable, "-u", str(plot_script)])
                    else:
                        print("??  Channel profile plot script not found; skipping plot step")

                    _mark_done(state, key)

                sub["channel_suite"].update(1); sub["channel_suite"].close(); overall.update(1)

            if args.validate_theory:
                if master_cancelled.is_set():
                    _safe_close_progress(pm, overall, sub)
                    print("?? Master pipeline stopped by user")
                    sys.exit(130)

                key = "validate_theory"
                if not (args.resume and state.get(key, {}).get("done")):
                    scripts: List[Path] = [
                        project_root / "analysis" / "validate_analytics.py",
                        project_root / "analysis" / "validate_transport_against_fick.py",
                        project_root / "analysis" / "rebuild_oect_figs.py",
                        project_root / "analysis" / "rebuild_binding_figs.py",
                        project_root / "analysis" / "rebuild_transport_figs.py",
                        project_root / "analysis" / "rebuild_pipeline_figs.py",
                    ]
                    for script in scripts:
                        if not script.exists():
                            print(f"??  Script not found: {script.name}")
                            continue
                        cmd = [sys.executable, "-u", str(script)]
                        rc = _run_tracked(cmd)
                        if rc == 130:
                            _safe_close_progress(pm, overall, sub, key)
                            print("?? Master pipeline stopped by user")
                            sys.exit(130)
                        elif rc != 0:
                            _safe_close_progress(pm, overall, sub, key)
                            sys.exit(rc)
                    _mark_done(state, key)

                sub[key].update(1); sub[key].close(); overall.update(1)


        # Check cancellation before each major step
        if master_cancelled.is_set():
            _safe_close_progress(pm, overall, sub)
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

        if "plots" in sub:
            # --- Comparative plots (Fig.7/10/11) ---
            if not (args.resume and state.get("plots", {}).get("done")):
                rc = _run_tracked([sys.executable, "-u", "analysis/generate_comparative_plots.py"])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "plots")
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "plots")
                    sys.exit(rc)
                _mark_done(state, "plots")
            sub["plots"].update(1); sub["plots"].close(); overall.update(1)

        if "gain" in sub:
            # --- Input-output gain plots ---
            if master_cancelled.is_set():
                _safe_close_progress(pm, overall, sub)
                print("\U0001f6d1 Master pipeline stopped by user")
                sys.exit(130)

            if not (args.resume and state.get("gain", {}).get("done")):
                gain_cmd = [sys.executable, "-u", "analysis/plot_input_output_gain.py"]
                rc = _run_tracked(gain_cmd)
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "gain")
                    print("\U0001f6d1 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "gain")
                    sys.exit(rc)
                _mark_done(state, "gain")
            sub["gain"].update(1); sub["gain"].close(); overall.update(1)

        # --- SNR vs Ts sweep ---
        if master_cancelled.is_set():
            _safe_close_progress(pm, overall, sub)
            print("\U0001f6d1 Master pipeline stopped by user")
            sys.exit(130)

        if "snr_ts" in sub:
            if not (args.resume and state.get("snr_ts", {}).get("done")):
                snr_modes = ["MoSK", "CSK", "Hybrid"] if args.modes.lower() == "all" else [args.modes]
                for mode in snr_modes:
                    ts_cmd = [sys.executable, "-u", "analysis/sweep_snr_vs_ts.py", "--mode", mode]
                    rc = _run_tracked(ts_cmd)
                    if rc == 130:
                        _safe_close_progress(pm, overall, sub, "snr_ts")
                        print("\U0001f6d1 Master pipeline stopped by user")
                        sys.exit(130)
                    elif rc != 0:
                        _safe_close_progress(pm, overall, sub, "snr_ts")
                        sys.exit(rc)
                _mark_done(state, "snr_ts")
            sub["snr_ts"].update(1); sub["snr_ts"].close(); overall.update(1)

        # --- SNR summary panels ---
        if master_cancelled.is_set():
            _safe_close_progress(pm, overall, sub)
            print("\U0001f6d1 Master pipeline stopped by user")
            sys.exit(130)

        if "snr_panels" in sub:
            if not (args.resume and state.get("snr_panels", {}).get("done")):
                panel_cmd = [sys.executable, "-u", "analysis/plot_snr_panels.py"]
                modes_arg = args.modes if hasattr(args, "modes") else getattr(args, "mode", "MoSK")
                if modes_arg:
                    panel_cmd.extend(["--modes", modes_arg])
                rc = _run_tracked(panel_cmd)
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "snr_panels")
                    print("\U0001f6d1 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "snr_panels")
                    sys.exit(rc)
                _mark_done(state, "snr_panels")
            sub["snr_panels"].update(1); sub["snr_panels"].close(); overall.update(1)

        if "device_fom" in sub:
            if master_cancelled.is_set():
                _safe_close_progress(pm, overall, sub)
                print("\U0001f6d1 Master pipeline stopped by user")
                sys.exit(130)
            if not (args.resume and state.get("device_fom", {}).get("done")):
                rc = _run_tracked([sys.executable, "-u", "analysis/plot_device_fom.py"])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "device_fom")
                    print("\U0001f6d1 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "device_fom")
                    sys.exit(rc)
                _mark_done(state, "device_fom")
            sub["device_fom"].update(1); sub["device_fom"].close(); overall.update(1)

        # --- OECT PSD & notebook replica figures ---
        if master_cancelled.is_set():
            _safe_close_progress(pm, overall, sub)
            print("\U0001f6d1 Master pipeline stopped by user")
            sys.exit(130)

        if "oect_psd" in sub:
            if not (args.resume and state.get("oect_psd", {}).get("done")):
                rc = _run_tracked([sys.executable, "-u", "analysis/rebuild_oect_figs.py"])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "oect_psd")
                    print("\U0001f6d1 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "oect_psd")
                    sys.exit(rc)
                _mark_done(state, "oect_psd")
                _log_oect_figures()
            else:
                _log_oect_figures(prefix="[oect-cached]")
            sub["oect_psd"].update(1); sub["oect_psd"].close(); overall.update(1)

        # Check cancellation before ISI step
        if master_cancelled.is_set():
            _safe_close_progress(pm, overall, sub)
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

        # Check cancellation before guard-frontier plots
        if master_cancelled.is_set():
            _safe_close_progress(pm, overall, sub)
            print("?? Master pipeline stopped by user")
            sys.exit(130)

        # --- Guard frontier plots ---
        if "guard_frontier" in sub:
            if not (args.resume and state.get("guard_frontier", {}).get("done")):
                guard_script = project_root / "analysis" / "plot_guard_frontier.py"
                guard_commands = _build_guard_frontier_commands(guard_script)
                if guard_commands:
                    for cmd in guard_commands:
                        rc = _run_tracked(cmd)
                        if rc == 130:
                            _safe_close_progress(pm, overall, sub, "guard_frontier")
                            print("?? Master pipeline stopped by user")
                            sys.exit(130)
                        elif rc != 0:
                            _safe_close_progress(pm, overall, sub, "guard_frontier")
                            sys.exit(rc)
                else:
                    print("?? Guard frontier plot skipped: no guard_frontier CSVs detected.")
                _mark_done(state, "guard_frontier")
            sub["guard_frontier"].update(1); sub["guard_frontier"].close(); overall.update(1)

        if "isi" in sub:
            # --- ISI trade-off ---
            if not (args.resume and state.get("isi", {}).get("done")):
                rc = _run_tracked([sys.executable, "-u", "analysis/plot_isi_tradeoff.py"])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "isi")
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "isi")
                    sys.exit(rc)
                _mark_done(state, "isi")
            sub["isi"].update(1); sub["isi"].close(); overall.update(1)

        # Check cancellation before hybrid step
        if master_cancelled.is_set():
            _safe_close_progress(pm, overall, sub)
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

        if "hybrid" in sub:
            # --- Hybrid multidimensional benchmarks (Stage 10) ---
            if not (args.resume and state.get("hybrid", {}).get("done")):
                hybrid_script = None
                for cand in ["analysis/plot_hybrid_multidim_benchmarks.py",
                             "analysis/generate_hybrid_multidim_benchmarks.py"]:
                    if (project_root / cand).exists():
                        hybrid_script = cand; break
                if hybrid_script is None:
                    print("✗ Hybrid benchmark script not found"); sys.exit(2)
                hybrid_cmd = [sys.executable, "-u", hybrid_script]
                if args.realistic_onsi:
                    hybrid_cmd.append("--realistic-onsi")
                rc = _run_tracked(hybrid_cmd)
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "hybrid")
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "hybrid")
                    sys.exit(rc)
                _mark_done(state, "hybrid")
            sub["hybrid"].update(1); sub["hybrid"].close(); overall.update(1)

        if "onsi_vs_rho" in sub:
            if master_cancelled.is_set():
                _safe_close_progress(pm, overall, sub)
                print("🛑 Master pipeline stopped by user")
                sys.exit(130)
            key = "onsi_vs_rho"
            if not (args.resume and state.get(key, {}).get("done")):
                rc = _run_tracked([sys.executable, "-u", "analysis/fig_onsi_vs_rho_cc.py"])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, key)
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, key)
                    sys.exit(rc)
                _mark_done(state, key)
            sub[key].update(1); sub[key].close(); overall.update(1)

        # Check cancellation before nb_replicas step
        if master_cancelled.is_set():
            _safe_close_progress(pm, overall, sub)
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

        if "nb_replicas" in sub:
            # --- Optional notebook-replica panels (if present) ---
            if not (args.resume and state.get("nb_replicas", {}).get("done")):
                scripts = [
                    project_root / "analysis" / "rebuild_oect_figs.py",
                    project_root / "analysis" / "rebuild_binding_figs.py",
                    project_root / "analysis" / "rebuild_transport_figs.py",
                    project_root / "analysis" / "rebuild_pipeline_figs.py",
                ]
                for script_path in scripts:
                    if not script_path.exists():
                        print(f"??  Script not found: {script_path.name}"); continue
                    rc = _run_tracked([sys.executable, "-u", str(script_path)])
                    if rc == 130:
                        _safe_close_progress(pm, overall, sub, "nb_replicas")
                        print("?? Master pipeline stopped by user")
                        sys.exit(130)
                    elif rc != 0:
                        _safe_close_progress(pm, overall, sub, "nb_replicas")
                        sys.exit(rc)
            sub["nb_replicas"].update(1); sub["nb_replicas"].close(); overall.update(1)

        # Check cancellation before tables step
        if master_cancelled.is_set():
            _safe_close_progress(pm, overall, sub)
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

        if "tables" in sub:
            # --- Tables (Table I & II) ---
            if not (args.resume and state.get("tables", {}).get("done")):
                rc = _run_tracked([sys.executable, "-u", "analysis/param_table.py"])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "tables")
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "tables")
                    sys.exit(rc)
                rc = _run_tracked([sys.executable, "-u", "analysis/table_maker.py"])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "tables")
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "tables")
                    sys.exit(rc)
                _mark_done(state, "tables")
            sub["tables"].update(1); sub["tables"].close(); overall.update(1)

        if "study_sensitivity" in sub:
            if master_cancelled.is_set():
                _safe_close_progress(pm, overall, sub)
                print("[stop] Master pipeline stopped by user")
                sys.exit(130)
            key = "study_sensitivity"
            if not (args.resume and state.get(key, {}).get("done")):
                print()
                print("[study] Running parameter sensitivity sweeps...")
                rc = _run_tracked([sys.executable, "-u", "analysis/sensitivity_study.py", "--progress", args.progress])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, key)
                    print("[stop] Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, key)
                    sys.exit(rc)
                _mark_done(state, key)
            sub[key].update(1); sub[key].close(); overall.update(1)

        if "study_capacity" in sub:
            if master_cancelled.is_set():
                _safe_close_progress(pm, overall, sub)
                print("[stop] Master pipeline stopped by user")
                sys.exit(130)
            key = "study_capacity"
            if not (args.resume and state.get(key, {}).get("done")):
                print()
                print("[study] Running capacity analysis...")
                rc = _run_tracked([sys.executable, "-u", "analysis/capacity_analysis.py", "--progress", args.progress])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, key)
                    print("[stop] Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, key)
                    sys.exit(rc)
                _mark_done(state, key)
            sub[key].update(1); sub[key].close(); overall.update(1)

        if "study_isi_analytic" in sub:
            if master_cancelled.is_set():
                _safe_close_progress(pm, overall, sub)
                print("[stop] Master pipeline stopped by user")
                sys.exit(130)
            key = "study_isi_analytic"
            if not (args.resume and state.get(key, {}).get("done")):
                print()
                print("[study] Building analytic ISI overlay...")
                rc = _run_tracked([sys.executable, "-u", "analysis/isi_analytic_model.py", "--progress", args.progress])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, key)
                    print("[stop] Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, key)
                    sys.exit(rc)
                _mark_done(state, key)
            sub[key].update(1); sub[key].close(); overall.update(1)

        if "organoid_sensitivity" in sub:
            if master_cancelled.is_set():
                _safe_close_progress(pm, overall, sub)
                print("[stop] Master pipeline stopped by user")
                sys.exit(130)
            key = "organoid_sensitivity"
            if not (args.resume and state.get(key, {}).get("done")):
                print()
                print("[study] Running organoid sensitivity sweeps...")
                organoid_cmd = [sys.executable, "-u", "analysis/organoid_sensitivity.py"]
                rc = _run_tracked(organoid_cmd)
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, key)
                    print("[stop] Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, key)
                    sys.exit(rc)
                _mark_done(state, key)
            sub[key].update(1); sub[key].close(); overall.update(1)

        # --- Supplementary (optional) ---
        if "supplementary" in sub:
            if master_cancelled.is_set():
                _safe_close_progress(pm, overall, sub)
                print("[stop] Master pipeline stopped by user")
                sys.exit(130)
            if not (args.resume and state.get("supplementary", {}).get("done")):
                supp_cmd = [sys.executable, "-u", "analysis/generate_supplementary_figures.py"]
                if args.supplementary_include_synthetic:
                    supp_cmd.append("--include-synthetic")
                if args.supplementary_allow_fallback:
                    supp_cmd.append("--allow-fallback")
                rc = _run_tracked(supp_cmd)
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "supplementary")
                    print("[stop] Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "supplementary")
                    sys.exit(rc)
                _mark_done(state, "supplementary")
            sub["supplementary"].update(1); sub["supplementary"].close(); overall.update(1)

        if "appendix" in sub:
            if master_cancelled.is_set():
                _safe_close_progress(pm, overall, sub)
                print("🛑 Master pipeline stopped by user")
                sys.exit(130)
            if not (args.resume and state.get("appendix", {}).get("done")):
                rc = _run_tracked([sys.executable, "-u", "analysis/diagnose_csk.py"])
                if rc == 130:
                    _safe_close_progress(pm, overall, sub, "appendix")
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    _safe_close_progress(pm, overall, sub, "appendix")
                    sys.exit(rc)
                _mark_done(state, "appendix")
            sub["appendix"].update(1); sub["appendix"].close(); overall.update(1)

    except KeyboardInterrupt:
        print("\n🛑 Master pipeline interrupted")
        _safe_close_progress(pm, overall, sub)
        sys.exit(130)
    finally:
        _safe_close_progress(pm, overall, sub)
        try:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        except Exception:
            pass

    if master_cancelled.is_set():
        print("🛑 Master pipeline stopped by user")
        sys.exit(130)

    elapsed = (time.time() - t0) / 60.0
    print(f"\n✓ All steps completed in {elapsed:.1f} min")
    print(f"Results in: {project_root / 'results'}")
    _log_oect_figures(prefix="[oect-summary]")

if __name__ == "__main__":
    main()
