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
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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

STATE_FILE = project_root / "results" / "cache" / "run_master_state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = project_root / "results"

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(data), indent=2), encoding='utf-8')
    tmp.replace(path)

def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}

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
    if args.progress == "gui" and args.parallel_modes and args.parallel_modes > 1:
        # Only force fallback on macOS, where Tk must run in the main thread.
        if platform.system() == "Darwin":
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
    if args.resume and not args.reset:
        cmd.append("--resume")
    if args.recalibrate:
        cmd.append("--recalibrate")
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
    # Forward new optimization flags
    if args.distances:
        cmd.extend(["--distances", args.distances])
    if args.lod_num_seeds is not None:
        cmd.extend(["--lod-num-seeds", str(args.lod_num_seeds)])
    if args.lod_seq_len is not None:
        cmd.extend(["--lod-seq-len", str(args.lod_seq_len)])
    # Add AFTER the existing max-ts-for-lod pass-through:
    if getattr(args, 'max_ts_for_lod', None) is not None:
        cmd.extend(["--max-ts-for-lod", str(args.max_ts_for_lod)])

    # ADD THESE MISSING LINES:
    if getattr(args, 'max_lod_validation_seeds', None) is not None:
        cmd.extend(["--max-lod-validation-seeds", str(args.max_lod_validation_seeds)])
    if getattr(args, 'max_symbol_duration_s', None) is not None:
        cmd.extend(["--max-symbol-duration-s", str(args.max_symbol_duration_s)])

    # Pass logging controls through to child...
    return cmd

def _build_run_final_cmd_for_mode(args: argparse.Namespace, mode: str, use_ctrl: bool) -> List[str]:
    """Assemble the run_final_analysis.py command line for a single mode (no --parallel-modes)."""
    # GUI limitation: only force fallback on macOS, where Tk must run in the main thread
    child_progress = args.progress
    if args.progress == "gui" and platform.system() == "Darwin":
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
    # Resume / recalibrate
    if args.resume and not args.reset:
        cmd.append("--resume")
    if args.recalibrate:
        cmd.append("--recalibrate")
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
    # Forward new optimization flags
    if args.distances:
        cmd.extend(["--distances", args.distances])
    if args.lod_num_seeds is not None:
        cmd.extend(["--lod-num-seeds", str(args.lod_num_seeds)])
    if args.lod_seq_len is not None:
        cmd.extend(["--lod-seq-len", str(args.lod_seq_len)])
    # Add AFTER the existing max-ts-for-lod pass-through:
    if getattr(args, 'max_ts_for_lod', None) is not None:
        cmd.extend(["--max-ts-for-lod", str(args.max_ts_for_lod)])

    # ADD THESE MISSING LINES:
    if getattr(args, 'max_lod_validation_seeds', None) is not None:
        cmd.extend(["--max-lod-validation-seeds", str(args.max_lod_validation_seeds)])
    if getattr(args, 'max_symbol_duration_s', None) is not None:
        cmd.extend(["--max-symbol-duration-s", str(args.max_symbol_duration_s)])

    # Pass logging controls through to child...
    return cmd

def main() -> None:
    p = argparse.ArgumentParser(description="Master pipeline for tri-channel OECT paper")
    p.add_argument("--progress", choices=["gui", "rich", "tqdm", "none"], default="rich")
    p.add_argument("--resume", action="store_true", help="Resume completed steps")
    p.add_argument("--preset", choices=["ieee", "verify"], help="Apply preset configurations (ieee: publication-grade, verify: fast sanity)")
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--sequence-length", type=int, default=1000)
    p.add_argument("--recalibrate", action="store_true", help="Force recalibration (ignore JSON cache)")
    p.add_argument("--supplementary", action="store_true", help="Also generate supplementary figures")
    # Modes
    p.add_argument("--modes", choices=["MoSK", "CSK", "Hybrid", "all"], default="all")
    p.add_argument("--parallel-modes", type=int, default=1,
                   help="Run modes concurrently within each ablation run (e.g., 3 for all three)")
    # CTRL ablation controller
    p.add_argument("--ablation", choices=["both", "on", "off"], default="both",
                   help="Run with CTRL (on), without CTRL (off), or both (default)")
    p.add_argument("--ablation-parallel", action="store_true",
                   help="Launch CTRL-on and CTRL-off runs concurrently (use with care)")
    # Device/plots options
    p.add_argument("--realistic-onsi", action="store_true",
                   help="Use cached simulation noise for ONSI calculation in hybrid benchmarks")
    p.add_argument("--nt-pairs", type=str, default="",
                   help="Comma-separated NT pairs for CSK sweeps, e.g. GLU-GABA,GLU-DA")
    # Performance pass-through
    p.add_argument("--extreme-mode", action="store_true", help="Pass through to run_final_analysis (max P-core threads)")
    p.add_argument("--beast-mode", action="store_true", help="Pass through to run_final_analysis (P-cores minus margin)")
    p.add_argument("--max-workers", type=int, default=None, help="Override worker count in run_final_analysis")
    # Stage-13 tuning pass-through
    p.add_argument("--target-ci", type=float, default=0.0,
                   help="Stop adding seeds once Wilson 95% CI half-width <= target; 0 disables (pass-through)")
    p.add_argument("--min-ci-seeds", type=int, default=6,
                   help="Minimum seeds before CI stopping can trigger (pass-through)")
    p.add_argument("--lod-screen-delta", type=float, default=1e-4,
                   help="Hoeffding screening significance for LoD binary search (pass-through)")
    p.add_argument("--distances", type=str, default="",
                   help="Comma-separated distance grid in µm for LoD (pass-through)")
    p.add_argument("--lod-num-seeds", type=int, default=None,
                   help="Use only this many seeds for LoD binary search (pass-through)")
    p.add_argument("--lod-seq-len", type=int, default=None,
                   help="Override sequence_length during LoD search only (pass-through)")
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
    p.add_argument("--keep-display-on", action="store_true",
                   help="Also keep the display awake (Windows/macOS)")

    args = p.parse_args()

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
            # Publication-grade statistical parameters
            _set_if_default("num_seeds", 50)
            _set_if_default("sequence_length", 2000)
            _set_if_default("target_ci", 0.002)       # Tighter confidence intervals
            _set_if_default("lod_screen_delta", 1e-3)  # Stronger Hoeffding screen
            
            # NEW: LoD search accelerators (search uses fewer resources, validation uses full)
            _set_if_default("lod_num_seeds", 8)     # search uses 8 seeds, later validated with full 50
            _set_if_default("lod_seq_len", 300)     # search uses 300 symbols/seed
            
                # NEW: Add these lines at the end of the IEEE preset:
            _set_if_default("max_lod_validation_seeds", 12)    # Cap expensive validation retries
            _set_if_default("max_symbol_duration_s", 180.0)    # Skip when Ts > 3 minutes
            
            # Avoid infeasible tails: skip LoD when Ts is too large (no physics change)
            if not hasattr(args, 'distances') or args.distances == "":
                # keep long points optional; user can restore by passing --max-ts-for-lod=None
                args.distances = "25,35,45,55,65,75,85,95,105,125,150,175"
            
            # Force comprehensive coverage
            args.modes = "all"
            args.ablation = "both"
            args.supplementary = True
            
            # Optimize performance (but allow manual override)
            if args.max_workers is None and not args.extreme_mode and not args.beast_mode:
                args.extreme_mode = True  # Use max P-core threads (changed from beast_mode)
            
            print("🏆 IEEE preset applied: publication-grade configuration")
            print(f"   • Seeds: {args.num_seeds}, Sequences: {args.sequence_length}")
            print(f"   • LoD search: {getattr(args, 'lod_num_seeds', 8)} seeds × {getattr(args, 'lod_seq_len', 300)} symbols")
            print(f"   • Target CI: {args.target_ci}, All modes, Both ablations")
            print(f"   • Supplementary: {args.supplementary}, Performance: {'extreme-mode' if args.extreme_mode else 'default'}")

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

    # Resolve ablation plan
    if args.ablation == "on":
        ablation_runs = [True]   # CTRL ON only
    elif args.ablation == "off":
        ablation_runs = [False]  # CTRL OFF only
    else:
        ablation_runs = [True, False]  # BOTH (default)

    # Enhanced session metadata for GUI header
    session_meta = {
        "modes": (["MoSK","CSK","Hybrid"] if args.modes.lower()=="all" else [args.modes]),
        "progress": args.progress,
        "resume": bool(args.resume and not args.reset),
        "with_ctrl": None if len(ablation_runs) == 2 else bool(ablation_runs[0]),
        "isi": True,
        "flags": [
            f"--num-seeds={args.num_seeds}",
            f"--sequence-length={args.sequence_length}",
            f"--ablation={args.ablation}",
            f"--parallel-modes={args.parallel_modes}",
            f"--target-ci={args.target_ci}",
            f"--min-ci-seeds={args.min_ci_seeds}",
            f"--lod-screen-delta={args.lod_screen_delta}",
        ] + ([f"--preset={args.preset}"] if args.preset else []) +  # NEW: Add preset to flags
            ([f"--nt-pairs={args.nt_pairs}"] if args.nt_pairs else []) +
            (["--recalibrate"] if args.recalibrate else []) +
            (["--extreme-mode"] if args.extreme_mode else (["--beast-mode"] if args.beast_mode else [])) +
            ([f"--max-workers={args.max_workers}"] if args.max_workers is not None else [])
    }

    # Dynamic step plan: split simulate into per-ablation steps
    steps: List[str] = []
    steps.extend(["simulate_ctrl_on" if use else "simulate_ctrl_off" for use in ablation_runs])
    steps += ["plots", "isi", "hybrid", "nb_replicas", "tables"]
    if args.supplementary:
        steps.extend(["supplementary", "appendix"])

    pm = ProgressManager(mode=args.progress, gui_session_meta=session_meta)
    
    # Add global stop mechanism
    master_cancelled = threading.Event()
    current_process: Optional[subprocess.Popen] = None
    process_lock = threading.Lock()
    
    def stop_callback():
        """Called when GUI stop button is pressed."""
        print("\n🛑 GUI Stop button pressed - initiating graceful shutdown...")
        master_cancelled.set()
        
        # Terminate current subprocess if any
        with process_lock:
            if current_process is not None:
                try:
                    print("   → Terminating active subprocess...")
                    if os.name == "nt":
                        current_process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        killpg = getattr(os, 'killpg', None)
                        if killpg and hasattr(current_process, 'pid'):
                            killpg(current_process.pid, signal.SIGTERM)
                        else:
                            current_process.terminate()
                except Exception as e:
                    print(f"   ⚠️  Error terminating subprocess: {e}")
                    try:
                        current_process.terminate()
                    except Exception:
                        pass
    
    # Connect stop callback to progress manager
    if hasattr(pm, 'set_stop_callback'):
        pm.set_stop_callback(stop_callback)
    else:
        print("⚠️  Stop button not available in this ProgressManager version")
    
    overall = pm.task(total=len(steps), description="Master Pipeline", key="overall", kind="overall")
    sub = {s: pm.task(total=1, description=s.replace("_"," ").title(),
                      parent="overall", key=("step", s), kind="mode") for s in steps}

    state: Dict[str, Any] = _load_state() if args.resume and not args.reset else {}
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
            nonlocal current_process
            
            if master_cancelled.is_set():
                print("🛑 Run cancelled before starting")
                return 130  # Standard cancellation exit code
            
            creationflags = 0
            preexec_fn = None
            if os.name == "nt":
                creationflags = 0x00000200  # CREATE_NEW_PROCESS_GROUP
            else:
                preexec_fn = getattr(os, 'setsid', None)
            
            with process_lock:
                current_process = subprocess.Popen(
                    cmd, cwd=project_root,
                    creationflags=creationflags, 
                    preexec_fn=preexec_fn
                )
            
            try:
                # Poll for completion or cancellation
                while current_process.poll() is None:
                    if master_cancelled.is_set():
                        print("🛑 Cancelling due to stop button...")
                        try:
                            if os.name == "nt":
                                current_process.send_signal(signal.CTRL_BREAK_EVENT)
                            else:
                                killpg = getattr(os, 'killpg', None)
                                if killpg:
                                    killpg(current_process.pid, signal.SIGTERM)
                                else:
                                    current_process.terminate()
                        except Exception:
                            current_process.terminate()
                        return 130
                    time.sleep(0.1)  # Check every 100ms
                
                return current_process.returncode
                
            except KeyboardInterrupt:
                print("\n^C received — stopping child process...", flush=True)
                try:
                    if os.name == "nt":
                        current_process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        killpg = getattr(os, 'killpg', None)
                        if killpg:
                            killpg(current_process.pid, signal.SIGTERM)
                        else:
                            current_process.terminate()
                except Exception:
                    current_process.terminate()
                return current_process.wait()
            finally:
                with process_lock:
                    current_process = None

        # --- Simulations per ablation state ---
        def _do_one_ablation(use_ctrl: bool) -> int:
            if master_cancelled.is_set():
                return 130
                
            skey = "simulate_ctrl_on" if use_ctrl else "simulate_ctrl_off"
            if args.resume and state.get(skey, {}).get("done"):
                print(f"↩️  Resume: skipping {skey} (already done)")
                return 0
            
            # Determine which modes to run
            modes = ["MoSK", "CSK", "Hybrid"] if args.modes.lower() == "all" else [args.modes]
            print(f"\n🧪 Simulate ({'CTRL' if use_ctrl else 'NoCTRL'}) — modes: {modes}")

            # Run per-mode children concurrently (isolated process pools)
            maxp = min(args.parallel_modes or 1, len(modes))
            rcs = []
            
            if maxp > 1 and len(modes) > 1:
                # Concurrent mode execution
                with ThreadPoolExecutor(max_workers=maxp) as tpool:
                    futs = []
                    for m in modes:
                        cmd = _build_run_final_cmd_for_mode(args, m, use_ctrl)
                        print(f"  $ {' '.join(cmd)}")
                        futs.append(tpool.submit(_run_tracked, cmd))
                    
                    for f in as_completed(futs):
                        if master_cancelled.is_set():
                            return 130
                        rc = f.result()
                        rcs.append(rc)
                        if rc == 130:
                            print(f"🛑 Mode cancelled: {skey}")
                            return 130
                        elif rc != 0:
                            print(f"✗ Mode failed with exit code {rc}: {skey}")
                            return rc
            else:
                # Sequential mode execution (fallback)
                for m in modes:
                    if master_cancelled.is_set():
                        return 130
                    cmd = _build_run_final_cmd_for_mode(args, m, use_ctrl)
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
                    if rc == 130:
                        sub[key].close(); overall.close(); pm.stop()
                        print("🛑 Master pipeline stopped by user")
                        sys.exit(130)
                    elif rc != 0:
                        sub[key].close(); overall.close(); pm.stop()
                        sys.exit(rc)
                    sub[key].update(1); sub[key].close(); overall.update(1)
        else:
            for use in ablation_runs:
                if master_cancelled.is_set():
                    break
                rc = _do_one_ablation(use)
                key = "simulate_ctrl_on" if use else "simulate_ctrl_off"
                if rc == 130:  # Cancellation
                    sub[key].close(); overall.close(); pm.stop()
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    sub[key].close(); overall.close(); pm.stop()
                    sys.exit(rc)
                sub[key].update(1); sub[key].close(); overall.update(1)

        # Check cancellation before each major step
        if master_cancelled.is_set():
            overall.close(); pm.stop()
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

        # --- Comparative plots (Fig.7/10/11) ---
        if not (args.resume and state.get("plots", {}).get("done")):
            rc = _run_tracked([sys.executable, "-u", "analysis/generate_comparative_plots.py"])
            if rc == 130:
                sub["plots"].close(); overall.close(); pm.stop()
                print("🛑 Master pipeline stopped by user")
                sys.exit(130)
            elif rc != 0:
                sub["plots"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "plots")
        sub["plots"].update(1); sub["plots"].close(); overall.update(1)

        # Check cancellation before ISI step
        if master_cancelled.is_set():
            overall.close(); pm.stop()
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

        # --- ISI trade-off ---
        if not (args.resume and state.get("isi", {}).get("done")):
            rc = _run_tracked([sys.executable, "-u", "analysis/plot_isi_tradeoff.py"])
            if rc == 130:
                sub["isi"].close(); overall.close(); pm.stop()
                print("🛑 Master pipeline stopped by user")
                sys.exit(130)
            elif rc != 0:
                sub["isi"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "isi")
        sub["isi"].update(1); sub["isi"].close(); overall.update(1)

        # Check cancellation before hybrid step
        if master_cancelled.is_set():
            overall.close(); pm.stop()
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

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
                sub["hybrid"].close(); overall.close(); pm.stop()
                print("🛑 Master pipeline stopped by user")
                sys.exit(130)
            elif rc != 0:
                sub["hybrid"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "hybrid")
        sub["hybrid"].update(1); sub["hybrid"].close(); overall.update(1)

        # Check cancellation before nb_replicas step
        if master_cancelled.is_set():
            overall.close(); pm.stop()
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

        # --- Optional notebook-replica panels (if present) ---
        if not (args.resume and state.get("nb_replicas", {}).get("done")):
            for script in [
                "analysis/rebuild_oect_figs.py",
                "analysis/rebuild_binding_figs.py",
                "analysis/rebuild_transport_figs.py",
                "analysis/rebuild_pipeline_figs.py",
            ]:
                script_path = Path(script)
                if not script_path.exists():
                    print(f"⚠️  Script not found: {script}"); continue
                rc = _run_tracked([sys.executable, "-u", script])
                if rc == 130:
                    sub["nb_replicas"].close(); overall.close(); pm.stop()
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    sub["nb_replicas"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "nb_replicas")
        sub["nb_replicas"].update(1); sub["nb_replicas"].close(); overall.update(1)

        # Check cancellation before tables step
        if master_cancelled.is_set():
            overall.close(); pm.stop()
            print("🛑 Master pipeline stopped by user")
            sys.exit(130)

        # --- Tables (Table I & II) ---
        if not (args.resume and state.get("tables", {}).get("done")):
            rc = _run_tracked([sys.executable, "-u", "analysis/param_table.py"])
            if rc == 130:
                sub["tables"].close(); overall.close(); pm.stop()
                print("🛑 Master pipeline stopped by user")
                sys.exit(130)
            elif rc != 0:
                sub["tables"].close(); overall.close(); pm.stop(); sys.exit(rc)
            rc = _run_tracked([sys.executable, "-u", "analysis/table_maker.py"])
            if rc == 130:
                sub["tables"].close(); overall.close(); pm.stop()
                print("🛑 Master pipeline stopped by user")
                sys.exit(130)
            elif rc != 0:
                sub["tables"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "tables")
        sub["tables"].update(1); sub["tables"].close(); overall.update(1)

        # --- Supplementary (optional) ---
        if "supplementary" in sub:
            if master_cancelled.is_set():
                overall.close(); pm.stop()
                print("🛑 Master pipeline stopped by user")
                sys.exit(130)
            if not (args.resume and state.get("supplementary", {}).get("done")):
                rc = _run_tracked([sys.executable, "-u", "analysis/generate_supplementary_figures.py",
                           "--strict", "--only-data"])
                if rc == 130:
                    sub["supplementary"].close(); overall.close(); pm.stop()
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    sub["supplementary"].close(); overall.close(); pm.stop(); sys.exit(rc)
                _mark_done(state, "supplementary")
            sub["supplementary"].update(1); sub["supplementary"].close(); overall.update(1)

        if "appendix" in sub:
            if master_cancelled.is_set():
                overall.close(); pm.stop()
                print("🛑 Master pipeline stopped by user")
                sys.exit(130)
            if not (args.resume and state.get("appendix", {}).get("done")):
                rc = _run_tracked([sys.executable, "-u", "analysis/diagnose_csk.py"])
                if rc == 130:
                    sub["appendix"].close(); overall.close(); pm.stop()
                    print("🛑 Master pipeline stopped by user")
                    sys.exit(130)
                elif rc != 0:
                    sub["appendix"].close(); overall.close(); pm.stop(); sys.exit(rc)
                _mark_done(state, "appendix")
            sub["appendix"].update(1); sub["appendix"].close(); overall.update(1)

    except KeyboardInterrupt:
        print("\n🛑 Master pipeline interrupted")
        overall.close()
        pm.stop()
        sys.exit(130)
    finally:
        overall.close()
        pm.stop()
        try:
            if ctx is not None:  # Changed from checking if 'ctx' in locals()
                ctx.__exit__(None, None, None)
        except Exception:
            pass

    if master_cancelled.is_set():
        print("🛑 Master pipeline stopped by user")
        sys.exit(130)

    elapsed = (time.time() - t0) / 60.0
    print(f"\n✓ All steps completed in {elapsed:.1f} min")
    print(f"Results in: {project_root / 'results'}")

if __name__ == "__main__":
    main()