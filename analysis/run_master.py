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

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import time
import argparse
import shutil
import os
import stat
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    # GUI limitation: if user asked parallel_modes>1 + gui, fall back to rich for the child
    child_progress = "rich" if (args.progress == "gui" and args.parallel_modes and args.parallel_modes > 1) else args.progress
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
    # Interleaving across modes
    if args.parallel_modes and args.parallel_modes > 1:
        cmd.extend(["--parallel-modes", str(args.parallel_modes)])
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
    # Pass logging controls through to child as environment variables if needed (stdout is already tee'd)
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

    args = p.parse_args()

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
            
            # Force comprehensive coverage
            args.modes = "all"
            args.ablation = "both"
            args.supplementary = True
            
            # Optimize performance (but allow manual override)
            if args.max_workers is None and not args.extreme_mode and not args.beast_mode:
                args.beast_mode = True  # Use P-cores with margin
            
            print("🏆 IEEE preset applied: publication-grade configuration")
            print(f"   • Seeds: {args.num_seeds}, Sequences: {args.sequence_length}")
            print(f"   • Target CI: {args.target_ci}, All modes, Both ablations")
            print(f"   • Supplementary: {args.supplementary}, Performance: {'beast-mode' if args.beast_mode else 'default'}")

    # Initialize master-level logging
    if not args.no_log:
        setup_tee_logging(Path(args.logdir), prefix="run_master", fsync=args.fsync_logs)

    # Handle reset before any state is read or steps run
    if args.reset:
        print(f"⚠  Reset requested: {args.reset}. Deleting saved data under {RESULTS_DIR} ...")
        _reset_state(args.reset)

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
    overall = pm.task(total=len(steps), description="Master Pipeline", key="overall", kind="overall")
    sub = {s: pm.task(total=1, description=s.replace("_"," ").title(),
                      parent="overall", key=("step", s), kind="mode") for s in steps}

    state: Dict[str, Any] = _load_state() if args.resume and not args.reset else {}
    t0 = time.time()

    try:
        # --- Simulations per ablation state ---
        def _do_one_ablation(use_ctrl: bool) -> int:
            skey = "simulate_ctrl_on" if use_ctrl else "simulate_ctrl_off"
            if args.resume and state.get(skey, {}).get("done"):
                print(f"↩️  Resume: skipping {skey} (already done)")
                return 0
            cmd = _build_run_final_cmd(args, use_ctrl=use_ctrl)
            print(f"\n🧪 Simulate ({'CTRL' if use_ctrl else 'NoCTRL'}):\n  $ {' '.join(cmd)}\n")
            rc = _run(cmd)
            if rc == 0:
                _mark_done(state, skey)
            return rc

        if args.ablation_parallel and len(ablation_runs) == 2:
            # Launch CTRL ON/OFF concurrently (each will internally interleave modes if requested)
            with ThreadPoolExecutor(max_workers=2) as tpool:
                futs = [tpool.submit(_do_one_ablation, use) for use in ablation_runs]
                for f, use in zip(as_completed(futs), ablation_runs):
                    rc = f.result()
                    key = "simulate_ctrl_on" if use else "simulate_ctrl_off"
                    if rc != 0:
                        sub[key].close(); overall.close(); pm.stop(); sys.exit(rc)
                    sub[key].update(1); sub[key].close(); overall.update(1)
        else:
            for use in ablation_runs:
                rc = _do_one_ablation(use)
                key = "simulate_ctrl_on" if use else "simulate_ctrl_off"
                if rc != 0:
                    sub[key].close(); overall.close(); pm.stop(); sys.exit(rc)
                sub[key].update(1); sub[key].close(); overall.update(1)

        # --- Comparative plots (Fig.7/10/11) ---
        if not (args.resume and state.get("plots", {}).get("done")):
            rc = _run([sys.executable, "-u", "analysis/generate_comparative_plots.py"])
            if rc != 0:
                sub["plots"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "plots")
        sub["plots"].update(1); sub["plots"].close(); overall.update(1)

        # --- ISI trade-off ---
        if not (args.resume and state.get("isi", {}).get("done")):
            rc = _run([sys.executable, "-u", "analysis/plot_isi_tradeoff.py"])
            if rc != 0:
                sub["isi"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "isi")
        sub["isi"].update(1); sub["isi"].close(); overall.update(1)

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
            rc = _run(hybrid_cmd)
            if rc != 0:
                sub["hybrid"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "hybrid")
        sub["hybrid"].update(1); sub["hybrid"].close(); overall.update(1)

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
                    print(f"⚠️  Skipping optional notebook script: {script} (not found)")
                    continue
                rc = _run([sys.executable, "-u", script])
                if rc != 0:
                    sub["nb_replicas"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "nb_replicas")
        sub["nb_replicas"].update(1); sub["nb_replicas"].close(); overall.update(1)

        # --- Tables (Table I & II) ---
        if not (args.resume and state.get("tables", {}).get("done")):
            rc = _run([sys.executable, "-u", "analysis/param_table.py"])
            if rc != 0:
                sub["tables"].close(); overall.close(); pm.stop(); sys.exit(rc)
            rc = _run([sys.executable, "-u", "analysis/table_maker.py"])
            if rc != 0:
                sub["tables"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "tables")
        sub["tables"].update(1); sub["tables"].close(); overall.update(1)

        # --- Supplementary (optional) ---
        if "supplementary" in sub:
            if not (args.resume and state.get("supplementary", {}).get("done")):
                rc = _run([sys.executable, "-u", "analysis/generate_supplementary_figures.py",
                           "--strict", "--only-data"])
                if rc != 0:
                    sub["supplementary"].close(); overall.close(); pm.stop(); sys.exit(rc)
                _mark_done(state, "supplementary")
            sub["supplementary"].update(1); sub["supplementary"].close(); overall.update(1)

        if "appendix" in sub:
            if not (args.resume and state.get("appendix", {}).get("done")):
                rc = _run([sys.executable, "-u", "analysis/diagnose_csk.py"])
                if rc != 0:
                    sub["appendix"].close(); overall.close(); pm.stop(); sys.exit(rc)
                _mark_done(state, "appendix")
            sub["appendix"].update(1); sub["appendix"].close(); overall.update(1)

    finally:
        overall.close()
        pm.stop()

    elapsed = (time.time() - t0) / 60.0
    print(f"\n✓ All steps completed in {elapsed:.1f} min")
    print(f"Results in: {project_root / 'results'}")

if __name__ == "__main__":
    main()
