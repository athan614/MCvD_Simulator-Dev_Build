# analysis/run_master.py
"""
Master driver (Stage 11) — single command to reproduce all main-text results.

This script standardizes the end-to-end workflow:
  1) Simulate all modes with crash-safe resume
  2) Generate comparative figures (Fig. 7/10/11)
  3) Plot ISI trade-off
  4) Plot Hybrid multidimensional benchmarks (Stage 10)
  5) Build Table I and Table II
  6) Optionally: build Supplementary figures (gated, Stage 12)

Usage (full suite):
    python analysis/run_master.py --modes all --resume --progress rich --supplementary
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

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.ui_progress import ProgressManager

STATE_FILE = project_root / "results" / "cache" / "run_master_state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = project_root / "results"

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(data), indent=2), encoding='utf-8')  # Add encoding='utf-8'
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
    """Run a command streaming output; return process return code."""
    proc = subprocess.Popen(cmd, cwd=project_root)
    return proc.wait()

def _on_rm_error(func, path, exc_info):
    """Make rmtree robust to read-only files on Windows, etc."""
    try:
        os.chmod(path, stat.S_IWUSR)
        func(path)
    except Exception:
        pass  # best effort

def _safe_rmtree(path: Path) -> None:
    """
    Remove a directory tree under results/ with guardrails.
    Refuses to delete outside results/.
    """
    root = (project_root / "results").resolve()
    target = path.resolve()
    if not str(target).startswith(str(root)):
        raise RuntimeError(f"Refusing to delete outside results/: {target}")
    if target.exists():
        shutil.rmtree(target, onerror=_on_rm_error)

def _reset_state(mode: str = "all") -> None:
    """
    Reset simulator state.
      - mode == 'cache': remove caches and state only
      - mode == 'all' : remove entire results/ tree
    Re-creates minimal skeleton needed for subsequent steps.
    """
    mode = (mode or "all").lower()
    if mode not in ("all", "cache"):
        raise ValueError(f"Unknown --reset mode: {mode}")

    if mode == "all":
        # Remove the entire results directory
        _safe_rmtree(RESULTS_DIR)
    else:
        # Cache-only: remove cache subdir(s) and the state file
        _safe_rmtree(RESULTS_DIR / "cache")
        try:
            STATE_FILE.unlink(missing_ok=True)
        except TypeError:
            # Python <3.8 compatibility
            if STATE_FILE.exists():
                STATE_FILE.unlink()

        # Optionally clear known intermediate/cached sub-dirs if you use them
        for maybe_cached in ("simulations", "tmp", "intermediate"):
            _safe_rmtree(RESULTS_DIR / maybe_cached)

    # Re-create minimum skeleton expected by the pipeline
    (RESULTS_DIR / "cache").mkdir(parents=True, exist_ok=True)

def main() -> None:
    p = argparse.ArgumentParser(description="Master pipeline for tri-channel OECT paper")
    p.add_argument("--progress", choices=["gui", "rich", "tqdm", "none"], default="rich")
    p.add_argument("--resume", action="store_true", help="Resume completed steps")
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--sequence-length", type=int, default=1000)
    p.add_argument("--recalibrate", action="store_true", help="Force recalibration (ignore JSON cache)")
    p.add_argument("--supplementary", action="store_true", help="Also generate supplementary figures")
    p.add_argument("--with-ctrl", dest="use_ctrl", action="store_true", help="Use CTRL differential subtraction")
    p.add_argument("--no-ctrl", dest="use_ctrl", action="store_false", help="Disable CTRL subtraction (ablation)")
    p.add_argument("--nt-pairs", type=str, default="", help="Comma-separated NT pairs for CSK sweeps, e.g. GLU-GABA,GLU-DA")
    p.add_argument("--modes", choices=["MoSK", "CSK", "Hybrid", "all"], default="all")
    p.add_argument("--realistic-onsi", action="store_true", 
                   help="Use cached simulation noise for ONSI calculation")
    p.add_argument(
        "--reset",
        nargs="?",
        choices=["cache", "all"],
        const="all",
        help="Reset simulator state. 'cache' removes caches/state only; 'all' (default) removes results/*"
    )
    p.set_defaults(use_ctrl=True)
    args = p.parse_args()

    # Handle reset before any state is read or steps run
    if args.reset:
        print(f"⚠  Reset requested: {args.reset}. Deleting saved data under {RESULTS_DIR} ...")
        _reset_state(args.reset)

    steps: List[str] = ["simulate", "plots", "isi", "hybrid", "tables"]
    if args.supplementary:
        steps.extend(["supplementary", "appendix"])

    pm = ProgressManager(mode=args.progress)
    overall = pm.task(total=len(steps), description="Overall")
    sub = {s: pm.task(total=1, description=s.capitalize()) for s in steps}

    state: Dict[str, Any] = _load_state() if args.resume and not args.reset else {}
    t0 = time.time()

    try:
        # 1) Simulation
        if not (args.resume and state.get("simulate", {}).get("done")):
            cmd = [
                sys.executable, "-u", "analysis/run_final_analysis.py",
                "--mode", "ALL" if args.modes.lower() == "all" else args.modes,
                "--num-seeds", str(args.num_seeds),
                "--sequence-length", str(args.sequence_length),
                "--progress", "rich" if args.progress == "gui" else args.progress,
                "--target-ci", "0.005",  # Add this line for Stage 13 adaptive seeds
            ]
            # Append '--resume' only if we didn't just reset and the user asked to resume
            if args.resume and not args.reset:
                cmd.append("--resume")
            if args.recalibrate:
                cmd.append("--recalibrate")
            if not args.use_ctrl:
                cmd.extend(["--no-ctrl"])
            if args.nt_pairs:
                cmd.extend(["--nt-pairs", args.nt_pairs])
            rc = _run(cmd)
            if rc != 0:
                sub["simulate"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "simulate")
        sub["simulate"].update(1); sub["simulate"].close(); overall.update(1)

        # 2) Comparative plots
        if not (args.resume and state.get("plots", {}).get("done")):
            rc = _run([sys.executable, "-u", "analysis/generate_comparative_plots.py"])
            if rc != 0:
                sub["plots"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "plots")
        sub["plots"].update(1); sub["plots"].close(); overall.update(1)

        # 3) ISI trade-off
        if not (args.resume and state.get("isi", {}).get("done")):
            rc = _run([sys.executable, "-u", "analysis/plot_isi_tradeoff.py"])
            if rc != 0:
                sub["isi"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "isi")
        sub["isi"].update(1); sub["isi"].close(); overall.update(1)

        # 4) Hybrid multidimensional benchmarks
        if not (args.resume and state.get("hybrid", {}).get("done")):
            # Tolerate either file name
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

        # 5) Tables
        if not (args.resume and state.get("tables", {}).get("done")):
            rc = _run([sys.executable, "-u", "analysis/param_table.py"])
            if rc != 0:
                sub["tables"].close(); overall.close(); pm.stop(); sys.exit(rc)
            rc = _run([sys.executable, "-u", "analysis/table_maker.py"])
            if rc != 0:
                sub["tables"].close(); overall.close(); pm.stop(); sys.exit(rc)
            _mark_done(state, "tables")
        sub["tables"].update(1); sub["tables"].close(); overall.update(1)

        # 6) Supplementary (optional, gated)
        if "supplementary" in steps:
            if not (args.resume and state.get("supplementary", {}).get("done")):
                rc = _run([sys.executable, "-u", "analysis/generate_supplementary_figures.py",
                           "--strict", "--only-data"])
                if rc != 0:
                    sub["supplementary"].close(); overall.close(); pm.stop(); sys.exit(rc)
                _mark_done(state, "supplementary")
            sub["supplementary"].update(1); sub["supplementary"].close(); overall.update(1)

        # 7) Appendix (CSK diagnostic; optional with supplementary)
        if "appendix" in steps:
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
