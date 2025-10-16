#!/usr/bin/env python3
"""
Utility script to probe the LoD ceiling for each modulation mode.

The tool reuses the LoD search pipeline from ``analysis/run_final_analysis.py`` and
evaluates successively larger distances (default 5 μm increments) until the LoD
search can no longer meet the target SER within the configured Nm ceiling.
Use this to size a sensible distance grid before running the full analysis suite.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import yaml


# Keep BLAS/OpenMP usage predictable (matches analysis scripts).
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import heavy analysis helpers lazily after setting up sys.path.
from analysis import run_final_analysis as rfa  # noqa: E402


DEFAULT_SCHEDULE = "<=100:6,<=150:8,>150:10"
DEFAULT_MODES: Tuple[str, ...] = ("MoSK", "CSK", "Hybrid")
TARGET_SER_DEFAULT = 0.01
DEFAULT_STEP = 5


def _parse_schedule(schedule: str) -> List[str]:
    """Normalize the schedule string into rule tokens."""
    return [token.strip() for token in schedule.split(",") if token.strip()]


def _choose_seed_subset(
    distance_um: float,
    seeds: Sequence[int],
    schedule: Optional[str],
) -> List[int]:
    """
    Reproduce the LoD seed scheduling logic from run_final_analysis.

    Supports the three canonical formats:
      * "N" -> fixed number of seeds
      * "min,max" -> interpolate between min and max across 25–200 μm
      * rule list, e.g. "<=100:6,<=150:8,>150:10"
    """
    if schedule is None:
        return list(seeds)

    schedule = schedule.strip()
    if not schedule:
        return list(seeds)

    # Fast path: single integer request.
    if schedule.isdigit():
        count = max(1, int(schedule))
        return list(seeds[:count])

    tokens = _parse_schedule(schedule)

    # Format "min,max"
    if len(tokens) == 2 and all(token.replace("_", "").isdigit() for token in tokens):
        min_seeds, max_seeds = (int(token) for token in tokens)
        min_seeds = max(1, min_seeds)
        max_seeds = max(min_seeds, max_seeds)
        # Linear interpolation across 25–200 μm (default grid span).
        ratio = (distance_um - 25.0) / (200.0 - 25.0)
        ratio = max(0.0, min(1.0, ratio))
        count = int(round(min_seeds + ratio * (max_seeds - min_seeds)))
        return list(seeds[:max(1, count)])

    # Rule-based schedule: evaluate in order and stop at first match.
    for token in tokens:
        if ":" not in token:
            continue
        condition, value = token.split(":", 1)
        condition = condition.strip()
        try:
            count = max(1, int(value.strip()))
        except ValueError:
            continue

        try:
            threshold = float(condition.lstrip("<>="))
        except ValueError:
            threshold = None

        if threshold is None:
            continue

        if condition.startswith("<=") and distance_um <= threshold:
            return list(seeds[:count])
        if condition.startswith(">=") and distance_um >= threshold:
            return list(seeds[:count])
        if condition.startswith("<") and distance_um < threshold:
            return list(seeds[:count])
        if condition.startswith(">") and distance_um > threshold:
            return list(seeds[:count])

    # Fallback: last token might still be a bare integer.
    try:
        fallback = int(tokens[-1])
        return list(seeds[:max(1, fallback)])
    except (ValueError, IndexError):
        return list(seeds)


def _load_default_config(config_path: Path) -> Dict[str, Any]:
    """Load and preprocess the YAML configuration."""
    with config_path.open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)
    return rfa.preprocess_config_full(raw_cfg)


def _build_args_namespace(
    allow_ts_exceed: bool,
    lod_nm_max: int,
    max_ts_for_lod: Optional[float],
    full_seeds: Sequence[int],
    lod_seq_len: Optional[int],
) -> SimpleNamespace:
    """Create the lightweight Namespace expected by process_distance_for_lod."""
    return SimpleNamespace(
        allow_ts_exceed=allow_ts_exceed,
        ts_warn_only=False,
        lod_seq_len=lod_seq_len,
        lod_max_nm=lod_nm_max,
        max_ts_for_lod=max_ts_for_lod,
        full_seeds=list(full_seeds),
    )


def run_lod_ceiling_sweep(
    cfg_base: Dict[str, Any],
    mode: str,
    seeds: Sequence[int],
    distances: Iterable[float],
    target_ser: float,
    schedule: Optional[str],
    fail_patience: int,
    lod_seq_len: Optional[int],
    max_ts_for_lod: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Iterate over distances and record the LoD result at each point.

    Stops early once ``fail_patience`` consecutive distances fail to meet the
    target SER.
    """
    cfg_mode: Dict[str, Any] = deepcopy(cfg_base)
    pipeline_cfg = cast(Dict[str, Any], cfg_mode.setdefault("pipeline", {}))
    analysis_cfg = cast(Dict[str, Any], cfg_mode.setdefault("analysis", {}))

    pipeline_cfg["modulation"] = mode
    pipeline_cfg["enable_isi"] = bool(pipeline_cfg.get("enable_isi", True))
    pipeline_cfg["show_progress"] = False
    pipeline_cfg["verbose"] = False
    cfg_mode["disable_progress"] = True

    # Control channel defaults: MoSK uses differential statistics without CTRL.
    if mode == "MoSK":
        pipeline_cfg["use_control_channel"] = False
    else:
        pipeline_cfg["use_control_channel"] = bool(pipeline_cfg.get("use_control_channel", True))

    allow_ts_exceed = bool(analysis_cfg.get("allow_ts_exceed", False))
    lod_nm_max = int(pipeline_cfg.get("lod_nm_max", 1_000_000))

    args_ns = _build_args_namespace(
        allow_ts_exceed=allow_ts_exceed,
        lod_nm_max=lod_nm_max,
        max_ts_for_lod=max_ts_for_lod,
        full_seeds=seeds,
        lod_seq_len=lod_seq_len,
    )

    results: List[Dict[str, Any]] = []
    warm_guess: Optional[int] = None
    consecutive_failures = 0

    for dist in distances:
        dist = float(dist)
        lod_seeds = _choose_seed_subset(dist, seeds, schedule)
        res = cast(
            Dict[str, Any],
            rfa.process_distance_for_lod(
                dist_um=dist,
                cfg_base=cfg_mode,
                seeds=lod_seeds,
                target_ser=target_ser,
                debug_calibration=False,
                progress_cb=None,
                resume=False,
                args=args_ns,
                warm_lod_guess=warm_guess,
            ),
        )

        # Track warm start for faster convergence.
        lod_nm_val = res.get("lod_nm")
        if isinstance(lod_nm_val, (int, float, np.floating)):
            lod_nm_float = float(lod_nm_val)
        else:
            lod_nm_float = float("nan")
        if math.isfinite(lod_nm_float) and lod_nm_float > 0:
            warm_guess = int(lod_nm_float)
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        results.append(res)

        if fail_patience and consecutive_failures >= fail_patience:
            break

    return results


def summarise_results(results: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """Return the maximum distance with a valid LoD and its Nm value."""
    valid: List[Tuple[float, float]] = []
    for row in results:
        try:
            dist_val = float(row.get("distance_um", float("nan")))
            lod_val = float(row.get("lod_nm", float("nan")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dist_val) or not math.isfinite(lod_val) or lod_val <= 0:
            continue
        valid.append((dist_val, lod_val))
    if not valid:
        return None, None
    valid.sort(key=lambda pair: pair[0])
    return valid[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the LoD ceiling by sweeping distance in 5 μm increments.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to the YAML configuration (default: config/default.yaml).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated list of modes to scan (default: MoSK,CSK,Hybrid).",
    )
    parser.add_argument(
        "--start-um",
        type=float,
        default=15.0,
        help="Starting distance in μm (default: 15).",
    )
    parser.add_argument(
        "--stop-um",
        type=float,
        default=250.0,
        help="Final distance in μm to consider (default: 250).",
    )
    parser.add_argument(
        "--step-um",
        type=float,
        default=DEFAULT_STEP,
        help="Distance increment in μm (default: 5).",
    )
    parser.add_argument(
        "--target-ser",
        type=float,
        default=TARGET_SER_DEFAULT,
        help="Target SER for the LoD search (default: 0.01).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
        help="Number of Monte Carlo seeds to generate (default: 20).",
    )
    parser.add_argument(
        "--seed-schedule",
        type=str,
        default=DEFAULT_SCHEDULE,
        help="LoD seed schedule, e.g., '<=100:6,<=150:8,>150:10'.",
    )
    parser.add_argument(
        "--fail-patience",
        type=int,
        default=2,
        help="Stop after this many consecutive failed distances (default: 2).",
    )
    parser.add_argument(
        "--lod-seq-len",
        type=int,
        default=None,
        help="Optional sequence length override for the bracketing search.",
    )
    parser.add_argument(
        "--max-ts-for-lod",
        type=float,
        default=None,
        help="Skip distances whose symbol period exceeds this value (seconds).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV file to store per-distance results.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reserved for future use (no effect yet).",
    )
    return parser.parse_args()


def build_distance_grid(start_um: float, stop_um: float, step_um: float) -> List[float]:
    """Generate an inclusive distance grid honoring floating point steps."""
    if step_um <= 0:
        raise ValueError("--step-um must be positive")
    n_steps = int(math.floor((stop_um - start_um) / step_um)) + 1
    return [start_um + i * step_um for i in range(max(0, n_steps))]


def main() -> None:
    args = parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        if mode not in DEFAULT_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from {DEFAULT_MODES}.")

    cfg = _load_default_config(args.config)

    # Reproduce the deterministic seed schedule from run_final_analysis.
    seed_sequence = np.random.SeedSequence(2026)
    base_seeds = [int(seed) for seed in seed_sequence.generate_state(args.num_seeds)]

    distances = build_distance_grid(args.start_um, args.stop_um, args.step_um)

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for mode in modes:
        print(f"\n=== {mode} ===")
        results = run_lod_ceiling_sweep(
            cfg_base=cfg,
            mode=mode,
            seeds=base_seeds,
            distances=distances,
            target_ser=args.target_ser,
            schedule=args.seed_schedule,
            fail_patience=max(0, args.fail_patience),
            lod_seq_len=args.lod_seq_len,
            max_ts_for_lod=args.max_ts_for_lod,
        )
        all_results[mode] = results

        limit_dist, limit_nm = summarise_results(results)
        if limit_dist is None:
            print("  No LoD found within the scanned distance range.")
        else:
            print(f"  Max LoD distance: {limit_dist:.0f} μm with Nm ≈ {limit_nm:.0f}")

        for row in results:
            status = "OK" if row.get("lod_found") else f"FAIL ({row.get('skipped_reason')})"
            try:
                dist_display = float(row.get("distance_um", float("nan")))
            except (TypeError, ValueError):
                dist_display = float("nan")
            try:
                lod_display = float(row.get("lod_nm", float("nan")))
            except (TypeError, ValueError):
                lod_display = float("nan")
            try:
                ser_display = float(row.get("ser_at_lod", float("nan")))
            except (TypeError, ValueError):
                ser_display = float("nan")
            print(
                f"    d={dist_display:6.1f} μm | LoD={lod_display:>10.0f} | SER={ser_display:>7.4f} | {status}"
            )

    if args.output:
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - optional dependency
            raise SystemExit("pandas is required to write CSV output; please install it.")

        rows: List[Dict[str, Any]] = []
        for mode, records in all_results.items():
            for rec in records:
                row = dict(rec)
                row["mode"] = mode
                rows.append(row)
        df = pd.DataFrame(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
