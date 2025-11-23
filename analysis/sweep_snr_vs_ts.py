#!/usr/bin/env python3
"""
Sweep symbol period scaling and record SNR/SER performance.

Generates results/data/snr_vs_ts_{mode}.csv and corresponding figures under
results/figures/, allowing review of SNR(dB) vs Ts trade-offs at a fixed
distance/Nm anchor.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures"

from analysis.ieee_plot_style import apply_ieee_style
from analysis.run_final_analysis import (
    preprocess_config_full,
    calculate_dynamic_symbol_period,
    calibrate_thresholds_cached,
    run_single_instance,
    calculate_snr_from_stats,
)
from src.pipeline import _resolve_decision_window


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SNR vs symbol period.")
    parser.add_argument(
        "--mode",
        choices=["MoSK", "CSK", "Hybrid"],
        default="MoSK",
        help="Modulation mode to evaluate.",
    )
    parser.add_argument(
        "--distance-um",
        type=float,
        default=None,
        help="Anchor distance (µm). Defaults to config or sensitivity anchor.",
    )
    parser.add_argument(
        "--nm",
        type=float,
        default=None,
        help="Fixed Nm per symbol (defaults to config Nm).",
    )
    parser.add_argument(
        "--ts-grid",
        type=str,
        default="0.25x,0.35x,0.5x,0.7x,1x,1.4x,2x,2.8x,4x,5.5x",
        help="Comma-separated Ts multipliers (defaults to 10 log-ish samples from 0.25x to 5.5x).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=6,
        help="Number of seeds per Ts point (default: 6).",
    )
    parser.add_argument(
        "--detector-mode",
        choices=["raw", "zscore", "whitened"],
        default=None,
        help="Optional detector override (default: config).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="snr_vs_ts",
        help="Base filename prefix for CSV and figure.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix for CSV/figure names.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if CSV already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip Ts points already present in the output CSV.",
    )
    return parser.parse_args()


def _load_cfg() -> Dict[str, Any]:
    cfg_path = project_root / "config" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg_raw = yaml.safe_load(fh)
    cfg = preprocess_config_full(cfg_raw)
    cfg.setdefault("pipeline", {})
    cfg.setdefault("sim", {})
    cfg.setdefault("detection", {})
    return cfg


def _anchor_distance(cfg: Dict[str, Any], mode: str) -> Tuple[float, float]:
    base_distance = float(cfg["pipeline"].get("distance_um", 50.0))
    base_nm = float(cfg["pipeline"].get("Nm_per_symbol", 2000.0))
    data_path = data_dir / f"ser_vs_nm_{mode.lower()}.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        if "distance_um" in df.columns and df["distance_um"].notna().any():
            majority = df["distance_um"].dropna().astype(float)
            if not majority.empty:
                base_distance = float(majority.mode().iloc[0])
        nm_cols = [
            "pipeline_Nm_per_symbol",
            "pipeline.Nm_per_symbol",
            "Nm_per_symbol",
        ]
        for col in nm_cols:
            if col in df.columns and df[col].notna().any():
                base_nm = float(df[col].median())
                break
    return base_distance, base_nm


def _calc_metric(results: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics = {
        "snr_db": float("nan"),
        "ser": float("nan"),
        "delta_Q_diff": float("nan"),
        "delta_I_diff": float("nan"),
        "delta_over_sigma_Q": float("nan"),
        "delta_over_sigma_I": float("nan"),
    }
    if not results:
        return metrics
    all_da = []
    all_sero = []
    charge_da = []
    charge_sero = []
    current_da = []
    current_sero = []
    total_symbols = 0.0
    total_errors = 0.0
    for res in results:
        all_da.extend(res.get("stats_da", []))
        all_sero.extend(res.get("stats_sero", []))
        charge_da.extend(res.get("stats_charge_da", []))
        charge_sero.extend(res.get("stats_charge_sero", []))
        current_da.extend(res.get("stats_current_da", []))
        current_sero.extend(res.get("stats_current_sero", []))
        seq_len = float(res.get("sequence_length", 0))
        total_symbols += seq_len
        errors = float(res.get("errors", seq_len * float(res.get("ser", 0.0))))
        total_errors += errors

    snr_lin = calculate_snr_from_stats(all_da, all_sero) if all_da and all_sero else 0.0
    if snr_lin > 0:
        metrics["snr_db"] = 10.0 * np.log10(snr_lin)

    if charge_da and charge_sero:
        mean_da = np.nanmean(charge_da)
        mean_sero = np.nanmean(charge_sero)
        metrics["delta_Q_diff"] = float(mean_da - mean_sero)
    if current_da and current_sero:
        mean_da = np.nanmean(current_da)
        mean_sero = np.nanmean(current_sero)
        metrics["delta_I_diff"] = float(mean_da - mean_sero)

    ser = total_errors / total_symbols if total_symbols > 0 else float("nan")
    metrics["ser"] = max(0.0, min(1.0, ser)) if math.isfinite(ser) else float("nan")
    return metrics


def _collect_point(
    cfg_base: Dict[str, Any],
    Ts: float,
    nm_value: float,
    seeds: Sequence[int],
) -> Dict[str, float]:
    cfg = copy.deepcopy(cfg_base)
    cfg.setdefault("pipeline", {})
    cfg.setdefault("detection", {})
    cfg.setdefault("sim", {})
    cfg["pipeline"]["symbol_period_s"] = Ts
    cfg["pipeline"]["Nm_per_symbol"] = int(max(1, round(nm_value)))
    dt = float(cfg["sim"]["dt_s"])
    detection_win = _resolve_decision_window(cfg, Ts, dt)
    cfg["detection"]["decision_window_s"] = detection_win
    cfg["pipeline"]["time_window_s"] = max(cfg["pipeline"].get("time_window_s", 0.0), detection_win)
    cfg["sim"]["time_window_s"] = max(cfg["sim"].get("time_window_s", Ts), Ts)

    cal_seeds = list(range(max(6, len(seeds))))
    thresholds = calibrate_thresholds_cached(cfg, cal_seeds)
    for key, val in thresholds.items():
        if isinstance(key, str) and key.startswith("noise."):
            cfg.setdefault("noise", {})[key.split(".", 1)[1]] = val
        else:
            cfg["pipeline"][key] = val

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["pipeline"]["random_seed"] = int(seed)
        res = run_single_instance(run_cfg, int(seed), attach_isi_meta=True)
        if res:
            results.append(res)

    metrics = _calc_metric(results)
    metrics.update(
        {
            "symbol_period_s": Ts,
            "decision_window_s": detection_win,
            "Nm_per_symbol": cfg["pipeline"]["Nm_per_symbol"],
        }
    )
    return metrics


def _plot_results(df: pd.DataFrame, mode: str, out_png: Path) -> None:
    if df.empty:
        return
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(df["symbol_period_s"], df["snr_db"], marker="o", label="SNR (dB)")
    ax.set_xscale("log")
    ax.set_xlabel("Symbol period Ts (s)")
    ax.set_ylabel("SNR (dB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"{mode} : SNR vs Ts")
    if "ser" in df.columns and np.isfinite(df["ser"]).any():
        ax2 = ax.twinx()
        ax2.plot(df["symbol_period_s"], df["ser"], color="tab:red", linestyle="--", marker="s", label="SER")
        ax2.set_ylabel("SER")
        ax2.set_yscale("log")
        ax2.grid(False)
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax.legend(loc="upper left", fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_suffix(out_png.suffix + ".tmp")
    fig.savefig(tmp, dpi=400, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    tmp.replace(out_png)
    print(f"[saved] {out_png}")


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg()
    cfg["pipeline"]["modulation"] = args.mode
    seeds = list(range(args.seeds))

    anchor_d, anchor_nm = _anchor_distance(cfg, args.mode)
    if args.distance_um is not None:
        anchor_d = float(args.distance_um)
    if args.nm is not None:
        anchor_nm = float(args.nm)

    cfg["pipeline"]["distance_um"] = anchor_d
    Ts_base = calculate_dynamic_symbol_period(anchor_d, cfg)
    cfg["pipeline"]["symbol_period_s"] = Ts_base
    cfg["sim"]["time_window_s"] = max(cfg["sim"].get("time_window_s", Ts_base), Ts_base)
    cfg["detection"]["decision_window_s"] = _resolve_decision_window(cfg, Ts_base, float(cfg["sim"]["dt_s"]))
    if args.detector_mode:
        cfg["pipeline"]["detector_mode"] = args.detector_mode

    multipliers = []
    for token in args.ts_grid.split(","):
        clean = token.strip().lower()
        if not clean:
            continue
        if clean.endswith("x"):
            clean = clean[:-1]
        try:
            multipliers.append(float(clean))
        except ValueError:
            print(f"?? Invalid Ts multiplier '{token}', skipping.")
    if not multipliers:
        multipliers = [0.5, 1.0, 2.0]

    csv_path = data_dir / f"{args.output_prefix}_{args.mode.lower()}{args.suffix}.csv"
    fig_path = fig_dir / f"fig_{args.output_prefix}_{args.mode.lower()}{args.suffix}.png"
    existing_rows: List[Dict[str, Any]] = []
    done_multipliers: set[float] = set()
    if args.resume and not args.force and csv_path.exists():
        try:
            df_prev = pd.read_csv(csv_path)
            existing_rows = cast(List[Dict[str, Any]], df_prev.to_dict("records"))
            done_multipliers = set(float(x) for x in df_prev.get("Ts_multiplier", []) if pd.notna(x))
            print(f"[resume] Loaded {len(df_prev)} existing Ts points from {csv_path}")
        except Exception as exc:
            print(f"[resume] Failed to load existing CSV ({exc}); recomputing.")
            existing_rows, done_multipliers = [], set()
    elif not args.force and csv_path.exists():
        df_prev = pd.read_csv(csv_path)
        _plot_results(df_prev, args.mode, fig_path)
        return

    rows: List[Dict[str, float]] = cast(List[Dict[str, float]], existing_rows)
    for mult in multipliers:
        if args.resume and mult in done_multipliers:
            print(f"[resume] skip Ts multiplier {mult} (cached)")
            continue
        Ts = Ts_base * mult
        metrics = _collect_point(cfg, Ts, anchor_nm, seeds)
        metrics["Ts_multiplier"] = mult
        metrics["distance_um"] = anchor_d
        metrics["mode"] = args.mode
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["Ts_multiplier"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = csv_path.with_suffix(".tmp")
    df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)
    print(f"[saved] {csv_path}")

    _plot_results(df, args.mode, fig_path)


if __name__ == "__main__":
    main()
