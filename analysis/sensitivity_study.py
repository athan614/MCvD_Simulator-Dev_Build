#!/usr/bin/env python3
"""
Parameter sensitivity sweeps for LoD versus temperature, diffusion, and binding kinetics.

Each sweep reuses the LoD search (with resume support) so the outputs remain
consistent with the main analysis pipeline.

Outputs (per metric suffix: "", "_snr", "_ser")
-------
results/data/sensitivity_T{suffix}.csv
results/data/sensitivity_D{suffix}.csv
results/data/sensitivity_binding{suffix}.csv
results/data/sensitivity_device{suffix}.csv
results/data/sensitivity_corr{suffix}.csv
results/figures/fig_sensitivity_T{suffix}.png
results/figures/fig_sensitivity_D{suffix}.png
results/figures/fig_sensitivity_binding_kon{scale}{suffix}.png
results/figures/fig_sensitivity_device{suffix}.png
results/figures/fig_sensitivity_corr{suffix}.png
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set, cast

import os
import matplotlib as mpl
if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style
from analysis.ui_progress import ProgressManager
from analysis.log_utils import setup_tee_logging
from analysis.run_final_analysis import (
    preprocess_config_full,
    calculate_dynamic_symbol_period,
    find_lod_for_ser,
    calibrate_thresholds_cached,
    run_single_instance,
    calculate_snr_from_stats,
)
from src.pipeline import _resolve_decision_window

MetricDict = Dict[str, float]


def _metric_suffix(metric: str) -> str:
    return "" if metric == "lod" else f"_{metric}"


def _inject_thresholds(cfg: Dict[str, Any], thresholds: Dict[str, Any]) -> None:
    for key, value in thresholds.items():
        if key is None:
            continue
        if isinstance(key, str) and key.startswith("noise."):
            _, sub = key.split(".", 1)
            cfg.setdefault("noise", {})[sub] = value
        else:
            cfg.setdefault("pipeline", {})[key] = value


def _nan_metric_dict() -> MetricDict:
    return {
        "snr_db": float("nan"),
        "snr_q_db": float("nan"),
        "snr_i_db": float("nan"),
        "delta_Q_diff": float("nan"),
        "delta_I_diff": float("nan"),
        "delta_over_sigma_Q": float("nan"),
        "delta_over_sigma_I": float("nan"),
        "sigma_Q_diff": float("nan"),
        "sigma_I_diff": float("nan"),
        "ser_eval": float("nan"),
    }


def _snr_from_ratio(ratio: float) -> float:
    if not math.isfinite(ratio) or ratio == 0.0:
        return float("nan")
    return 10.0 * math.log10(ratio * ratio)


def _aggregate_results(cfg: Dict[str, Any], results: List[Dict[str, Any]]) -> MetricDict:
    metrics = _nan_metric_dict()
    if not results:
        return metrics

    total_symbols = 0.0
    total_errors = 0.0
    all_a: List[float] = []
    all_b: List[float] = []
    charge_da_all: List[float] = []
    charge_sero_all: List[float] = []
    current_da_all: List[float] = []
    current_sero_all: List[float] = []
    noise_charge_sigmas: List[float] = []
    noise_current_sigmas: List[float] = []

    for res in results:
        seq_len = float(res.get("sequence_length", cfg["pipeline"].get("sequence_length", 0)))
        ser_val = res.get("ser", res.get("SER", float("nan")))
        errors = res.get("errors", float("nan"))
        if not math.isfinite(errors) and math.isfinite(ser_val) and seq_len > 0:
            errors = float(ser_val) * seq_len
        if math.isfinite(errors):
            total_errors += float(errors)
        if math.isfinite(seq_len):
            total_symbols += float(seq_len)

        all_a.extend([float(x) for x in res.get("stats_da", [])])
        all_b.extend([float(x) for x in res.get("stats_sero", [])])
        charge_da_all.extend([float(x) for x in res.get("stats_charge_da", [])])
        charge_sero_all.extend([float(x) for x in res.get("stats_charge_sero", [])])
        current_da_all.extend([float(x) for x in res.get("stats_current_da", [])])
        current_sero_all.extend([float(x) for x in res.get("stats_current_sero", [])])
        noise_charge_sigmas.append(float(res.get("noise_sigma_diff_charge", float("nan"))))
        det_win = float(res.get("decision_window_s", cfg["detection"].get("decision_window_s", float("nan"))))
        sigma_charge = float(res.get("noise_sigma_diff_charge", float("nan")))
        sigma_current = float("nan")
        if math.isfinite(det_win) and det_win > 0 and math.isfinite(sigma_charge):
            sigma_current = sigma_charge / det_win
        noise_current_sigmas.append(sigma_current)

    ser_eval = float("nan")
    if total_symbols > 0:
        ser_eval = max(0.0, min(1.0, total_errors / total_symbols))

    snr_lin = calculate_snr_from_stats(all_a, all_b) if all_a and all_b else 0.0
    snr_db = 10.0 * float(np.log10(snr_lin)) if snr_lin > 0 else float("nan")

    mean_charge_da = float(np.nanmean(charge_da_all)) if charge_da_all else float("nan")
    mean_charge_sero = float(np.nanmean(charge_sero_all)) if charge_sero_all else float("nan")
    mean_current_da = float(np.nanmean(current_da_all)) if current_da_all else float("nan")
    mean_current_sero = float(np.nanmean(current_sero_all)) if current_sero_all else float("nan")

    delta_q_diff = float("nan")
    if np.isfinite(mean_charge_da) and np.isfinite(mean_charge_sero):
        delta_q_diff = float(mean_charge_da - mean_charge_sero)

    delta_i_diff = float("nan")
    if np.isfinite(mean_current_da) and np.isfinite(mean_current_sero):
        delta_i_diff = float(mean_current_da - mean_current_sero)

    sigma_q = float(np.nanmedian(np.asarray(noise_charge_sigmas, dtype=float))) if noise_charge_sigmas else float("nan")
    sigma_i = float(np.nanmedian(np.asarray(noise_current_sigmas, dtype=float))) if noise_current_sigmas else float("nan")

    delta_over_sigma_q = float("nan")
    if np.isfinite(delta_q_diff) and np.isfinite(sigma_q) and sigma_q > 0:
        delta_over_sigma_q = float(delta_q_diff / sigma_q)

    delta_over_sigma_i = float("nan")
    if np.isfinite(delta_i_diff) and np.isfinite(sigma_i) and sigma_i > 0:
        delta_over_sigma_i = float(delta_i_diff / sigma_i)

    metrics.update(
        {
            "snr_db": snr_db,
            "snr_q_db": _snr_from_ratio(delta_over_sigma_q),
            "snr_i_db": _snr_from_ratio(delta_over_sigma_i),
            "delta_Q_diff": delta_q_diff,
            "delta_I_diff": delta_i_diff,
            "delta_over_sigma_Q": delta_over_sigma_q,
            "delta_over_sigma_I": delta_over_sigma_i,
            "sigma_Q_diff": sigma_q,
            "sigma_I_diff": sigma_i,
            "ser_eval": ser_eval,
        }
    )
    return metrics


def _collect_metric_profile(cfg: Dict[str, Any], seeds: Sequence[int]) -> MetricDict:
    cfg_eval = copy.deepcopy(cfg)
    metrics = _nan_metric_dict()
    if not seeds:
        return metrics
    cfg_eval.setdefault("pipeline", {})
    cfg_eval.setdefault("detection", {})
    cal_seeds = list(range(max(6, len(seeds))))
    thresholds = calibrate_thresholds_cached(cfg_eval, cal_seeds)
    _inject_thresholds(cfg_eval, thresholds)
    results: List[Dict[str, Any]] = []
    for seed in seeds:
        res = run_single_instance(cfg_eval, int(seed), attach_isi_meta=True)
        if res:
            results.append(res)
    return _aggregate_results(cfg_eval, results)


def _resolve_metric_column(
    df: pd.DataFrame,
    metric: str,
    prefer: str = "q",
) -> Tuple[Optional[str], str, bool]:
    metric = metric.lower()
    prefer = prefer.lower()
    if metric == "lod":
        if "lod_nm" in df.columns:
            return "lod_nm", "LoD (Nm)", False
        return None, "", False
    if metric == "ser":
        candidates = ["ser_eval", "ser_at_lod", "ser"]
        for col in candidates:
            if col in df.columns and np.isfinite(df[col]).any():
                return col, "SER", True
        return None, "", True
    if metric == "snr":
        if prefer == "i":
            candidates = ["snr_i_db", "snr_q_db", "snr_db"]
        else:
            candidates = ["snr_q_db", "snr_i_db", "snr_db"]
        for col in candidates:
            if col in df.columns and np.isfinite(df[col]).any():
                ylabel = "SNR_I (dB)" if "snr_i" in col else "SNR_Q (dB)"
                return col, ylabel, False
        return None, "SNR (dB)", False
    return None, "", False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sensitivity sweeps for LoD.")
    parser.add_argument(
        "--target-ser",
        type=float,
        default=0.01,
        help="Target SER for LoD search (default: 0.01).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="Number of seeds for LoD estimation (default: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cached CSVs already exist.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a partially computed sweep by skipping rows already present in CSV outputs.",
    )
    parser.add_argument(
        "--progress",
        choices=["gui", "rich", "tqdm", "none"],
        default="tqdm",
        help="Progress backend (gui, rich, tqdm, none).",
    )
    parser.add_argument(
        "--logdir",
        default=str(project_root / "results" / "logs"),
        help="Directory for log files.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable file logging.",
    )
    parser.add_argument(
        "--fsync-logs",
        action="store_true",
        help="Force fsync on each write.",
    )
    parser.add_argument(
        "--freeze-calibration",
        action="store_true",
        help="Reuse baseline detector thresholds/noise during sweeps.",
    )
    parser.add_argument(
        "--detector-mode",
        choices=["raw", "zscore", "whitened"],
        default=None,
        help="Detector mode to apply for the T/D/ρ sweeps (device/binding use raw).",
    )
    parser.add_argument(
        "--metric",
        choices=["lod", "snr", "ser"],
        default="snr",
        help="Metric for sensitivity panels (lod, snr, or ser).",
    )
    parser.add_argument(
        "--dual-ctrl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run sweeps with CTRL on and off (default: on; use --no-dual-ctrl for CTRL-on only).",
    )
    return parser.parse_args()


def water_viscosity_pa_s(T_K: float) -> float:
    """Approximate water viscosity (Pa*s) using an Andrade-like fit (0-60 C)."""
    T_C = T_K - 273.15
    A, B, C = 2.414e-5, 247.8, 140.0
    return A * 10.0 ** (B / (T_C + C))


def scale_diffusion_for_temperature(D_ref: float, T_ref: float, T_new: float) -> float:
    """Scale diffusion coefficient with temperature/viscosity (Stokes-Einstein trend)."""
    mu_ref = water_viscosity_pa_s(T_ref)
    mu_new = water_viscosity_pa_s(T_new)
    return D_ref * (T_new / T_ref) * (mu_ref / mu_new)


def _load_cfg() -> Dict[str, Any]:
    cfg_path = project_root / "config" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return preprocess_config_full(cfg)


def _save_csv_atomic(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)


def _append_row_atomic(dest: Path, row: Dict[str, Any], subset: Sequence[str]) -> None:
    """Append one row to CSV with deduplication and atomic replace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            df = pd.read_csv(dest)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    if not df.empty:
        df = df.drop_duplicates(subset=list(subset), keep="last")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)


def _load_existing_rows(dest: Path, key_cols: Sequence[str]) -> Tuple[List[Dict[str, Any]], Set[Tuple[Any, ...]]]:
    if not dest.exists():
        return [], set()
    try:
        df = pd.read_csv(dest)
    except Exception:
        return [], set()
    rows = cast(List[Dict[str, Any]], df.to_dict("records"))
    keys: Set[Tuple[Any, ...]] = set()
    for _, r in df.iterrows():
        try:
            keys.add(tuple(r.get(col) for col in key_cols))
        except Exception:
            continue
    return rows, keys


class _NoopBar:
    def __init__(self) -> None:
        self._completed = 0
    def update(self, n: int = 1, description: Optional[str] = None) -> None:
        self._completed += n
    def close(self) -> None:
        pass
    def set_description(self, text: str) -> None:
        pass


def _make_progress(pm: Optional[ProgressManager], total: int, desc: str, kind: str = "sweep"):
    if pm is not None:
        return pm.task(total=total, description=desc, kind=kind)
    try:
        return tqdm(total=total, desc=desc, leave=False)
    except Exception:
        return _NoopBar()


def _resolve_detection_window(cfg: Dict[str, Any], Ts: float) -> float:
    dt = float(cfg["sim"]["dt_s"])
    win = _resolve_decision_window(cfg, Ts, dt)
    cfg["pipeline"]["time_window_s"] = max(
        float(cfg["pipeline"].get("time_window_s", 0.0)),
        win,
    )
    cfg["sim"]["time_window_s"] = max(float(cfg["sim"].get("time_window_s", Ts)), Ts)
    return win


def _apply_sensitivity_overrides(
    cfg: Dict[str, Any],
    detector_mode: Optional[str] = None,
    freeze_calibration: bool = False,
) -> None:
    pipeline_cfg = cfg.setdefault("pipeline", {})
    if detector_mode:
        pipeline_cfg["detector_mode"] = detector_mode
    if freeze_calibration:
        analysis_cfg = cfg.setdefault("analysis", {})
        analysis_cfg["freeze_calibration"] = True


def _plot_param(
    df: pd.DataFrame,
    x: str,
    out_png: Path,
    xlabel: str,
    metric: str,
    prefer: str = "q",
    label: Optional[str] = None,
) -> None:
    if df.empty:
        return
    col, ylabel, logy = _resolve_metric_column(df, metric, prefer)
    if not col:
        return
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    if "use_ctrl" in df.columns:
        for uc, group in df.groupby("use_ctrl"):
            df_sorted = group.sort_values(by=x)
            if not np.isfinite(df_sorted[col]).any():
                continue
            series_label = f"{'CTRL on' if uc else 'CTRL off'}"
            ax.plot(df_sorted[x], df_sorted[col], marker="o", label=series_label)
    else:
        df_sorted = df.sort_values(by=x)
        series_label = label or col
        if np.isfinite(df_sorted[col]).any():
            ax.plot(df_sorted[x], df_sorted[col], marker="o", label=series_label)
        else:
            plt.close(fig)
            return
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_suffix(out_png.suffix + ".tmp")
    fig.savefig(tmp, dpi=400, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    os.replace(tmp, out_png)
    print(f"[saved] {out_png}")


def _pick_anchor(cfg: Dict[str, Any]) -> Tuple[int, float]:
    """Choose an anchor distance/Nm from existing LoD CSVs, falling back to config defaults."""
    data_dir = project_root / "results" / "data"
    target_dist = float(cfg.get("pipeline", {}).get("distance_um", 50.0))
    d_ref = int(round(target_dist))
    nm_ref = float(cfg["pipeline"].get("Nm_per_symbol", 2000.0))

    # Prefer the current modulation, then fall back to other modes
    cfg_mode = str(cfg.get("pipeline", {}).get("modulation", "")).lower()
    mode_order = []
    if cfg_mode in ("mosk", "csk", "hybrid"):
        mode_order.append(cfg_mode)
    for m in ("mosk", "csk", "hybrid"):
        if m not in mode_order:
            mode_order.append(m)

    for mode_suffix in mode_order:
        csv_path = data_dir / f"lod_vs_distance_{mode_suffix}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "distance_um" not in df.columns or "lod_nm" not in df.columns:
            continue
        df = df.dropna(subset=["lod_nm"])
        if df.empty:
            continue
        dist_vals = pd.to_numeric(df["distance_um"], errors="coerce")
        lod_vals = pd.to_numeric(df["lod_nm"], errors="coerce")
        valid = dist_vals.notna() & lod_vals.notna()
        if not valid.any():
            continue
        dist_clean = dist_vals[valid].astype(float)
        lod_clean = lod_vals[valid].astype(float)
        # Use desired distance if present; otherwise, interpolate/pick nearest
        d_ref = int(round(target_dist))
        exact = lod_clean[dist_clean.astype(int) == d_ref]
        if not exact.empty:
            nm_ref = float(exact.iloc[-1])
            break
        if len(dist_clean) >= 2:
            try:
                x = np.array(dist_clean, dtype=float)
                y = np.log(np.array(lod_clean, dtype=float))
                coeffs = np.polyfit(x, y, 1)
                nm_pred = float(np.exp(np.polyval(coeffs, target_dist)))
                if math.isfinite(nm_pred) and nm_pred > 0:
                    nm_ref = nm_pred
                    break
            except Exception:
                pass
        idx = (dist_clean - target_dist).abs().idxmin()
        d_ref = int(round(dist_clean.loc[idx]))
        nm_val = float(lod_clean.loc[idx])
        if math.isfinite(nm_val) and nm_val > 0:
            nm_ref = nm_val
            break

    return d_ref, nm_ref


def _warm_start(cfg: Dict[str, Any], nm_hint: float) -> None:
    if math.isfinite(nm_hint) and nm_hint > 0:
        cfg["pipeline"]["Nm_per_symbol"] = int(nm_hint)
        cfg["_warm_lod_guess"] = int(nm_hint)


def _compute_lod(
    cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    cache_label: str,
) -> Tuple[float, float]:
    cfg_local = copy.deepcopy(cfg)
    try:
        lod_nm, ser_at_lod, _, _ = find_lod_for_ser(
            cfg_local,
            list(seeds),
            target_ser=target_ser,
            resume=True,
            cache_tag=cache_label,
        )
    except Exception as exc:
        print(f"[warn] LoD search failed for {cache_label}: {exc}")
        return float("nan"), float("nan")

    if isinstance(lod_nm, (int, float)) and math.isfinite(lod_nm) and float(lod_nm) > 0:
        return float(lod_nm), float(ser_at_lod)
    return float("nan"), float("nan")


def run_sensitivity_T(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    anchor_distance: int,
    nm_hint: float,
    detector_mode: Optional[str],
    freeze_calibration: bool,
    metric: str,
    ctrl_states: Sequence[bool],
    force: bool = False,
    resume: bool = False,
    pm: Optional[ProgressManager] = None,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_T{suffix}.csv"
    out_fig = project_root / "results" / "figures" / f"fig_sensitivity_T{suffix}.png"
    if not force and out_csv.exists() and not resume:
        df_cached = pd.read_csv(out_csv)
        _plot_param(df_cached, "T_K", out_fig, "Temperature (K)", metric, prefer="q")
        return df_cached
    existing_rows, done = _load_existing_rows(out_csv, ["T_K", "use_ctrl"]) if (resume and not force) else ([], set())

    # Ten-point temperature sweep centered on physiological 310 K
    T_grid = [285.0, 290.0, 295.0, 300.0, 305.0, 310.0, 315.0, 320.0, 325.0, 330.0]
    D_refs = {
        nt: float(params["D_m2_s"])
        for nt, params in base_cfg["neurotransmitters"].items()
        if "D_m2_s" in params
    }
    T_ref = float(base_cfg["sim"].get("temperature_K", 310.0))

    rows: List[Dict[str, float]] = cast(List[Dict[str, float]], existing_rows)
    progress = _make_progress(pm, total=len(T_grid) * len(ctrl_states), desc="T sweep")
    for T in T_grid:
        for use_ctrl in ctrl_states:
            key = (T, use_ctrl)
            if resume and key in done:
                print(f"[resume] skip T={T}K (cached, CTRL={'on' if use_ctrl else 'off'})")
                progress.update(1)
                continue
            cfg_t = copy.deepcopy(base_cfg)
            cfg_t["pipeline"]["use_control_channel"] = bool(use_ctrl)
            _apply_sensitivity_overrides(cfg_t, detector_mode, freeze_calibration)
            cfg_t["sim"]["temperature_K"] = T

            for nt, D_ref in D_refs.items():
                cfg_t["neurotransmitters"][nt]["D_m2_s"] = scale_diffusion_for_temperature(D_ref, T_ref, T)

            cfg_t["pipeline"]["distance_um"] = anchor_distance
            Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_t)
            cfg_t["pipeline"]["symbol_period_s"] = Ts
            _resolve_detection_window(cfg_t, Ts)

            if math.isfinite(nm_hint) and nm_hint > 0:
                _warm_start(cfg_t, nm_hint)

            ctrl_tag = "wctrl" if use_ctrl else "noctrl"
            cache_label = f"T_{int(T)}K_{ctrl_tag}"
            lod_nm, ser_at_lod = _compute_lod(cfg_t, seeds, target_ser, cache_label)

            nm_eval = lod_nm if math.isfinite(lod_nm) and lod_nm > 0 else nm_hint
            metrics = _nan_metric_dict()
            if math.isfinite(nm_eval) and nm_eval > 0:
                cfg_eval = copy.deepcopy(cfg_t)
                cfg_eval["pipeline"]["Nm_per_symbol"] = int(max(1, round(nm_eval)))
                metrics = _collect_metric_profile(cfg_eval, seeds)

            row = {
                "T_K": T,
                "lod_nm": lod_nm,
                "ser_at_lod": ser_at_lod,
                "symbol_period_s": Ts,
                "decision_window_s": float(cfg_t["detection"]["decision_window_s"]),
                "distance_um": anchor_distance,
                "use_ctrl": bool(use_ctrl),
            }
            row.update(metrics)
            rows.append(row)
            _append_row_atomic(out_csv, row, ["T_K", "use_ctrl"])
            progress.update(1)
    progress.close()

    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(rows)
    _save_csv_atomic(df, out_csv)
    _plot_param(df, "T_K", out_fig, "Temperature (K)", metric, prefer="q")
    return df


def run_sensitivity_D(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    anchor_distance: int,
    nm_hint: float,
    detector_mode: Optional[str],
    freeze_calibration: bool,
    metric: str,
    ctrl_states: Sequence[bool],
    force: bool = False,
    resume: bool = False,
    pm: Optional[ProgressManager] = None,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_D{suffix}.csv"
    out_fig = project_root / "results" / "figures" / f"fig_sensitivity_D{suffix}.png"
    if not force and out_csv.exists() and not resume:
        df_cached = pd.read_csv(out_csv)
        _plot_param(df_cached, "D_scale", out_fig, "Diffusion scale (x)", metric, prefer="q")
        return df_cached
    existing_rows, done = _load_existing_rows(out_csv, ["D_scale", "use_ctrl"]) if (resume and not force) else ([], set())

    # Log-friendly diffusion scaling factors (10 pts from 0.5x to 2x)
    scales = [0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.55, 1.75, 2.0]
    D_refs = {
        nt: float(base_cfg["neurotransmitters"][nt]["D_m2_s"])
        for nt in ("DA", "SERO", "CTRL")
        if nt in base_cfg["neurotransmitters"] and "D_m2_s" in base_cfg["neurotransmitters"][nt]
    }

    rows: List[Dict[str, float]] = cast(List[Dict[str, float]], existing_rows)
    progress = _make_progress(pm, total=len(scales) * len(ctrl_states), desc="D sweep")
    for scale in scales:
        for use_ctrl in ctrl_states:
            key = (scale, use_ctrl)
            if resume and key in done:
                print(f"[resume] skip D_scale={scale} (cached, CTRL={'on' if use_ctrl else 'off'})")
                progress.update(1)
                continue
            cfg_s = copy.deepcopy(base_cfg)
            cfg_s["pipeline"]["use_control_channel"] = bool(use_ctrl)
            _apply_sensitivity_overrides(cfg_s, detector_mode, freeze_calibration)
            for nt, D_ref in D_refs.items():
                cfg_s["neurotransmitters"][nt]["D_m2_s"] = D_ref * scale

            cfg_s["pipeline"]["distance_um"] = anchor_distance
            Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_s)
            cfg_s["pipeline"]["symbol_period_s"] = Ts
            _resolve_detection_window(cfg_s, Ts)

            if math.isfinite(nm_hint) and nm_hint > 0:
                _warm_start(cfg_s, nm_hint)

            ctrl_tag = "wctrl" if use_ctrl else "noctrl"
            cache_label = f"Dscale_{scale:.2f}_{ctrl_tag}"
            lod_nm, ser_at_lod = _compute_lod(cfg_s, seeds, target_ser, cache_label)

            nm_eval = lod_nm if math.isfinite(lod_nm) and lod_nm > 0 else nm_hint
            metrics = _nan_metric_dict()
            if math.isfinite(nm_eval) and nm_eval > 0:
                cfg_eval = copy.deepcopy(cfg_s)
                cfg_eval["pipeline"]["Nm_per_symbol"] = int(max(1, round(nm_eval)))
                metrics = _collect_metric_profile(cfg_eval, seeds)

            row = {
                "D_scale": scale,
                "lod_nm": lod_nm,
                "ser_at_lod": ser_at_lod,
                "symbol_period_s": Ts,
                "decision_window_s": float(cfg_s["detection"]["decision_window_s"]),
                "distance_um": anchor_distance,
                "use_ctrl": bool(use_ctrl),
            }
            row.update(metrics)
            rows.append(row)
            _append_row_atomic(out_csv, row, ["D_scale", "use_ctrl"])
            progress.update(1)
    progress.close()

    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(rows)
    _save_csv_atomic(df, out_csv)
    _plot_param(df, "D_scale", out_fig, "Diffusion scale (x)", metric, prefer="q")
    return df


def run_sensitivity_binding(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    anchor_distance: int,
    nm_hint: float,
    freeze_calibration: bool,
    ctrl_states: Sequence[bool],
    metric: str,
    force: bool = False,
    resume: bool = False,
    pm: Optional[ProgressManager] = None,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_binding{suffix}.csv"
    fig_base = project_root / "results" / "figures"
    if not force and out_csv.exists() and not resume:
        df_cached = pd.read_csv(out_csv)
        for s_on in sorted(df_cached["kon_scale"].unique()):
            suffix_val = str(s_on).replace(".", "p")
            fig_path = fig_base / f"fig_sensitivity_binding_kon{suffix_val}{suffix}.png"
            subset = df_cached[df_cached["kon_scale"] == s_on]
            _plot_param(subset, "koff_scale", fig_path, f"koff scale (kon x {s_on:g})", metric, prefer="q")
        return df_cached
    existing_rows, done = _load_existing_rows(out_csv, ["kon_scale", "koff_scale", "use_ctrl"]) if (resume and not force) else ([], set())

    # Ten-point on/off-rate multipliers to capture gradual kinetic changes
    scales = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    k_on_refs = {
        nt: float(params.get("k_on_M_s", 0.0))
        for nt, params in base_cfg["neurotransmitters"].items()
    }
    k_off_refs = {
        nt: float(params.get("k_off_s", 0.0))
        for nt, params in base_cfg["neurotransmitters"].items()
    }
    binding_ref = {
        "k_on_M_s": float(base_cfg.get("binding", {}).get("k_on_M_s", 0.0)),
        "k_off_s": float(base_cfg.get("binding", {}).get("k_off_s", 0.0)),
    }

    rows: List[Dict[str, float]] = cast(List[Dict[str, float]], existing_rows)
    progress = _make_progress(pm, total=len(scales) * len(scales) * len(ctrl_states), desc="Binding sweep")
    for s_on in scales:
        for s_off in scales:
            for use_ctrl in ctrl_states:
                key = (s_on, s_off, use_ctrl)
                if resume and key in done:
                    progress.update(1)
                    continue
                cfg_b = copy.deepcopy(base_cfg)
                cfg_b["pipeline"]["use_control_channel"] = bool(use_ctrl)
                _apply_sensitivity_overrides(cfg_b, detector_mode=None, freeze_calibration=freeze_calibration)

                for nt, val in k_on_refs.items():
                    cfg_b["neurotransmitters"][nt]["k_on_M_s"] = val * s_on
                for nt, val in k_off_refs.items():
                    cfg_b["neurotransmitters"][nt]["k_off_s"] = val * s_off
                if "binding" in cfg_b:
                    cfg_b["binding"]["k_on_M_s"] = binding_ref["k_on_M_s"] * s_on
                    cfg_b["binding"]["k_off_s"] = binding_ref["k_off_s"] * s_off

                cfg_b["pipeline"]["distance_um"] = anchor_distance
                Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_b)
                cfg_b["pipeline"]["symbol_period_s"] = Ts
                _resolve_detection_window(cfg_b, Ts)

                if math.isfinite(nm_hint) and nm_hint > 0:
                    _warm_start(cfg_b, nm_hint)

                ctrl_tag = "wctrl" if use_ctrl else "noctrl"
                cache_label = f"binding_kon{s_on:.2f}_koff{s_off:.2f}_{ctrl_tag}"
                lod_nm, ser_at_lod = _compute_lod(cfg_b, seeds, target_ser, cache_label)

                nm_eval = lod_nm if math.isfinite(lod_nm) and lod_nm > 0 else nm_hint
                metrics = _nan_metric_dict()
                if math.isfinite(nm_eval) and nm_eval > 0:
                    cfg_eval = copy.deepcopy(cfg_b)
                    cfg_eval["pipeline"]["Nm_per_symbol"] = int(max(1, round(nm_eval)))
                    metrics = _collect_metric_profile(cfg_eval, seeds)

                row = {
                    "kon_scale": s_on,
                    "koff_scale": s_off,
                    "lod_nm": lod_nm,
                    "ser_at_lod": ser_at_lod,
                    "symbol_period_s": Ts,
                    "decision_window_s": float(cfg_b["detection"]["decision_window_s"]),
                    "distance_um": anchor_distance,
                    "use_ctrl": bool(use_ctrl),
                }
                row.update(metrics)
                rows.append(row)
                _append_row_atomic(out_csv, row, ["kon_scale", "koff_scale", "use_ctrl"])
                progress.update(1)
    progress.close()

    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["kon_scale", "koff_scale", "use_ctrl"])
    _save_csv_atomic(df, out_csv)

    for s_on in sorted(df["kon_scale"].unique()):
        df_on = df[df["kon_scale"] == s_on].sort_values("koff_scale")
        suffix_val = str(s_on).replace(".", "p")
        fig_path = fig_base / f"fig_sensitivity_binding_kon{suffix_val}{suffix}.png"
        _plot_param(df_on, "koff_scale", fig_path, f"koff scale (kon x {s_on:g})", metric, prefer="q")

    return df


def _plot_device_curves(df: pd.DataFrame, out_fig: Path, metric: str) -> None:
    if df is None or df.empty:
        return
    col, ylabel, logy = _resolve_metric_column(df, metric, prefer="i")
    required = {"gm_S", "C_tot_F"}
    if col is None or not required.issubset(df.columns) or col not in df.columns:
        return
    subset = df.dropna(subset=[col])
    if subset.empty:
        return
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    groups = subset.groupby("use_ctrl") if "use_ctrl" in subset.columns else [(None, subset)]
    for uc, df_uc in groups:
        pivot = df_uc.pivot_table(index="gm_S", columns="C_tot_F", values=col, aggfunc="mean")
        if pivot.empty:
            continue
        for c_val in sorted(pivot.columns):
            label = f"C_tot={c_val:.0e} F"
            if uc is not None:
                label += f" ({'CTRL on' if uc else 'CTRL off'})"
            ax.plot(pivot.index, pivot[c_val], marker="o", label=label)
    ax.set_xlabel("gm (S)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_fig.with_suffix(out_fig.suffix + ".tmp")
    fig.savefig(tmp, dpi=400, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    os.replace(tmp, out_fig)
    print(f"[saved] {out_fig}")


def _plot_correlation_curves(df: pd.DataFrame, out_fig: Path, metric: str) -> None:
    if df is None or df.empty:
        return
    col, ylabel, logy = _resolve_metric_column(df, metric, prefer="i")
    required = {"rho_corr", "rho_post"}
    if col is None or not required.issubset(df.columns) or col not in df.columns:
        return
    subset = df.dropna(subset=[col])
    if subset.empty:
        return
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    groups = subset.groupby("use_ctrl") if "use_ctrl" in subset.columns else [(None, subset)]
    for uc, df_uc in groups:
        pivot = df_uc.pivot_table(index="rho_corr", columns="rho_post", values=col, aggfunc="mean")
        if pivot.empty:
            continue
        for col_val in sorted(pivot.columns):
            label = f"ρ_post={col_val:.2f}"
            if uc is not None:
                label += f" ({'CTRL on' if uc else 'CTRL off'})"
            ax.plot(pivot.index, pivot[col_val], marker="o", label=label)
    ax.set_xlabel("ρ pre-CTRL")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_fig.with_suffix(out_fig.suffix + ".tmp")
    fig.savefig(tmp, dpi=400, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    os.replace(tmp, out_fig)
    print(f"[saved] {out_fig}")


def run_sensitivity_device(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    anchor_distance: int,
    nm_hint: float,
    freeze_calibration: bool,
    metric: str,
    ctrl_states: Sequence[bool],
    force: bool = False,
    resume: bool = False,
    pm: Optional[ProgressManager] = None,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_device{suffix}.csv"
    out_fig = project_root / "results" / "figures" / f"fig_sensitivity_device{suffix}.png"
    if not force and out_csv.exists() and not resume:
        df_cached = pd.read_csv(out_csv)
        _plot_device_curves(df_cached, out_fig, metric)
        return df_cached
    existing_rows, done = _load_existing_rows(out_csv, ["gm_S", "C_tot_F", "use_ctrl"]) if (resume and not force) else ([], set())

    # Ten gm values (roughly log-spaced) and ten capacitances for smoother FoMs
    gm_grid = [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, 3.0e-3, 4.0e-3, 5.0e-3, 6.5e-3, 8.0e-3, 1.0e-2]
    c_grid = [1.0e-8, 1.5e-8, 2.0e-8, 2.5e-8, 3.0e-8, 4.0e-8, 5.0e-8, 6.5e-8, 8.0e-8, 1.0e-7]
    rows: list[dict[str, float]] = cast(List[Dict[str, float]], existing_rows)
    progress = _make_progress(pm, total=len(gm_grid) * len(c_grid) * len(ctrl_states), desc="Device sweep")

    for gm in gm_grid:
        for c_tot in c_grid:
            for use_ctrl in ctrl_states:
                if resume and (gm, c_tot, use_ctrl) in done:
                    progress.update(1)
                    continue
                cfg_d = copy.deepcopy(base_cfg)
                cfg_d["pipeline"]["use_control_channel"] = bool(use_ctrl)
                _apply_sensitivity_overrides(cfg_d, detector_mode=None, freeze_calibration=freeze_calibration)
                cfg_d.setdefault("oect", {})
                cfg_d["oect"]["gm_S"] = gm
                cfg_d["oect"]["C_tot_F"] = c_tot
                cfg_d["pipeline"]["distance_um"] = anchor_distance
                Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_d)
                cfg_d["pipeline"]["symbol_period_s"] = Ts
                _resolve_detection_window(cfg_d, Ts)
                _warm_start(cfg_d, nm_hint)
                ctrl_tag = "wctrl" if use_ctrl else "noctrl"
                cache_label = f"gm_{gm:.2e}_C_{c_tot:.2e}_{ctrl_tag}"
                lod_nm, ser_at_lod = _compute_lod(cfg_d, seeds, target_ser, cache_label)

                nm_eval = lod_nm if math.isfinite(lod_nm) and lod_nm > 0 else nm_hint
                metrics = _nan_metric_dict()
                if math.isfinite(nm_eval) and nm_eval > 0:
                    cfg_eval = copy.deepcopy(cfg_d)
                    cfg_eval["pipeline"]["Nm_per_symbol"] = int(max(1, round(nm_eval)))
                    metrics = _collect_metric_profile(cfg_eval, seeds)

                rows.append({
                    "gm_S": gm,
                    "C_tot_F": c_tot,
                    "lod_nm": lod_nm,
                    "ser_at_lod": ser_at_lod,
                    "symbol_period_s": Ts,
                    "decision_window_s": float(cfg_d["detection"]["decision_window_s"]),
                    "distance_um": anchor_distance,
                    "use_ctrl": bool(use_ctrl),
                    **metrics,
                })
                _append_row_atomic(out_csv, rows[-1], ["gm_S", "C_tot_F", "use_ctrl"])
                progress.update(1)

    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["gm_S", "C_tot_F", "use_ctrl"])
    progress.close()
    _save_csv_atomic(df, out_csv)
    _plot_device_curves(df, out_fig, metric)
    return df


def run_sensitivity_correlation(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    anchor_distance: int,
    nm_hint: float,
    detector_mode: Optional[str],
    freeze_calibration: bool,
    metric: str,
    ctrl_states: Sequence[bool],
    force: bool = False,
    resume: bool = False,
    pm: Optional[ProgressManager] = None,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_corr{suffix}.csv"
    out_fig = project_root / "results" / "figures" / f"fig_sensitivity_corr{suffix}.png"
    if not force and out_csv.exists() and not resume:
        df_cached = pd.read_csv(out_csv)
        _plot_correlation_curves(df_cached, out_fig, metric)
        return df_cached
    existing_rows, done = _load_existing_rows(out_csv, ["rho_corr", "rho_post", "use_ctrl"]) if (resume and not force) else ([], set())

    # Ten-point correlation grids (pre/post CTRL) for smoother capacity curves
    rho_pre = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    rho_post = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows: list[dict[str, float]] = cast(List[Dict[str, float]], existing_rows)
    progress = _make_progress(pm, total=len(rho_pre) * len(rho_post) * len(ctrl_states), desc="Corr sweep")

    for r0 in rho_pre:
        for r1 in rho_post:
            for use_ctrl in ctrl_states:
                if resume and (r0, r1, use_ctrl) in done:
                    progress.update(1)
                    continue
                cfg_c = copy.deepcopy(base_cfg)
                cfg_c["pipeline"]["use_control_channel"] = bool(use_ctrl)
                _apply_sensitivity_overrides(cfg_c, detector_mode, freeze_calibration)
                cfg_c.setdefault("noise", {})
                cfg_c["noise"]["rho_corr"] = r0
                cfg_c["noise"]["rho_between_channels_after_ctrl"] = r1
                cfg_c["pipeline"]["distance_um"] = anchor_distance
                Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_c)
                cfg_c["pipeline"]["symbol_period_s"] = Ts
                _resolve_detection_window(cfg_c, Ts)
                _warm_start(cfg_c, nm_hint)
                ctrl_tag = "wctrl" if use_ctrl else "noctrl"
                cache_label = f"rho_pre_{r0:.2f}_rho_post_{r1:.2f}_{ctrl_tag}"
                lod_nm, ser_at_lod = _compute_lod(cfg_c, seeds, target_ser, cache_label)

                nm_eval = lod_nm if math.isfinite(lod_nm) and lod_nm > 0 else nm_hint
                metrics = _nan_metric_dict()
                if math.isfinite(nm_eval) and nm_eval > 0:
                    cfg_eval = copy.deepcopy(cfg_c)
                    cfg_eval["pipeline"]["Nm_per_symbol"] = int(max(1, round(nm_eval)))
                    metrics = _collect_metric_profile(cfg_eval, seeds)

                rows.append({
                    "rho_corr": r0,
                    "rho_post": r1,
                    "lod_nm": lod_nm,
                    "ser_at_lod": ser_at_lod,
                    "symbol_period_s": Ts,
                    "decision_window_s": float(cfg_c["detection"]["decision_window_s"]),
                    "distance_um": anchor_distance,
                    "use_ctrl": bool(use_ctrl),
                    **metrics,
                })
                _append_row_atomic(out_csv, rows[-1], ["rho_corr", "rho_post", "use_ctrl"])
                progress.update(1)

    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["rho_corr", "rho_post", "use_ctrl"])
    progress.close()
    _save_csv_atomic(df, out_csv)
    _plot_correlation_curves(df, out_fig, metric)
    return df


def main() -> None:
    args = _parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be a positive integer")
    if not args.no_log:
        setup_tee_logging(Path(args.logdir), prefix="sensitivity_study", fsync=args.fsync_logs)
    else:
        print("[log] File logging disabled by --no-log")

    pm = ProgressManager(args.progress, gui_session_meta={"resume": args.resume, "progress": args.progress})

    base_cfg = _load_cfg()
    anchor_distance, nm_hint = _pick_anchor(base_cfg)
    if not math.isfinite(nm_hint) or nm_hint <= 0:
        nm_hint = float(base_cfg["pipeline"].get("Nm_per_symbol", 1e4))

    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["pipeline"]["distance_um"] = anchor_distance
    Ts_anchor = calculate_dynamic_symbol_period(anchor_distance, base_cfg)
    base_cfg["pipeline"]["symbol_period_s"] = Ts_anchor
    _resolve_detection_window(base_cfg, Ts_anchor)
    base_cfg["sim"]["time_window_s"] = max(float(base_cfg["sim"].get("time_window_s", Ts_anchor)), Ts_anchor)

    seeds = list(range(args.seeds))
    base_ctrl = bool(base_cfg["pipeline"].get("use_control_channel", True))
    ctrl_states: List[bool] = [True, False] if args.dual_ctrl else [base_ctrl]

    print(f"[anchor] distance={anchor_distance} um, Nm hint={nm_hint:.3g}")

    run_sensitivity_T(
        base_cfg,
        seeds,
        args.target_ser,
        anchor_distance,
        nm_hint,
        detector_mode=args.detector_mode,
        freeze_calibration=args.freeze_calibration,
        metric=args.metric,
        ctrl_states=ctrl_states,
        force=args.force,
        resume=args.resume,
        pm=pm,
    )
    run_sensitivity_D(
        base_cfg,
        seeds,
        args.target_ser,
        anchor_distance,
        nm_hint,
        detector_mode=args.detector_mode,
        freeze_calibration=args.freeze_calibration,
        metric=args.metric,
        ctrl_states=ctrl_states,
        force=args.force,
        resume=args.resume,
        pm=pm,
    )
    run_sensitivity_binding(
        base_cfg,
        seeds,
        args.target_ser,
        anchor_distance,
        nm_hint,
        freeze_calibration=True,
        metric=args.metric,
        ctrl_states=ctrl_states,
        force=args.force,
        resume=args.resume,
        pm=pm,
    )
    run_sensitivity_device(
        base_cfg,
        seeds,
        args.target_ser,
        anchor_distance,
        nm_hint,
        freeze_calibration=True,
        metric=args.metric,
        ctrl_states=ctrl_states,
        force=args.force,
        resume=args.resume,
        pm=pm,
    )
    run_sensitivity_correlation(
        base_cfg,
        seeds,
        args.target_ser,
        anchor_distance,
        nm_hint,
        detector_mode=args.detector_mode,
        freeze_calibration=args.freeze_calibration,
        metric=args.metric,
        ctrl_states=ctrl_states,
        force=args.force,
        resume=args.resume,
        pm=pm,
    )

    print("[done] Parameter sensitivity sweeps completed.")
    pm.stop()


if __name__ == "__main__":
    main()
