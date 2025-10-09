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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style
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
        "--progress",
        default="none",
        help="Placeholder for run_master compatibility.",
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
    df_sorted = df.sort_values(by=x)
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
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
    if label is not None:
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
    d_ref = int(cfg["pipeline"].get("distance_um", 50))
    nm_ref = float(cfg["pipeline"].get("Nm_per_symbol", 2000.0))

    for mode_suffix in ("mosk", "csk", "hybrid"):
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
        dist_int = pd.to_numeric(df["distance_um"], errors="coerce").dropna().astype(int)
        if dist_int.empty:
            continue
        d_ref = int(np.median(dist_int))
        match = df.loc[dist_int == d_ref, "lod_nm"]
        if not match.empty:
            nm_ref = float(match.iloc[-1])
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
        lod_nm, ser_at_lod, _ = find_lod_for_ser(
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
    force: bool = False,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_T{suffix}.csv"
    out_fig = project_root / "results" / "figures" / f"fig_sensitivity_T{suffix}.png"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        _plot_param(df_cached, "T_K", out_fig, "Temperature (K)", metric, prefer="q")
        return df_cached

    T_grid = [297.0, 303.0, 310.0, 315.0]
    D_refs = {
        nt: float(params["D_m2_s"])
        for nt, params in base_cfg["neurotransmitters"].items()
        if "D_m2_s" in params
    }
    T_ref = float(base_cfg["sim"].get("temperature_K", 310.0))

    rows: List[Dict[str, float]] = []
    for T in T_grid:
        cfg_t = copy.deepcopy(base_cfg)
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

        cache_label = f"T_{int(T)}K"
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
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
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
    force: bool = False,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_D{suffix}.csv"
    out_fig = project_root / "results" / "figures" / f"fig_sensitivity_D{suffix}.png"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        _plot_param(df_cached, "D_scale", out_fig, "Diffusion scale (x)", metric, prefer="q")
        return df_cached

    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    D_refs = {
        nt: float(base_cfg["neurotransmitters"][nt]["D_m2_s"])
        for nt in ("DA", "SERO", "CTRL")
        if nt in base_cfg["neurotransmitters"] and "D_m2_s" in base_cfg["neurotransmitters"][nt]
    }

    rows: List[Dict[str, float]] = []
    for scale in scales:
        cfg_s = copy.deepcopy(base_cfg)
        _apply_sensitivity_overrides(cfg_s, detector_mode, freeze_calibration)
        for nt, D_ref in D_refs.items():
            cfg_s["neurotransmitters"][nt]["D_m2_s"] = D_ref * scale

        cfg_s["pipeline"]["distance_um"] = anchor_distance
        Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_s)
        cfg_s["pipeline"]["symbol_period_s"] = Ts
        _resolve_detection_window(cfg_s, Ts)

        if math.isfinite(nm_hint) and nm_hint > 0:
            _warm_start(cfg_s, nm_hint)

        cache_label = f"Dscale_{scale:.2f}"
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
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
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
    metric: str,
    force: bool = False,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_binding{suffix}.csv"
    fig_base = project_root / "results" / "figures"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        for s_on in sorted(df_cached["kon_scale"].unique()):
            suffix_val = str(s_on).replace(".", "p")
            fig_path = fig_base / f"fig_sensitivity_binding_kon{suffix_val}{suffix}.png"
            subset = df_cached[df_cached["kon_scale"] == s_on]
            _plot_param(subset, "koff_scale", fig_path, f"koff scale (kon x {s_on:g})", metric, prefer="q")
        return df_cached

    scales = [0.5, 1.0, 2.0]
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

    rows: List[Dict[str, float]] = []
    for s_on in scales:
        for s_off in scales:
            cfg_b = copy.deepcopy(base_cfg)
            _apply_sensitivity_overrides(cfg_b, detector_mode="raw", freeze_calibration=freeze_calibration)
            cfg_b["pipeline"]["use_control_channel"] = False

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

            cache_label = f"binding_kon{s_on:.2f}_koff{s_off:.2f}"
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
            }
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
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
    pivot = subset.pivot_table(index="gm_S", columns="C_tot_F", values=col, aggfunc="mean")
    if pivot.empty:
        return
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for c_val in sorted(pivot.columns):
        label = f"C_tot={c_val:.0e} F"
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
    pivot = subset.pivot_table(index="rho_corr", columns="rho_post", values=col, aggfunc="mean")
    if pivot.empty:
        return
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for col_val in sorted(pivot.columns):
        label = f"ρ_post={col_val:.2f}"
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
    force: bool = False,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_device{suffix}.csv"
    out_fig = project_root / "results" / "figures" / f"fig_sensitivity_device{suffix}.png"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        _plot_device_curves(df_cached, out_fig, metric)
        return df_cached

    gm_grid = [1e-3, 3e-3, 5e-3, 1e-2]
    c_grid = [1e-8, 3e-8, 5e-8, 1e-7]
    rows: list[dict[str, float]] = []

    for gm in gm_grid:
        for c_tot in c_grid:
            cfg_d = copy.deepcopy(base_cfg)
            _apply_sensitivity_overrides(cfg_d, detector_mode="raw", freeze_calibration=freeze_calibration)
            cfg_d.setdefault("oect", {})
            cfg_d["oect"]["gm_S"] = gm
            cfg_d["oect"]["C_tot_F"] = c_tot
            cfg_d["pipeline"]["distance_um"] = anchor_distance
            Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_d)
            cfg_d["pipeline"]["symbol_period_s"] = Ts
            _resolve_detection_window(cfg_d, Ts)
            _warm_start(cfg_d, nm_hint)
            cache_label = f"gm_{gm:.2e}_C_{c_tot:.2e}"
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
                **metrics,
            })

    df = pd.DataFrame(rows)
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
    force: bool = False,
) -> pd.DataFrame:
    suffix = _metric_suffix(metric)
    out_csv = project_root / "results" / "data" / f"sensitivity_corr{suffix}.csv"
    out_fig = project_root / "results" / "figures" / f"fig_sensitivity_corr{suffix}.png"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        _plot_correlation_curves(df_cached, out_fig, metric)
        return df_cached

    rho_pre = [0.2, 0.5, 0.7, 0.9]
    rho_post = [0.0, 0.2, 0.4, 0.6]
    rows: list[dict[str, float]] = []

    for r0 in rho_pre:
        for r1 in rho_post:
            cfg_c = copy.deepcopy(base_cfg)
            _apply_sensitivity_overrides(cfg_c, detector_mode, freeze_calibration)
            cfg_c.setdefault("noise", {})
            cfg_c["noise"]["rho_corr"] = r0
            cfg_c["noise"]["rho_between_channels_after_ctrl"] = r1
            cfg_c["pipeline"]["distance_um"] = anchor_distance
            Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_c)
            cfg_c["pipeline"]["symbol_period_s"] = Ts
            _resolve_detection_window(cfg_c, Ts)
            _warm_start(cfg_c, nm_hint)
            cache_label = f"rho_pre_{r0:.2f}_rho_post_{r1:.2f}"
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
                **metrics,
            })

    df = pd.DataFrame(rows)
    _save_csv_atomic(df, out_csv)
    _plot_correlation_curves(df, out_fig, metric)
    return df


def main() -> None:
    args = _parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be a positive integer")

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
        force=args.force,
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
        force=args.force,
    )
    run_sensitivity_binding(
        base_cfg,
        seeds,
        args.target_ser,
        anchor_distance,
        nm_hint,
        freeze_calibration=True,
        metric=args.metric,
        force=args.force,
    )
    run_sensitivity_device(
        base_cfg,
        seeds,
        args.target_ser,
        anchor_distance,
        nm_hint,
        freeze_calibration=True,
        metric=args.metric,
        force=args.force,
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
        force=args.force,
    )

    print("[done] Parameter sensitivity sweeps completed.")


if __name__ == "__main__":
    main()
