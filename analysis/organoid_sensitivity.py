#!/usr/bin/env python3
"""
Organoid-focused sensitivity sweeps for the tri-channel OECT receiver.

This script reproduces key sweeps from the 3D FinFET receiver study:
  1. 1/f noise amplitude (Hooge parameter / trap density) with SER vs Nm.
  2. Drain-current / gate-bias sweep highlighting noise-aware bias trade-offs.
  3. Ionic strength (effective charge) sweep illustrating Debye screening.

In addition to SER/SNR metrics, the script captures noise PSDs for each sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
from numbers import Real

import os
import matplotlib as mpl
if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import math
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from analysis.log_utils import setup_tee_logging
from analysis.ui_progress import ProgressManager
from analysis.run_final_analysis import (
    calibrate_thresholds_cached,
    _apply_thresholds_into_cfg,
    _thresholds_filename,
)
from src.config_utils import preprocess_config
from src.pipeline import run_sequence


@dataclass
class SweepConfig:
    values: Sequence[float]
    label: str
    param_path: Tuple[str, ...]


_THRESHOLD_CACHE: Dict[str, Dict[str, Any]] = {}


def _calibration_key(cfg: Dict[str, Any]) -> str:
    """
    Build a deterministic cache key that matches the threshold filename logic so
    cached calibrations line up with on-disk JSON artifacts.
    """
    cfg_copy = deepcopy(cfg)
    return str(_thresholds_filename(cfg_copy))


def _resolve_calibration_seeds(seeds: Sequence[int]) -> List[int]:
    resolved: List[int] = []
    for idx, seed in enumerate(seeds):
        try:
            resolved.append(int(seed))
        except Exception:
            resolved.append(idx)
    if not resolved:
        resolved = list(range(6))
    return resolved


def _get_thresholds_for_cfg(cfg: Dict[str, Any],
                            seeds: Sequence[int],
                            recalibrate: bool) -> Dict[str, Any]:
    """
    Calibrate (or reuse) thresholds for the provided configuration.
    """
    key = _calibration_key(cfg)
    if recalibrate:
        _THRESHOLD_CACHE.pop(key, None)

    thresholds = _THRESHOLD_CACHE.get(key)
    if thresholds is None:
        cfg_cal = deepcopy(cfg)
        cal_seeds = _resolve_calibration_seeds(seeds)
        thresholds = calibrate_thresholds_cached(cfg_cal, cal_seeds, recalibrate=recalibrate)
        _THRESHOLD_CACHE[key] = deepcopy(thresholds)
    return deepcopy(thresholds)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_existing(path: Path, key_cols: Sequence[str]) -> Tuple[pd.DataFrame, set[tuple]]:
    if not path.exists():
        return pd.DataFrame(), set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(), set()
    keys: set[tuple] = set()
    for _, row in df.iterrows():
        try:
            keys.add(tuple(row.get(col) for col in key_cols))
        except Exception:
            continue
    return df, keys


class _NoopBar:
    def __init__(self) -> None:
        self._completed = 0
    def update(self, n: int = 1, description: Optional[str] = None) -> None:
        self._completed += n
    def close(self) -> None:
        pass
    def set_description(self, text: str) -> None:
        pass


def _make_progress(pm: Optional[ProgressManager], total: int, desc: str, kind: str = "sweep", parent: Any = None):
    if pm is not None:
        return pm.task(total=total, description=desc, kind=kind, parent=parent)
    try:
        return tqdm(total=total, desc=desc, leave=False)
    except Exception:
        return _NoopBar()


def _load_base_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)
    cfg = preprocess_config(raw_cfg)
    cfg.setdefault("pipeline", {})
    cfg.setdefault("noise", {})
    cfg.setdefault("neurotransmitters", {})
    return cfg


def _set_nested(cfg: Dict[str, Any], path: Tuple[str, ...], value: float) -> None:
    target = cfg
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return float("nan")
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def _snr_db(delta: float, sigma: float) -> float:
    if not (np.isfinite(delta) and np.isfinite(sigma)) or sigma <= 0:
        return float("nan")
    ratio = (delta / sigma) ** 2
    if ratio <= 0:
        return float("nan")
    return float(10.0 * np.log10(ratio))


def _aggregate_run(results: List[Dict[str, Any]]) -> Dict[str, float]:
    ser_vals: List[float] = []
    delta_i_vals: List[float] = []
    delta_q_vals: List[float] = []
    sigma_i_vals: List[float] = []
    sigma_q_vals: List[float] = []
    snr_i_vals: List[float] = []
    snr_q_vals: List[float] = []

    for res in results:
        ser_val = res.get("SER")
        if isinstance(ser_val, Real):
            ser_f = float(ser_val)
            if math.isfinite(ser_f):
                ser_vals.append(ser_f)

        stats_i_da = np.asarray(res.get("stats_current_da", []), dtype=float)
        stats_i_sero = np.asarray(res.get("stats_current_sero", []), dtype=float)
        stats_q_da = np.asarray(res.get("stats_charge_da", []), dtype=float)
        stats_q_sero = np.asarray(res.get("stats_charge_sero", []), dtype=float)

        if stats_i_da.size:
            stats_i_da = stats_i_da[np.isfinite(stats_i_da)]
        if stats_i_sero.size:
            stats_i_sero = stats_i_sero[np.isfinite(stats_i_sero)]
        if stats_q_da.size:
            stats_q_da = stats_q_da[np.isfinite(stats_q_da)]
        if stats_q_sero.size:
            stats_q_sero = stats_q_sero[np.isfinite(stats_q_sero)]

        if stats_i_da.size and stats_i_sero.size:
            delta_i = float(np.mean(stats_i_da) - np.mean(stats_i_sero))
            delta_i_vals.append(delta_i)
            sigma_i_raw = res.get("noise_sigma_I_diff", float("nan"))
            sigma_i = float(sigma_i_raw) if isinstance(sigma_i_raw, Real) else float("nan")
            if np.isfinite(sigma_i) and sigma_i > 0:
                sigma_i_vals.append(sigma_i)
                snr_i_vals.append(_snr_db(delta_i, sigma_i))

        if stats_q_da.size and stats_q_sero.size:
            delta_q = float(np.mean(stats_q_da) - np.mean(stats_q_sero))
            delta_q_vals.append(delta_q)
            sigma_q_raw = res.get("noise_sigma_diff_charge", res.get("noise_sigma_I_diff", float("nan")))
            sigma_q = float(sigma_q_raw) if isinstance(sigma_q_raw, Real) else float("nan")
            if np.isfinite(sigma_q) and sigma_q > 0:
                sigma_q_vals.append(sigma_q)
                snr_q_vals.append(_snr_db(delta_q, sigma_q))

    return {
        "ser_mean": _mean(ser_vals),
        "ser_sem": _sem(ser_vals),
        "delta_i_mean": _mean(delta_i_vals),
        "delta_q_mean": _mean(delta_q_vals),
        "sigma_i_median": _median(sigma_i_vals),
        "sigma_q_median": _median(sigma_q_vals),
        "snr_i_db_median": _median(snr_i_vals),
        "snr_q_db_median": _median(snr_q_vals),
        "runs": len(results),
    }


def run_monte_carlo(cfg: Dict[str, Any],
                    seeds: Sequence[int],
                    sequence_length: int,
                    recalibrate: bool,
                    progress: Optional[Any] = None) -> Dict[str, float]:
    thresholds = _get_thresholds_for_cfg(cfg, seeds, recalibrate)
    results: List[Dict[str, Any]] = []
    for seed in seeds:
        cfg_seed = deepcopy(cfg)
        _apply_thresholds_into_cfg(cfg_seed, thresholds)
        pipe = cfg_seed.setdefault('pipeline', {})
        pipe['random_seed'] = int(seed)
        pipe['sequence_length'] = sequence_length
        cfg_seed['disable_progress'] = True
        res = run_sequence(cfg_seed)
        results.append(res)
        if progress is not None:
            try:
                progress.update(1)
            except Exception:
                pass
    return _aggregate_run(results)


def _average_psd(payloads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not payloads:
        return None
    freq = np.asarray(payloads[0].get("frequency_hz", []), dtype=float)
    if freq.size == 0:
        return None
    keys = [key for key in payloads[0].keys() if key != "frequency_hz"]
    averaged: Dict[str, Any] = {"frequency_hz": freq}
    for key in keys:
        spectra: List[np.ndarray] = []
        for payload in payloads:
            values = np.asarray(payload.get(key, []), dtype=float)
            if values.size != freq.size:
                continue
            spectra.append(values)
        if not spectra:
            continue
        stacked = np.vstack(spectra)
        averaged[key] = np.mean(stacked, axis=0)
    return averaged


def capture_noise_psd(cfg: Dict[str, Any],
                      seeds: Sequence[int],
                      recalibrate: bool) -> Optional[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    thresholds = _get_thresholds_for_cfg(cfg, seeds, recalibrate)
    for seed in seeds:
        cfg_noise = deepcopy(cfg)
        _apply_thresholds_into_cfg(cfg_noise, thresholds)
        pipe = cfg_noise.setdefault("pipeline", {})
        pipe["_noise_only_run"] = True
        pipe["_collect_noise_components"] = True
        pipe["_collect_psd"] = True
        pipe["sequence_length"] = 1
        pipe["Nm_per_symbol"] = 0
        pipe["enable_molecular_noise"] = False
        pipe["random_seed"] = int(seed)
        cfg_noise['disable_progress'] = True
        res = run_sequence(cfg_noise)
        payload = res.get("psd_payload")
        if isinstance(payload, dict):
            payloads.append(payload)
    return _average_psd(payloads)


def psd_to_dataframe(psd: Dict[str, Any], sweep_label: str, sweep_value: float) -> pd.DataFrame:
    freq = np.asarray(psd.get("frequency_hz", []), dtype=float)
    rows: List[Dict[str, Union[float, str]]] = []
    for key, values in psd.items():
        if key == "frequency_hz":
            continue
        spectrum = np.asarray(values, dtype=float)
        if spectrum.size != freq.size:
            continue
        for f, power in zip(freq, spectrum):
            rows.append({
                "frequency_hz": float(f),
                "power_a2_per_hz": float(power),
                "spectrum": str(key),
                sweep_label: float(sweep_value),
            })
    return pd.DataFrame(rows)


def plot_ser_vs_nm(df: pd.DataFrame, sweep_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ctrl_groups = df.groupby("use_ctrl") if "use_ctrl" in df.columns else [(None, df)]
    for uc, df_ctrl in ctrl_groups:
        for value, group in df_ctrl.groupby(sweep_label):
            group_sorted = group.sort_values("Nm_per_symbol")
            label = f"{sweep_label}={value:g}"
            if uc is not None:
                label += f" ({'CTRL on' if uc else 'CTRL off'})"
            ax.errorbar(
                group_sorted["Nm_per_symbol"],
                group_sorted["ser_mean"],
                yerr=group_sorted["ser_sem"],
                marker="o",
                linewidth=1.2,
                label=label,
            )
    ax.set_xlabel("Nm per symbol")
    ax.set_ylabel("SER")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_metric_vs_param(df: pd.DataFrame, param: str, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ctrl_groups: List[Tuple[Any, pd.DataFrame]] = list(df.groupby("use_ctrl")) if "use_ctrl" in df.columns else [(None, df)]
    for uc, group in ctrl_groups:
        df_sorted = group.sort_values(param)
        label = None
        if uc is not None:
            label = "CTRL on" if uc else "CTRL off"
        ax.plot(df_sorted[param], df_sorted[metric], marker="o", linewidth=1.2, label=label)
    ax.set_xlabel(param)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if any(g is not None for g, _ in ctrl_groups):
        ax.legend()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_psd(df: pd.DataFrame, sweep_label: str, spectra_filter: Sequence[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    spectra = set(spectra_filter)
    group_cols = [sweep_label] + (["use_ctrl"] if "use_ctrl" in df.columns else [])
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label_prefix = f"{sweep_label}={keys[0]:g}"
        if len(keys) > 1:
            label_prefix += f" ({'CTRL on' if keys[1] else 'CTRL off'})"
        for spectrum, spec_group in group[group["spectrum"].isin(spectra)].groupby("spectrum"):
            spec_sorted = spec_group.sort_values("frequency_hz")
            ax.loglog(
                spec_sorted["frequency_hz"],
                spec_sorted["power_a2_per_hz"],
                label=f"{label_prefix} | {spectrum}",
                linewidth=1.2,
            )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (A^2/Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small")
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run_alpha_sweep(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    args: argparse.Namespace,
    ctrl_states: Sequence[bool],
    existing_ser: Optional[pd.DataFrame] = None,
    existing_psd: Optional[pd.DataFrame] = None,
    resume: bool = False,
    pm: Optional[ProgressManager] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nm_values = args.nm_grid if args.nm_grid else base_cfg.get("Nm_range", {}).get("MoSK", [200, 350, 650, 1100, 1750])
    rows: List[Dict[str, float]] = cast(List[Dict[str, float]], existing_ser.to_dict("records")) if resume and existing_ser is not None else []
    psd_frames: List[pd.DataFrame] = []
    done_ser: set[tuple] = set()
    done_psd: set[tuple] = set()
    if resume and existing_ser is not None:
        done_ser = {(float(r["alpha_H"]), float(r["Nm_per_symbol"]), bool(r.get("use_ctrl", True))) for _, r in existing_ser.iterrows() if "alpha_H" in r and "Nm_per_symbol" in r}
    if resume and existing_psd is not None:
        done_psd = {(float(r["alpha_H"]), bool(r.get("use_ctrl", True))) for _, r in existing_psd.iterrows() if "alpha_H" in r}
        psd_frames.append(existing_psd)

    progress = _make_progress(pm, total=len(args.alpha_values) * len(ctrl_states) * len(nm_values), desc="Organoid alpha")
    for alpha in args.alpha_values:
        for use_ctrl in ctrl_states:
            for nm in nm_values:
                seed_bar = None
                if args.progress != "none":
                    desc = f"alpha={alpha:g}, Nm={nm:g}, {'CTRL' if use_ctrl else 'NoCTRL'}"
                    seed_bar = _make_progress(pm, total=len(seeds), desc=desc, kind="worker")
                key = (float(alpha), float(nm), bool(use_ctrl))
                if resume and key in done_ser:
                    progress.update(1)
                    if seed_bar:
                        seed_bar.close()
                    continue
                cfg = deepcopy(base_cfg)
                cfg['pipeline']['use_control_channel'] = bool(use_ctrl)
                cfg['noise']['alpha_H'] = float(alpha)
                cfg['pipeline']['Nm_per_symbol'] = int(nm)
                agg = run_monte_carlo(cfg, seeds, args.sequence_length, recalibrate=args.recalibrate, progress=seed_bar)
                agg.update({"alpha_H": float(alpha), "Nm_per_symbol": int(nm), "use_ctrl": bool(use_ctrl)})
                rows.append(agg)
                progress.update(1)
                if seed_bar:
                    seed_bar.close()

            if resume and (float(alpha), bool(use_ctrl)) in done_psd:
                continue
            psd_cfg = deepcopy(base_cfg)
            psd_cfg['pipeline']['use_control_channel'] = bool(use_ctrl)
            psd_cfg['noise']['alpha_H'] = float(alpha)
            psd_cfg['pipeline']['Nm_per_symbol'] = int(nm_values[0]) if nm_values else base_cfg['pipeline'].get('Nm_per_symbol', 1000)
            psd = capture_noise_psd(psd_cfg, seeds, recalibrate=args.recalibrate)
            if psd:
                frame = psd_to_dataframe(psd, "alpha_H", float(alpha))
                frame["use_ctrl"] = bool(use_ctrl)
                psd_frames.append(frame)
    progress.close()

    df_ser = pd.DataFrame(rows)
    df_ser = df_ser.drop_duplicates(subset=["alpha_H", "Nm_per_symbol", "use_ctrl"]) if not df_ser.empty else df_ser
    df_psd = pd.concat(psd_frames, ignore_index=True) if psd_frames else pd.DataFrame()
    return df_ser, df_psd


def run_bias_sweep(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    args: argparse.Namespace,
    ctrl_states: Sequence[bool],
    existing_ser: Optional[pd.DataFrame] = None,
    existing_psd: Optional[pd.DataFrame] = None,
    resume: bool = False,
    pm: Optional[ProgressManager] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = cast(List[Dict[str, float]], existing_ser.to_dict("records")) if resume and existing_ser is not None else []
    psd_frames: List[pd.DataFrame] = []
    done_ser: set[tuple] = set()
    done_psd: set[tuple] = set()
    if resume and existing_ser is not None:
        done_ser = {(float(r["I_dc_A"]), bool(r.get("use_ctrl", True))) for _, r in existing_ser.iterrows() if "I_dc_A" in r}
    if resume and existing_psd is not None:
        done_psd = {(float(r["I_dc_A"]), bool(r.get("use_ctrl", True))) for _, r in existing_psd.iterrows() if "I_dc_A" in r}
        psd_frames.append(existing_psd)

    oect_cfg = base_cfg.get('oect', {})
    base_i_dc = float(oect_cfg.get('I_dc_A', 1.0e-4) or 1.0e-4)
    if not math.isfinite(base_i_dc) or base_i_dc <= 0.0:
        base_i_dc = 1.0e-4
    base_gm = float(oect_cfg.get('gm_S', 5.0e-3) or 5.0e-3)
    if not math.isfinite(base_gm) or base_gm <= 0.0:
        base_gm = 5.0e-3

    gm_mode = getattr(args, "bias_gm_scaling", "sqrt")

    progress = _make_progress(pm, total=len(args.bias_values) * len(ctrl_states), desc="Organoid bias")
    for bias in args.bias_values:
        bias_val = float(bias)
        for use_ctrl in ctrl_states:
            seed_bar = None
            if args.progress != "none":
                desc = f"I_dc={bias_val:.2e}, {'CTRL' if use_ctrl else 'NoCTRL'}"
                seed_bar = _make_progress(pm, total=len(seeds), desc=desc, kind="worker")
            cfg = deepcopy(base_cfg)
            if resume and (bias_val, use_ctrl) in done_ser:
                progress.update(1)
                if seed_bar:
                    seed_bar.close()
                continue
            cfg['oect']['I_dc_A'] = bias_val
            cfg['pipeline']['use_control_channel'] = bool(use_ctrl)

            gm_scale = 1.0
            if gm_mode != "none" and math.isfinite(bias_val) and bias_val > 0.0:
                ratio = max(bias_val, 1.0e-18) / base_i_dc
                if gm_mode == "linear":
                    gm_scale = ratio
                else:
                    gm_scale = math.sqrt(ratio)
            gm_scaled = base_gm * gm_scale
            cfg['oect']['gm_S'] = gm_scaled

            cfg['pipeline']['Nm_per_symbol'] = args.bias_nm
            agg = run_monte_carlo(cfg, seeds, args.sequence_length, recalibrate=args.recalibrate, progress=seed_bar)
            agg.update({
                "I_dc_A": bias_val,
                "gm_S": gm_scaled,
                "gm_scale": gm_scale,
                "use_ctrl": bool(use_ctrl),
            })
            rows.append(agg)

            if not (resume and (bias_val, use_ctrl) in done_psd):
                psd_cfg = deepcopy(cfg)
                psd = capture_noise_psd(psd_cfg, seeds, recalibrate=args.recalibrate)
                if psd:
                    frame = psd_to_dataframe(psd, "I_dc_A", float(bias))
                    frame["gm_S"] = gm_scaled
                    frame["gm_scale"] = gm_scale
                    frame["use_ctrl"] = bool(use_ctrl)
                    psd_frames.append(frame)
            progress.update(1)
            if seed_bar:
                seed_bar.close()
    progress.close()

    df_bias = pd.DataFrame(rows)
    if not df_bias.empty:
        df_bias = df_bias.drop_duplicates(subset=["I_dc_A", "use_ctrl"])
    df_psd = pd.concat(psd_frames, ignore_index=True) if psd_frames else pd.DataFrame()
    return df_bias, df_psd


def run_qeff_sweep(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    args: argparse.Namespace,
    ctrl_states: Sequence[bool],
    existing_ser: Optional[pd.DataFrame] = None,
    existing_psd: Optional[pd.DataFrame] = None,
    resume: bool = False,
    pm: Optional[ProgressManager] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = cast(List[Dict[str, float]], existing_ser.to_dict("records")) if resume and existing_ser is not None else []
    psd_frames: List[pd.DataFrame] = []
    done_ser: set[tuple] = set()
    done_psd: set[tuple] = set()
    if resume and existing_ser is not None:
        done_ser = {(float(r["q_eff_scale"]), bool(r.get("use_ctrl", True))) for _, r in existing_ser.iterrows() if "q_eff_scale" in r}
    if resume and existing_psd is not None:
        done_psd = {(float(r["q_eff_scale"]), bool(r.get("use_ctrl", True))) for _, r in existing_psd.iterrows() if "q_eff_scale" in r}
        psd_frames.append(existing_psd)

    base_q_da = float(base_cfg['neurotransmitters']['DA']['q_eff_e'])
    base_q_sero = float(base_cfg['neurotransmitters']['SERO']['q_eff_e'])

    progress = _make_progress(pm, total=len(args.qeff_scales) * len(ctrl_states), desc="Organoid q_eff")
    for scale in args.qeff_scales:
        for use_ctrl in ctrl_states:
            seed_bar = None
            if args.progress != "none":
                desc = f"q_eff={scale:g}, {'CTRL' if use_ctrl else 'NoCTRL'}"
                seed_bar = _make_progress(pm, total=len(seeds), desc=desc, kind="worker")
            cfg = deepcopy(base_cfg)
            if resume and (float(scale), use_ctrl) in done_ser:
                progress.update(1)
                if seed_bar:
                    seed_bar.close()
                continue
            cfg['pipeline']['use_control_channel'] = bool(use_ctrl)
            cfg['neurotransmitters']['DA']['q_eff_e'] = base_q_da * float(scale)
            cfg['neurotransmitters']['SERO']['q_eff_e'] = base_q_sero * float(scale)
            cfg['pipeline']['Nm_per_symbol'] = args.qeff_nm

            agg = run_monte_carlo(cfg, seeds, args.sequence_length, recalibrate=args.recalibrate, progress=seed_bar)
            agg.update({"q_eff_scale": float(scale), "use_ctrl": bool(use_ctrl)})
            rows.append(agg)

            if not (resume and (float(scale), use_ctrl) in done_psd):
                psd_cfg = deepcopy(cfg)
                psd = capture_noise_psd(psd_cfg, seeds, recalibrate=args.recalibrate)
                if psd:
                    frame = psd_to_dataframe(psd, "q_eff_scale", float(scale))
                    frame["use_ctrl"] = bool(use_ctrl)
                    psd_frames.append(frame)
            progress.update(1)
            if seed_bar:
                seed_bar.close()

    df_qeff = pd.DataFrame(rows)
    if not df_qeff.empty:
        df_qeff = df_qeff.drop_duplicates(subset=["q_eff_scale", "use_ctrl"])
    df_psd = pd.concat(psd_frames, ignore_index=True) if psd_frames else pd.DataFrame()
    progress.close()
    return df_qeff, df_psd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organoid sensitivity sweeps (SER + PSD).")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"), help="Configuration YAML.")
    parser.add_argument("--sequence-length", type=int, default=400, help="Symbols per Monte-Carlo run.")
    parser.add_argument("--seeds", type=int, default=6, help="Number of random seeds per sweep point.")
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[3e-4, 4e-4, 5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3, 2.8e-3, 4e-3, 5.6e-3],
        dest="alpha_values",
    )
    parser.add_argument(
        "--bias-values",
        type=float,
        nargs="+",
        default=[4e-5, 6e-5, 8e-5, 1.0e-4, 1.3e-4, 1.7e-4, 2.3e-4, 3.0e-4, 4.0e-4, 5.5e-4],
        dest="bias_values",
    )
    parser.add_argument("--bias-nm", type=int, default=2000, help="Nm per symbol for bias sweep.")
    parser.add_argument(
        "--bias-gm-scaling",
        choices=["none", "sqrt", "linear"],
        default="sqrt",
        help="How to remap gm when sweeping I_dc (default sqrt scaling).",
    )
    parser.add_argument(
        "--qeff-scales",
        type=float,
        nargs="+",
        default=[1.25, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        dest="qeff_scales",
    )
    parser.add_argument("--qeff-nm", type=int, default=2000, help="Nm per symbol for q_eff sweep.")
    parser.add_argument("--nm-grid", type=int, nargs="+", default=None, help="Optional Nm values for alpha sweep.")
    parser.add_argument("--output-root", type=Path, default=Path("results"), help="Root directory for outputs.")
    parser.add_argument(
        "--progress",
        choices=["gui", "rich", "tqdm", "none"],
        default="tqdm",
        help="Progress backend (gui, rich, tqdm, none).",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=project_root / "results" / "logs",
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
        "--recalibrate",
        action="store_true",
        help="Force threshold recalibration (ignore cached JSON files).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sweep points already present in the output CSVs.",
    )
    parser.add_argument(
        "--dual-ctrl",
        dest="dual_ctrl",
        action="store_true",
        default=True,
        help="Run sweeps with CTRL on and off (default: enabled).",
    )
    parser.add_argument(
        "--no-dual-ctrl",
        dest="dual_ctrl",
        action="store_false",
        help="Disable dual CTRL runs (CTRL-on only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_log:
        setup_tee_logging(Path(args.logdir), prefix="organoid_sensitivity", fsync=args.fsync_logs)
    else:
        print("[log] File logging disabled by --no-log")
    pm = ProgressManager(args.progress, gui_session_meta={"resume": args.resume, "progress": args.progress})
    base_cfg = _load_base_config(args.config)
    seeds = [base_cfg['pipeline'].get('random_seed', 2025) + i for i in range(args.seeds)]
    base_ctrl = bool(base_cfg["pipeline"].get("use_control_channel", True))
    ctrl_states: List[bool] = [True, False] if args.dual_ctrl else [base_ctrl]

    out_data = args.output_root / "data"
    out_fig = args.output_root / "figures"
    # Preload existing data for resume
    alpha_csv = out_data / "organoid_alpha_sensitivity_ser.csv"
    alpha_psd_csv = out_data / "organoid_alpha_sensitivity_psd.csv"
    bias_csv = out_data / "organoid_bias_sensitivity.csv"
    bias_psd_csv = out_data / "organoid_bias_sensitivity_psd.csv"
    qeff_csv = out_data / "organoid_qeff_sensitivity.csv"
    qeff_psd_csv = out_data / "organoid_qeff_sensitivity_psd.csv"

    alpha_existing, alpha_psd_existing = (pd.DataFrame(), pd.DataFrame())
    bias_existing, bias_psd_existing = (pd.DataFrame(), pd.DataFrame())
    qeff_existing, qeff_psd_existing = (pd.DataFrame(), pd.DataFrame())
    if args.resume:
        alpha_existing, _ = _load_existing(alpha_csv, ["alpha_H", "Nm_per_symbol", "use_ctrl"])
        alpha_psd_existing, _ = _load_existing(alpha_psd_csv, ["alpha_H", "use_ctrl"])
        bias_existing, _ = _load_existing(bias_csv, ["I_dc_A", "use_ctrl"])
        bias_psd_existing, _ = _load_existing(bias_psd_csv, ["I_dc_A", "use_ctrl"])
        qeff_existing, _ = _load_existing(qeff_csv, ["q_eff_scale", "use_ctrl"])
        qeff_psd_existing, _ = _load_existing(qeff_psd_csv, ["q_eff_scale", "use_ctrl"])

    # --- Alpha (1/f noise) sweep ---
    df_alpha, df_alpha_psd = run_alpha_sweep(
        base_cfg,
        seeds,
        args,
        ctrl_states,
        existing_ser=alpha_existing,
        existing_psd=alpha_psd_existing,
        resume=args.resume,
        pm=pm,
    )
    _ensure_dir(alpha_csv)
    df_alpha.to_csv(alpha_csv, index=False)
    if not df_alpha.empty:
        plot_ser_vs_nm(df_alpha, "alpha_H", out_fig / "organoid_alpha_ser.png")
        df_alpha_snr = df_alpha.groupby("alpha_H", as_index=False).agg({"snr_i_db_median": "mean"})
        plot_metric_vs_param(df_alpha_snr, "alpha_H", "snr_i_db_median", "SNR (dB)", out_fig / "organoid_alpha_snr.png")
    if not df_alpha_psd.empty:
        _ensure_dir(alpha_psd_csv)
        df_alpha_psd.to_csv(alpha_psd_csv, index=False)
        plot_psd(df_alpha_psd, "alpha_H", ["total_diff", "flicker_diff"], out_fig / "organoid_alpha_psd.png")

    # --- Bias sweep ---
    df_bias, df_bias_psd = run_bias_sweep(
        base_cfg,
        seeds,
        args,
        ctrl_states,
        existing_ser=bias_existing,
        existing_psd=bias_psd_existing,
        resume=args.resume,
        pm=pm,
    )
    _ensure_dir(bias_csv)
    df_bias.to_csv(bias_csv, index=False)
    if not df_bias.empty:
        plot_metric_vs_param(df_bias, "I_dc_A", "ser_mean", "SER", out_fig / "organoid_bias_ser.png")
        df_bias_snr = df_bias.groupby("I_dc_A", as_index=False).agg({"snr_i_db_median": "mean"})
        plot_metric_vs_param(df_bias_snr, "I_dc_A", "snr_i_db_median", "SNR (dB)", out_fig / "organoid_bias_snr.png")
    if not df_bias_psd.empty:
        _ensure_dir(bias_psd_csv)
        df_bias_psd.to_csv(bias_psd_csv, index=False)
        plot_psd(df_bias_psd, "I_dc_A", ["total_diff", "thermal_diff"], out_fig / "organoid_bias_psd.png")

    # --- q_eff (ionic strength) sweep ---
    df_qeff, df_qeff_psd = run_qeff_sweep(
        base_cfg,
        seeds,
        args,
        ctrl_states,
        existing_ser=qeff_existing,
        existing_psd=qeff_psd_existing,
        resume=args.resume,
        pm=pm,
    )
    _ensure_dir(qeff_csv)
    df_qeff.to_csv(qeff_csv, index=False)
    if not df_qeff.empty:
        plot_metric_vs_param(df_qeff, "q_eff_scale", "ser_mean", "SER", out_fig / "organoid_qeff_ser.png")
        df_qeff_snr = df_qeff.groupby("q_eff_scale", as_index=False).agg({"snr_i_db_median": "mean"})
        plot_metric_vs_param(df_qeff_snr, "q_eff_scale", "snr_i_db_median", "SNR (dB)", out_fig / "organoid_qeff_snr.png")
    if not df_qeff_psd.empty:
        _ensure_dir(qeff_psd_csv)
        df_qeff_psd.to_csv(qeff_psd_csv, index=False)
        plot_psd(df_qeff_psd, "q_eff_scale", ["total_diff"], out_fig / "organoid_qeff_psd.png")

    summary = {
        "alpha_points": len(df_alpha),
        "bias_points": len(df_bias),
        "qeff_points": len(df_qeff),
    }
    print(json.dumps(summary, indent=2))
    pm.stop()


if __name__ == "__main__":
    main()
