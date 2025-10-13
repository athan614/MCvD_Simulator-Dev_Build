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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import math

from src.config_utils import preprocess_config
from src.pipeline import run_sequence


@dataclass
class SweepConfig:
    values: Sequence[float]
    label: str
    param_path: Tuple[str, ...]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def run_monte_carlo(cfg: Dict[str, Any], seeds: Sequence[int], sequence_length: int) -> Dict[str, float]:
    results: List[Dict[str, Any]] = []
    for seed in seeds:
        cfg_seed = deepcopy(cfg)
        cfg_seed['pipeline']['random_seed'] = int(seed)
        cfg_seed['pipeline']['sequence_length'] = sequence_length
        cfg_seed['disable_progress'] = True
        res = run_sequence(cfg_seed)
        results.append(res)
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


def capture_noise_psd(cfg: Dict[str, Any], seeds: Sequence[int]) -> Optional[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for seed in seeds:
        cfg_noise = deepcopy(cfg)
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
    for value, group in df.groupby(sweep_label):
        group_sorted = group.sort_values("Nm_per_symbol")
        ax.errorbar(
            group_sorted["Nm_per_symbol"],
            group_sorted["ser_mean"],
            yerr=group_sorted["ser_sem"],
            marker="o",
            linewidth=1.2,
            label=f"{sweep_label}={value:g}",
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
    df_sorted = df.sort_values(param)
    ax.plot(df_sorted[param], df_sorted[metric], marker="o", linewidth=1.2)
    ax.set_xlabel(param)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_psd(df: pd.DataFrame, sweep_label: str, spectra_filter: Sequence[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    spectra = set(spectra_filter)
    for value, group in df.groupby(sweep_label):
        for spectrum, spec_group in group[group["spectrum"].isin(spectra)].groupby("spectrum"):
            spec_sorted = spec_group.sort_values("frequency_hz")
            ax.loglog(
                spec_sorted["frequency_hz"],
                spec_sorted["power_a2_per_hz"],
                label=f"{sweep_label}={value:g} | {spectrum}",
                linewidth=1.2,
            )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (A^2/Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small")
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run_alpha_sweep(base_cfg: Dict[str, Any], seeds: Sequence[int], args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nm_values = args.nm_grid if args.nm_grid else base_cfg.get("Nm_range", {}).get("MoSK", [200, 350, 650, 1100, 1750])
    rows: List[Dict[str, float]] = []
    psd_frames: List[pd.DataFrame] = []

    for alpha in args.alpha_values:
        for nm in nm_values:
            cfg = deepcopy(base_cfg)
            cfg['noise']['alpha_H'] = float(alpha)
            cfg['pipeline']['Nm_per_symbol'] = int(nm)
            agg = run_monte_carlo(cfg, seeds, args.sequence_length)
            agg.update({"alpha_H": float(alpha), "Nm_per_symbol": int(nm)})
            rows.append(agg)

        psd_cfg = deepcopy(base_cfg)
        psd_cfg['noise']['alpha_H'] = float(alpha)
        psd_cfg['pipeline']['Nm_per_symbol'] = int(nm_values[0]) if nm_values else base_cfg['pipeline'].get('Nm_per_symbol', 1000)
        psd = capture_noise_psd(psd_cfg, seeds)
        if psd:
            psd_frames.append(psd_to_dataframe(psd, "alpha_H", float(alpha)))

    df_ser = pd.DataFrame(rows)
    df_psd = pd.concat(psd_frames, ignore_index=True) if psd_frames else pd.DataFrame()
    return df_ser, df_psd


def run_bias_sweep(base_cfg: Dict[str, Any], seeds: Sequence[int], args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    psd_frames: List[pd.DataFrame] = []

    oect_cfg = base_cfg.get('oect', {})
    base_i_dc = float(oect_cfg.get('I_dc_A', 1.0e-4) or 1.0e-4)
    if not math.isfinite(base_i_dc) or base_i_dc <= 0.0:
        base_i_dc = 1.0e-4
    base_gm = float(oect_cfg.get('gm_S', 5.0e-3) or 5.0e-3)
    if not math.isfinite(base_gm) or base_gm <= 0.0:
        base_gm = 5.0e-3

    gm_mode = getattr(args, "bias_gm_scaling", "sqrt")

    for bias in args.bias_values:
        cfg = deepcopy(base_cfg)
        bias_val = float(bias)
        cfg['oect']['I_dc_A'] = bias_val

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
        agg = run_monte_carlo(cfg, seeds, args.sequence_length)
        agg.update({
            "I_dc_A": bias_val,
            "gm_S": gm_scaled,
            "gm_scale": gm_scale,
        })
        rows.append(agg)

        psd_cfg = deepcopy(base_cfg)
        psd_cfg['oect']['I_dc_A'] = bias_val
        psd_cfg['oect']['gm_S'] = gm_scaled
        psd_cfg['pipeline']['Nm_per_symbol'] = args.bias_nm
        psd = capture_noise_psd(psd_cfg, seeds)
        if psd:
            frame = psd_to_dataframe(psd, "I_dc_A", float(bias))
            frame["gm_S"] = gm_scaled
            frame["gm_scale"] = gm_scale
            psd_frames.append(frame)

    df_bias = pd.DataFrame(rows)
    df_psd = pd.concat(psd_frames, ignore_index=True) if psd_frames else pd.DataFrame()
    return df_bias, df_psd


def run_qeff_sweep(base_cfg: Dict[str, Any], seeds: Sequence[int], args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    psd_frames: List[pd.DataFrame] = []

    base_q_da = float(base_cfg['neurotransmitters']['DA']['q_eff_e'])
    base_q_sero = float(base_cfg['neurotransmitters']['SERO']['q_eff_e'])

    for scale in args.qeff_scales:
        cfg = deepcopy(base_cfg)
        cfg['neurotransmitters']['DA']['q_eff_e'] = base_q_da * float(scale)
        cfg['neurotransmitters']['SERO']['q_eff_e'] = base_q_sero * float(scale)
        cfg['pipeline']['Nm_per_symbol'] = args.qeff_nm

        agg = run_monte_carlo(cfg, seeds, args.sequence_length)
        agg.update({"q_eff_scale": float(scale)})
        rows.append(agg)

        psd_cfg = deepcopy(cfg)
        psd = capture_noise_psd(psd_cfg, seeds)
        if psd:
            psd_frames.append(psd_to_dataframe(psd, "q_eff_scale", float(scale)))

    df_qeff = pd.DataFrame(rows)
    df_psd = pd.concat(psd_frames, ignore_index=True) if psd_frames else pd.DataFrame()
    return df_qeff, df_psd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organoid sensitivity sweeps (SER + PSD).")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"), help="Configuration YAML.")
    parser.add_argument("--sequence-length", type=int, default=400, help="Symbols per Monte-Carlo run.")
    parser.add_argument("--seeds", type=int, default=6, help="Number of random seeds per sweep point.")
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[5e-4, 1e-3, 2e-3, 4e-3], dest="alpha_values")
    parser.add_argument("--bias-values", type=float, nargs="+", default=[5e-5, 1e-4, 2e-4, 4e-4], dest="bias_values")
    parser.add_argument("--bias-nm", type=int, default=2000, help="Nm per symbol for bias sweep.")
    parser.add_argument(
        "--bias-gm-scaling",
        choices=["none", "sqrt", "linear"],
        default="sqrt",
        help="How to remap gm when sweeping I_dc (default sqrt scaling).",
    )
    parser.add_argument("--qeff-scales", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25], dest="qeff_scales")
    parser.add_argument("--qeff-nm", type=int, default=2000, help="Nm per symbol for q_eff sweep.")
    parser.add_argument("--nm-grid", type=int, nargs="+", default=None, help="Optional Nm values for alpha sweep.")
    parser.add_argument("--output-root", type=Path, default=Path("results"), help="Root directory for outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = _load_base_config(args.config)
    seeds = [base_cfg['pipeline'].get('random_seed', 2025) + i for i in range(args.seeds)]

    out_data = args.output_root / "data"
    out_fig = args.output_root / "figures"

    # --- Alpha (1/f noise) sweep ---
    df_alpha, df_alpha_psd = run_alpha_sweep(base_cfg, seeds, args)
    alpha_csv = out_data / "organoid_alpha_sensitivity_ser.csv"
    _ensure_dir(alpha_csv)
    df_alpha.to_csv(alpha_csv, index=False)
    if not df_alpha.empty:
        plot_ser_vs_nm(df_alpha, "alpha_H", out_fig / "organoid_alpha_ser.png")
        df_alpha_snr = df_alpha.groupby("alpha_H", as_index=False).agg({"snr_i_db_median": "mean"})
        plot_metric_vs_param(df_alpha_snr, "alpha_H", "snr_i_db_median", "SNR (dB)", out_fig / "organoid_alpha_snr.png")
    if not df_alpha_psd.empty:
        psd_csv = out_data / "organoid_alpha_sensitivity_psd.csv"
        _ensure_dir(psd_csv)
        df_alpha_psd.to_csv(psd_csv, index=False)
        plot_psd(df_alpha_psd, "alpha_H", ["total_diff", "flicker_diff"], out_fig / "organoid_alpha_psd.png")

    # --- Bias sweep ---
    df_bias, df_bias_psd = run_bias_sweep(base_cfg, seeds, args)
    bias_csv = out_data / "organoid_bias_sensitivity.csv"
    _ensure_dir(bias_csv)
    df_bias.to_csv(bias_csv, index=False)
    if not df_bias.empty:
        plot_metric_vs_param(df_bias, "I_dc_A", "ser_mean", "SER", out_fig / "organoid_bias_ser.png")
        df_bias_snr = df_bias.groupby("I_dc_A", as_index=False).agg({"snr_i_db_median": "mean"})
        plot_metric_vs_param(df_bias_snr, "I_dc_A", "snr_i_db_median", "SNR (dB)", out_fig / "organoid_bias_snr.png")
    if not df_bias_psd.empty:
        psd_csv = out_data / "organoid_bias_sensitivity_psd.csv"
        _ensure_dir(psd_csv)
        df_bias_psd.to_csv(psd_csv, index=False)
        plot_psd(df_bias_psd, "I_dc_A", ["total_diff", "thermal_diff"], out_fig / "organoid_bias_psd.png")

    # --- q_eff (ionic strength) sweep ---
    df_qeff, df_qeff_psd = run_qeff_sweep(base_cfg, seeds, args)
    qeff_csv = out_data / "organoid_qeff_sensitivity.csv"
    _ensure_dir(qeff_csv)
    df_qeff.to_csv(qeff_csv, index=False)
    if not df_qeff.empty:
        plot_metric_vs_param(df_qeff, "q_eff_scale", "ser_mean", "SER", out_fig / "organoid_qeff_ser.png")
        df_qeff_snr = df_qeff.groupby("q_eff_scale", as_index=False).agg({"snr_i_db_median": "mean"})
        plot_metric_vs_param(df_qeff_snr, "q_eff_scale", "snr_i_db_median", "SNR (dB)", out_fig / "organoid_qeff_snr.png")
    if not df_qeff_psd.empty:
        psd_csv = out_data / "organoid_qeff_sensitivity_psd.csv"
        _ensure_dir(psd_csv)
        df_qeff_psd.to_csv(psd_csv, index=False)
        plot_psd(df_qeff_psd, "q_eff_scale", ["total_diff"], out_fig / "organoid_qeff_psd.png")

    summary = {
        "alpha_points": len(df_alpha),
        "bias_points": len(df_bias),
        "qeff_points": len(df_qeff),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
