#!/usr/bin/env python3
"""
Parameter sensitivity sweeps for LoD versus temperature, diffusion, and binding kinetics.

Each sweep reuses the LoD search (with resume support) so the outputs remain
consistent with the main analysis pipeline.

Outputs
-------
results/data/sensitivity_T.csv
results/data/sensitivity_D.csv
results/data/sensitivity_binding.csv
results/data/sensitivity_device.csv
results/data/sensitivity_corr.csv
results/figures/fig_sensitivity_T.png
results/figures/fig_sensitivity_D.png
results/figures/fig_sensitivity_binding_kon{scale}.png
results/figures/fig_sensitivity_device.png
results/figures/fig_sensitivity_corr.png
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
)


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
    detection = cfg.setdefault("detection", {})
    dt = float(cfg["sim"]["dt_s"])
    min_pts = int(cfg.get("_min_decision_points", 4))
    policy = str(detection.get("decision_window_policy", "full_Ts")).lower()

    if policy in ("fraction_of_ts", "fraction", "tail_fraction", "tail"):
        frac = float(detection.get("decision_window_fraction", 0.9))
        min_win = max(min_pts * dt, Ts * frac)
    elif policy == "full_ts":
        min_win = max(min_pts * dt, Ts)
    else:
        explicit = float(detection.get("decision_window_s", Ts))
        min_win = max(min_pts * dt, explicit if explicit > 0 else Ts)

    detection["decision_window_s"] = max(
        float(detection.get("decision_window_s", min_win)), min_win
    )
    cfg["pipeline"]["time_window_s"] = max(
        float(cfg["pipeline"].get("time_window_s", 0.0)), detection["decision_window_s"]
    )
    return detection["decision_window_s"]


def _plot_param(df: pd.DataFrame, x: str, y_cols: Sequence[str], out_png: Path, xlabel: str) -> None:
    if df.empty:
        return
    df_sorted = df.sort_values(by=x)
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    drew_any = False
    for col in y_cols:
        if col in df_sorted.columns and np.isfinite(df_sorted[col]).any():
            ax.plot(df_sorted[x], df_sorted[col], marker="o", label=col)
            drew_any = True
    if not drew_any:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel("LoD (Nm)")
    ax.grid(True, alpha=0.3)
    if len(y_cols) > 1:
        ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_suffix(out_png.suffix + ".tmp")
    fig.savefig(tmp, dpi=400, bbox_inches="tight", pad_inches=0.02)
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
    force: bool = False,
) -> pd.DataFrame:
    out_csv = project_root / "results" / "data" / "sensitivity_T.csv"
    out_fig = project_root / "results" / "figures" / "fig_sensitivity_T.png"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        _plot_param(df_cached, "T_K", ["lod_nm"], out_fig, "Temperature (K)")
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
        cfg_t["sim"]["temperature_K"] = T

        for nt, D_ref in D_refs.items():
            cfg_t["neurotransmitters"][nt]["D_m2_s"] = scale_diffusion_for_temperature(D_ref, T_ref, T)

        cfg_t["pipeline"]["distance_um"] = anchor_distance
        Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_t)
        cfg_t["pipeline"]["symbol_period_s"] = Ts
        _resolve_detection_window(cfg_t, Ts)
        cfg_t["sim"]["time_window_s"] = max(float(cfg_t["sim"].get("time_window_s", Ts)), Ts)

        if math.isfinite(nm_hint) and nm_hint > 0:
            _warm_start(cfg_t, nm_hint)

        cache_label = f"T_{int(T)}K"
        lod_nm, ser_at_lod = _compute_lod(cfg_t, seeds, target_ser, cache_label)

        rows.append(
            {
                "T_K": T,
                "lod_nm": lod_nm,
                "ser_at_lod": ser_at_lod,
                "symbol_period_s": Ts,
                "decision_window_s": float(cfg_t["detection"]["decision_window_s"]),
                "distance_um": anchor_distance,
            }
        )

    df = pd.DataFrame(rows)
    _save_csv_atomic(df, out_csv)
    _plot_param(df, "T_K", ["lod_nm"], out_fig, "Temperature (K)")
    return df


def run_sensitivity_D(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    anchor_distance: int,
    nm_hint: float,
    force: bool = False,
) -> pd.DataFrame:
    out_csv = project_root / "results" / "data" / "sensitivity_D.csv"
    out_fig = project_root / "results" / "figures" / "fig_sensitivity_D.png"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        _plot_param(df_cached, "D_scale", ["lod_nm"], out_fig, "Diffusion scale (x)")
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
        for nt, D_ref in D_refs.items():
            cfg_s["neurotransmitters"][nt]["D_m2_s"] = D_ref * scale

        cfg_s["pipeline"]["distance_um"] = anchor_distance
        Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_s)
        cfg_s["pipeline"]["symbol_period_s"] = Ts
        _resolve_detection_window(cfg_s, Ts)
        cfg_s["sim"]["time_window_s"] = max(float(cfg_s["sim"].get("time_window_s", Ts)), Ts)

        if math.isfinite(nm_hint) and nm_hint > 0:
            _warm_start(cfg_s, nm_hint)

        cache_label = f"Dscale_{scale:.2f}"
        lod_nm, ser_at_lod = _compute_lod(cfg_s, seeds, target_ser, cache_label)

        rows.append(
            {
                "D_scale": scale,
                "lod_nm": lod_nm,
                "ser_at_lod": ser_at_lod,
                "symbol_period_s": Ts,
                "decision_window_s": float(cfg_s["detection"]["decision_window_s"]),
                "distance_um": anchor_distance,
            }
        )

    df = pd.DataFrame(rows)
    _save_csv_atomic(df, out_csv)
    _plot_param(df, "D_scale", ["lod_nm"], out_fig, "Diffusion scale (x)")
    return df


def run_sensitivity_binding(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    anchor_distance: int,
    nm_hint: float,
    force: bool = False,
) -> pd.DataFrame:
    out_csv = project_root / "results" / "data" / "sensitivity_binding.csv"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        for s_on in sorted(df_cached["kon_scale"].unique()):
            suffix = str(s_on).replace(".", "p")
            fig_path = project_root / "results" / "figures" / f"fig_sensitivity_binding_kon{suffix}.png"
            _plot_param(
                df_cached[df_cached["kon_scale"] == s_on],
                "koff_scale",
                ["lod_nm"],
                fig_path,
                f"koff scale (kon x {s_on:g})",
            )
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
            cfg_b["sim"]["time_window_s"] = max(float(cfg_b["sim"].get("time_window_s", Ts)), Ts)

            if math.isfinite(nm_hint) and nm_hint > 0:
                _warm_start(cfg_b, nm_hint)

            cache_label = f"binding_kon{s_on:.2f}_koff{s_off:.2f}"
            lod_nm, ser_at_lod = _compute_lod(cfg_b, seeds, target_ser, cache_label)

            rows.append(
                {
                    "kon_scale": s_on,
                    "koff_scale": s_off,
                    "lod_nm": lod_nm,
                    "ser_at_lod": ser_at_lod,
                    "symbol_period_s": Ts,
                    "decision_window_s": float(cfg_b["detection"]["decision_window_s"]),
                    "distance_um": anchor_distance,
                }
            )

    df = pd.DataFrame(rows)
    _save_csv_atomic(df, out_csv)

    for s_on in sorted(df["kon_scale"].unique()):
        df_on = df[df["kon_scale"] == s_on].sort_values("koff_scale")
        suffix = str(s_on).replace(".", "p")
        fig_path = project_root / "results" / "figures" / f"fig_sensitivity_binding_kon{suffix}.png"
        _plot_param(df_on, "koff_scale", ["lod_nm"], fig_path, f"koff scale (kon x {s_on:g})")

    return df


def _plot_device_curves(df: pd.DataFrame, out_fig: Path) -> None:
    if df is None or df.empty:
        return
    required = {"gm_S", "C_tot_F", "lod_nm"}
    if not required.issubset(df.columns):
        return
    subset = df.dropna(subset=["lod_nm"])
    if subset.empty:
        return
    pivot = subset.pivot_table(index="gm_S", columns="C_tot_F", values="lod_nm", aggfunc="mean")
    if pivot.empty:
        return
    plot_df = pivot.reset_index().sort_values("gm_S")
    original_cols = [c for c in pivot.columns]
    y_cols: list[str] = []
    for col in original_cols:
        new_col = f"C_tot_F={col:.0e}F"
        plot_df[new_col] = plot_df[col]
        y_cols.append(new_col)
    plot_df = plot_df.drop(columns=original_cols)
    _plot_param(plot_df, "gm_S", y_cols, out_fig, "gm (S)")


def _plot_correlation_curves(df: pd.DataFrame, out_fig: Path) -> None:
    if df is None or df.empty:
        return
    required = {"rho_corr", "rho_post", "lod_nm"}
    if not required.issubset(df.columns):
        return
    subset = df.dropna(subset=["lod_nm"])
    if subset.empty:
        return
    pivot = subset.pivot_table(index="rho_corr", columns="rho_post", values="lod_nm", aggfunc="mean")
    if pivot.empty:
        return
    plot_df = pivot.reset_index().sort_values("rho_corr")
    original_cols = [c for c in pivot.columns]
    y_cols: list[str] = []
    for col in original_cols:
        new_col = f"rho_post={col:.2f}"
        plot_df[new_col] = plot_df[col]
        y_cols.append(new_col)
    plot_df = plot_df.drop(columns=original_cols)
    _plot_param(plot_df, "rho_corr", y_cols, out_fig, "rho pre-CTRL")


def run_sensitivity_device(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    anchor_distance: int,
    nm_hint: float,
    force: bool = False,
) -> pd.DataFrame:
    out_csv = project_root / "results" / "data" / "sensitivity_device.csv"
    out_fig = project_root / "results" / "figures" / "fig_sensitivity_device.png"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        _plot_device_curves(df_cached, out_fig)
        return df_cached

    gm_grid = [1e-3, 3e-3, 5e-3, 1e-2]
    c_grid = [1e-8, 3e-8, 5e-8, 1e-7]
    rows: list[dict[str, float]] = []

    for gm in gm_grid:
        for c_tot in c_grid:
            cfg_d = copy.deepcopy(base_cfg)
            cfg_d.setdefault("oect", {})
            cfg_d["oect"]["gm_S"] = gm
            cfg_d["oect"]["C_tot_F"] = c_tot
            cfg_d["pipeline"]["distance_um"] = anchor_distance
            Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_d)
            cfg_d["pipeline"]["symbol_period_s"] = Ts
            _resolve_detection_window(cfg_d, Ts)
            cfg_d["sim"]["time_window_s"] = max(float(cfg_d["sim"].get("time_window_s", Ts)), Ts)
            _warm_start(cfg_d, nm_hint)
            cache_label = f"gm_{gm:.2e}_C_{c_tot:.2e}"
            lod_nm, ser_at_lod = _compute_lod(cfg_d, seeds, target_ser, cache_label)
            rows.append({
                "gm_S": gm,
                "C_tot_F": c_tot,
                "lod_nm": lod_nm,
                "ser_at_lod": ser_at_lod,
                "symbol_period_s": Ts,
                "decision_window_s": float(cfg_d["detection"]["decision_window_s"]),
                "distance_um": anchor_distance,
            })

    df = pd.DataFrame(rows)
    _save_csv_atomic(df, out_csv)
    _plot_device_curves(df, out_fig)
    return df


def run_sensitivity_correlation(
    base_cfg: Dict[str, Any],
    seeds: Sequence[int],
    target_ser: float,
    anchor_distance: int,
    nm_hint: float,
    force: bool = False,
) -> pd.DataFrame:
    out_csv = project_root / "results" / "data" / "sensitivity_corr.csv"
    out_fig = project_root / "results" / "figures" / "fig_sensitivity_corr.png"
    if not force and out_csv.exists():
        df_cached = pd.read_csv(out_csv)
        _plot_correlation_curves(df_cached, out_fig)
        return df_cached

    rho_pre = [0.2, 0.5, 0.7, 0.9]
    rho_post = [0.0, 0.2, 0.4, 0.6]
    rows: list[dict[str, float]] = []

    for r0 in rho_pre:
        for r1 in rho_post:
            cfg_c = copy.deepcopy(base_cfg)
            cfg_c.setdefault("noise", {})
            cfg_c["noise"]["rho_corr"] = r0
            cfg_c["noise"]["rho_between_channels_after_ctrl"] = r1
            cfg_c["pipeline"]["distance_um"] = anchor_distance
            Ts = calculate_dynamic_symbol_period(anchor_distance, cfg_c)
            cfg_c["pipeline"]["symbol_period_s"] = Ts
            _resolve_detection_window(cfg_c, Ts)
            cfg_c["sim"]["time_window_s"] = max(float(cfg_c["sim"].get("time_window_s", Ts)), Ts)
            _warm_start(cfg_c, nm_hint)
            cache_label = f"rho_pre_{r0:.2f}_rho_post_{r1:.2f}"
            lod_nm, ser_at_lod = _compute_lod(cfg_c, seeds, target_ser, cache_label)
            rows.append({
                "rho_corr": r0,
                "rho_post": r1,
                "lod_nm": lod_nm,
                "ser_at_lod": ser_at_lod,
                "symbol_period_s": Ts,
                "decision_window_s": float(cfg_c["detection"]["decision_window_s"]),
                "distance_um": anchor_distance,
            })

    df = pd.DataFrame(rows)
    _save_csv_atomic(df, out_csv)
    _plot_correlation_curves(df, out_fig)
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

    run_sensitivity_T(base_cfg, seeds, args.target_ser, anchor_distance, nm_hint, force=args.force)
    run_sensitivity_D(base_cfg, seeds, args.target_ser, anchor_distance, nm_hint, force=args.force)
    run_sensitivity_binding(base_cfg, seeds, args.target_ser, anchor_distance, nm_hint, force=args.force)
    run_sensitivity_device(base_cfg, seeds, args.target_ser, anchor_distance, nm_hint, force=args.force)
    run_sensitivity_correlation(base_cfg, seeds, args.target_ser, anchor_distance, nm_hint, force=args.force)

    print("[done] Parameter sensitivity sweeps completed.")


if __name__ == "__main__":
    main()
