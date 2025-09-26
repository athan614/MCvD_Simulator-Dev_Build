#!/usr/bin/env python3
"""Analytical ISI model versus simulation sweep."""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style
from analysis.run_final_analysis import preprocess_config_full, calculate_dynamic_symbol_period
from src.pipeline import calculate_proper_noise_sigma
from src.mc_channel.isi_analytic import window_coefficients, gaussian_ser_binary, predicted_stats_mosk


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare analytic ISI model with simulation results.")
    parser.add_argument("--points", type=int, default=11, help="Number of guard factor samples (default: 11).")
    parser.add_argument("--force", action="store_true", help="Ignore existing CSV and recompute.")
    parser.add_argument("--progress", default="none", help="Placeholder for run_master compatibility.")
    return parser.parse_args()


def _load_cfg() -> Dict[str, Any]:
    cfg_path = project_root / "config" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return preprocess_config_full(cfg)


def _resolve_detection_window(cfg: Dict[str, Any], Ts: float) -> float:
    detection = cfg.setdefault("detection", {})
    dt = float(cfg["sim"].get("dt_s", 0.01))
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
        float(cfg["pipeline"].get("time_window_s", 0.0)),
        detection["decision_window_s"],
    )
    return detection["decision_window_s"]


def _pick_anchor(cfg: Dict[str, Any]) -> tuple[float, float]:
    data_dir = project_root / "results" / "data"
    lod_csv = data_dir / "lod_vs_distance_mosk.csv"
    distance = float(cfg["pipeline"].get("distance_um", 50.0))
    nm_hint = float(cfg["pipeline"].get("Nm_per_symbol", 1e4))

    if lod_csv.exists():
        try:
            df = pd.read_csv(lod_csv)
            df = df.dropna(subset=["distance_um"])
        except Exception:
            df = pd.DataFrame()
        if not df.empty and "lod_nm" in df.columns:
            dist_vals = pd.to_numeric(df["distance_um"], errors="coerce").dropna().astype(int)
            if not dist_vals.empty:
                distance = float(np.median(dist_vals))
                match = df.loc[dist_vals == int(distance), "lod_nm"]
                if not match.empty:
                    nm_hint = float(match.iloc[-1])
    return distance, nm_hint


def main() -> None:
    args = _parse_args()
    if args.points <= 1:
        raise ValueError("--points must be greater than 1")

    cfg = _load_cfg()
    distance_um, nm_hint = _pick_anchor(cfg)
    nm_hint = float(nm_hint if math.isfinite(nm_hint) and nm_hint > 0 else cfg["pipeline"].get("Nm_per_symbol", 1e4))

    guard_vals = np.linspace(0.0, 1.0, args.points)
    analytic_rows = []

    for gf in guard_vals:
        cfg_g = copy.deepcopy(cfg)
        cfg_g["pipeline"]["guard_factor"] = float(gf)
        cfg_g["pipeline"]["distance_um"] = float(distance_um)
        cfg_g["pipeline"]["Nm_per_symbol"] = float(nm_hint)

        Ts = calculate_dynamic_symbol_period(distance_um, cfg_g)
        cfg_g["pipeline"]["symbol_period_s"] = Ts
        win = _resolve_detection_window(cfg_g, Ts)
        cfg_g["sim"]["time_window_s"] = max(float(cfg_g["sim"].get("time_window_s", Ts)), Ts)

        h_vec = window_coefficients(distance_m=distance_um * 1e-6, Ts=Ts, win_s=win, cfg=cfg_g, k_max=int(cfg_g["pipeline"].get("isi_memory_cap_symbols", 60)))
        mu_da, mu_sero, _, _ = predicted_stats_mosk(nm_hint, h_vec, cfg_g)
        sigma_da, sigma_sero = calculate_proper_noise_sigma(cfg_g, win)
        sigma_eff = math.sqrt(max(0.5 * (sigma_da ** 2 + sigma_sero ** 2), 1e-18))
        ser_est = gaussian_ser_binary(mu_da, mu_sero, sigma_eff, sigma_eff)

        analytic_rows.append({
            "guard_factor": gf,
            "ser_analytic": ser_est,
            "symbol_period_s": Ts,
            "decision_window_s": win,
        })

    df_analytic = pd.DataFrame(analytic_rows)
    out_csv = project_root / "results" / "data" / "isi_analytic_model.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    df_analytic.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    print(f"[saved] {out_csv}")

    sim_csv = project_root / "results" / "data" / "isi_tradeoff_mosk.csv"
    df_sim = pd.DataFrame()
    if sim_csv.exists():
        try:
            df_sim = pd.read_csv(sim_csv)
        except Exception:
            df_sim = pd.DataFrame()

    gf_col = None
    if not df_sim.empty:
        if "guard_factor" in df_sim.columns:
            gf_col = "guard_factor"
        elif "pipeline.guard_factor" in df_sim.columns:
            gf_col = "pipeline.guard_factor"

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(df_analytic["guard_factor"], df_analytic["ser_analytic"], marker="o", label="Analytic")
    if gf_col:
        ax.plot(df_sim[gf_col], df_sim["ser"], marker="s", label="Simulation")
    ax.axhline(0.01, linestyle="--", linewidth=1.0, label="SER = 1%")
    ax.set_xlabel("Guard factor")
    ax.set_ylabel("SER")
    ax.set_ylim(0.0, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png = project_root / "results" / "figures" / "fig_isi_analytic_vs_sim.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = out_png.with_suffix(out_png.suffix + ".tmp")
    fig.savefig(tmp_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    os.replace(tmp_png, out_png)
    print(f"[saved] {out_png}")
    print("[done] ISI analytic model built.")


if __name__ == "__main__":
    main()
