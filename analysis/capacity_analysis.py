#!/usr/bin/env python3
"""
Capacity estimations from confusion matrices and calibration statistics.

Outputs
-------
results/data/capacity_bounds.csv
results/data/capacity_bounds_table.tex
results/figures/fig_capacity_{mode}.png
results/figures/fig_capacity_all.png
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import os
import matplotlib as mpl
if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numbers import Real
from analysis.ui_progress import ProgressManager

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style
from analysis.run_final_analysis import (
    preprocess_config_full,
    run_calibration_symbols,
    calculate_dynamic_symbol_period,
    canonical_value_key,
)

ALL_MODES: Tuple[str, ...] = ("MoSK", "CSK", "Hybrid")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate capacity bounds for MoSK, CSK, and Hybrid modes.")
    parser.add_argument("--target-ser", type=float, default=0.01, help="SER operating point (default: 0.01).")
    parser.add_argument("--samples", type=int, default=800, help="Calibration symbols per class (default: 800).")
    parser.add_argument("--mc", type=int, default=5000, help="Monte Carlo samples for MI estimation (default: 5000).")
    parser.add_argument("--force", action="store_true", help="Ignore existing CSV and recompute.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing rows and only compute missing mode/grid points.")
    parser.add_argument(
        "--progress",
        choices=["gui", "rich", "tqdm", "none"],
        default="tqdm",
        help="Progress backend (gui, rich, tqdm, none).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="all",
        help="Comma-separated list of modes (MoSK,CSK,Hybrid) or 'all' (default).",
    )
    return parser.parse_args()


def _resolve_modes(spec: Optional[str]) -> List[str]:
    if not spec:
        return list(ALL_MODES)
    tokens = [tok.strip() for tok in spec.split(",") if tok.strip()]
    if not tokens:
        return list(ALL_MODES)
    if len(tokens) == 1 and tokens[0].lower() == "all":
        return list(ALL_MODES)
    resolved: List[str] = []
    for token in tokens:
        token_upper = token.upper()
        if token_upper == "MOSK":
            resolved.append("MoSK")
        elif token_upper == "CSK":
            resolved.append("CSK")
        elif token_upper == "HYBRID":
            resolved.append("Hybrid")
        else:
            raise ValueError(f"Unsupported mode '{token}'. Expected MoSK, CSK, Hybrid, or 'all'.")
    # Preserve order but remove duplicates while keeping first occurrence
    deduped: List[str] = []
    for mode in resolved:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def _load_cfg() -> Dict[str, Any]:
    cfg_path = project_root / "config" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return preprocess_config_full(cfg)


def _resolve_detection_window(cfg: Dict[str, Any], Ts: float) -> float:
    detection = cfg.setdefault("detection", {})
    pipeline_cfg = cfg.setdefault("pipeline", {})
    dt = float(cfg["sim"]["dt_s"])
    min_pts = int(cfg.get("_min_decision_points", 4))
    policy = str(detection.get("decision_window_policy", "full_ts")).lower()

    if policy in ("fraction_of_ts", "fraction", "tail_fraction", "tail", "frac"):
        frac = float(detection.get("decision_window_fraction", 0.9))
        frac = min(max(frac, 0.1), 1.0)
        win_s = frac * Ts
        anchor = "tail"
    elif policy in ("full_ts", "full", "ts"):
        win_s = Ts
        anchor = detection.get("decision_window_anchor", "start")
    else:
        win_s = float(detection.get("decision_window_s", Ts))
        anchor = detection.get("decision_window_anchor", "start")

    min_win = max(min_pts * dt, min(win_s, Ts))
    detection["decision_window_s"] = max(
        float(detection.get("decision_window_s", min_win)), min_win
    )
    detection["decision_window_anchor"] = str(anchor).lower()

    pipeline_cfg["time_window_s"] = max(
        float(pipeline_cfg.get("time_window_s", 0.0)),
        detection["decision_window_s"],
    )
    return detection["decision_window_s"]


def _nm_column(df: pd.DataFrame) -> Optional[str]:
    for col in (
        "pipeline_Nm_per_symbol",
        "pipeline.Nm_per_symbol",
        "Nm_per_symbol",
        "nm_per_symbol",
    ):
        if col in df.columns:
            return col
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (float, int, np.floating, np.integer, bool, np.bool_)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, Real):
        return float(value)
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr.item())
    except Exception:
        return None
    return None



def _alphabet_size(mode: str) -> int:
    return 2 if mode == "MoSK" else 4


def hard_decision_mi_from_confusion(C: np.ndarray) -> float:
    total = float(C.sum())
    if total <= 0:
        return float("nan")
    Pxy = C / total
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    I = 0.0
    for i in range(Pxy.shape[0]):
        for j in range(Pxy.shape[1]):
            p = float(Pxy[i, j])
            if p <= 0 or Px[i, 0] <= 0 or Py[0, j] <= 0:
                continue
            I += p * (math.log2(p) - math.log2(Px[i, 0]) - math.log2(Py[0, j]))
    return float(I)


def symmetric_channel_ceiling(M: int, ser: float) -> float:
    if M <= 1 or not math.isfinite(ser):
        return float("nan")
    p_err = min(max(float(ser), 0.0), 1.0)
    if p_err >= 1.0:
        return 0.0
    eps = 1e-12
    p_corr = max(1.0 - p_err, eps)
    if M == 1:
        return 0.0
    p_wrong = max(p_err / (M - 1), eps)
    return math.log2(M) + p_corr * math.log2(p_corr) + p_err * math.log2(p_wrong)


def _select_operating_point(df: pd.DataFrame, target_ser: float) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    nm_col = _nm_column(df)
    if nm_col is None or "ser" not in df.columns:
        return None, None, None
    df_valid = df.dropna(subset=[nm_col]).copy()
    if df_valid.empty:
        return None, None, None
    ser_numeric = pd.to_numeric(df_valid["ser"], errors="coerce")
    df_valid = df_valid.loc[ser_numeric.notna()].copy()
    if df_valid.empty:
        return None, None, None
    df_valid["ser_diff"] = (ser_numeric.loc[df_valid.index] - target_ser).abs()
    idx = df_valid["ser_diff"].idxmin()
    nm = _coerce_float(df_valid.loc[idx, nm_col])
    if nm is None or not math.isfinite(nm):
        return None, None, None
    distance: Optional[float] = None
    if "distance_um" in df_valid.columns:
        distance = _coerce_float(df_valid.loc[idx, "distance_um"])
    use_ctrl: Optional[bool] = None
    if "use_ctrl" in df_valid.columns:
        ctrl_val = df_valid.loc[idx, "use_ctrl"]
        if isinstance(ctrl_val, (bool, np.bool_)):
            use_ctrl = bool(ctrl_val)
        elif isinstance(ctrl_val, (int, float, np.integer, np.floating)):
            use_ctrl = bool(ctrl_val)
    return nm, distance, use_ctrl


def _select_operating_grid(
    df: pd.DataFrame,
    target_ser: float,
    desired_ctrl: Optional[bool],
    max_points: int = 10,
) -> List[Tuple[float, float, Optional[bool]]]:
    nm_col = _nm_column(df)
    if nm_col is None or "ser" not in df.columns or "distance_um" not in df.columns:
        return []
    work = df.copy()
    work["distance_um"] = pd.to_numeric(work["distance_um"], errors="coerce")
    work["ser"] = pd.to_numeric(work["ser"], errors="coerce")
    work[nm_col] = pd.to_numeric(work[nm_col], errors="coerce")
    work = work.dropna(subset=["distance_um", "ser", nm_col])
    if work.empty:
        return []
    if desired_ctrl is not None and "use_ctrl" in work.columns:
        work = work[work["use_ctrl"] == bool(desired_ctrl)]
    if work.empty:
        return []
    distances = sorted(work["distance_um"].unique().tolist())
    combos: List[Tuple[float, float, Optional[bool]]] = []
    for dist in distances:
        subset = work[np.isclose(work["distance_um"], dist)]
        if subset.empty:
            continue
        idx = (subset["ser"] - target_ser).abs().idxmin()
        row = subset.loc[idx]
        nm_scalar = _coerce_float(row.get(nm_col))
        if nm_scalar is None or not math.isfinite(nm_scalar):
            continue
        nm_val = float(nm_scalar)
        ctrl_val: Optional[bool] = None
        ctrl_raw = row.get("use_ctrl")
        if isinstance(ctrl_raw, (bool, np.bool_)):
            ctrl_val = bool(ctrl_raw)
        elif isinstance(ctrl_raw, (int, float, np.integer, np.floating)):
            val = float(ctrl_raw)
            if math.isfinite(val):
                ctrl_val = bool(val)
        combos.append((nm_val, float(dist), ctrl_val if ctrl_val is not None else desired_ctrl))
        if len(combos) >= max_points:
            break
    return combos


def _augment_with_lod(
    df_lod: pd.DataFrame,
    desired_ctrl: Optional[bool],
    combos: List[Tuple[float, float, Optional[bool]]],
    max_points: int = 10,
) -> List[Tuple[float, float, Optional[bool]]]:
    if df_lod is None or df_lod.empty or len(combos) >= max_points:
        return combos
    work = df_lod.copy()
    work["distance_um"] = pd.to_numeric(work["distance_um"], errors="coerce")
    work["lod_nm"] = pd.to_numeric(work["lod_nm"], errors="coerce")
    work = work.dropna(subset=["distance_um", "lod_nm"])
    if work.empty:
        return combos
    if desired_ctrl is not None and "use_ctrl" in work.columns:
        work = work[work["use_ctrl"] == bool(desired_ctrl)]
    if work.empty:
        return combos
    seen = {round(float(dist), 6) for _, dist, _ in combos}
    for dist in sorted(work["distance_um"].unique().tolist()):
        if len(combos) >= max_points:
            break
        key = round(float(dist), 6)
        if key in seen:
            continue
        subset = work[np.isclose(work["distance_um"], dist)]
        if subset.empty:
            continue
        nm_vals = pd.to_numeric(subset["lod_nm"], errors="coerce").dropna()
        if nm_vals.empty:
            continue
        nm_candidate = float(nm_vals.iloc[-1])
        ctrl_val: Optional[bool] = None
        if "use_ctrl" in subset.columns and subset["use_ctrl"].notna().any():
            ctrl_val = bool(subset["use_ctrl"].iloc[-1])
        combos.append((nm_candidate, float(dist), ctrl_val if ctrl_val is not None else desired_ctrl))
        seen.add(key)
    return combos


def _iter_ser_segments(mode: str) -> List[Path]:
    base = project_root / "results" / "cache" / mode.lower()
    segments: List[Path] = []
    if not base.exists():
        return segments
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("ser_vs_nm"):
            segments.append(child)
        else:
            for sub in child.iterdir():
                if sub.is_dir() and sub.name.startswith("ser_vs_nm"):
                    segments.append(sub)
    return segments


def _segment_matches_ctrl(name: str, use_ctrl: Optional[bool]) -> bool:
    if use_ctrl is True:
        return "wctrl" in name
    if use_ctrl is False:
        return "noctrl" in name
    return True


def _collect_confusion_from_cache(
    mode: str,
    nm: float,
    use_ctrl: Optional[bool],
    tolerance: float = 0.1,
) -> Tuple[np.ndarray, Optional[float], int]:
    segments = [
        seg for seg in _iter_ser_segments(mode)
        if _segment_matches_ctrl(seg.name, use_ctrl)
    ]
    if not segments:
        size = _alphabet_size(mode)
        return np.zeros((size, size), dtype=int), None, 0

    best_dir: Optional[Path] = None
    best_diff = float("inf")

    for seg in segments:
        target_key = canonical_value_key(nm)
        candidate = seg / target_key
        if candidate.exists():
            best_dir = candidate
            best_diff = 0.0
            break
        for child in seg.iterdir():
            if not child.is_dir():
                continue
            try:
                value = float(child.name)
            except ValueError:
                continue
            diff = abs(value - nm)
            rel = diff / nm if nm not in (0.0, 0) else diff
            if rel <= tolerance and diff < best_diff:
                best_dir = child
                best_diff = diff

    if best_dir is None or not best_dir.exists():
        size = _alphabet_size(mode)
        return np.zeros((size, size), dtype=int), None, 0

    tx_all: List[int] = []
    rx_all: List[int] = []

    for seed_file in best_dir.glob("seed_*.json"):
        try:
            with seed_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        tx = data.get("symbols_tx")
        rx = data.get("symbols_rx")
        if not isinstance(tx, list) or not isinstance(rx, list):
            continue
        if len(tx) != len(rx):
            continue
        try:
            tx_all.extend(int(v) for v in tx)
            rx_all.extend(int(v) for v in rx)
        except Exception:
            continue

    if not tx_all or not rx_all:
        size = _alphabet_size(mode)
        return np.zeros((size, size), dtype=int), None, 0

    tx_arr = np.asarray(tx_all, dtype=int)
    rx_arr = np.asarray(rx_all, dtype=int)
    max_symbol = int(max(tx_arr.max(initial=0), rx_arr.max(initial=0))) + 1
    size = max(_alphabet_size(mode), max_symbol)
    C = np.zeros((size, size), dtype=int)
    for t, r in zip(tx_arr, rx_arr):
        if 0 <= t < size and 0 <= r < size:
            C[t, r] += 1

    try:
        nm_cache = float(best_dir.name)
    except ValueError:
        nm_cache = None

    return C, nm_cache, int(tx_arr.size)


def soft_mi_from_Q(
    cfg: Dict[str, Any],
    mode: str,
    distance_um: float,
    nm: float,
    samples: int,
    mc_samples: int,
    use_ctrl: Optional[bool],
) -> float:
    if not math.isfinite(nm) or nm <= 0:
        return float("nan")

    cfg_run = copy.deepcopy(cfg)
    cfg_run["pipeline"]["modulation"] = mode
    cfg_run["pipeline"]["distance_um"] = float(distance_um)
    cfg_run["pipeline"]["Nm_per_symbol"] = float(nm)
    if use_ctrl is not None:
        cfg_run["pipeline"]["use_control_channel"] = bool(use_ctrl)

    Ts = calculate_dynamic_symbol_period(float(distance_um), cfg_run)
    cfg_run["pipeline"]["symbol_period_s"] = Ts
    _resolve_detection_window(cfg_run, Ts)
    cfg_run["sim"]["time_window_s"] = max(float(cfg_run["sim"].get("time_window_s", Ts)), Ts)

    M = _alphabet_size(mode)
    mus: List[float] = []
    sigs: List[float] = []
    pri = np.ones(M) / M

    for symbol in range(M):
        try:
            res = run_calibration_symbols(cfg_run, symbol=symbol, mode=mode, num_symbols=samples)
        except Exception:
            res = None
        q_vals: List[float] = []
        if isinstance(res, dict):
            if isinstance(res.get("q_values"), list):
                q_vals = res["q_values"]
            elif isinstance(res.get("decision_stats"), list):
                q_vals = res["decision_stats"]
        if not q_vals:
            mus.append(0.0)
            sigs.append(1.0)
            continue
        arr = np.asarray(q_vals, dtype=float)
        if arr.size == 0:
            mus.append(0.0)
            sigs.append(1.0)
            continue
        mus.append(float(np.mean(arr)))
        std = float(np.std(arr, ddof=1) if arr.size > 1 else 0.0)
        sigs.append(std if std > 0 else 1.0)

    rng = np.random.default_rng(12345)
    xs = rng.integers(0, M, size=mc_samples)
    ys = np.zeros(mc_samples, dtype=float)
    for idx, sym in enumerate(xs):
        ys[idx] = rng.normal(loc=mus[sym], scale=sigs[sym])

    def log_py(y_val: float) -> float:
        total = 0.0
        for i in range(M):
            coef = pri[i] / (math.sqrt(2.0 * math.pi) * sigs[i])
            total += coef * math.exp(-0.5 * ((y_val - mus[i]) / sigs[i]) ** 2)
        return math.log(max(total, 1e-300))

    bits = 0.0
    for sym, y_val in zip(xs, ys):
        log_cond = -0.5 * ((y_val - mus[sym]) / sigs[sym]) ** 2 - math.log(math.sqrt(2.0 * math.pi) * sigs[sym])
        bits += (log_cond - log_py(y_val)) / math.log(2.0)
    return float(max(bits / mc_samples, 0.0))


def _plot_capacity(df_mode: pd.DataFrame, mode: str) -> None:
    if df_mode.empty:
        return
    df_sorted = df_mode.sort_values("distance_um")
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(df_sorted["distance_um"], df_sorted["I_soft_bits"], marker="o", label="Soft MI (Q)")
    if "I_hd_bits" in df_sorted.columns and df_sorted["I_hd_bits"].notna().any():
        ax.plot(df_sorted["distance_um"], df_sorted["I_hd_bits"], marker="s", label="Hard MI (confusion)")
    if "I_sym_ceiling_bits" in df_sorted.columns and df_sorted["I_sym_ceiling_bits"].notna().any():
        ax.plot(df_sorted["distance_um"], df_sorted["I_sym_ceiling_bits"], linestyle="--", label="Symmetric ceiling")
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Bits per use")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png = project_root / "results" / "figures" / f"fig_capacity_{mode.lower()}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_suffix(out_png.suffix + ".tmp")
    fig.savefig(tmp, dpi=400, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    os.replace(tmp, out_png)
    print(f"[saved] {out_png}")


def _plot_capacity_dashboard(df: pd.DataFrame, modes: Sequence[str]) -> None:
    if df.empty or not modes:
        return
    data = [(m, df[df["mode"] == m].sort_values("distance_um")) for m in modes]
    if not any(not d.empty for _, d in data):
        return
    apply_ieee_style()
    fig, axes = plt.subplots(1, len(modes), figsize=(9.6, 2.6), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    legend_handles = None
    legend_labels = None
    for ax, (mode, df_mode) in zip(axes, data):
        ax.set_title(mode)
        if df_mode.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Distance (um)")
            continue
        ax.plot(df_mode["distance_um"], df_mode["I_soft_bits"], marker="o", label="Soft MI (Q)")
        if df_mode.get("I_hd_bits") is not None and df_mode["I_hd_bits"].notna().any():
            ax.plot(df_mode["distance_um"], df_mode["I_hd_bits"], marker="s", label="Hard MI")
        if df_mode.get("I_sym_ceiling_bits") is not None and df_mode["I_sym_ceiling_bits"].notna().any():
            ax.plot(df_mode["distance_um"], df_mode["I_sym_ceiling_bits"], linestyle="--", label="Symmetric ceiling")
        ax.set_xlabel("Distance (um)")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            legend_handles, legend_labels = handles, labels
    axes[0].set_ylabel("Bits per use")
    if legend_handles and legend_labels:
        axes[-1].legend(legend_handles, legend_labels, loc="upper right")
    out_png = project_root / "results" / "figures" / "fig_capacity_all.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_suffix(out_png.suffix + ".tmp")
    fig.savefig(tmp, dpi=400, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    os.replace(tmp, out_png)
    print(f"[saved] {out_png}")


def _write_capacity_table(df: pd.DataFrame, modes: Sequence[str]) -> None:
    if df.empty or not modes:
        return
    rows = []
    for mode in modes:
        df_mode = df[df["mode"] == mode]
        if df_mode.empty:
            rows.append((mode, "--", "--", "--"))
            continue
        last = df_mode.sort_values("distance_um").iloc[-1]
        def fmt(val: Any) -> str:
            if val is None:
                return "--"
            try:
                fval = float(val)
            except Exception:
                return "--"
            if not math.isfinite(fval):
                return "--"
            return f"{fval:.2f}"
        rows.append((mode, fmt(last.get("I_soft_bits")), fmt(last.get("I_hd_bits")), fmt(last.get("I_sym_ceiling_bits"))))
    table_path = project_root / "results" / "data" / "capacity_bounds_table.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    slash = "\\" * 2
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(f"Mode & Soft MI (bits) & Hard MI (bits) & Symmetric ceiling (bits) {slash}")
    lines.append(r"\hline")
    for mode, soft_val, hard_val, ceil_val in rows:
        row = f"{mode} & {soft_val} & {hard_val} & {ceil_val}"
        lines.append(row + " " + slash)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    tmp = table_path.with_suffix(table_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, table_path)
    print(f"[saved] {table_path}")



def main() -> None:
    args = _parse_args()
    if args.samples <= 0 or args.mc <= 0:
        raise ValueError("--samples and --mc must be positive integers")

    selected_modes = _resolve_modes(args.modes)
    if not selected_modes:
        selected_modes = list(ALL_MODES)

    pm = ProgressManager(args.progress, gui_session_meta={"resume": args.resume, "progress": args.progress})
    overall_bar = pm.task(total=len(selected_modes), description="Capacity modes", kind="overall") if args.progress != "none" else None

    cfg = _load_cfg()
    data_dir = project_root / "results" / "data"

    out_csv = project_root / "results" / "data" / "capacity_bounds.csv"
    existing_df = pd.DataFrame()
    existing_keys: set[tuple] = set()
    if args.resume and not args.force and out_csv.exists():
        try:
            existing_df = pd.read_csv(out_csv)
            for _, row in existing_df.iterrows():
                nm_val = row.get("Nm", float("nan"))
                for alt_nm_col in ("Nm_per_symbol", "pipeline_Nm_per_symbol", "pipeline.Nm_per_symbol"):
                    if (pd.isna(nm_val) or not math.isfinite(float(nm_val))) and alt_nm_col in row:
                        nm_val = row.get(alt_nm_col, nm_val)
                existing_keys.add(
                    (
                        str(row.get("mode", "")).strip(),
                        float(row.get("distance_um", float("nan"))),
                        float(nm_val),
                        bool(row.get("use_ctrl", True)),
                    )
                )
            print(f"[resume] Loaded {len(existing_df)} existing capacity rows from {out_csv}")
        except Exception as exc:
            print(f"[resume] Failed to load existing capacity CSV ({exc}); recomputing.")
            existing_df = pd.DataFrame()
            existing_keys = set()

    records: List[Dict[str, Any]] = cast(List[Dict[str, Any]], existing_df.to_dict("records")) if not existing_df.empty else []

    for mode in selected_modes:
        ser_csv = data_dir / f"ser_vs_nm_{mode.lower()}.csv"
        desired_ctrl = bool(cfg["pipeline"].get("use_control_channel", True))
        if mode == "MoSK":
            desired_ctrl = False

        combos: List[Tuple[float, float, Optional[bool]]] = []
        nm_hint: Optional[float] = None
        distance_hint: Optional[float] = None
        ctrl_hint: Optional[bool] = None
        df_ser = pd.DataFrame()

        if ser_csv.exists():
            try:
                df_ser = pd.read_csv(ser_csv)
                nm_hint, distance_hint, ctrl_hint = _select_operating_point(df_ser, args.target_ser)
                combos = _select_operating_grid(df_ser, args.target_ser, desired_ctrl)
            except Exception:
                df_ser = pd.DataFrame()
                nm_hint, distance_hint, ctrl_hint = None, None, None

        lod_csv = data_dir / f"lod_vs_distance_{mode.lower()}.csv"
        df_lod = pd.DataFrame()
        if lod_csv.exists():
            try:
                df_lod = pd.read_csv(lod_csv)
            except Exception:
                df_lod = pd.DataFrame()
        combos = _augment_with_lod(df_lod, desired_ctrl, combos)

        distance_fallback = distance_hint if (distance_hint is not None and math.isfinite(distance_hint)) else None
        nm_fallback = nm_hint if (nm_hint is not None and math.isfinite(nm_hint)) else None

        if (distance_fallback is None or nm_fallback is None) and not df_lod.empty and "distance_um" in df_lod.columns:
            dist_vals = pd.to_numeric(df_lod["distance_um"], errors="coerce").dropna()
            if not dist_vals.empty:
                distance_fallback = float(np.median(dist_vals))
                if (nm_fallback is None or not math.isfinite(nm_fallback)) and "lod_nm" in df_lod.columns:
                    match = df_lod.loc[np.isclose(dist_vals, distance_fallback), "lod_nm"]
                    match = pd.to_numeric(match, errors="coerce").dropna()
                    if not match.empty:
                        nm_fallback = float(match.iloc[-1])

        if distance_fallback is None or not math.isfinite(distance_fallback):
            fallback_distance = _coerce_float(cfg["pipeline"].get("distance_um", 50.0))
            distance_fallback = fallback_distance if (fallback_distance is not None and math.isfinite(fallback_distance)) else 50.0
        if nm_fallback is None or not math.isfinite(nm_fallback):
            fallback_nm = _coerce_float(cfg["pipeline"].get("Nm_per_symbol", 1e4))
            nm_fallback = fallback_nm if (fallback_nm is not None and math.isfinite(fallback_nm)) else 1e4
        ctrl_fallback = ctrl_hint if isinstance(ctrl_hint, (bool, np.bool_)) else desired_ctrl

        if not combos:
            combos.append((float(nm_fallback), float(distance_fallback), ctrl_fallback))

        combos = combos[:10]
        mode_bar = pm.task(total=len(combos), description=f"{mode} capacity", kind="sweep") if args.progress != "none" else None

        for point_idx, (nm_value, distance_value, ctrl_value) in enumerate(combos, start=1):
            use_ctrl = desired_ctrl if ctrl_value is None else bool(ctrl_value)
            key = (mode, float(distance_value), float(nm_value), bool(use_ctrl))
            if args.resume and key in existing_keys:
                print(f"[resume] Skipping {mode} distance={distance_value} Nm={nm_value} use_ctrl={use_ctrl} (cached)")
                if mode_bar:
                    mode_bar.update(1)
                continue
            print(f"[info] {mode}: point {point_idx:02d} distance={distance_value:.1f} um, Nm={nm_value:.3g}, use_ctrl={use_ctrl}")

            confusion, nm_cache, total_samples = _collect_confusion_from_cache(mode, nm_value, use_ctrl)
            nm_cache_val = _coerce_float(nm_cache)
            nm_eff = nm_cache_val if (nm_cache_val is not None and math.isfinite(nm_cache_val)) else nm_value

            ser_cache = float("nan")
            if total_samples > 0:
                ser_cache = 1.0 - (np.trace(confusion) / total_samples)

            I_hd = hard_decision_mi_from_confusion(confusion) if total_samples > 0 else float("nan")
            I_sym = symmetric_channel_ceiling(confusion.shape[0], ser_cache) if total_samples > 0 else float("nan")
            I_soft = soft_mi_from_Q(cfg, mode, distance_value, nm_eff, args.samples, args.mc, use_ctrl)

            records.append({
                "mode": mode,
                "distance_um": distance_value,
                "Nm": nm_eff,
                "use_ctrl": use_ctrl,
                "total_samples": total_samples,
                "ser_from_cache": ser_cache,
                "I_hd_bits": I_hd,
                "I_sym_ceiling_bits": I_sym,
                "I_soft_bits": I_soft,
                "point_index": point_idx,
            })
            if mode_bar:
                mode_bar.update(1)
        if mode_bar:
            mode_bar.close()
        if overall_bar:
            overall_bar.update(1)

    df_out = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    df_out.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    print(f"[saved] {out_csv}")

    for mode in selected_modes:
        df_mode = df_out[df_out["mode"] == mode]
        _plot_capacity(df_mode, mode)

    _plot_capacity_dashboard(df_out, selected_modes)
    _write_capacity_table(df_out, selected_modes)
    pm.stop()

    print("[done] Capacity analysis complete.")


if __name__ == "__main__":
    main()
