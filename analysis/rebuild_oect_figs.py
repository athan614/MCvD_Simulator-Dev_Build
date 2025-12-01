# analysis/rebuild_oect_figs.py
"""
Notebook-replica OECT figures using real device noise (no synthetic placeholders).

Outputs (in results/figures/notebook_replicas/):
  - oect_differential_psd.png       (Noise PSD before vs after CTRL subtraction)
  - oect_noise_breakdown.png        (Thermal / 1/f / 1/f^2 budget + total)

Notes:
  • Uses oect.oect_trio/oect_current to synthesize *noise-only* currents.
  • PSD via Welch (density) on long records; IEEE style for TMBMC figures.

Run directly or via run_master.py (steps: oect_psd or notebook_replicas).
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import os
import matplotlib as mpl
if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch  #type: ignore
from typing import Any, Dict, cast
import yaml
import os

# Project import path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# IEEE style
from analysis.ieee_plot_style import apply_ieee_style  # :contentReference[oaicite:3]{index=3}

# Device model
try:
    from src.mc_receiver import oect as oect_mod
except Exception:
    # Fallback if your project is laid out flat for local dev
    import src.mc_receiver.oect as oect_mod


def _load_cfg() -> dict:
    cfg: dict[str, Any] = {}
    cfg_path = project_root / "config" / "default.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Fill only missing keys; keep YAML authoritative so figures match pipeline sweeps
    cfg.setdefault("sim", {}).setdefault("dt_s", 0.01)

    cfg.setdefault("oect", {})
    for k, v in oect_mod.default_params().items():
        cfg["oect"].setdefault(k, v)

    cfg.setdefault("noise", {})
    # Respect aliases without overriding explicit YAML choices
    if "rho_correlated" not in cfg["noise"] and "rho_corr" in cfg["noise"]:
        cfg["noise"]["rho_correlated"] = cfg["noise"]["rho_corr"]
    cfg["noise"].setdefault("rho_correlated", 0.9)
    cfg["noise"].setdefault("rho_corr", cfg["noise"]["rho_correlated"])

    cfg.setdefault("neurotransmitters", {
        "DA": {"q_eff_e": -1.0}, "SERO": {"q_eff_e": +1.0}, "CTRL": {"q_eff_e": 0.0}
    })
    return cfg


def _psd(x: np.ndarray, fs: float):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 16384), detrend="constant", scaling="density")
    return f, Pxx


def _atomic_save(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_suffix(outpath.suffix + ".tmp")
    fig.savefig(tmp, dpi=300, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    os.replace(tmp, outpath)
    print(f"✓ Saved: {outpath}")


def fig_differential_psd(cfg: dict):
    """Before/after CTRL subtraction on a noise-only DA trace."""
    apply_ieee_style()
    dt = float(cfg["sim"]["dt_s"])
    fs = 1.0 / dt

    # Long noise record for LF resolution (>= 200 s)
    T = 200.0
    n = int(T / dt)
    rng = np.random.default_rng(123)

    # Bound sites = 0 → signal=0; still get thermal/flicker/drift noise
    zeros = np.zeros(n)
    bound_sites = np.vstack([zeros, zeros, zeros])  # DA, SERO, CTRL
    nts = ("DA", "SERO", "CTRL")

    trio = cast(Dict[str, np.ndarray], oect_mod.oect_trio(bound_sites, nts, cfg, rng, rho=cfg["noise"]["rho_correlated"]))
    i_da, i_ctrl = trio["DA"], trio["CTRL"]

    # PSD before subtraction (DA alone) vs after subtraction (DA-CTRL)
    f1, p_da = _psd(i_da, fs)
    f2, p_diff = _psd(i_da - i_ctrl, fs)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.loglog(f1, p_da, label="Before subtraction", lw=1.8)
    ax.loglog(f2, p_diff, label="After subtraction", lw=1.8)

    # Light shading for very‑low‑frequency "drift band"
    ax.axvspan(0.01, 0.1, color="0.8", alpha=0.25)
    ax.set_xlim(1e-2, min(fs/2, 1e2))
    ax.set_ylim(1e-25, 1e-20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (A²/Hz)")
    ax.set_title("Noise PSD: impact of CTRL channel subtraction")
    ax.grid(True, which="both", ls="--", alpha=0.25)
    ax.legend()

    _atomic_save(fig, project_root / "results/figures/notebook_replicas/oect_differential_psd.png")


def fig_noise_budget(cfg: dict):
    """Thermal / 1/f / 1/f² components for one (DA) pixel."""
    apply_ieee_style()
    dt = float(cfg["sim"]["dt_s"])
    fs = 1.0 / dt
    T = 200.0
    n = int(T / dt)

    zeros = np.zeros(n)
    out = oect_mod.oect_current(zeros, "DA", cfg, seed=456)  # returns dict of components

    fT, pT = _psd(out["thermal"], fs)
    fF, pF = _psd(out["flicker"], fs)
    fD, pD = _psd(out["drift"], fs)
    fTot, pTot = _psd(out["total"], fs)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.loglog(fT, pT, label="Thermal (white)")
    ax.loglog(fF, pF, label="Flicker 1/f")
    ax.loglog(fD, pD, label="Drift 1/f²")
    ax.loglog(fTot, pTot, "k--", lw=2.0, label="Total")

    # Slope guides (−1 and −2) for visual sanity
    f0 = 0.05
    y0 = pF[np.searchsorted(fF, f0)] if len(fF) else 1e-26
    ax.loglog([f0, 10*f0], [y0, y0/10], color="0.4", lw=1.0)   # ~−1 slope guide
    ax.text(f0*1.2, y0*0.8, "−1", fontsize=8)
    ax.loglog([f0, 10*f0], [y0, y0/100], color="0.4", lw=1.0)  # ~−2 slope guide
    ax.text(f0*1.2, y0*0.1, "−2", fontsize=8)

    ax.set_xlim(1e-2, min(fs/2, 1e2))
    ax.set_ylim(1e-30, 1e-20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (A²/Hz)")
    ax.set_title("OECT Pixel Noise Budget (DA channel)")
    ax.grid(True, which="both", ls="--", alpha=0.25)
    ax.legend(loc="best")

    _atomic_save(fig, project_root / "results/figures/notebook_replicas/oect_noise_breakdown.png")


def main():
    cfg = _load_cfg()
    fig_differential_psd(cfg)
    fig_noise_budget(cfg)


if __name__ == "__main__":
    main()
