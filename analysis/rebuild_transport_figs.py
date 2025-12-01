# analysis/rebuild_transport_figs.py
"""
Notebook-replica propagation/transport figures using the real diffusion model.

Outputs (results/figures/notebook_replicas/):
  - concentration_profiles.png
  - delay_factors_analysis.png
  - distance_scaling_analysis.png
"""

from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # headless-safe backend to avoid native UI crashes
import matplotlib.pyplot as plt
import yaml
from typing import Any

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style  # :contentReference[oaicite:6]{index=6}
try:
    from src.mc_channel import transport as transport_mod
except Exception:
    import src.mc_channel.transport as transport_mod


def _load_cfg() -> dict:
    p = project_root / "config" / "default.yaml"
    cfg: dict[str, Any] = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    cfg.setdefault("sim", {}).setdefault("dt_s", 0.01)
    cfg.setdefault("T_release_ms", 10.0)
    cfg.setdefault("burst_shape", "rect")
    cfg.setdefault("gamma_shape_k", 2.0)
    cfg.setdefault("gamma_scale_theta", 5e-3)
    cfg.setdefault("alpha", 0.2)
    cfg.setdefault("clearance_rate", 0.1)
    return cfg


def _atomic_save(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_suffix(outpath.suffix + ".tmp")
    fig.savefig(tmp, dpi=300, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    os.replace(tmp, outpath)
    print(f"✓ Saved: {outpath}")


def fig_concentration_profiles(cfg: dict):
    apply_ieee_style()
    dt = float(cfg["sim"]["dt_s"])
    t = np.arange(int(10.0/dt)) * dt
    Nm_small = 3e3
    dists = [50e-6, 100e-6, 200e-6]

    # (A) DA/SERO – rect burst; (B) same – gamma burst
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.0))
    for j, (nt, title) in enumerate([("DA", "DA"), ("SERO", "SERO")]):
        for i, dist in enumerate(dists):
            cfg_rect = {**cfg, "burst_shape": "rect"}
            c_rect = transport_mod.finite_burst_concentration(Nm_small, dist, t, cfg_rect, nt)
            tpk = t[np.argmax(c_rect)]
            axes[0, j].plot(t, c_rect, label=f"{int(dist*1e6)} µm (peak: {tpk:.2f}s)")
        axes[0, j].set_title(f"{title} - Rect Burst")
        axes[0, j].set_xlabel("Time (s)")
        axes[0, j].set_ylabel("Concentration (nM)")
        axes[0, j].legend(fontsize=8)

        for i, dist in enumerate(dists):
            cfg_gamma = {**cfg, "burst_shape": "gamma"}
            c_gam = transport_mod.finite_burst_concentration(Nm_small*2, dist, t, cfg_gamma, nt)
            tpk = t[np.argmax(c_gam)]
            axes[1, j].plot(t, c_gam, label=f"{int(dist*1e6)} µm (peak: {tpk:.2f}s)")
        axes[1, j].set_title(f"{title} - Gamma Burst")
        axes[1, j].set_xlabel("Time (s)")
        axes[1, j].set_ylabel("Concentration (nM)")
        axes[1, j].legend(fontsize=8)

    fig.tight_layout()
    _atomic_save(fig, project_root / "results/figures/notebook_replicas/concentration_profiles.png")

    # Normalized shapes (overlay): small inset figure
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for nt, ax in [("DA", ax1), ("SERO", ax2)]:
        c = transport_mod.finite_burst_concentration(Nm_small, 100e-6, t, cfg, nt)
        ax.plot(t, c/np.max(c))
        ax.axhline(0.5, color="0.5", ls=":", lw=1.0)
        ax.set_title("Normalized Profiles (Shape Comparison)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Concentration")
    fig2.tight_layout()
    _atomic_save(fig2, project_root / "results/figures/notebook_replicas/delay_factors_analysis.png")


def fig_distance_scaling(cfg: dict):
    apply_ieee_style()
    dt = float(cfg["sim"]["dt_s"])
    t = np.arange(int(20.0/dt)) * dt
    d_grid = np.linspace(10e-6, 320e-6, 12)  # 10–320 µm
    Nm = 1e4

    def t_peak(nt: str):
        vals = []
        for d in d_grid:
            c = transport_mod.finite_burst_concentration(Nm, d, t, cfg, nt)
            vals.append(t[np.argmax(c)])
        return np.array(vals)

    tpk_da = t_peak("DA"); tpk_sero = t_peak("SERO")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    axL.plot(d_grid*1e6, tpk_da, label="DA")
    axL.plot(d_grid*1e6, tpk_sero, label="SERO")
    axL.set_xlabel("Distance (µm)"); axL.set_ylabel("Time to Peak (s)")
    axL.set_title("Propagation Delay vs Distance"); axL.legend(); axL.grid(True, ls="--", alpha=0.25)

    axR.loglog(d_grid*1e6, tpk_da, label="DA")
    axR.loglog(d_grid*1e6, tpk_sero, label="SERO")
    # Reference slopes (r^2 and r^1.5) for visual comparison
    r = d_grid*1e6
    axR.loglog(r, 1e-3*(r/r[0])**2, "k--", alpha=0.6, label="∝ r²")
    axR.loglog(r, 1e-3*(r/r[0])**1.5, "k:", alpha=0.6, label="∝ r^1.5")
    axR.set_xlabel("Distance (µm)"); axR.set_ylabel("Time to Peak (s)")
    axR.set_title("Log–Log Scaling Analysis"); axR.legend()
    fig.tight_layout()
    _atomic_save(fig, project_root / "results/figures/notebook_replicas/distance_scaling_analysis.png")


def main():
    cfg = _load_cfg()
    fig_concentration_profiles(cfg)
    fig_distance_scaling(cfg)


if __name__ == "__main__":
    main()
