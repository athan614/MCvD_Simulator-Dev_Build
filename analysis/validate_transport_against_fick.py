#!/usr/bin/env python3
"""Validate diffusion solver against the closed-form Fick solution (k_clear = 0)."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import os
import matplotlib as mpl
if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")
import matplotlib.pyplot as plt
import yaml

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style
from src.mc_channel.transport import finite_burst_concentration
from src.constants import AVOGADRO

fig_dir = project_root / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


def _load_config() -> dict:
    cfg_path = project_root / "config" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg.setdefault("alpha", 0.2)
    cfg.setdefault("clearance_rate", 0.0)
    cfg.setdefault("T_release_ms", 1.0)
    cfg["burst_shape"] = "instant"
    cfg["clearance_rate"] = 0.0

    nts = cfg.setdefault("neurotransmitters", {})
    da = nts.setdefault("DA", {})
    da.setdefault("D_m2_s", 5.0e-10)
    da.setdefault("lambda", 1.6)

    return cfg


def _dirac_concentration(Nm: float, r_m: float, t_s: np.ndarray, D: float, lam: float, alpha: float) -> np.ndarray:
    tt = np.asarray(t_s, dtype=float)
    tt = np.clip(tt, 1e-9, None)
    deff = D / (lam ** 2)
    pref = Nm / (alpha * (4.0 * np.pi * deff * tt) ** 1.5)
    expo = np.exp(-(r_m ** 2) / (4.0 * deff * tt))
    conc = pref * expo
    return conc / (AVOGADRO * 1000.0)


def main() -> None:
    cfg = _load_config()
    apply_ieee_style()

    Nm = 1.0e4
    distances_um = [25.0, 50.0, 100.0]
    t = np.linspace(0.05, 60.0, 600)

    # Extract transport parameters for DA (representative)
    nt_params = cfg["neurotransmitters"]["DA"]
    D = float(nt_params["D_m2_s"])
    lam = float(nt_params["lambda"])
    alpha = float(cfg["alpha"])

    plt.figure(figsize=(4.2, 3.2))
    for dist_um in distances_um:
        r_m = dist_um * 1e-6
        sim = finite_burst_concentration(Nm, r_m, t, cfg, "DA")
        theory = _dirac_concentration(Nm, r_m, t, D, lam, alpha)
        mask = t > 0.1
        denom = np.maximum(np.abs(theory[mask]), 1e-18)
        mape = 100.0 * np.mean(np.abs(sim[mask] - theory[mask]) / denom)
        plt.plot(t, sim, label=f"{dist_um:.0f} um sim (MAPE {mape:.1f}%)")
        plt.plot(t, theory, linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (M)")
    plt.title("Instantaneous release: simulation vs. Fick solution")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = fig_dir / "fig_transport_fick_validation.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
