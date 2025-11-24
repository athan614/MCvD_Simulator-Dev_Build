# analysis/rebuild_binding_figs.py
"""
Notebook-replica binding figures using the real stochastic binder + analytic PSD.

Outputs (results/figures/notebook_replicas/):
  - binding_mean_vs_mc.png
  - binding_psd_vs_analytic.png
"""

from __future__ import annotations
import sys, os
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch  # type: ignore
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style  # :contentReference[oaicite:5]{index=5}
try:
    from src.mc_receiver import binding as binding_mod
    from src.mc_channel import transport as transport_mod
    from src.mc_receiver import oect as oect_mod
except Exception:
    import src.mc_receiver.binding as binding_mod
    import src.mc_channel.transport as transport_mod
    import src.mc_receiver.oect as oect_mod


def _load_cfg() -> dict:
    cfg: dict[str, Any] = {}
    p = project_root / "config" / "default.yaml"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    cfg.setdefault("sim", {}).setdefault("dt_s", 0.01)
    cfg.setdefault("pipeline", {}).setdefault("symbol_period_s", 20.0)
    cfg.setdefault("oect", {})
    for k, v in oect_mod.default_params().items():
        cfg["oect"].setdefault(k, v)
    # Minimum neurotransmitter params for DA
    cfg.setdefault("neurotransmitters", {}).setdefault("DA", {
        "k_on_M_s": 1e5, "k_off_s": 0.02, "q_eff_e": -1.0
    })
    cfg.setdefault("N_apt", 2e8)
    return cfg


def _atomic_save(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_suffix(outpath.suffix + ".tmp")
    fig.savefig(tmp, dpi=300, bbox_inches="tight", pad_inches=0.02, format="png")
    plt.close(fig)
    os.replace(tmp, outpath)
    print(f"✓ Saved: {outpath}")


def fig_mean_vs_mc(cfg: dict):
    apply_ieee_style()
    dt = float(cfg["sim"]["dt_s"])
    Ts = float(cfg["pipeline"]["symbol_period_s"])
    t = np.arange(int(10.0 / dt)) * dt  # 10 s window

    # Use a realistic concentration transient at 100 µm:
    conc = transport_mod.finite_burst_concentration(
        Nm=1e4, r=100e-6, t_vec=t, config=cfg, nt_type="DA"
    )

    # Deterministic mean
    mean_b = binding_mod.mean_binding(conc, "DA", cfg)
    mu = float(np.mean(mean_b))

    # Three MC traces
    traces = []
    for seed in (101, 202, 303):
        b_t, _, _ = binding_mod.bernoulli_binding(conc, "DA", cfg, np.random.default_rng(seed))
        traces.append(b_t)

    # Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t, mean_b, "k-", lw=2.0, label=f"Deterministic mean (μ≈{mu:.3e})")
    for i, tr in enumerate(traces, 1):
        ax.plot(t, tr, lw=1.2, label=f"MC trace {i}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bound sites")
    ax.set_title("Binding Kinetics: Deterministic vs Monte Carlo")
    ax.grid(True, ls="--", alpha=0.25)
    ax.legend(loc="best", ncol=2)
    _atomic_save(fig, project_root / "results/figures/notebook_replicas/binding_mean_vs_mc.png")


def fig_psd_empirical_vs_analytic(cfg: dict):
    apply_ieee_style()
    dt = float(cfg["sim"]["dt_s"])
    fs = 1.0 / dt
    T = 200.0
    t = np.arange(int(T / dt)) * dt

    # Stationary binding around equilibrium concentration (nM)
    C_eq = 10e-9
    conc = np.full_like(t, C_eq, dtype=float)

    # MC binding → current via OECT gain (use *signal* path only)
    b_t, _, _ = binding_mod.bernoulli_binding(conc, "DA", cfg, np.random.default_rng(404))
    gm = float(cfg["oect"]["gm_S"]); Ctot = float(cfg["oect"]["C_tot_F"])
    q_eff = float(cfg["neurotransmitters"]["DA"]["q_eff_e"])
    e = 1.602176634e-19
    i_sig = gm * q_eff * e * b_t / Ctot
    i_sig = i_sig - np.mean(i_sig)  # remove DC

    # Empirical PSD
    f_emp, p_emp = welch(i_sig, fs=fs, nperseg=min(len(i_sig), 16384), detrend="constant")

    # Analytic binding PSD in A^2/Hz (provided by binding module)
    f_th = np.logspace(np.log10(max(1e-2, f_emp[1])), np.log10(fs/2), 500)
    p_th = binding_mod.binding_noise_psd("DA", cfg, f_th, C_eq=C_eq)

    # Corner frequency estimate for label
    k_on = float(cfg["neurotransmitters"]["DA"]["k_on_M_s"])
    k_off = float(cfg["neurotransmitters"]["DA"]["k_off_s"])
    f_c = (k_on * C_eq + k_off) / (2*np.pi)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.loglog(f_emp, p_emp, lw=1.8, label="Welch estimate")
    ax.loglog(f_th, p_th, "k--", lw=2.0, label="Analytical Lorentzian")
    ax.axvline(f_c, color="0.5", ls=":", lw=1.2)
    ax.text(f_c*1.05, p_emp.max()*0.5, f"$f_c$ ≈ {f_c:.2f} Hz", fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (A²/Hz)")
    ax.set_title("Binding Noise PSD: Empirical vs Analytical")
    ax.grid(True, which="both", ls="--", alpha=0.25)
    ax.legend(loc="best")
    _atomic_save(fig, project_root / "results/figures/notebook_replicas/binding_psd_vs_analytic.png")


def main():
    cfg = _load_cfg()
    fig_mean_vs_mc(cfg)
    fig_psd_empirical_vs_analytic(cfg)


if __name__ == "__main__":
    main()
