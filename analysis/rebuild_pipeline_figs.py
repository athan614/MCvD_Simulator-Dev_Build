# analysis/rebuild_pipeline_figs.py
"""
Notebook-replica end-to-end pipeline panel: concentrations → binding → OECT currents
with ISI enabled and decision distributions.

Output:
  - fixed_with_isi_v1_format.png  (results/figures/notebook_replicas/)
"""

from __future__ import annotations
import sys, os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde  # type: ignore
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style  # :contentReference[oaicite:7]{index=7}

# Model components
try:
    from src.mc_channel.transport import finite_burst_concentration
    from src.mc_receiver.binding import mean_binding
    from src.mc_receiver.oect import oect_current
    # We use the same helper used by the runner for consistent decision stats:
    from analysis.run_final_analysis import run_calibration_symbols  # for q-value sampling
except Exception:
    from src.mc_channel.transport import finite_burst_concentration
    from src.mc_receiver.binding import mean_binding
    from src.mc_receiver.oect import oect_current
    from analysis.run_final_analysis import run_calibration_symbols


def _load_cfg() -> dict:
    p = project_root / "config" / "default.yaml"
    with open(p, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # minimal sanity defaults
    cfg.setdefault("sim", {}).setdefault("dt_s", 0.01)
    cfg.setdefault("pipeline", {}).setdefault("symbol_period_s", 20.0)
    cfg["pipeline"].setdefault("modulation", "MoSK")
    cfg["pipeline"].setdefault("use_control_channel", True)
    cfg.setdefault("neurotransmitters", {"GLU": {"q_eff_e": -1.0}, "GABA": {"q_eff_e": 1.0}})
    return cfg


def _atomic_save(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_suffix(outpath.suffix + ".tmp")
    fig.savefig(tmp, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    os.replace(tmp, outpath)
    print(f"✓ Saved: {outpath}")


def main() -> None:
    apply_ieee_style()
    cfg = _load_cfg()
    dt = float(cfg["sim"]["dt_s"])
    Ts = float(cfg["pipeline"]["symbol_period_s"])
    t = np.arange(int(20.0/dt)) * dt
    rng = np.random.default_rng(7)

    # --- Build a simple ISI history (3 previous symbols spaced by Ts)
    Nm0 = float(cfg.get("pipeline", {}).get("Nm_per_symbol", 1e4))
    history: List[Tuple[int, float]] = [(1, Nm0*0.9), (0, Nm0*0.8), (1, Nm0*1.1)]  # (symbol, Nm)

    # Current symbol: GLU (0) vs GABA (1) for MoSK
    # Concentration at each channel (with/without ISI)
    def conc_at_channels(s_tx: int, Nm: float):
        conc_glu = finite_burst_concentration(Nm, 100e-6, t, cfg, "GLU")
        conc_gaba = finite_burst_concentration(Nm, 100e-6, t, cfg, "GABA")
        # ISI contribution: reuse the same finite burst with negative time shifts (−k*Ts)
        conc_glu_isi = conc_glu.copy()*0
        conc_gaba_isi = conc_gaba.copy()*0
        for k, (sym, Nm_hist) in enumerate(history, start=1):
            tau = k*Ts
            if sym == 0:
                conc_glu_isi += finite_burst_concentration(Nm_hist, 100e-6, t + tau, cfg, "GLU")
            else:
                conc_gaba_isi += finite_burst_concentration(Nm_hist, 100e-6, t + tau, cfg, "GABA")
        # Place current symbol on the right channel
        conc_glu_tot = conc_glu + (conc_glu_isi if s_tx == 0 else conc_glu_isi)
        conc_gaba_tot = conc_gaba + (conc_gaba_isi if s_tx == 1 else conc_gaba_isi)
        conc_ctrl = np.zeros_like(t)  # control has no signal
        return conc_glu_tot, conc_gaba_tot, conc_ctrl, conc_glu, conc_gaba

    Nm = Nm0
    cG_w, cB_w, cC_w, cG_no, cB_no = conc_at_channels(0, Nm)  # GLU symbol

    # --- Binding means (deterministic) for cleaner depiction
    bG_w = mean_binding(cG_w, "GLU", cfg)
    bB_w = mean_binding(cB_w, "GABA", cfg)
    bC_w = np.zeros_like(bG_w)

    bG_no = mean_binding(cG_no, "GLU", cfg)
    bB_no = mean_binding(cB_no, "GABA", cfg)

    # --- OECT currents (use 'signal' mapping only for clarity)
    iG_w = oect_current(bG_w, "GLU", cfg, seed=11)["signal"]
    iB_w = oect_current(bB_w, "GABA", cfg, seed=12)["signal"]
    iC_w = oect_current(bC_w, "CTRL", cfg, seed=13)["signal"]

    iG_no = oect_current(bG_no, "GLU", cfg, seed=21)["signal"]
    iB_no = oect_current(bB_no, "GABA", cfg, seed=22)["signal"]

    # --- Decision distributions (reuse runner helper for q-values across seeds)
    #     This ensures perfect consistency with “official” detection statistics.
    cal = run_calibration_symbols(cfg, symbol=0, mode="MoSK", num_symbols=300)
    q_glu = np.array(cal["q_values"]) if cal and "q_values" in cal else np.zeros(1)
    cal2 = run_calibration_symbols(cfg, symbol=1, mode="MoSK", num_symbols=300)
    q_gaba = np.array(cal2["q_values"]) if cal2 and "q_values" in cal2 else np.zeros(1)

    # --- Plot panel
    fig = plt.figure(figsize=(7.0, 9.4))
    gs = fig.add_gridspec(6, 1, hspace=0.5)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, cG_w, label="GLU at GLU-CH (with ISI)")
    ax1.plot(t, cB_w, label="GABA at GABA-CH (with ISI)")
    ax1.plot(t, cG_no, "--", alpha=0.5, label="GLU at GLU-CH (no ISI)")
    ax1.plot(t, cB_no, "--", alpha=0.5, label="GABA at GABA-CH (no ISI)")
    ax1.set_ylabel("Concentration [nM]"); ax1.set_title("Concentration with ISI")
    ax1.grid(True, ls="--", alpha=0.25); ax1.legend(fontsize=7, ncol=2)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, bG_w, label="GLU-CH (with ISI)")
    ax2.plot(t, bB_w, label="GABA-CH (with ISI)")
    ax2.plot(t, bG_no, "--", alpha=0.6, label="GLU-CH (no ISI)")
    ax2.plot(t, bB_no, "--", alpha=0.6, label="GABA-CH (no ISI)")
    ax2.set_ylabel("Bound sites"); ax2.set_title("Aptamer occupancy with ISI")
    ax2.grid(True, ls="--", alpha=0.25); ax2.legend(fontsize=7, ncol=2)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(t, iG_w*1e9, label="GLU-CH (with ISI)")
    ax3.plot(t, iB_w*1e9, label="GABA-CH (with ISI)")
    ax3.set_ylabel("Current [nA]"); ax3.set_title("OECT currents (signal paths)")
    ax3.grid(True, ls="--", alpha=0.25); ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(gs[3, 0])
    # Decision distributions via KDE
    if len(q_glu) > 5 and len(q_gaba) > 5:
        for vec, lab in [(q_glu, "GLU sent"), (q_gaba, "GABA sent")]:
            kde = gaussian_kde(vec)
            xs = np.linspace(np.quantile(vec, 0.01), np.quantile(vec, 0.99), 300)
            ax4.plot(xs, kde(xs), label=lab)
        ax4.set_title("Decision distributions (Monte‑Carlo KDE)")
        ax4.set_xlabel("Decision statistic Q"); ax4.set_ylabel("Probability density")
        ax4.grid(True, ls="--", alpha=0.25); ax4.legend(fontsize=7)
    else:
        ax4.text(0.5, 0.5, "Not enough q-values", transform=ax4.transAxes, ha="center")

    ax5 = fig.add_subplot(gs[4, 0])
    ax5.plot(t, (iG_w - iC_w)*1e9, label="GLU-CH − CTRL (with ISI)")
    ax5.plot(t, (iB_w - iC_w)*1e9, label="GABA-CH − CTRL (with ISI)")
    ax5.plot(t, (iG_no - 0)*1e9, "--", alpha=0.6, label="GLU-CH (no ISI)")
    ax5.plot(t, (iB_no - 0)*1e9, "--", alpha=0.6, label="GABA-CH (no ISI)")
    ax5.set_ylabel("Differential current [nA]")
    ax5.set_title("Differential measurement: ISI vs no‑ISI")
    ax5.grid(True, ls="--", alpha=0.25); ax5.legend(fontsize=7, ncol=2)

    ax6 = fig.add_subplot(gs[5, 0])
    ax6.plot(t, (iG_w*1e9), label="GLU-CH raw (GLU sent)")
    ax6.plot(t, ((iG_w - iC_w)*1e9), label="Differential (reduced noise)")
    ax6.set_ylabel("Current [nA]"); ax6.set_xlabel("Time [s]")
    ax6.set_title("Noise Reduction via Differential Measurement")
    ax6.grid(True, ls="--", alpha=0.25); ax6.legend(fontsize=7)

    fig.tight_layout()
    _atomic_save(fig, project_root / "results/figures/notebook_replicas/fixed_with_isi_v1_format.png")


if __name__ == "__main__":
    main()
