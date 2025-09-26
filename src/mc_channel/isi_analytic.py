"""Analytical ISI helpers for the passive diffusion channel."""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .transport import greens_function_3d_vectorized

__all__ = [
    "window_coefficients",
    "gaussian_ser_binary",
    "predicted_stats_mosk",
]


def window_coefficients(
    distance_m: float,
    Ts: float,
    win_s: float,
    cfg: Dict[str, Any],
    k_max: int = 200,
) -> np.ndarray:
    """Return windowed ISI coefficients h_k = integral of g(t) over each decision window."""
    sim_cfg = cfg.get("sim", {})
    dt = float(sim_cfg.get("dt_s", 0.01))
    if dt <= 0:
        raise ValueError("dt_s must be positive")

    nt_cfg = cfg.get("neurotransmitters", {}).get("DA", {})
    D = float(nt_cfg.get("D_m2_s", 4.9e-10))
    lam = float(nt_cfg.get("lambda", 1.6))
    alpha = float(cfg.get("alpha", 0.2))
    k_clear = float(cfg.get("clearance_rate", 0.1))

    # Local time vector inside each decision window
    t_local = np.arange(0.0, max(win_s, dt), dt)
    if t_local.size == 0:
        t_local = np.array([0.0])

    h_vals = []
    max_ref = 0.0

    for k in range(int(max(0, k_max)) + 1):
        t0 = k * Ts + (Ts - win_s)
        t_abs = t0 + t_local
        g_vals = greens_function_3d_vectorized(distance_m, t_abs, D, lam, alpha, k_clear)
        area = float(np.trapz(g_vals, dx=dt))
        h_vals.append(area)
        max_ref = max(max_ref, abs(area))
        if k > 5 and max_ref > 0 and abs(area) < 1e-6 * max_ref:
            break

    return np.asarray(h_vals, dtype=float)


def gaussian_ser_binary(mu0: float, mu1: float, sigma0: float, sigma1: float) -> float:
    """Approximate SER for two Gaussian hypotheses with unequal variances."""
    var = max(0.5 * (sigma0 ** 2 + sigma1 ** 2), 1e-18)
    arg = abs(mu1 - mu0) / (math.sqrt(2.0 * var))
    return 0.5 * math.erfc(arg)


def predicted_stats_mosk(Nm: float, h: np.ndarray, cfg: Dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (mu_DA, mu_SERO, var_DA, var_SERO) for the MoSK contrast statistic."""
    oect = cfg.get("oect", {})
    gm = float(oect.get("gm_S", 0.001))
    C_tot = float(oect.get("C_tot_F", 1e-7))
    gain = gm / C_tot if C_tot else gm

    nt_cfg = cfg.get("neurotransmitters", {})
    q_da = float(nt_cfg.get("DA", {}).get("q_eff_e", -0.35))
    q_sero = float(nt_cfg.get("SERO", {}).get("q_eff_e", 0.35))
    own = Nm * (h[0] if h.size > 0 else 0.0)

    mu_da = gain * q_da * own
    mu_sero = gain * q_sero * own
    return mu_da, mu_sero, 0.0, 0.0
