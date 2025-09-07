# src/pipeline.py - BOTH FIXES APPLIED + ISI metrics plumbing
"""
End-to-end simulator for the tri-channel OECT receiver.

COMPLETE FIX VERSION:
1. Uniform CSK level mapping to avoid zero molecules
2. Correct Poisson noise application at the level, not scaling factor
3. (New) ISI metric collection hook (no physics changes)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from numpy.random import default_rng
import math
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# Import vectorized transport functions including batch time
from .mc_channel.transport import finite_burst_concentration, finite_burst_concentration_batch, finite_burst_concentration_batch_time
from .mc_receiver.binding import bernoulli_binding
from .mc_receiver.oect import oect_trio
from .constants import get_nt_params

logger = logging.getLogger(__name__)

def to_int(value):
    """Convert value to int, handling scientific notation strings"""
    if isinstance(value, str):
        return int(float(value))
    elif isinstance(value, float):
        return int(value)
    else:
        return value

def _resolve_decision_window(cfg: Dict[str, Any], Ts: float, dt: float) -> float:
    """
    Resolve decision window based on policy: 'fixed', 'fraction_of_Ts', or 'full_Ts'.
    Default is 'fixed' for backward compatibility.
    """
    det = cfg.get('detection', {})
    policy = str(det.get('decision_window_policy', 'fixed')).lower()
    
    if policy in ('full_ts', 'full', 'ts'):
        win_s = Ts
    elif policy in ('fraction_of_ts', 'fraction', 'frac'):
        frac = float(det.get('decision_window_fraction', 0.9))
        frac = min(max(frac, 0.1), 1.0)  # clamp to 10–100%
        win_s = frac * Ts
    else:  # 'fixed' (legacy)
        win_s = float(det.get('decision_window_s', Ts))

    # Guardrails: at least N samples; never exceed Ts
    min_samples = int(det.get('min_decision_samples', 16))
    win_s = max(min_samples * dt, min(win_s, Ts))
    return win_s

def _csk_dual_channel_Q(
    q_da: float, q_sero: float,
    sigma_da: float, sigma_sero: float,
    rho_cc: float,
    combiner: str = "zscore",
    leakage_frac: float = 0.0,
    target: str = "DA"
) -> float:
    """
    Compute dual-channel CSK decision statistic Q_comb = w_t Q_t + w_o Q_o.
    
    Args:
        q_da, q_sero: Single-channel charge statistics
        sigma_da, sigma_sero: Noise standard deviations
        rho_cc: Cross-channel correlation coefficient after CTRL
        combiner: "zscore", "whitened", or "leakage"
        leakage_frac: For leakage combiner, fraction of signal on 'other' channel
        target: "DA" or "SERO" - which channel carries the primary signal
    """
    # Nominal means direction (amplitude-only lives on 'target' axis)
    # μ = [μ_t, μ_o]; for pure CSK (no leakage): μ = [μ_t, 0]; with leakage: μ_o = leakage_frac * μ_t
    # Work in "signed" charges (you already use signed q via q_eff in oect path)
    Qt, Qo = (q_da, q_sero) if target == "DA" else (q_sero, q_da)
    sg_t, sg_o = (sigma_da, sigma_sero) if target == "DA" else (sigma_sero, sigma_da)
    rho = float(np.clip(rho_cc, -0.999, 0.999))

    if combiner == "zscore":
        # Numerically well-conditioned, scale-free:
        # Q = (Qt/σt) − ρ * (Qo/σo)
        return (Qt / max(sg_t, 1e-30)) - rho * (Qo / max(sg_o, 1e-30))

    elif combiner == "whitened":
        # Fisher LDA for ordered classes; w ∝ Σ^{-1} μ, Σ = [[σt^2, ρσtσo],[ρσtσo, σo^2]]
        # For pure CSK (μ=[μt,0]), up to scale: w ∝ [1/σt^2, -ρ/(σtσo)].
        # Implement via the scale-free z-score equivalent for stability:
        return (Qt / max(sg_t, 1e-30)) - rho * (Qo / max(sg_o, 1e-30))

    elif combiner == "leakage":
        # General Σ^{-1} μ with μ = [μt, μo] and μo = leakage_frac * μt
        # Solve w = Σ^{-1} μ up to a common scale (the sign/ordering is what matters).
        st2, so2 = sg_t*sg_t, sg_o*sg_o
        s_to = rho * sg_t * sg_o
        # Inverse of 2x2 Σ
        det = max(st2*so2 - s_to*s_to, 1e-60)
        inv00 =  so2 / det
        inv01 = -s_to / det
        inv10 = -s_to / det
        inv11 =  st2 / det
        mu_t = 1.0
        mu_o = float(leakage_frac)
        w_t = inv00 * mu_t + inv01 * mu_o
        w_o = inv10 * mu_t + inv11 * mu_o
        return w_t * Qt + w_o * Qo

    else:
        # Fallback: legacy single-channel behavior
        return Qt

def calculate_proper_noise_sigma(cfg: Dict[str, Any], detection_window_s: float, symbol_index: int = -1) -> Tuple[float, float]:
    """Calculate physics-based noise sigma (robust implementation)"""
    k_B = 1.38e-23
    
    # Robustly access configuration parameters with config override support
    T = cfg.get('sim', {}).get('temperature_K', 310)
    
    oect_cfg = cfg.get('oect', {})
    R_ch = oect_cfg.get('R_ch_Ohm', 100)
    gm = oect_cfg.get('gm_S', 0.005)
    V_g_bias = oect_cfg.get('V_g_bias_V', -0.02)

    I_dc = gm * abs(V_g_bias)
    I_dc = max(I_dc, 1e-6)

    noise_cfg = cfg.get('noise', {})
    alpha_H = noise_cfg.get('alpha_H', 1e-3)
    N_c = noise_cfg.get('N_c', 4e12)
    rho_corr = noise_cfg.get('rho_correlated', 0.9)
    K_d = 0.0

    dt = cfg['sim']['dt_s']
    f_samp = 1.0 / dt
    # Enhanced: Allow config override of detection bandwidth
    B_det = cfg.get('detection_bandwidth_Hz', noise_cfg.get('detection_bandwidth_Hz', 100))
    T_int = detection_window_s
    f_min = 1.0 / T_int
    f_max = min(B_det, f_samp/2)
    
    effective_B = 0.25 / T_int
    
    psd_johnson = 4 * k_B * T / R_ch
    johnson_charge_var = psd_johnson * T_int
    
    K_f = alpha_H / N_c
    flicker_charge_var = K_f * I_dc**2 * max(0.0, np.log(2 * np.pi * T_int)) * T_int if f_max > f_min else 0
    
    drift_charge_var = 0
    
    total_single_var = johnson_charge_var + flicker_charge_var + drift_charge_var
    
    # Enhanced: Mode-aware noise calculation
    use_ctrl_flag = bool(cfg.get('pipeline', {}).get('use_control_channel', True))
    mod = str(cfg.get('pipeline', {}).get('modulation', '')).upper()
    effective_rho = float(np.clip(noise_cfg.get('effective_correlation', rho_corr), -0.999, 0.999))  # Allow override with clamp

    # For MoSK we never subtract CTRL from the charges, so noise is single-ended even if CTRL is "enabled".
    use_ctrl_for_noise = (use_ctrl_flag and mod != 'MOSK')

    if use_ctrl_for_noise:
        # Differential measurement: benefits from common-mode rejection
        differential_var = 2 * total_single_var * (1 - effective_rho)
        sigma = np.sqrt(differential_var)
    else:
        # Single-ended measurement: no common-mode rejection
        sigma = np.sqrt(total_single_var)
    
    # Increased floor slightly for robustness
    sigma = max(sigma, 1e-15)
    
    return sigma, sigma


def _calculate_isi_vectorized(
    tx_history: List[Tuple[int, float]], 
    t_vec: np.ndarray,
    cfg: Dict[str, Any],
    distance_da_m: float,
    distance_sero_m: float,
    distance_ctrl_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FULLY VECTORIZED: Calculate ISI contributions from transmission history.
    """
    if not tx_history:
        return np.zeros_like(t_vec), np.zeros_like(t_vec), np.zeros_like(t_vec)
    
    mod = cfg['pipeline']['modulation']
    Ts = cfg['pipeline']['symbol_period_s']
    
    # Convert history to arrays for vectorization
    past_symbols = np.array([h[0] for h in tx_history], dtype=np.int32)
    Nm_history = np.array([h[1] for h in tx_history], dtype=np.float64)
    n_history = len(tx_history)
    
    # Pre-allocate concentration arrays
    conc_at_da_ch = np.zeros_like(t_vec)
    conc_at_sero_ch = np.zeros_like(t_vec)
    conc_at_ctrl_ch = np.zeros_like(t_vec)  # Always zero (no signal contamination)
    
    # Calculate all time offsets at once
    k_indices = np.arange(n_history)
    time_offsets = (k_indices + 1) * Ts
    
    if mod == "MoSK":
        # Separate DA and SERO transmissions
        da_mask = (past_symbols == 0)
        sero_mask = ~da_mask
        
        # Process DA contributions in batch
        if np.any(da_mask):
            da_Nm = Nm_history[da_mask]
            da_offsets = time_offsets[da_mask]
            da_residuals = finite_burst_concentration_batch_time(
                da_Nm, distance_da_m, t_vec, da_offsets, cfg, 'DA'
            )
            conc_at_da_ch = np.sum(da_residuals, axis=0)
        
        # Process SERO contributions in batch
        if np.any(sero_mask):
            sero_Nm = Nm_history[sero_mask]
            sero_offsets = time_offsets[sero_mask]
            sero_residuals = finite_burst_concentration_batch_time(
                sero_Nm, distance_sero_m, t_vec, sero_offsets, cfg, 'SERO'
            )
            conc_at_sero_ch = np.sum(sero_residuals, axis=0)
                
    elif mod.startswith("CSK"):
        # ISI uses actual transmitted Nm values from history
        target_channel = cfg['pipeline'].get('csk_target_channel', 'DA')
        
        # Process non-zero contributions
        nonzero_mask = Nm_history > 0
        if np.any(nonzero_mask):
            active_Nm = Nm_history[nonzero_mask]
            active_offsets = time_offsets[nonzero_mask]
            
            if target_channel == 'DA':
                residuals = finite_burst_concentration_batch_time(
                    active_Nm, distance_da_m, t_vec, active_offsets, cfg, 'DA'
                )
                conc_at_da_ch = np.sum(residuals, axis=0)
            else:  # SERO
                residuals = finite_burst_concentration_batch_time(
                    active_Nm, distance_sero_m, t_vec, active_offsets, cfg, 'SERO'
                )
                conc_at_sero_ch = np.sum(residuals, axis=0)
                    
    elif mod == "Hybrid":
        # Extract molecule type from history
        mol_type_bits = past_symbols >> 1
        
        # Process DA contributions
        da_mask = (mol_type_bits == 0) & (Nm_history > 0)
        if np.any(da_mask):
            da_amps = Nm_history[da_mask]
            da_offsets = time_offsets[da_mask]
            da_residuals = finite_burst_concentration_batch_time(
                da_amps, distance_da_m, t_vec, da_offsets, cfg, 'DA'
            )
            conc_at_da_ch = np.sum(da_residuals, axis=0)
        
        # Process SERO contributions
        sero_mask = (mol_type_bits == 1) & (Nm_history > 0)
        if np.any(sero_mask):
            sero_amps = Nm_history[sero_mask]
            sero_offsets = time_offsets[sero_mask]
            sero_residuals = finite_burst_concentration_batch_time(
                sero_amps, distance_sero_m, t_vec, sero_offsets, cfg, 'SERO'
            )
            conc_at_sero_ch = np.sum(sero_residuals, axis=0)
    
    return conc_at_da_ch, conc_at_sero_ch, conc_at_ctrl_ch


def _single_symbol_currents(s_tx: int, tx_history: List[Tuple[int, float]], cfg: Dict[str, Any], rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate DA, SERO, CTRL drain-current traces for one symbol interval.
    COMPLETE FIX: 
    1. Uniform CSK level mapping to avoid zero molecules
    2. Correct Poisson noise application at the level, not scaling factor
    3. (New) If cfg['collect_isi_metrics'] is True, collect ISI-to-signal ratio proxy
    """
    # Initialization
    dt = cfg['sim']['dt_s']
    Ts = cfg['pipeline']['symbol_period_s']
    n_samples = int(Ts / dt)
    t_vec = np.arange(n_samples) * dt
    
    # Initialize concentration at each channel location
    conc_at_da_ch = np.zeros(n_samples)
    conc_at_sero_ch = np.zeros(n_samples)
    conc_at_ctrl_ch = np.zeros(n_samples)  # Pure noise reference
    
    # Keep separate holders for metrics
    isi_da = np.zeros(n_samples)
    isi_sero = np.zeros(n_samples)
    sym_da = np.zeros(n_samples)
    sym_sero = np.zeros(n_samples)
    
    # Get channel-specific distances
    if 'channel_distances' in cfg:
        distance_da_m = cfg['channel_distances']['DA'] * 1e-6
        distance_sero_m = cfg['channel_distances']['SERO'] * 1e-6
        distance_ctrl_m = cfg['channel_distances']['CTRL'] * 1e-6
    else:
        distance_m = cfg['pipeline']['distance_um'] * 1e-6
        distance_da_m = distance_m
        distance_sero_m = distance_m
        distance_ctrl_m = distance_m
    
    mod = cfg['pipeline']['modulation']
    level_scheme = cfg['pipeline'].get('csk_level_scheme', 'uniform')
    
    # ========== CRITICAL FIX: Correct Energy Fairness and Noise Application ==========
    
    # Step 1: Calculate DETERMINISTIC Nm_peak (scaling factor for energy fairness)
    if mod == "Hybrid":
        levels = [0.5, 1.0]
        mean_amp = float(np.mean(levels))
        Nm_peak = cfg['pipeline']['Nm_per_symbol'] / mean_amp
    elif mod.startswith("CSK"):
        M = cfg['pipeline']['csk_levels']
        if level_scheme == 'uniform':
            # Average of [1/M, 2/M, ..., M/M]
            mean_amp = float((M + 1) / (2 * M))
        else:  # zero-based
            if M > 1:
                mean_amp = float(np.mean(np.arange(M) / (M - 1)))
            else:
                mean_amp = 0.5
        Nm_peak = cfg['pipeline']['Nm_per_symbol'] / mean_amp
    else:  # MoSK
        Nm_peak = cfg['pipeline']['Nm_per_symbol']
    
    Nm_peak = float(Nm_peak)  # Keep deterministic - NO NOISE YET!
    
    # Step 2: Calculate DETERMINISTIC Nm_level based on transmitted symbol
    if mod == "MoSK":
        # MoSK always uses full Nm_peak
        Nm_level = Nm_peak
        
    elif mod.startswith("CSK"):
        M = cfg['pipeline']['csk_levels']
        if level_scheme == 'uniform':
            # FIX 1: Map symbols 0,1,2,3 to fractions 1/M, 2/M, 3/M, M/M
            Nm_level = Nm_peak * ((s_tx + 1) / M)
        else:  # zero-based
            if M > 1:
                Nm_level = Nm_peak * (s_tx / (M - 1))
            else:
                Nm_level = Nm_peak if s_tx == 0 else 0
                
    elif mod == "Hybrid":
        mol_type_bit = s_tx >> 1
        amp_bit = s_tx & 1
        Nm_level = Nm_peak if amp_bit == 1 else Nm_peak * 0.5
        
    else:
        Nm_level = Nm_peak
    
    # Step 3: Apply Poisson noise to the SPECIFIC LEVEL (not the scaling factor!)
    if cfg['pipeline'].get('enable_molecular_noise', True) and Nm_level > 0:
        Nm_actual = float(rng.poisson(Nm_level))
    else:
        Nm_actual = float(Nm_level)
    
    # VECTORIZED ISI Calculation
    if cfg['pipeline'].get('enable_isi', False):
        if 'isi_memory_symbols' not in cfg['pipeline']:
            Ts = cfg['pipeline']['symbol_period_s']
            
            # Diffusion memory
            D_da = cfg['neurotransmitters']['DA']['D_m2_s']
            D_sero = cfg['neurotransmitters']['SERO']['D_m2_s']
            D = (D_da + D_sero) / 2
            avg_distance_m = (distance_da_m + distance_sero_m) / 2
            decay95_diff = 3.0 * avg_distance_m**2 / D
            
            # Binding memory (NEW - missing component)
            k_off_da = cfg['neurotransmitters']['DA']['k_off_s']
            k_off_sero = cfg['neurotransmitters']['SERO']['k_off_s']
            k_off_avg = (k_off_da + k_off_sero) / 2
            decay95_binding = 5.0 / k_off_avg  # 5 time constants for 99% decay
            
            # Total ISI decay time (max of both mechanisms)
            decay95_total = max(decay95_diff, decay95_binding)
            
            guard_factor = cfg['pipeline'].get('guard_factor', 0.3)
            isi_memory = math.ceil((1 + guard_factor) * decay95_total / Ts)
            
            # Apply soft cap to prevent Ts explosion during LoD sweeps
            cap = int(cfg['pipeline'].get('isi_memory_cap_symbols', 60))
            if cap > 0:
                isi_memory = min(isi_memory, cap)
        else:
            isi_memory = cfg['pipeline']['isi_memory_symbols']
        
        relevant_history = tx_history[-isi_memory:] if len(tx_history) >= isi_memory else tx_history
        
        # Use vectorized ISI calculation
        if relevant_history:
            isi_da, isi_sero, _ = _calculate_isi_vectorized(
                relevant_history, t_vec, cfg, 
                distance_da_m, distance_sero_m, distance_ctrl_m
            )
            conc_at_da_ch += isi_da
            conc_at_sero_ch += isi_sero
    
    # Current Symbol Generation using Nm_actual (the noisy realization)
    if mod == "MoSK":
        if s_tx == 0:  # DA
            conc_da = finite_burst_concentration(Nm_actual, distance_da_m, t_vec, cfg, 'DA')
            conc_at_da_ch += conc_da
            sym_da += conc_da
        else:  # SERO
            conc_sero = finite_burst_concentration(Nm_actual, distance_sero_m, t_vec, cfg, 'SERO')
            conc_at_sero_ch += conc_sero
            sym_sero += conc_sero
            
    elif mod.startswith("CSK"):
        target_channel = cfg['pipeline'].get('csk_target_channel', 'DA')
        if Nm_actual > 0:
            if target_channel == 'DA':
                conc_da = finite_burst_concentration(Nm_actual, distance_da_m, t_vec, cfg, 'DA')
                conc_at_da_ch += conc_da
                sym_da += conc_da
            else:  # SERO
                conc_sero = finite_burst_concentration(Nm_actual, distance_sero_m, t_vec, cfg, 'SERO')
                conc_at_sero_ch += conc_sero
                sym_sero += conc_sero
            
    elif mod == "Hybrid":
        mol_type_bit = s_tx >> 1
        if mol_type_bit == 0:  # DA
            conc_da = finite_burst_concentration(Nm_actual, distance_da_m, t_vec, cfg, 'DA')
            conc_at_da_ch += conc_da
            sym_da += conc_da
        else:  # SERO
            conc_sero = finite_burst_concentration(Nm_actual, distance_sero_m, t_vec, cfg, 'SERO')
            conc_at_sero_ch += conc_sero
            sym_sero += conc_sero
    
    # INDEPENDENT Binding simulation for each channel
    cfg_da = cfg.copy()
    cfg_da['N_apt'] = cfg.get('binding', {}).get('N_sites_da', cfg.get('N_apt'))
    bound_da_ch, _, _ = bernoulli_binding(conc_at_da_ch, 'DA', cfg_da, rng)
    
    cfg_sero = cfg.copy()
    cfg_sero['N_apt'] = cfg.get('binding', {}).get('N_sites_sero', cfg.get('N_apt'))
    bound_sero_ch, _, _ = bernoulli_binding(conc_at_sero_ch, 'SERO', cfg_sero, rng)
    
    cfg_ctrl = cfg.copy()
    cfg_ctrl['N_apt'] = 0  # No aptamer sites
    bound_ctrl_ch, _, _ = bernoulli_binding(conc_at_ctrl_ch, 'CTRL', cfg_ctrl, rng)
    
    # OECT current generation
    bound_sites_trio = np.vstack([bound_da_ch, bound_sero_ch, bound_ctrl_ch])
    currents = oect_trio(
        bound_sites_trio,
        nts=("DA", "SERO", "CTRL"),
        cfg=cfg,
        rng=rng
    )

    # === ISI METRIC (proxy) ==========================================
    # If requested, push an ISI-to-signal "energy" ratio proxy into cfg['_metrics'].
    # We compute it in concentration domain (integration over decision window);
    # this avoids any physics change and is sufficient for reporting/gating.
    if cfg.get('collect_isi_metrics', False):
        try:
            dt = cfg['sim']['dt_s']
            win_s = cfg['detection'].get('decision_window_s', cfg['pipeline']['symbol_period_s'])
            n_int = min(int(win_s / dt), n_samples)
            isi_mass = float(np.trapezoid(isi_da[:n_int], dx=dt) + np.trapezoid(isi_sero[:n_int], dx=dt))
            sym_mass = float(np.trapezoid(sym_da[:n_int], dx=dt) + np.trapezoid(sym_sero[:n_int], dx=dt))
            denom = max(sym_mass, 1e-30)
            ratio = isi_mass / denom
            metrics = cfg.setdefault('_metrics', {})
            lst = metrics.setdefault('isi_ratio_values', [])
            lst.append(ratio)
        except Exception:
            pass
    # ================================================================

    return currents["DA"], currents["SERO"], currents["CTRL"], Nm_actual


def run_sequence(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate <sequence_length> symbols and compute BER/SEP.
    Detection logic unchanged - the fixes are in signal generation.
    """
    
    # Guard against missing detection config
    cfg.setdefault('detection', {})
        
    mod = cfg['pipeline']['modulation']
    L = cfg['pipeline']['sequence_length']
    rng = default_rng(cfg['pipeline'].get('random_seed'))
    
    tx_history: List[Tuple[int, float]] = []
    subsymbol_errors = {'mosk': 0, 'csk': 0}
    thresholds_used = {}
    
    # Generate transmitted symbols
    if mod == 'MoSK':
        tx_symbols = rng.integers(0, 2, size=L)
    elif mod.startswith('CSK'):
        M = cfg['pipeline']['csk_levels']
        tx_symbols = rng.integers(0, M, size=L)
    elif mod == 'Hybrid':
        tx_symbols = rng.integers(0, 4, size=L)
    else:
        raise ValueError(f"Unknown modulation: {mod}")
    
    rx_symbols = np.zeros(L, dtype=int)
    stats_da = []
    stats_sero = []

    # enable metric collection for this run
    cfg['collect_isi_metrics'] = True
    metrics_ref = cfg.setdefault('_metrics', {})
    metrics_ref.setdefault('isi_ratio_values', [])
    
    # Get polarities
    q_eff_da = get_nt_params(cfg, 'DA')['q_eff_e']
    q_eff_sero = get_nt_params(cfg, 'SERO')['q_eff_e']

    # Stage 14: Compute noise sigma ONCE before the loop
    detection_window_s = _resolve_decision_window(
        cfg,
        cfg['pipeline']['symbol_period_s'],
        cfg['sim']['dt_s']
    )
    sigma_da, sigma_sero = calculate_proper_noise_sigma(cfg, detection_window_s)

    # ✨ NEW: also compute single-ended noise for MoSK statistic (even when mode='Hybrid')
    from copy import deepcopy
    cfg_mosk = deepcopy(cfg)
    cfg_mosk['pipeline']['modulation'] = 'MoSK'  # force single-ended sigmas
    sigma_da_mosk, sigma_sero_mosk = calculate_proper_noise_sigma(cfg_mosk, detection_window_s)

    # Enhanced: Mode-aware correlation selection
    noise_cfg = cfg.get('noise', {})
    use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))

    # Correlation to use in the *denominator*:
    # - MoSK: use pre-CTRL correlation (matches calibration)
    # - CSK/Hybrid: use post-CTRL correlation when CTRL subtraction is on; otherwise pre-CTRL
    rho_pre   = float(noise_cfg.get('rho_corr', noise_cfg.get('rho_correlated', 0.9)))
    rho_post  = float(noise_cfg.get('rho_between_channels_after_ctrl', 0.0))
    if mod == 'MoSK':
        rho_for_diff = rho_pre
    else:
        rho_for_diff = rho_post if use_ctrl else rho_pre

    rho_for_diff = max(-1.0, min(1.0, rho_for_diff))

    var_diff = (sigma_da * sigma_da +
                sigma_sero * sigma_sero -
                2.0 * rho_for_diff * sigma_da * sigma_sero)
    sigma_diff = float(math.sqrt(max(var_diff, 0.0)))

    # Pre-CTRL correlation for MoSK denominator
    sigma_diff_mosk = float(math.sqrt(max(
        sigma_da_mosk*sigma_da_mosk + sigma_sero_mosk*sigma_sero_mosk
        - 2.0 * rho_pre * sigma_da_mosk * sigma_sero_mosk,
        0.0
    )))
    
    # Keep rho_cc for CSK dual-channel combiner (preserve existing behavior)
    rho_cc = rho_post
    rho_cc = max(-1.0, min(1.0, rho_cc))

    # Process symbols
    for i, s_tx in enumerate(tqdm(tx_symbols, desc=f"Simulating {mod}", disable=cfg.get('disable_progress', False))):
        ig, ia, ic, Nm_realised = _single_symbol_currents(s_tx, tx_history, cfg, rng)
        tx_history.append((s_tx, Nm_realised))
        
        # Detection and decision statistics
        dt = cfg['sim']['dt_s']
        # detection_window_s already computed above
    
        # NOTE: sigma_da, sigma_sero, sigma_diff already computed once above
        
        n_total_samples = len(ig)
        n_detect_samples = min(int(detection_window_s / dt), n_total_samples)

        if n_detect_samples <= 1:
            if i == 0: print("Warning: Insufficient detection samples. Check dt_s and symbol_period_s.")
            # Defensive: set rx_symbols[i] to transmitted symbol to avoid false error
            # This should be rare due to _enforce_min_window, but provides safety
            rx_symbols[i] = s_tx
            continue
        
        use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
        # For MoSK, do NOT subtract CTRL from the charges; for CSK/Hybrid keep the existing behavior.
        subtract_for_q = (mod != "MoSK") and use_ctrl
        
        # Calculate charges (mode-specific CTRL handling)
        sig_da = (ig - ic) if subtract_for_q else ig
        sig_sero = (ia - ic) if subtract_for_q else ia
        q_da = float(np.trapezoid(sig_da[:n_detect_samples], dx=dt))
        q_sero = float(np.trapezoid(sig_sero[:n_detect_samples], dx=dt))

        # Patch 4: ISI Memory Equalizer with K-tap tail cancellation
        if cfg['pipeline'].get('enable_isi_equalizer', False) and len(tx_history) > 1:
            K = cfg['pipeline'].get('isi_equalizer_taps', 3)  # 3-tap default
            alpha = cfg['pipeline'].get('isi_equalizer_alpha', 0.1)  # 10% cancellation
            
            # Get last K *previous* symbols, excluding current
            recent_history = tx_history[:-1][-K:]
            
            for hist_symbol, hist_Nm in recent_history:
                # Calculate ISI contribution from this historical symbol
                isi_factor = alpha * (hist_Nm / cfg['pipeline']['Nm_per_symbol'])
                
                if hist_symbol == 0:  # Previous DA transmission
                    q_da -= isi_factor * abs(q_da)  # Subtract DA ISI
                else:  # Previous SERO transmission  
                    q_sero -= isi_factor * abs(q_sero)  # Subtract SERO ISI
        
        # sigma_da, sigma_sero already available from above

        if mod == 'MoSK':
            # Same statistic used during calibration (sign-aware difference; no CTRL subtraction here)
            q_eff_da = get_nt_params(cfg, 'DA')['q_eff_e']
            q_eff_sero = get_nt_params(cfg, 'SERO')['q_eff_e']
            sign_da = 1.0 if q_eff_da >= 0 else -1.0
            sign_sero = 1.0 if q_eff_sero >= 0 else -1.0

            numer = (sign_da * q_da) - (sign_sero * q_sero)
            decision_stat = numer / (sigma_diff if sigma_diff > 1e-15 else 1e-15)

            # Honor comparator direction saved by calibration (default to '>')
            threshold = float(cfg['pipeline'].get('mosk_threshold', 0.0))
            comparator = str(cfg['pipeline'].get('mosk_comparator', '>'))
            if comparator == '>':
                s_rx = 0 if decision_stat > threshold else 1  # DA when stat exceeds threshold
            else:
                s_rx = 0 if decision_stat < threshold else 1
            rx_symbols[i] = s_rx
            
            # Patch 5: Guardrails to catch future regressions
            if i == 0:  # Log once per sequence
                # Validate decision statistic makes sense
                stat_magnitude = abs(decision_stat)
                charge_magnitude = max(abs(q_da), abs(q_sero))
                
                # FIX: Compare like-for-like since decision_stat is normalized
                if cfg['pipeline'].get('mosk_statistic', 'sign_aware_diff') == 'sign_aware_diff':
                    # Decision statistic is normalized (dimensionless), compare to its typical scale (~O(1))
                    ref_scale = 10.0  # Reasonable for normalized statistics
                else:
                    # Raw units - compare to charge scale
                    ref_scale = 1000 * charge_magnitude
                
                if stat_magnitude > ref_scale:
                    logger.warning(f"MoSK decision statistic suspiciously large: {decision_stat:.2e} "
                                f"vs expected scale ~{ref_scale:.1f}. Check for normalization error.")
                
                threshold_magnitude = abs(threshold)
                if cfg['pipeline'].get('mosk_statistic', 'sign_aware_diff') == 'sign_aware_diff':
                    # Threshold should also be normalized (dimensionless)
                    threshold_ref_scale = 10.0  # Reasonable for normalized thresholds
                else:
                    # Raw threshold units
                    threshold_ref_scale = 1000 * charge_magnitude
                    
                if threshold_magnitude > threshold_ref_scale:
                    logger.warning(f"MoSK threshold suspiciously large: {threshold:.2e} "
                                f"vs expected scale ~{threshold_ref_scale:.1f}. Check for units.")
                
                # Validate threshold in matching units. When using the normalized MoSK
                # statistic ('sign_aware_diff'), the threshold is dimensionless, so
                # comparing to a Coulomb scale is meaningless (avoid false positives).
                if cfg['pipeline'].get('mosk_statistic', 'sign_aware_diff') != 'sign_aware_diff':
                    if abs(threshold) > charge_magnitude * 10:
                        logger.warning(f"MoSK threshold suspiciously large: {threshold:.2e} "
                                       f"vs typical charge {charge_magnitude:.2e}")
                
                logger.info(f"MoSK decision: stat={decision_stat:.2e}, thresh={threshold:.2e}, "
                        f"charges=[{q_da:.2e}, {q_sero:.2e}]")
            if s_tx == 0:
                stats_da.append(decision_stat)
            else:
                stats_sero.append(decision_stat)
          
        elif mod == 'CSK':
            # Get CSK configuration parameters
            M = cfg['pipeline']['csk_levels']
            target_channel = cfg['pipeline'].get('csk_target_channel', 'DA')
            combiner_cfg = cfg['pipeline'].get('csk_combiner', 'zscore')
            combiner = cfg['pipeline'].get('csk_selected_combiner', combiner_cfg)  # Use selected if present
            use_dual = bool(cfg['pipeline'].get('csk_dual_channel', True))
            leakage = float(cfg['pipeline'].get('csk_leakage_frac', 0.0))
            
            # Tail-gated integration
            tail = float(cfg['pipeline'].get('csk_tail_fraction', 1.0))
            tail = min(max(tail, 0.1), 1.0)  # clamp
            i0 = int((1.0 - tail) * n_detect_samples)

            use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
            # CSK always uses CTRL subtraction when enabled
            sig_da = (ig - ic) if use_ctrl else ig
            sig_sero = (ia - ic) if use_ctrl else ia

            # Tail-gated trapezoid integration
            q_da = float(np.trapezoid(sig_da[i0:n_detect_samples], dx=dt))
            q_sero = float(np.trapezoid(sig_sero[i0:n_detect_samples], dx=dt))

            # Compute dual-channel decision statistic Q
            if use_dual:
                Q = _csk_dual_channel_Q(
                    q_da=q_da, q_sero=q_sero,
                    sigma_da=sigma_da, sigma_sero=sigma_sero,
                    rho_cc=rho_cc, combiner=combiner, leakage_frac=leakage,
                    target=target_channel
                )
            else:
                # Legacy single-channel behavior
                Q = q_da if target_channel == 'DA' else q_sero

            # Get thresholds from configuration
            threshold_key = f'csk_thresholds_{target_channel.lower()}'
            thresholds = cfg['pipeline'].get(threshold_key, [])
            if not thresholds:
                logger.warning(f"CSK thresholds missing for {threshold_key}. Symbol detection will default to level 0.")
            
            # Use data-driven orientation (not q_eff)
            q_eff_target = get_nt_params(cfg, target_channel)['q_eff_e']
            increasing = bool(cfg['pipeline'].get('csk_thresholds_increasing',
                                                True if q_eff_target > 0 else False))
            
            # CSK symbol detection
            s_rx = 0
            # Use the orientation determined during calibration
            for thresh in thresholds:
                if increasing:
                    if Q > thresh:
                        s_rx += 1
                    else:
                        break
                else:
                    if Q < thresh:
                        s_rx += 1
                    else:
                        break
            rx_symbols[i] = s_rx
            
            # Store decision statistics for analysis
            (stats_da if s_tx < M/2 else stats_sero).append(Q)

            # Log configuration on first symbol (for debugging)
            if i == 0:
                cb = cfg['pipeline'].get('csk_selected_combiner', combiner)
                logger.info(f"CSK combiner={cb}, dual={use_dual}, σ=[{sigma_da:.2e},{sigma_sero:.2e}], "
                            f"ρcc={rho_cc:+.2f}, increasing={increasing}, leakage={leakage:.2f}")
              
        elif mod == 'Hybrid':

            q_da_raw = float(np.trapezoid(ig[:n_detect_samples], dx=dt))
            q_sero_raw = float(np.trapezoid(ia[:n_detect_samples], dx=dt))

            # Use MoSK decision statistic with proper single-ended noise
            sign_da = 1.0 if q_eff_da >= 0 else -1.0
            sign_sero = 1.0 if q_eff_sero >= 0 else -1.0
            decision_stat = (sign_da * q_da_raw - sign_sero * q_sero_raw) / max(sigma_diff_mosk, 1e-15)

            threshold_mosk = float(cfg['pipeline'].get('mosk_threshold', 0.0))
            comparator = str(cfg['pipeline'].get('mosk_comparator', '>'))
            b_hat = (0 if (decision_stat > threshold_mosk) else 1) if comparator == '>' \
                    else (0 if (decision_stat < threshold_mosk) else 1)

            # Amplitude decision still uses CTRL-differenced charges
            threshold_da = cfg['pipeline'].get('hybrid_threshold_da', 0.0)
            threshold_sero = cfg['pipeline'].get('hybrid_threshold_sero', 0.0)
            
            if b_hat == 0:
                # DA channel active: sign-aware comparator
                l_hat = 1 if ((q_da > threshold_da) if q_eff_da >= 0 else (q_da < threshold_da)) else 0
            else:
                # SERO channel active: sign-aware comparator
                l_hat = 1 if ((q_sero > threshold_sero) if q_eff_sero >= 0 else (q_sero < threshold_sero)) else 0

            # Construct final symbol
            s_rx = (b_hat << 1) | l_hat
            rx_symbols[i] = s_rx

            # ✅ For SNR diagnostics: store the *MoSK decision statistic* by TRUE molecule class (not by decision)
            true_mol_bit = (s_tx >> 1)
            if true_mol_bit == 0:
                stats_da.append(decision_stat)
            else:
                stats_sero.append(decision_stat)

            true_mol_bit = s_tx >> 1
            true_amp_bit = s_tx & 1
            if b_hat != true_mol_bit:
                subsymbol_errors['mosk'] += 1
            elif l_hat != true_amp_bit:
                subsymbol_errors['csk'] += 1
                
    # Calculate errors
    errors = np.sum(tx_symbols != rx_symbols)

    # summarize isi metric for this run
    isi_vals = metrics_ref.get('isi_ratio_values', [])
    isi_ratio_mean = float(np.mean(isi_vals)) if len(isi_vals) else float('nan')
    isi_ratio_median = float(np.median(isi_vals)) if len(isi_vals) else float('nan')
    
    result = {
        "modulation": mod,
        "symbols_tx": tx_symbols.tolist(),
        "symbols_rx": rx_symbols.tolist(),
        "errors": int(errors),
        "SER": errors / L if L > 0 else 0,
        "stats_da": stats_da,
        "stats_sero": stats_sero,
        "subsymbol_errors": subsymbol_errors,
        "thresholds_used": thresholds_used,
        "isi_ratio_mean": isi_ratio_mean,
        "isi_ratio_median": isi_ratio_median,
        # Stage 14: reproducibility knobs
        'noise_sigma_da': float(sigma_da),
        'noise_sigma_sero': float(sigma_sero),
        'noise_sigma_I_diff': float(sigma_diff),
    }

    # Add MoSK sigma metadata for Hybrid mode
    if mod == 'Hybrid':
        result['mosk_sigma_diff_used'] = float(sigma_diff_mosk)
        result['mosk_stat_units'] = 'normalized_by_sigma_diff_single_ended_preCTRL'

    return result


def run_sequence_batch(cfg: Dict[str, Any], batch_size: int = 50) -> Dict[str, Any]:
    return run_sequence(cfg)


def run_sequence_batch_cpu(cfg_list: List[Dict[str, Any]], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {executor.submit(run_sequence, cfg): cfg for cfg in cfg_list}
        
        for future in as_completed(future_to_cfg):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                cfg = future_to_cfg[future]
                print(f'Configuration generated an exception: {exc}')
                results.append({
                    'error': str(exc),
                    'modulation': cfg.get('pipeline', {}).get('modulation', 'unknown'),
                    'SER': 1.0
                })
    
    return results


__all__ = ["run_sequence", "run_sequence_batch", "run_sequence_batch_cpu", "calculate_proper_noise_sigma", "_single_symbol_currents"]
