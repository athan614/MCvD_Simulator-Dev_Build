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
    flicker_charge_var = K_f * I_dc**2 * np.log(2 * np.pi * T_int) * T_int if f_max > f_min else 0
    
    drift_charge_var = 0
    
    total_single_var = johnson_charge_var + flicker_charge_var + drift_charge_var
    
    # Enhanced: Expose correlation coefficient via config
    use_ctrl = bool(cfg.get('pipeline', {}).get('use_control_channel', True))
    effective_rho = noise_cfg.get('effective_correlation', rho_corr)  # Allow override
    
    if use_ctrl:
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
    distance_glu_m: float,
    distance_gaba_m: float,
    distance_ctrl_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FULLY VECTORIZED: Calculate ISI contributions from transmission history.
    """
    if not tx_history:
        return np.zeros_like(t_vec), np.zeros_like(t_vec), np.zeros_like(t_vec)
    
    mod = cfg['pipeline']['modulation']
    Ts = cfg['pipeline']['symbol_period_s']
    level_scheme = cfg['pipeline'].get('csk_level_scheme', 'uniform')
    
    # Convert history to arrays for vectorization
    past_symbols = np.array([h[0] for h in tx_history], dtype=np.int32)
    Nm_history = np.array([h[1] for h in tx_history], dtype=np.float64)
    n_history = len(tx_history)
    
    # Pre-allocate concentration arrays
    conc_at_glu_ch = np.zeros_like(t_vec)
    conc_at_gaba_ch = np.zeros_like(t_vec)
    conc_at_ctrl_ch = np.zeros_like(t_vec)  # Always zero (no signal contamination)
    
    # Calculate all time offsets at once
    k_indices = np.arange(n_history)
    time_offsets = (k_indices + 1) * Ts
    
    if mod == "MoSK":
        # Separate GLU and GABA transmissions
        glu_mask = (past_symbols == 0)
        gaba_mask = ~glu_mask
        
        # Process GLU contributions in batch
        if np.any(glu_mask):
            glu_Nm = Nm_history[glu_mask]
            glu_offsets = time_offsets[glu_mask]
            glu_residuals = finite_burst_concentration_batch_time(
                glu_Nm, distance_glu_m, t_vec, glu_offsets, cfg, 'GLU'
            )
            conc_at_glu_ch = np.sum(glu_residuals, axis=0)
        
        # Process GABA contributions in batch
        if np.any(gaba_mask):
            gaba_Nm = Nm_history[gaba_mask]
            gaba_offsets = time_offsets[gaba_mask]
            gaba_residuals = finite_burst_concentration_batch_time(
                gaba_Nm, distance_gaba_m, t_vec, gaba_offsets, cfg, 'GABA'
            )
            conc_at_gaba_ch = np.sum(gaba_residuals, axis=0)
                
    elif mod.startswith("CSK"):
        # ISI uses actual transmitted Nm values from history
        target_channel = cfg['pipeline'].get('csk_target_channel', 'GLU')
        
        # Process non-zero contributions
        nonzero_mask = Nm_history > 0
        if np.any(nonzero_mask):
            active_Nm = Nm_history[nonzero_mask]
            active_offsets = time_offsets[nonzero_mask]
            
            if target_channel == 'GLU':
                residuals = finite_burst_concentration_batch_time(
                    active_Nm, distance_glu_m, t_vec, active_offsets, cfg, 'GLU'
                )
                conc_at_glu_ch = np.sum(residuals, axis=0)
            else:  # GABA
                residuals = finite_burst_concentration_batch_time(
                    active_Nm, distance_gaba_m, t_vec, active_offsets, cfg, 'GABA'
                )
                conc_at_gaba_ch = np.sum(residuals, axis=0)
                    
    elif mod == "Hybrid":
        # Extract molecule type from history
        mol_type_bits = past_symbols >> 1
        
        # Process GLU contributions
        glu_mask = (mol_type_bits == 0) & (Nm_history > 0)
        if np.any(glu_mask):
            glu_amps = Nm_history[glu_mask]
            glu_offsets = time_offsets[glu_mask]
            glu_residuals = finite_burst_concentration_batch_time(
                glu_amps, distance_glu_m, t_vec, glu_offsets, cfg, 'GLU'
            )
            conc_at_glu_ch = np.sum(glu_residuals, axis=0)
        
        # Process GABA contributions
        gaba_mask = (mol_type_bits == 1) & (Nm_history > 0)
        if np.any(gaba_mask):
            gaba_amps = Nm_history[gaba_mask]
            gaba_offsets = time_offsets[gaba_mask]
            gaba_residuals = finite_burst_concentration_batch_time(
                gaba_amps, distance_gaba_m, t_vec, gaba_offsets, cfg, 'GABA'
            )
            conc_at_gaba_ch = np.sum(gaba_residuals, axis=0)
    
    return conc_at_glu_ch, conc_at_gaba_ch, conc_at_ctrl_ch


def _single_symbol_currents(s_tx: int, tx_history: List[Tuple[int, float]], cfg: Dict[str, Any], rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate GLU, GABA, CTRL drain-current traces for one symbol interval.
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
    conc_at_glu_ch = np.zeros(n_samples)
    conc_at_gaba_ch = np.zeros(n_samples)
    conc_at_ctrl_ch = np.zeros(n_samples)  # Pure noise reference
    
    # Keep separate holders for metrics
    isi_glu = np.zeros(n_samples)
    isi_gaba = np.zeros(n_samples)
    sym_glu = np.zeros(n_samples)
    sym_gaba = np.zeros(n_samples)
    
    # Get channel-specific distances
    if 'channel_distances' in cfg:
        distance_glu_m = cfg['channel_distances']['GLU'] * 1e-6
        distance_gaba_m = cfg['channel_distances']['GABA'] * 1e-6
        distance_ctrl_m = cfg['channel_distances']['CTRL'] * 1e-6
    else:
        distance_m = cfg['pipeline']['distance_um'] * 1e-6
        distance_glu_m = distance_m
        distance_gaba_m = distance_m
        distance_ctrl_m = distance_m
    
    mod = cfg['pipeline']['modulation']
    level_scheme = cfg['pipeline'].get('csk_level_scheme', 'uniform')
    non_specific_factor = cfg['pipeline'].get('non_specific_binding_factor', 0.0)
    
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
            D_glu = cfg['neurotransmitters']['GLU']['D_m2_s']
            D_gaba = cfg['neurotransmitters']['GABA']['D_m2_s']
            D = (D_glu + D_gaba) / 2
            avg_distance_m = (distance_glu_m + distance_gaba_m) / 2
            decay95 = 3.0 * avg_distance_m**2 / D
            guard_factor = cfg['pipeline'].get('guard_factor', 0.3)
            isi_memory = math.ceil((1 + guard_factor) * decay95 / Ts)
        else:
            isi_memory = cfg['pipeline']['isi_memory_symbols']
        
        relevant_history = tx_history[-isi_memory:] if len(tx_history) >= isi_memory else tx_history
        
        # Use vectorized ISI calculation
        if relevant_history:
            isi_glu, isi_gaba, _ = _calculate_isi_vectorized(
                relevant_history, t_vec, cfg, 
                distance_glu_m, distance_gaba_m, distance_ctrl_m
            )
            conc_at_glu_ch += isi_glu
            conc_at_gaba_ch += isi_gaba
    
    # Current Symbol Generation using Nm_actual (the noisy realization)
    if mod == "MoSK":
        if s_tx == 0:  # GLU
            conc_glu = finite_burst_concentration(Nm_actual, distance_glu_m, t_vec, cfg, 'GLU')
            conc_at_glu_ch += conc_glu
            sym_glu += conc_glu
        else:  # GABA
            conc_gaba = finite_burst_concentration(Nm_actual, distance_gaba_m, t_vec, cfg, 'GABA')
            conc_at_gaba_ch += conc_gaba
            sym_gaba += conc_gaba
            
    elif mod.startswith("CSK"):
        target_channel = cfg['pipeline'].get('csk_target_channel', 'GLU')
        if Nm_actual > 0:
            if target_channel == 'GLU':
                conc_glu = finite_burst_concentration(Nm_actual, distance_glu_m, t_vec, cfg, 'GLU')
                conc_at_glu_ch += conc_glu
                sym_glu += conc_glu
            else:  # GABA
                conc_gaba = finite_burst_concentration(Nm_actual, distance_gaba_m, t_vec, cfg, 'GABA')
                conc_at_gaba_ch += conc_gaba
                sym_gaba += conc_gaba
            
    elif mod == "Hybrid":
        mol_type_bit = s_tx >> 1
        if mol_type_bit == 0:  # GLU
            conc_glu = finite_burst_concentration(Nm_actual, distance_glu_m, t_vec, cfg, 'GLU')
            conc_at_glu_ch += conc_glu
            sym_glu += conc_glu
        else:  # GABA
            conc_gaba = finite_burst_concentration(Nm_actual, distance_gaba_m, t_vec, cfg, 'GABA')
            conc_at_gaba_ch += conc_gaba
            sym_gaba += conc_gaba
    
    # INDEPENDENT Binding simulation for each channel
    cfg_glu = cfg.copy()
    cfg_glu['N_apt'] = cfg.get('binding', {}).get('N_sites_glu', cfg.get('N_apt'))
    bound_glu_ch, _, _ = bernoulli_binding(conc_at_glu_ch, 'GLU', cfg_glu, rng)
    
    cfg_gaba = cfg.copy()
    cfg_gaba['N_apt'] = cfg.get('binding', {}).get('N_sites_gaba', cfg.get('N_apt'))
    bound_gaba_ch, _, _ = bernoulli_binding(conc_at_gaba_ch, 'GABA', cfg_gaba, rng)
    
    cfg_ctrl = cfg.copy()
    cfg_ctrl['N_apt'] = 0  # No aptamer sites
    bound_ctrl_ch, _, _ = bernoulli_binding(conc_at_ctrl_ch, 'CTRL', cfg_ctrl, rng)
    
    # OECT current generation
    bound_sites_trio = np.vstack([bound_glu_ch, bound_gaba_ch, bound_ctrl_ch])
    currents = oect_trio(
        bound_sites_trio,
        nts=("GLU", "GABA", "CTRL"),
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
            isi_mass = float(np.trapezoid(isi_glu[:n_int], dx=dt) + np.trapezoid(isi_gaba[:n_int], dx=dt))
            sym_mass = float(np.trapezoid(sym_glu[:n_int], dx=dt) + np.trapezoid(sym_gaba[:n_int], dx=dt))
            denom = max(sym_mass, 1e-30)
            ratio = isi_mass / denom
            metrics = cfg.setdefault('_metrics', {})
            lst = metrics.setdefault('isi_ratio_values', [])
            lst.append(ratio)
        except Exception:
            pass
    # ================================================================

    return currents["GLU"], currents["GABA"], currents["CTRL"], Nm_actual


def run_sequence(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate <sequence_length> symbols and compute BER/SEP.
    Detection logic unchanged - the fixes are in signal generation.
    """
    try:
        from src.detection import calculate_ml_threshold
    except ImportError:
        from .detection import calculate_ml_threshold
        
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
    stats_glu = []
    stats_gaba = []

    # enable metric collection for this run
    cfg['collect_isi_metrics'] = True
    metrics_ref = cfg.setdefault('_metrics', {})
    metrics_ref.setdefault('isi_ratio_values', [])
    
    # Get polarities
    q_eff_glu = get_nt_params(cfg, 'GLU')['q_eff_e']
    q_eff_gaba = get_nt_params(cfg, 'GABA')['q_eff_e']

    # Stage 14: Compute noise sigma ONCE before the loop
    detection_window_s = cfg['detection'].get('decision_window_s',
                                              cfg['pipeline']['symbol_period_s'])
    sigma_glu, sigma_gaba = calculate_proper_noise_sigma(cfg, detection_window_s)
    
    # Enhanced: Allow residual cross-channel correlation after CTRL subtraction
    noise_cfg = cfg.get('noise', {})
    rho_cc = float(noise_cfg.get('rho_between_channels_after_ctrl', 0.0))
    # Hardening: enforce a valid correlation range (prevents bad config values)
    rho_cc = max(-1.0, min(1.0, rho_cc))  # NEW: clamp at source
    
    # Compute variance of the difference
    var_diff = (sigma_glu * sigma_glu +
                sigma_gaba * sigma_gaba -
                2.0 * rho_cc * sigma_glu * sigma_gaba)
    
    if var_diff < -1e-12:  # generous tolerance for FP noise
        logger.warning("Computed var_diff < 0 (%.3e). Check rho_cc and sigmas; clipping to 0.", var_diff)
    
    sigma_diff = float(math.sqrt(var_diff)) if var_diff > 0.0 else 0.0

    # Process symbols
    for i, s_tx in enumerate(tqdm(tx_symbols, desc=f"Simulating {mod}", disable=cfg.get('disable_progress', False))):
        ig, ia, ic, Nm_realised = _single_symbol_currents(s_tx, tx_history, cfg, rng)
        tx_history.append((s_tx, Nm_realised))
        
        # Detection and decision statistics
        dt = cfg['sim']['dt_s']
        # detection_window_s already computed above
    
        # NOTE: sigma_glu, sigma_gaba, sigma_diff already computed once above
        
        n_total_samples = len(ig)
        n_detect_samples = min(int(detection_window_s / dt), n_total_samples)

        if n_detect_samples <= 1:
             if i == 0: print("Warning: Insufficient detection samples. Check dt_s and symbol_period_s.")
             continue
        
        use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
        
        # Calculate differential charges (Signal - Control)
        sig_glu = (ig - ic) if use_ctrl else ig
        sig_gaba = (ia - ic) if use_ctrl else ia
        q_glu = np.trapezoid(sig_glu[:n_detect_samples], dx=dt)
        q_gaba = np.trapezoid(sig_gaba[:n_detect_samples], dx=dt)
        
        # sigma_glu, sigma_gaba already available from above

        if mod == 'MoSK':
            decision_stat = q_glu / sigma_glu + q_gaba / sigma_gaba
            threshold = cfg['pipeline'].get('mosk_threshold', 0.0)
            s_rx = 0 if decision_stat > threshold else 1
            rx_symbols[i] = s_rx
            if s_tx == 0:
                stats_glu.append(decision_stat)
            else:
                stats_gaba.append(decision_stat)
          
        elif mod == 'CSK':
            target_channel = cfg['pipeline'].get('csk_target_channel', 'GLU')
            M = cfg['pipeline']['csk_levels']
            if target_channel == 'GLU':
                Q = q_glu
                q_eff = q_eff_glu
            else:
                Q = q_gaba
                q_eff = q_eff_gaba
            thresholds = cfg['pipeline'].get(f'csk_thresholds_{target_channel.lower()}', [])
            s_rx = 0
            for thresh in thresholds:
                if q_eff > 0:
                    if Q > thresh: s_rx += 1
                    else: break
                else:
                    if Q < thresh: s_rx += 1
                    else: break
            rx_symbols[i] = s_rx
            if s_tx < M/2:
                stats_glu.append(Q)
            else:
                stats_gaba.append(Q)
              
        elif mod == 'Hybrid':
            decision_stat = q_glu / sigma_glu + q_gaba / sigma_gaba
            threshold_mosk = cfg['pipeline'].get('mosk_threshold', 0.0)
            b_hat = 0 if decision_stat > threshold_mosk else 1  # GLU if positive
            threshold_glu = cfg['pipeline'].get('hybrid_threshold_glu', 0.0)
            threshold_gaba = cfg['pipeline'].get('hybrid_threshold_gaba', 0.0)
            thresholds_used = {'GLU': threshold_glu, 'GABA': threshold_gaba}
            if b_hat == 0:
                q_target = q_glu
                l_hat = 1 if q_target > threshold_glu else 0
                stats_glu.append(q_target)
            else:
                q_target = q_gaba
                l_hat = 1 if q_target < threshold_gaba else 0
                stats_gaba.append(q_target)
            s_rx = (b_hat << 1) | l_hat
            rx_symbols[i] = s_rx
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
    
    return {
        "modulation": mod,
        "symbols_tx": tx_symbols.tolist(),
        "symbols_rx": rx_symbols,
        "errors": int(errors),
        "SER": errors / L if L > 0 else 0,
        "stats_glu": stats_glu,
        "stats_gaba": stats_gaba,
        "subsymbol_errors": subsymbol_errors,
        "thresholds_used": thresholds_used,
        "isi_ratio_mean": isi_ratio_mean,
        "isi_ratio_median": isi_ratio_median,
        # Stage 14: reproducibility knobs
        'noise_sigma_glu': float(sigma_glu),
        'noise_sigma_gaba': float(sigma_gaba),
        'noise_sigma_I_diff': float(sigma_diff),
    }


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
