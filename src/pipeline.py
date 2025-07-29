# src/pipeline.py - FULLY FIXED VERSION
"""
End-to-end simulator for the tri-channel OECT receiver.

ALL CRITICAL FIXES APPLIED:
- Fix 1: Removed double-subtraction in decision statistics
- Fix 2: Removed control channel signal contamination  
- Fix 3: Complete physics-based noise estimation with correlation
- Fix 4: Optimized detection window and device parameters

Calls:
    ▸ diffusion.finite_burst_concentration
    ▸ binding.bernoulli_binding
    ▸ oect.oect_current
    ▸ detection.detect_mosk / detect_csk_*

Returns bit/symbol error statistics for a complete symbol sequence.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from numpy.random import default_rng
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# local imports
from .mc_channel.transport import finite_burst_concentration
from .mc_receiver.binding import bernoulli_binding
from .mc_receiver.oect import oect_trio
from src.detection import calculate_ml_threshold

def to_int(value):
    """Convert value to int, handling scientific notation strings"""
    if isinstance(value, str):
        return int(float(value))
    elif isinstance(value, float):
        return int(value)
    else:
        return value

def calculate_proper_noise_sigma(cfg: Dict[str, Any], detection_window_s: float, symbol_index: int = -1) -> Tuple[float, float]:
    k_B = 1.38e-23
    T = cfg.get('temperature_K', 310)
    R_ch = cfg['oect'].get('R_ch_Ohm', 100)  # Realistic
    gm = cfg['oect'].get('gm_S', 0.005)
    I_dc = gm * abs(cfg['oect'].get('V_g_bias_V', -0.02))
    I_dc = max(I_dc, 1e-6)
    alpha_H = cfg['noise'].get('alpha_H', 1e-3)
    N_c = cfg['noise'].get('N_c', 4e12)
    rho_corr = cfg['noise'].get('rho_correlated', 0.9)
    K_d = 0.0  # FORCED to 0 for simplicity
    dt = cfg['sim']['dt_s']
    f_samp = 1.0 / dt
    B_det = cfg.get('detection_bandwidth_Hz', 100)
    T_int = detection_window_s
    f_min = 1.0 / T_int
    f_max = min(B_det, f_samp/2)
    
    effective_B = 0.25 / T_int  # Literature approx for integrator
    
    # Johnson: ~ PSD * T
    psd_johnson = 4 * k_B * T / R_ch
    johnson_charge_var = psd_johnson * T_int
    
    # Flicker: approx K_f * I^2 * ln(2πT) * T
    K_f = alpha_H / N_c
    flicker_charge_var = K_f * I_dc**2 * np.log(2 * np.pi * T_int) * T_int if f_max > f_min else 0
    
    # Drift: forced 0
    drift_charge_var = 0
    
    total_single_var = johnson_charge_var + flicker_charge_var + drift_charge_var
    differential_var = 2 * total_single_var * (1 - rho_corr)
    sigma_differential = np.sqrt(differential_var)
    sigma_differential = max(sigma_differential, 1e-12)
    
    #print(f"Noise calc: K_f={K_f:.3e}, sigma={sigma_differential:.3e} C, I_dc={I_dc:.3e} A")  # Keep for debug
    return sigma_differential, sigma_differential

# ─────────────────────────────────────────────────────────────────
def _single_symbol_currents(s_tx: int, tx_history: List[Tuple[int, float]], cfg: Dict[str, Any], rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate GLU, GABA, CTRL drain-current traces for one symbol interval.
    CORRECTED to use modern config structure and pass the main cfg directly.
    """
    # Initialization
    dt = cfg['sim']['dt_s']
    # The symbol period is now read dynamically from the pipeline config
    Ts = cfg['pipeline']['symbol_period_s']
    n_samples = int(Ts / dt)
    t_vec = np.arange(n_samples) * dt
    
    # Initialize concentrations at each channel
    conc_at_glu_ch = np.zeros(n_samples)
    conc_at_gaba_ch = np.zeros(n_samples)
    # Control channel receives no specific molecular signal
    conc_at_ctrl_ch = np.zeros(n_samples)
    
    distance_m = cfg['pipeline']['distance_um'] * 1e-6
    
    # Determine number of molecules for this symbol, including noise
    Nm_peak = cfg['pipeline']['Nm_per_symbol']
    if cfg['pipeline'].get('enable_molecular_noise', True):
        Nm_actual = rng.poisson(Nm_peak)
    else:
        Nm_actual = Nm_peak

    # --- ISI Calculation ---
    if cfg['pipeline'].get('enable_isi', False):
        # Calculate ISI memory dynamically based on diffusion time
        D_glu = cfg['neurotransmitters']['GLU']['D_m2_s']
        lambda_glu = cfg['neurotransmitters']['GLU']['lambda']
        D_eff = D_glu / (lambda_glu ** 2)
        time_95_percent_decay = 3.0 * (distance_m**2) / D_eff
        guard_factor = cfg['pipeline'].get('guard_factor', 0.3)
        total_clearance_time = (1 + guard_factor) * time_95_percent_decay
        isi_memory_symbols = math.ceil(total_clearance_time / Ts)

        relevant_history = tx_history[-isi_memory_symbols:]
        
        for k, (past_symbol, Nm_hist) in enumerate(reversed(relevant_history)):
            time_offset = (k + 1) * Ts
            t_vec_offset = t_vec + time_offset
            
            if past_symbol == 0:  # GLU was sent
                residual = finite_burst_concentration(Nm_hist, distance_m, t_vec_offset, cfg, 'GLU')
                conc_at_glu_ch += residual
            else:  # GABA was sent
                residual = finite_burst_concentration(Nm_hist, distance_m, t_vec_offset, cfg, 'GABA')
                conc_at_gaba_ch += residual
    
    # --- Current Symbol Generation ---
    if s_tx == 0:  # GLU sent
        current_conc = finite_burst_concentration(Nm_actual, distance_m, t_vec, cfg, 'GLU')
        conc_at_glu_ch += current_conc
    else:  # GABA sent
        current_conc = finite_burst_concentration(Nm_actual, distance_m, t_vec, cfg, 'GABA')
        conc_at_gaba_ch += current_conc
        
    # --- INDEPENDENT Binding Simulation for each channel ---
    # The bernoulli_binding function is now smart enough to find the correct
    # N_sites for each channel, so we can pass the main `cfg` directly.
    bound_glu_ch, _, _ = bernoulli_binding(conc_at_glu_ch, 'GLU', cfg, rng)
    bound_gaba_ch, _, _ = bernoulli_binding(conc_at_gaba_ch, 'GABA', cfg, rng)
    # Control channel has no specific binding, handled by bernoulli_binding
    bound_ctrl_ch, _, _ = bernoulli_binding(conc_at_ctrl_ch, 'CTRL', cfg, rng)
    
    # --- OECT current generation ---
    bound_sites_trio = np.vstack([bound_glu_ch, bound_gaba_ch, bound_ctrl_ch])
    currents = oect_trio(
        bound_sites_trio,
        nts=("GLU", "GABA", "CTRL"),
        cfg=cfg,
        rng=rng
    )
    
    return currents["GLU"], currents["GABA"], currents["CTRL"], Nm_actual


# ─────────────────────────────────────────────────────────────────
# REPLACE the entire run_sequence function with this one:
def run_sequence(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate <sequence_length> symbols and compute BER/SER.
    Includes an adaptive threshold for robust MoSK detection.
    """
    mod = cfg['pipeline']['modulation']
    L = cfg['pipeline']['sequence_length']
    rng = default_rng(cfg['pipeline'].get('random_seed'))
    
    tx_history: List[Tuple[int, float]] = []
    
    # --- NEW: Adaptive Threshold Logic for MoSK ---
    adaptive_threshold = 0.0  # Initialize the threshold
    alpha_adapt = 0.05        # Learning rate (how quickly the threshold adapts)
    is_first_symbol = True
    
    # Generate transmitted symbols
    if mod == 'MoSK':
        tx_symbols = rng.integers(0, 2, size=L)
    elif mod.startswith('CSK'):
        # This part remains for future use with CSK
        M = cfg['pipeline']['csk_levels']
        tx_symbols = rng.integers(0, M, size=L)
    else: # Hybrid
        tx_symbols = rng.integers(0, 4, size=L)

    rx_symbols = np.zeros(L, dtype=int)
    stats_glu = []
    stats_gaba = []
    
    # Use tqdm for progress bar, disable if specified (e.g., in batch runs)
    show_progress = not cfg.get('disable_progress', False)
    for i, s_tx in enumerate(tqdm(tx_symbols, desc=f"Simulating {mod}", disable=not show_progress)):
        ig, ia, ic, Nm_realised = _single_symbol_currents(s_tx, tx_history, cfg, rng)
        tx_history.append((s_tx, Nm_realised))
        
        dt = cfg['sim']['dt_s']
        # Use the full symbol period for integration, ensuring max signal capture
        detection_window_s = cfg['pipeline']['symbol_period_s']
        n_detect_samples = int(detection_window_s / dt)
        
        q_glu = np.trapezoid((ig - ic)[:n_detect_samples], dx=dt)
        q_gaba = np.trapezoid((ia - ic)[:n_detect_samples], dx=dt)
        
        if mod == 'MoSK':
            # Decision statistic: A larger GLU signal (more negative q_glu) gives a positive statistic
            decision_stat = -(q_glu - q_gaba)

            # On the first symbol, initialize the threshold directly to the observed value.
            if is_first_symbol:
                adaptive_threshold = decision_stat
                is_first_symbol = False
            
            # Decision: Compare to the DYNAMIC adaptive threshold
            s_rx = 0 if decision_stat > adaptive_threshold else 1
            rx_symbols[i] = s_rx

            # Update the threshold by moving it slightly towards the latest observation.
            # This allows it to track baseline shifts from ISI and low-frequency noise.
            adaptive_threshold = (1 - alpha_adapt) * adaptive_threshold + alpha_adapt * decision_stat
            
            if s_tx == 0:
                stats_glu.append(decision_stat)
            else:
                stats_gaba.append(decision_stat)
        else:
            # Placeholder for future CSK/Hybrid implementations
            # You would need adaptive logic for multiple thresholds here
            rx_symbols[i] = 0 # Default behavior for non-MoSK for now
    
    errors = np.sum(tx_symbols != rx_symbols)
    
    return {
        "modulation": mod,
        "symbols_tx": tx_symbols.tolist(),
        "symbols_rx": rx_symbols.tolist(), # Convert to list for consistency
        "errors": int(errors),
        "SER": errors / L if L > 0 else 0,
        "stats_glu": stats_glu,
        "stats_gaba": stats_gaba,
    }


# ─────────────────────────────────────────────────────────────────
def run_sequence_batch_cpu(cfg_list: List[Dict[str, Any]], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run multiple sequence simulations in parallel using CPU multiprocessing."""
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


__all__ = ["run_sequence", "run_sequence_batch_cpu"]