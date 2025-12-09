# src/pipeline.py - BOTH FIXES APPLIED + ISI metrics plumbing
"""
End-to-end simulator for the tri-channel OECT receiver.

COMPLETE FIX VERSION:
1. Uniform CSK level mapping to avoid zero molecules
2. Correct Poisson noise application at the level, not scaling factor
3. (New) ISI metric collection hook (no physics changes)
"""

from typing import Dict, Any, List, Tuple, Optional, cast
import numpy as np
from numpy.random import default_rng
import math
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
from collections import defaultdict

# Import vectorized transport functions including batch time
from .mc_channel.transport import finite_burst_concentration, finite_burst_concentration_batch, finite_burst_concentration_batch_time
from .mc_receiver.binding import bernoulli_binding
from .mc_receiver.oect import oect_trio
from .constants import get_nt_params

logger = logging.getLogger(__name__)


class RunningStat:
    """Online mean/variance tracker (Welford)."""

    __slots__ = ("count", "mean", "_m2")

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2

    def merge(self, other: "RunningStat") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self._m2 = other._m2
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self._m2 += other._m2 + delta * delta * self.count * other.count / total
        self.mean = (self.mean * self.count + other.mean * other.count) / total
        self.count = total

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self._m2 / (self.count - 1)

    def as_summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {"count": 0, "mean": float("nan"), "var": float("nan")}
        return {
            "count": float(self.count),
            "mean": float(self.mean),
            "var": float(self.variance),
        }

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

    The simulator now defaults to integrating over the full symbol period so that
    analytics and Monte Carlo experiments share the same decision window policy.
    The resolved window is written back into ``cfg['detection']['decision_window_s']``
    so downstream code (noise model, logging, plotting) observes a consistent value.
    """
    detection = cfg.setdefault('detection', {})
    pipeline_cfg = cfg.setdefault('pipeline', {})
    policy = str(detection.get('decision_window_policy', 'fraction_of_ts')).lower()

    if policy in ('full_ts', 'full', 'ts'):
        win_s = Ts
        anchor = detection.get('decision_window_anchor', 'start')
    elif policy in ('fraction_of_ts', 'fraction', 'frac', 'tail', 'tail_fraction'):
        frac = float(detection.get('decision_window_fraction', 0.9))
        frac = min(max(frac, 0.1), 1.0)
        win_s = frac * Ts
        anchor = 'tail'
    else:  # 'fixed' (legacy)
        win_s = float(detection.get('decision_window_s', Ts))
        anchor = detection.get('decision_window_anchor', 'start')

    min_samples = int(detection.get('min_decision_samples', 16))
    win_s = max(min_samples * dt, min(win_s, Ts))

    detection['decision_window_s'] = win_s
    detection['decision_window_anchor'] = str(anchor).lower()

    # Ensure generator produces at least a full symbol so the tail exists.
    pipeline_cfg['symbol_period_s'] = Ts

    cfg['detection'] = detection
    cfg['pipeline'] = pipeline_cfg
    return win_s

def _csk_dual_channel_Q(
    q_da: float, q_sero: float,
    sigma_da: float, sigma_sero: float,
    rho_cc: float,
    combiner: str = "zscore",
    leakage_frac: float = 0.0,
    target: str = "DA",
    cfg: Optional[Dict[str, Any]] = None  # FIX: Add cfg parameter
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
        cfg: Optional configuration dictionary for shrinkage parameters
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
        if abs(rho) >= 0.1:
            # FIX B: Use measured rho when available, with CI-aware shrinkage for stability
            rho_effective = rho
            
            # Optional: shrinkage based on sample size (if metadata available)
            # For very low Nm, shrink toward 0 to reduce over-subtraction risk
            if cfg is not None:  # FIX: Check cfg exists
                pipeline_cfg = cfg.get('pipeline', {})
                n_samples = pipeline_cfg.get('_noise_sample_size', 100)  # default to 100
                if n_samples < 50:  # few samples, apply shrinkage
                    shrinkage_factor = min(1.0, n_samples / 50.0)
                    rho_effective = shrinkage_factor * rho
            
            return Qt/sg_t - rho_effective * (Qo/sg_o)
        else:
            # keep variance normalization for stability
            return Qt / max(sg_t, 1e-30)

    elif combiner == "whitened":
        # Fisher LDA for ordered classes; w ∝ Σ^{-1} μ, Σ = [[σt^2, ρσtσo],[ρσtσo, σo^2]]
        # For pure CSK (μ=[μt,0]), up to scale: w ∝ [1/σt^2, -ρ/(σtσo)].
        # Implement via the scale-free z-score equivalent for stability:
        if abs(rho) >= 0.1:
            return (Qt / max(sg_t, 1e-30)) - rho * (Qo / max(sg_o, 1e-30))
        else:
            return (Qt / max(sg_t, 1e-30))  # no benefit from cross-channel term

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

def calculate_proper_noise_sigma(
    cfg: Dict[str, Any],
    detection_window_s: float,
    symbol_index: int = -1,
    components_out: Optional[Dict[str, float]] = None
) -> Tuple[float, float]:
    """Calculate physics-based noise sigma (robust implementation).

    Optionally populates ``components_out`` with a thermal/flicker breakdown for
    downstream logging while keeping the return contract unchanged.
    """
    k_B = 1.38e-23

    # Robustly access configuration parameters with config override support
    T = cfg.get('sim', {}).get('temperature_K', 310)

    oect_cfg = cfg.get('oect', {})
    R_ch = float(oect_cfg.get('R_ch_Ohm', 100))
    gm = float(oect_cfg.get('gm_S', 0.005))
    C_tot = float(oect_cfg.get('C_tot_F', 5e-8))
    V_g_bias = float(oect_cfg.get('V_g_bias_V', -0.2))

    I_dc_cfg = float(oect_cfg.get('I_dc_A', 0.0) or 0.0)
    if not np.isfinite(I_dc_cfg) or I_dc_cfg <= 0:
        I_dc = gm * abs(V_g_bias)
    else:
        I_dc = I_dc_cfg
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
    f_max = min(B_det, f_samp / 2)

    psd_johnson = 4 * k_B * T / R_ch
    johnson_charge_var = psd_johnson * T_int

    K_f = alpha_H / N_c
    log_span = math.log(f_max / f_min) if (f_max > f_min and f_min > 0.0) else 0.0
    flicker_charge_var = K_f * I_dc**2 * max(log_span, 0.0) * T_int

    drift_charge_var = 0.0

    thermal_var_single = max(johnson_charge_var, 0.0)
    flicker_var_single = max(flicker_charge_var, 0.0)
    drift_var_single = max(drift_charge_var, 0.0)
    total_single_var = thermal_var_single + flicker_var_single + drift_var_single
    sigma_single = math.sqrt(max(total_single_var, 0.0))

    pipeline_cfg = cfg.get('pipeline', {})
    modulation = str(pipeline_cfg.get('modulation', '')).upper()

    # Enhanced: Mode-aware noise calculation (MoSK never applies CTRL subtraction)
    use_ctrl_flag = bool(pipeline_cfg.get('use_control_channel', True))

    # Prefer measured post-CTRL correlation, fall back to config
    rho_measured = pipeline_cfg.get('rho_cc_measured', None)
    if rho_measured is None or not np.isfinite(rho_measured):
        rho_post_cfg = noise_cfg.get('rho_between_channels_after_ctrl', noise_cfg.get('effective_correlation', rho_corr))
        rho_post = float(np.clip(rho_post_cfg, -0.999, 0.999))
    else:
        rho_post = float(np.clip(rho_measured, -0.999, 0.999))

    effective_rho = float(np.clip(noise_cfg.get('effective_correlation', rho_corr), -0.999, 0.999))  # backward-compat fallback

    # Mirror detector behaviour: when the control channel is active, apply the
    # differential noise rejection. Otherwise remain single-ended.
    use_ctrl_for_noise = use_ctrl_flag and modulation not in ('MOSK',)

    if use_ctrl_for_noise:
        # Differential measurement: only correlated low-frequency noise is rejected.
        # Thermal (white) noise remains uncorrelated and is NOT cancelled by CTRL.
        rejection = max(0.0, 1.0 - rho_post)
        thermal_var_effective = 2.0 * thermal_var_single  # no CTRL cancellation for thermal
        flicker_var_effective = 2.0 * flicker_var_single * rejection
        drift_var_effective = 2.0 * drift_var_single * rejection
    else:
        # Single-ended measurement: no common-mode rejection
        thermal_var_effective = thermal_var_single
        flicker_var_effective = flicker_var_single
        drift_var_effective = drift_var_single

    total_effective_var = thermal_var_effective + flicker_var_effective + drift_var_effective
    sigma_effective = math.sqrt(max(total_effective_var, 0.0))

    # Increased floor slightly for robustness
    sigma = max(sigma_effective, 1e-15)

    rho_pre = float(np.clip(noise_cfg.get('rho_corr', rho_corr), -0.999, 0.999))
    rho_for_diff = rho_post if use_ctrl_for_noise else rho_pre
    sigma_diff_charge = math.sqrt(max(
        2.0 * total_single_var - 2.0 * rho_for_diff * (sigma_single ** 2),
        0.0
    ))

    if components_out is not None:
        components_out.clear()
        total_eff = max(total_effective_var, 0.0)
        components_out.update({
            'thermal_sigma': float(math.sqrt(max(thermal_var_effective, 0.0))),
            'flicker_sigma': float(math.sqrt(max(flicker_var_effective, 0.0))),
            'drift_sigma': float(math.sqrt(max(drift_var_effective, 0.0))),
            'thermal_fraction': float((thermal_var_effective / total_eff) if total_eff > 0 else 0.0),
            'flicker_fraction': float((flicker_var_effective / total_eff) if total_eff > 0 else 0.0),
            'drift_fraction': float((drift_var_effective / total_eff) if total_eff > 0 else 0.0),
            'total_sigma': float(sigma),
            'single_ended_sigma': float(max(sigma_single, 1e-15)),
            'use_ctrl_for_noise': float(1.0 if use_ctrl_for_noise else 0.0),
            'rho_used': float(rho_for_diff),
            'rho_pre_ctrl': float(rho_pre),
            'detection_window_s': float(T_int),
            'I_dc_used_A': float(I_dc),
            'V_g_bias_used_V': float(V_g_bias),
            'gm_S': float(gm),
            'C_tot_F': float(C_tot),
            'sigma_diff_charge': float(max(sigma_diff_charge, 1e-15)),
        })

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
    k_indices = np.arange(0, n_history, dtype=np.int32)
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


def _single_symbol_currents(
    s_tx: int,
    tx_history: List[Tuple[int, float]],
    cfg: Dict[str, Any],
    rng
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[Dict[str, np.ndarray]]]:
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
    t_vec = np.arange(0, n_samples, dtype=float) * dt
    
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
    
    profile = str(cfg['pipeline'].get('channel_profile', 'tri')).lower()
    sero_active = profile in ('dual', 'tri')
    ctrl_active = profile == 'tri'

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
                mean_amp = float(np.mean(np.arange(0, M, dtype=float) / (M - 1)))
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
            if sero_active:
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
                if sero_active:
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
            if sero_active:
                conc_sero = finite_burst_concentration(Nm_actual, distance_sero_m, t_vec, cfg, 'SERO')
                conc_at_sero_ch += conc_sero
                sym_sero += conc_sero
    
    # INDEPENDENT Binding simulation for each channel
    cfg_da = cfg.copy()
    cfg_da['N_apt'] = cfg.get('binding', {}).get('N_sites_da', cfg.get('N_apt'))
    bound_da_ch, _, _ = bernoulli_binding(conc_at_da_ch, 'DA', cfg_da, rng)

    if sero_active:
        cfg_sero = cfg.copy()
        cfg_sero['N_apt'] = cfg.get('binding', {}).get('N_sites_sero', cfg.get('N_apt'))
        bound_sero_ch, _, _ = bernoulli_binding(conc_at_sero_ch, 'SERO', cfg_sero, rng)
    else:
        bound_sero_ch = np.zeros(n_samples)

    if ctrl_active:
        cfg_ctrl = cfg.copy()
        cfg_ctrl['N_apt'] = 0  # No aptamer sites
        bound_ctrl_ch, _, _ = bernoulli_binding(conc_at_ctrl_ch, 'CTRL', cfg_ctrl, rng)
    else:
        bound_ctrl_ch = np.zeros(n_samples)

    # OECT current generation
    bound_sites_trio = np.vstack([bound_da_ch, bound_sero_ch, bound_ctrl_ch])
    collect_components = bool(cfg.get('pipeline', {}).get('_collect_noise_components', False))
    oect_result = oect_trio(
        bound_sites_trio,
        nts=("DA", "SERO", "CTRL"),
        cfg=cfg,
        rng=rng,
        return_components=collect_components
    )

    if collect_components:
        currents_dict_raw, component_traces = cast(Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]], oect_result)
    else:
        currents_dict_raw = cast(Dict[str, np.ndarray], oect_result)
        component_traces = None

    currents_dict = currents_dict_raw

    # === ISI METRIC (proxy) ==========================================
    # If requested, push an ISI-to-signal "energy" ratio proxy into cfg['_metrics'].
    # We compute it in concentration domain (integration over decision window);
    # this avoids any physics change and is sufficient for reporting/gating.
    if cfg.get('collect_isi_metrics', False):
        try:
            dt = cfg['sim']['dt_s']
            win_s = cfg['detection'].get('decision_window_s', cfg['pipeline']['symbol_period_s'])
            anchor_token = str(cfg['detection'].get('decision_window_anchor', 'start')).lower()
            tail_mode = anchor_token in ('tail', 'end')
            n_int = min(int(win_s / dt), n_samples)
            if n_int > 1:
                start = max(n_samples - n_int, 0) if tail_mode else 0
                end = start + n_int
                isi_mass = float(np.trapezoid(isi_da[start:end], dx=dt) + np.trapezoid(isi_sero[start:end], dx=dt))
                sym_mass = float(np.trapezoid(sym_da[start:end], dx=dt) + np.trapezoid(sym_sero[start:end], dx=dt))
                denom = max(sym_mass, 1e-30)
                ratio = isi_mass / denom
                metrics = cfg.setdefault('_metrics', {})
                lst = metrics.setdefault('isi_ratio_values', [])
                lst.append(ratio)
        except Exception:
            pass
    # ================================================================

    return currents_dict["DA"], currents_dict["SERO"], currents_dict["CTRL"], Nm_actual, component_traces


def run_sequence(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate <sequence_length> symbols and compute BER/SEP.
    Detection logic unchanged - the fixes are in signal generation.
    """
    
    profile = str(cfg.get('pipeline', {}).get('channel_profile', 'tri')).lower()
    if profile in ('single', 'dual'):
        cfg['pipeline']['use_control_channel'] = False

    # Guard against missing detection config
    cfg.setdefault('detection', {})

    pipeline_flags = cfg.get('pipeline', {})
    suppress_threshold_warnings = bool(pipeline_flags.get('_suppress_threshold_warnings', False))
    noise_only_mode = bool(pipeline_flags.get('_noise_only_run', False))
    if noise_only_mode:
        cfg['disable_progress'] = True

    collect_psd = bool(pipeline_flags.get('_collect_psd', False))
    
    mod = cfg['pipeline']['modulation']
    L = cfg['pipeline']['sequence_length']
    rng = default_rng(cfg['pipeline'].get('random_seed'))

    detector_mode = str(cfg['pipeline'].get('detector_mode', 'zscore')).lower()
    if detector_mode not in ('zscore', 'raw', 'whitened'):
        detector_mode = 'zscore'
    cfg['pipeline']['detector_mode'] = detector_mode

    tx_history: List[Tuple[int, float]] = []
    subsymbol_errors = {'mosk': 0, 'csk': 0}
    thresholds_used: Dict[str, Any] = {}
    mosk_correct = 0 
    
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
    stats_charge_da: List[float] = []
    stats_charge_sero: List[float] = []
    stats_current_da: List[float] = []
    stats_current_sero: List[float] = []
    stats_csk_levels: Optional[Dict[int, RunningStat]] = None
    stats_csk_levels_legacy: Optional[List[List[float]]] = None
    stats_hybrid_amp: Optional[Dict[Tuple[str, int], RunningStat]] = None
    stats_hybrid_amp_legacy: Optional[List[List[float]]] = None
    if mod == "CSK":
        level_count = int(cfg.get('pipeline', {}).get('csk_levels', 4))
        if level_count <= 0:
            level_count = 1
        stats_csk_levels = defaultdict(RunningStat)
        # Retain legacy list capture to keep backward-compatible payloads while new summaries mature
        stats_csk_levels_legacy = [[] for _ in range(level_count)]
    elif mod == "Hybrid":
        stats_hybrid_amp = defaultdict(RunningStat)
        stats_hybrid_amp_legacy = [[], []]
    constellation_points: List[Dict[str, float]] = []
    mosk_stats_raw: List[float] = []
    mosk_stats_z: List[float] = []
    mosk_stats_whitened: List[float] = []
    noise_charge_da_samples: List[float] = []
    noise_charge_sero_samples: List[float] = []
    noise_charge_diff_samples: List[float] = []
    noise_window_samples: List[float] = []

    component_names = ('thermal', 'flicker', 'drift')
    component_diff_samples: Dict[str, List[float]] = {}
    component_single_da_samples: Dict[str, List[float]] = {}
    component_single_sero_samples: Dict[str, List[float]] = {}
    if noise_only_mode:
        component_diff_samples = {name: [] for name in component_names}
        component_single_da_samples = {name: [] for name in component_names}
        component_single_sero_samples = {name: [] for name in component_names}

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
    noise_components: Dict[str, float] = {}
    sigma_da, sigma_sero = calculate_proper_noise_sigma(cfg, detection_window_s, components_out=noise_components)

    # ✨ NEW: also compute single-ended noise for MoSK statistic (even when mode='Hybrid')
    from copy import deepcopy
    cfg_mosk = deepcopy(cfg)
    cfg_mosk['pipeline']['modulation'] = 'MoSK'  # force single-ended sigmas
    cfg_mosk['pipeline']['use_control_channel'] = False
    sigma_da_mosk, sigma_sero_mosk = calculate_proper_noise_sigma(cfg_mosk, detection_window_s)

    # Enhanced: Mode-aware correlation selection
    noise_cfg = cfg.get('noise', {})
    use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))

    # Correlation to use in the *denominator*:
    # - MoSK: use pre-CTRL correlation (matches calibration)
    # - CSK/Hybrid: use post-CTRL correlation when CTRL subtraction is on; otherwise pre-CTRL
    rho_pre   = float(noise_cfg.get('rho_corr', noise_cfg.get('rho_correlated', 0.9)))
    
    # Enhancement 2: Use 0.5 fallback if noise model is symmetric
    # Conservative default: keep previous 0.0 behavior unless explicitly overridden
    rho_post_default = 0.0
    rho_post = float(noise_cfg.get('rho_between_channels_after_ctrl', rho_post_default))
    rho_post = float(cfg['pipeline'].get('rho_cc_measured', rho_post))
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

    frozen_noise = cfg['pipeline'].get('_frozen_noise')
    if isinstance(frozen_noise, dict):
        detection_window_s = float(frozen_noise.get('detection_window_s', detection_window_s))
        cfg.setdefault('detection', {})['decision_window_s'] = detection_window_s
        sigma_da = float(frozen_noise.get('sigma_da', sigma_da))
        sigma_sero = float(frozen_noise.get('sigma_sero', sigma_sero))
        sigma_diff = float(frozen_noise.get('sigma_diff', sigma_diff))
        sigma_diff_mosk = float(frozen_noise.get('sigma_diff_mosk', sigma_diff_mosk))
        rho_for_diff = float(frozen_noise.get('rho_for_diff', rho_for_diff))
        rho_override = frozen_noise.get('rho_cc')
        if rho_override is not None:
            rho_cc = float(rho_override)
        noise_components = dict(frozen_noise.get('noise_components', noise_components))

    # FIX B: Use measured ρ_post whenever available; fall back to config otherwise
    rho_cc_measured = cfg['pipeline'].get('rho_cc_measured')
    if rho_cc_measured is not None:
        rho_cc = float(rho_cc_measured)
    else:
        rho_cc = rho_post if use_ctrl else rho_pre
    rho_cc = max(-1.0, min(1.0, rho_cc))

    anchor_token_initial = str(cfg['detection'].get('decision_window_anchor', 'start')).lower()
    tail_mode_initial = anchor_token_initial in ('tail', 'end')
    subtract_ctrl_initial = (mod != "MoSK") and bool(cfg['pipeline'].get('use_control_channel', True))

    sigma_da_measured = float('nan')
    sigma_sero_measured = float('nan')
    sigma_diff_charge_measured = float('nan')
    sigma_single_measured = float('nan')
    rho_measured = float('nan')
    noise_sample_size = float('nan')
    psd_payload: Optional[Dict[str, Any]] = None

    # Process symbols
    for i, s_tx in enumerate(tqdm(tx_symbols, desc=f"Simulating {mod}", disable=cfg.get('disable_progress', False))):
        ig, ia, ic, Nm_realised, component_traces = _single_symbol_currents(s_tx, tx_history, cfg, rng)
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
        
        anchor_token = str(cfg['detection'].get('decision_window_anchor', 'start')).lower()
        tail_mode = anchor_token in ('tail', 'end')

        use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
        # For MoSK, do NOT subtract CTRL from the charges; for CSK/Hybrid keep the existing behavior.
        subtract_for_q = (mod != "MoSK") and use_ctrl
        
        # Calculate charges (mode-specific CTRL handling)
        sig_da = (ig - ic) if subtract_for_q else ig
        sig_sero = (ia - ic) if subtract_for_q else ia
        start_idx = max(n_total_samples - n_detect_samples, 0) if tail_mode else 0
        end_idx = n_total_samples if tail_mode else n_detect_samples
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = min(n_detect_samples, n_total_samples)

        if collect_psd and psd_payload is None and n_total_samples >= 2:
            freq = np.fft.rfftfreq(n_total_samples, d=dt)

            def _series_to_psd(series: np.ndarray) -> np.ndarray:
                series = np.asarray(series, dtype=float)
                if series.size < 2:
                    return np.zeros(freq.shape, dtype=float)
                series = series - np.mean(series)
                spectrum = np.fft.rfft(series)
                # Power spectral density (A^2/Hz) via periodogram
                psd = (np.abs(spectrum) ** 2) * dt / float(series.size)
                return psd

            psd_payload = {
                "frequency_hz": freq.tolist(),
                "total_diff": _series_to_psd(sig_da - sig_sero).tolist(),
                "total_da": _series_to_psd(sig_da).tolist(),
                "total_sero": _series_to_psd(sig_sero).tolist(),
            }

            if component_traces:
                for name, comp in component_traces.items():
                    if comp is None:
                        continue
                    comp_arr = np.asarray(comp)
                    if comp_arr.ndim != 2 or comp_arr.shape[0] < 2:
                        continue
                    comp_da = comp_arr[0]
                    comp_sero = comp_arr[1]
                    comp_ctrl = comp_arr[2] if comp_arr.shape[0] >= 3 else 0.0
                    comp_da_sig = comp_da - comp_ctrl if subtract_for_q else comp_da
                    comp_sero_sig = comp_sero - comp_ctrl if subtract_for_q else comp_sero
                    psd_payload[f"{name}_da"] = _series_to_psd(comp_da_sig).tolist()
                    psd_payload[f"{name}_sero"] = _series_to_psd(comp_sero_sig).tolist()
                    psd_payload[f"{name}_diff"] = _series_to_psd(comp_da_sig - comp_sero_sig).tolist()

        q_da = float(np.trapezoid(sig_da[start_idx:end_idx], dx=dt))
        q_sero = float(np.trapezoid(sig_sero[start_idx:end_idx], dx=dt))
        window_samples = max(end_idx - start_idx, 1)
        window_s = window_samples * dt
        mean_i_da = q_da / window_s if window_s > 0 else 0.0
        mean_i_sero = q_sero / window_s if window_s > 0 else 0.0

        if noise_only_mode:
            noise_charge_da_samples.append(q_da)
            noise_charge_sero_samples.append(q_sero)
            noise_charge_diff_samples.append(q_da - q_sero)
            noise_window_samples.append(window_s)

            if component_traces is not None:
                for name in component_names:
                    comp_opt = component_traces.get(name)
                    if comp_opt is None or len(comp_opt) < 3:
                        continue
                    comp = comp_opt
                    comp_da = comp[0]
                    comp_sero = comp[1]
                    comp_ctrl = comp[2]
                    comp_da_signal = comp_da - comp_ctrl if subtract_for_q else comp_da
                    comp_sero_signal = comp_sero - comp_ctrl if subtract_for_q else comp_sero
                    try:
                        comp_charge_da = float(np.trapezoid(comp_da_signal[start_idx:end_idx], dx=dt))
                        comp_charge_sero = float(np.trapezoid(comp_sero_signal[start_idx:end_idx], dx=dt))
                        comp_charge_da_single = float(np.trapezoid(comp_da[start_idx:end_idx], dx=dt))
                        comp_charge_sero_single = float(np.trapezoid(comp_sero[start_idx:end_idx], dx=dt))
                    except Exception:
                        continue
                    component_diff_samples.setdefault(name, []).append(comp_charge_da - comp_charge_sero)
                    component_single_da_samples.setdefault(name, []).append(comp_charge_da_single)
                    component_single_sero_samples.setdefault(name, []).append(comp_charge_sero_single)

        constellation_points.append({
            "symbol_index": float(i),
            "symbol_tx": float(s_tx),
            "q_da": q_da,
            "q_sero": q_sero,
            "use_ctrl": 1.0 if use_ctrl else 0.0,
            "subtract_ctrl": 1.0 if subtract_for_q else 0.0,
        })

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
            # Use the same statistic family employed during calibration (sign-aware difference).
            q_eff_da = get_nt_params(cfg, 'DA')['q_eff_e']
            q_eff_sero = get_nt_params(cfg, 'SERO')['q_eff_e']
            sign_da = 1.0 if q_eff_da >= 0 else -1.0
            sign_sero = 1.0 if q_eff_sero >= 0 else -1.0

            raw_stat = (sign_da * q_da) - (sign_sero * q_sero)
            sigma_norm = sigma_diff if sigma_diff > 1e-15 else 1e-15
            stat_zscore = raw_stat / sigma_norm
            # In 1-D the whitened statistic collapses to the z-score.
            stat_whitened = stat_zscore

            if detector_mode == 'raw':
                decision_stat = raw_stat
            elif detector_mode == 'whitened':
                decision_stat = stat_whitened
            else:
                decision_stat = stat_zscore

            mosk_stats_raw.append(raw_stat)
            mosk_stats_z.append(stat_zscore)
            mosk_stats_whitened.append(stat_whitened)

            # Honor comparator direction saved by calibration (default to '>')
            threshold = float(cfg['pipeline'].get('mosk_threshold', 0.0))
            if 'mosk_comparator' in cfg['pipeline']:
                comparator = str(cfg['pipeline']['mosk_comparator'])
            else:
                comparator = '>'
                if i == 0 and not suppress_threshold_warnings:
                    logger.warning(
                        "MoSK comparator flag missing; falling back to default ('>'). "
                        "Recalibrate thresholds to persist mosk_comparator."
                    )
            if comparator not in ('<', '>'):
                if i == 0 and not suppress_threshold_warnings:
                    logger.warning(f"MoSK comparator '{comparator}' unsupported; defaulting to '>'.")
                comparator = '>'
            if comparator == '>':
                s_rx = 0 if decision_stat > threshold else 1  # DA when stat exceeds threshold
            else:
                s_rx = 0 if decision_stat < threshold else 1
            rx_symbols[i] = s_rx
            
            # Guardrails to catch future regressions (log once per sequence)
            if i == 0 and not suppress_threshold_warnings:
                stat_magnitude = abs(decision_stat)
                charge_magnitude = max(abs(q_da), abs(q_sero))

                if detector_mode == 'raw':
                    ref_scale = max(charge_magnitude * 10.0, 1e-21)
                    threshold_ref_scale = ref_scale
                else:
                    ref_scale = 10.0  # Reasonable for dimensionless statistics
                    threshold_ref_scale = 10.0

                if stat_magnitude > ref_scale:
                    logger.warning(
                        f"MoSK decision statistic suspiciously large: {decision_stat:.2e} "
                        f"vs expected scale ~{ref_scale:.1f}. Check for normalization error."
                    )

                threshold_magnitude = abs(threshold)
                if threshold_magnitude > threshold_ref_scale:
                    logger.warning(
                        f"MoSK threshold suspiciously large: {threshold:.2e} "
                        f"vs expected scale ~{threshold_ref_scale:.1f}. Check for units."
                    )

                logger.info(
                    f"MoSK decision ({detector_mode}): stat={decision_stat:.2e}, "
                    f"thresh={threshold:.2e}, charges=[{q_da:.2e}, {q_sero:.2e}]"
                )

            if s_tx == 0:
                stats_da.append(decision_stat)
                stats_charge_da.append(q_da)
                stats_current_da.append(mean_i_da)
            else:
                stats_sero.append(decision_stat)
                stats_charge_sero.append(q_sero)
                stats_current_sero.append(mean_i_sero)
          
        elif mod == 'CSK':
            # Get CSK configuration parameters
            M = cfg['pipeline']['csk_levels']
            target_channel = cfg['pipeline'].get('csk_target_channel', 'DA')
            combiner_cfg = cfg['pipeline'].get('csk_combiner', 'zscore')
            combiner = cfg['pipeline'].get('csk_selected_combiner', combiner_cfg)  # Use selected if present
            use_dual = bool(cfg['pipeline'].get('csk_dual_channel', True))
            leakage = float(cfg['pipeline'].get('csk_leakage_frac', 0.0))

            use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))
            # CSK always uses CTRL subtraction when enabled
            sig_da = (ig - ic) if use_ctrl else ig
            sig_sero = (ia - ic) if use_ctrl else ia

            # Tail-gated trapezoid integration
            if tail_mode:
                csk_start = start_idx
                csk_end = end_idx
            else:
                tail = float(cfg['pipeline'].get('csk_tail_fraction', 1.0))
                tail = min(max(tail, 0.1), 1.0)
                csk_start = int((1.0 - tail) * n_detect_samples)
                csk_start = max(csk_start, 0)
                csk_end = n_detect_samples
            q_da = float(np.trapezoid(sig_da[csk_start:csk_end], dx=dt))
            q_sero = float(np.trapezoid(sig_sero[csk_start:csk_end], dx=dt))
            tail_samples = max(csk_end - csk_start, 0)
            tail_window_s = max(tail_samples * dt, dt)
            mean_i_da = q_da / tail_window_s if tail_window_s > 0 else 0.0
            mean_i_sero = q_sero / tail_window_s if tail_window_s > 0 else 0.0

            # Detector-mode aware statistics
            primary_is_da = target_channel.upper() == 'DA'
            primary_charge = q_da if primary_is_da else q_sero
            primary_current = mean_i_da if primary_is_da else mean_i_sero
            stat_raw = primary_charge

            if use_dual:
                stat_zscore = _csk_dual_channel_Q(
                    q_da=q_da, q_sero=q_sero,
                    sigma_da=sigma_da, sigma_sero=sigma_sero,
                    rho_cc=rho_cc, combiner=combiner, leakage_frac=leakage,
                    target=target_channel,
                    cfg=cfg
                )
                stat_whitened = _csk_dual_channel_Q(
                    q_da=q_da, q_sero=q_sero,
                    sigma_da=sigma_da, sigma_sero=sigma_sero,
                    rho_cc=rho_cc, combiner='whitened', leakage_frac=leakage,
                    target=target_channel,
                    cfg=cfg
                )
            else:
                sigma_primary = sigma_da if primary_is_da else sigma_sero
                sigma_primary = max(sigma_primary, 1e-15)
                stat_zscore = primary_charge / sigma_primary
                stat_whitened = stat_zscore

            stat_map = {
                'raw': stat_raw,
                'zscore': stat_zscore,
                'whitened': stat_whitened
            }
            decision_stat = stat_map.get(detector_mode, stat_zscore)

            # Get thresholds from configuration
            threshold_key = f'csk_thresholds_{target_channel.lower()}'
            thresholds = cfg['pipeline'].get(threshold_key, [])
            if not thresholds:
                logger.warning(f"CSK thresholds missing for {threshold_key}. Symbol detection will default to level 0.")

            # Use orientation measured during calibration; warn if missing and fallback to q_eff
            q_eff_target = get_nt_params(cfg, target_channel)['q_eff_e']
            if 'csk_thresholds_increasing' in cfg['pipeline']:
                increasing = bool(cfg['pipeline']['csk_thresholds_increasing'])
            else:
                increasing = True if q_eff_target > 0 else False
                if i == 0 and not suppress_threshold_warnings:
                    logger.warning(
                        "CSK orientation flag missing; falling back to q_eff sign. "
                        "Recalibrate thresholds to persist csk_thresholds_increasing."
                    )

            # CSK symbol detection
            s_rx = 0
            # Use the orientation determined during calibration
            for thresh in thresholds:
                if increasing:
                    if decision_stat > thresh:
                        s_rx += 1
                    else:
                        break
                else:
                    if decision_stat < thresh:
                        s_rx += 1
                    else:
                        break
            rx_symbols[i] = s_rx

            # Store decision statistics for analysis
            if s_tx < M/2:
                stats_da.append(decision_stat)
                stats_charge_da.append(primary_charge)
                stats_current_da.append(primary_current)
            else:
                stats_sero.append(decision_stat)
                stats_charge_sero.append(primary_charge)
                stats_current_sero.append(primary_current)

            true_level = int(s_tx)
            if stats_csk_levels is not None:
                stats_csk_levels[true_level].update(float(decision_stat))
            if stats_csk_levels_legacy is not None:
                try:
                    stats_csk_levels_legacy[true_level].append(float(decision_stat))
                except (ValueError, IndexError):
                    pass

            # Log configuration on first symbol (for debugging)
            if i == 0:
                cb = cfg['pipeline'].get('csk_selected_combiner', combiner)
                logger.info(
                    f"CSK combiner={cb}, dual={use_dual}, mode={detector_mode}, "
                    f"sigma=[{sigma_da:.2e},{sigma_sero:.2e}], rho_cc={rho_cc:+.2f}, "
                    f"increasing={increasing}, leakage={leakage:.2f}, stat={decision_stat:.2e}"
                )

        elif mod == 'Hybrid':
            # Align MoSK integration window with calibration (tail-gated)
            if tail_mode:
                raw_start = start_idx
                raw_end = end_idx
            else:
                tail = float(cfg['pipeline'].get('csk_tail_fraction', 1.0))
                tail = min(max(tail, 0.1), 1.0)
                raw_start = int((1.0 - tail) * n_detect_samples)
                raw_start = max(raw_start, 0)
                raw_end = n_detect_samples

            q_da_raw = float(np.trapezoid(ig[raw_start:raw_end], dx=dt))
            q_sero_raw = float(np.trapezoid(ia[raw_start:raw_end], dx=dt))
            tail_samples = max(raw_end - raw_start, 0)
            tail_window_s = max(tail_samples * dt, dt)
            mean_i_da_raw = q_da_raw / tail_window_s if tail_window_s > 0 else 0.0
            mean_i_sero_raw = q_sero_raw / tail_window_s if tail_window_s > 0 else 0.0

            # Use MoSK decision statistic with proper single-ended noise
            sign_da = 1.0 if q_eff_da >= 0 else -1.0
            sign_sero = 1.0 if q_eff_sero >= 0 else -1.0
            raw_stat_mosk = (sign_da * q_da_raw) - (sign_sero * q_sero_raw)
            sigma_norm_mosk = sigma_diff_mosk if sigma_diff_mosk > 1e-15 else 1e-15
            zscore_mosk = raw_stat_mosk / sigma_norm_mosk
            whitened_mosk = zscore_mosk

            if detector_mode == 'raw':
                decision_stat_mosk = raw_stat_mosk
            elif detector_mode == 'whitened':
                decision_stat_mosk = whitened_mosk
            else:
                decision_stat_mosk = zscore_mosk

            threshold_mosk = float(cfg['pipeline'].get('mosk_threshold', 0.0))
            comparator = str(cfg['pipeline'].get('mosk_comparator', '>'))
            if comparator == '>':
                b_hat = 0 if (decision_stat_mosk > threshold_mosk) else 1
            else:
                b_hat = 0 if (decision_stat_mosk < threshold_mosk) else 1

            # Recompute amplitude charges with tail-gated window and CTRL treatment
            use_ctrl = bool(cfg['pipeline'].get('use_control_channel', True))

            # Tail-gated window (same as used for MoSK statistic above)
            sig_da_amp = (ig - ic) if use_ctrl else ig
            sig_sero_amp = (ia - ic) if use_ctrl else ia
            q_da_amp = float(np.trapezoid(sig_da_amp[raw_start:raw_end], dx=dt))
            q_sero_amp = float(np.trapezoid(sig_sero_amp[raw_start:raw_end], dx=dt))
            mean_i_da_amp = q_da_amp / tail_window_s if tail_window_s > 0 else 0.0
            mean_i_sero_amp = q_sero_amp / tail_window_s if tail_window_s > 0 else 0.0

            # Use the dual-channel combiner for the amplitude bit (like CSK)
            combiner = str(cfg['pipeline'].get('hybrid_combiner', cfg['pipeline'].get('csk_combiner', 'zscore')))
            leakage = float(cfg['pipeline'].get('hybrid_leakage_frac', cfg['pipeline'].get('csk_leakage_frac', 0.0)))
            target = 'DA' if b_hat == 0 else 'SERO'
            stat_raw_amp = q_da_amp if target == 'DA' else q_sero_amp
            stat_zscore_amp = _csk_dual_channel_Q(
                q_da=q_da_amp, q_sero=q_sero_amp,
                sigma_da=sigma_da, sigma_sero=sigma_sero,
                rho_cc=rho_cc,
                combiner=combiner, leakage_frac=leakage,
                target=target, cfg=cfg
            )
            stat_whitened_amp = _csk_dual_channel_Q(
                q_da=q_da_amp, q_sero=q_sero_amp,
                sigma_da=sigma_da, sigma_sero=sigma_sero,
                rho_cc=rho_cc,
                combiner='whitened', leakage_frac=leakage,
                target=target, cfg=cfg
            )

            stat_map_amp = {
                'raw': stat_raw_amp,
                'zscore': stat_zscore_amp,
                'whitened': stat_whitened_amp
            }
            decision_stat_amp = stat_map_amp.get(detector_mode, stat_zscore_amp)

            # Use calibrated thresholds on the amplitude statistic with data-driven orientation
            if b_hat == 0:
                tau = float(cfg['pipeline'].get('hybrid_threshold_da', 0.0))
                inc_key = 'hybrid_threshold_da_increasing'
            else:
                tau = float(cfg['pipeline'].get('hybrid_threshold_sero', 0.0))
                inc_key = 'hybrid_threshold_sero_increasing'

            if inc_key in cfg['pipeline']:
                inc = bool(cfg['pipeline'][inc_key])
            else:
                inc = True
                if i == 0 and not suppress_threshold_warnings:
                    logger.warning(
                        "Hybrid amplitude orientation flag missing; falling back to default (increasing). "
                        "Recalibrate thresholds to persist hybrid_threshold_*_increasing."
                    )
            l_hat = 1 if ((decision_stat_amp > tau) if inc else (decision_stat_amp < tau)) else 0

            # Construct final symbol
            s_rx = (b_hat << 1) | l_hat
            rx_symbols[i] = s_rx

            # For SNR diagnostics: store the MoSK decision statistic by TRUE molecule class (not by decision)
            true_mol_bit = (s_tx >> 1)
            if true_mol_bit == 0:
                stats_da.append(decision_stat_mosk)
                stats_charge_da.append(q_da_raw)
                stats_current_da.append(mean_i_da_raw)
            else:
                stats_sero.append(decision_stat_mosk)
                stats_charge_sero.append(q_sero_raw)
                stats_current_sero.append(mean_i_sero_raw)

            true_mol_bit = int(s_tx >> 1)
            true_amp_bit = int(s_tx & 1)
            if stats_hybrid_amp is not None:
                mol_label = 'DA' if true_mol_bit == 0 else 'SERO'
                stats_hybrid_amp[(mol_label, true_amp_bit)].update(float(decision_stat_amp))
            if stats_hybrid_amp_legacy is not None:
                try:
                    stats_hybrid_amp_legacy[true_amp_bit].append(float(decision_stat_amp))
                except (ValueError, IndexError):
                    pass
            # Enhanced subsymbol error tracking:
            if b_hat != true_mol_bit:
                subsymbol_errors['mosk'] += 1
            else:
                mosk_correct += 1                # Count symbols that pass MoSK and expose CSK
                if l_hat != true_amp_bit:
                    subsymbol_errors['csk'] += 1

    if noise_only_mode:
        arr_da = np.asarray(noise_charge_da_samples, dtype=float)
        arr_sero = np.asarray(noise_charge_sero_samples, dtype=float)
        arr_diff = np.asarray(noise_charge_diff_samples, dtype=float)
        arr_win = np.asarray(noise_window_samples, dtype=float)

        def _finite_std_local(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            if finite.size >= 2:
                return float(np.std(finite, ddof=1))
            return float('nan')

        sigma_da_measured = _finite_std_local(arr_da)
        sigma_sero_measured = _finite_std_local(arr_sero)
        sigma_diff_charge_measured = _finite_std_local(arr_diff)
        if np.isfinite(sigma_da_measured) and np.isfinite(sigma_sero_measured):
            sigma_single_measured = float(math.sqrt(max((sigma_da_measured ** 2 + sigma_sero_measured ** 2) / 2.0, 0.0)))
        else:
            sigma_single_measured = float('nan')
        finite_mask = np.isfinite(arr_da) & np.isfinite(arr_sero)
        if finite_mask.sum() >= 2:
            try:
                rho_measured = float(np.corrcoef(arr_da[finite_mask], arr_sero[finite_mask])[0, 1])
            except (FloatingPointError, ValueError):
                rho_measured = float('nan')
        else:
            rho_measured = float('nan')
        noise_sample_size = float(finite_mask.sum())
        mean_window = float(np.nanmedian(arr_win)) if arr_win.size else float('nan')
        if np.isfinite(mean_window) and mean_window > 0.0 and np.isfinite(sigma_diff_charge_measured):
            sigma_diff_current_measured = sigma_diff_charge_measured / mean_window
        else:
            sigma_diff_current_measured = float('nan')

        component_sigma_diff: Dict[str, float] = {}
        component_sigma_single: Dict[str, float] = {}
        for name in component_names:
            diff_list = component_diff_samples.get(name, [])
            da_list = component_single_da_samples.get(name, [])
            sero_list = component_single_sero_samples.get(name, [])
            diff_arr = np.asarray(diff_list, dtype=float)
            da_arr = np.asarray(da_list, dtype=float)
            sero_arr = np.asarray(sero_list, dtype=float)
            component_sigma_diff[name] = _finite_std_local(diff_arr)
            std_da = _finite_std_local(da_arr)
            std_sero = _finite_std_local(sero_arr)
            if np.isfinite(std_da) and np.isfinite(std_sero):
                component_sigma_single[name] = float(math.sqrt(max((std_da ** 2 + std_sero ** 2) / 2.0, 0.0)))
            else:
                component_sigma_single[name] = float('nan')

    else:
        sigma_diff_current_measured = float('nan')
        component_sigma_diff = {}
        component_sigma_single = {}

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
        "stats_charge_da": stats_charge_da,
        "stats_charge_sero": stats_charge_sero,
        "stats_current_da": stats_current_da,
        "stats_current_sero": stats_current_sero,
        "stats_csk_levels": stats_csk_levels_legacy if stats_csk_levels_legacy is not None else [],
        "csk_level_stats": {int(level): stat.as_summary() for level, stat in (stats_csk_levels or {}).items()},
        "stats_hybrid_amp": stats_hybrid_amp_legacy if stats_hybrid_amp_legacy is not None else [],
        "hybrid_amp_stats": {
            f"{mol}_{amp}": stat.as_summary()
            for (mol, amp), stat in (stats_hybrid_amp or {}).items()
        },
        "constellation_points": constellation_points,
        "mosk_stats_raw": mosk_stats_raw,
        "mosk_stats_zscore": mosk_stats_z,
        "mosk_stats_whitened": mosk_stats_whitened,
        "subsymbol_errors": subsymbol_errors,
        "thresholds_used": thresholds_used,
        "isi_ratio_mean": isi_ratio_mean,
        "isi_ratio_median": isi_ratio_median,
        "detector_mode": detector_mode,
        # Stage 14: reproducibility knobs
        'noise_sigma_da': float(sigma_da),
        'noise_sigma_sero': float(sigma_sero),
        'noise_sigma_I_diff': float(sigma_diff),
        'noise_sigma_thermal': float(noise_components.get('thermal_sigma', float('nan'))),
        'noise_sigma_flicker': float(noise_components.get('flicker_sigma', float('nan'))),
        'noise_sigma_drift': float(noise_components.get('drift_sigma', float('nan'))),
        'noise_thermal_fraction': float(noise_components.get('thermal_fraction', float('nan'))),
        'noise_sigma_diff_charge': float(noise_components.get('sigma_diff_charge', float('nan'))),
        'noise_sigma_single': float(noise_components.get('single_ended_sigma', float('nan'))),
        'rho_pre_ctrl': float(noise_components.get('rho_pre_ctrl', float('nan'))),
        'I_dc_used_A': float(noise_components.get('I_dc_used_A', float('nan'))),
        'V_g_bias_V_used': float(noise_components.get('V_g_bias_used_V', float('nan'))),
        'gm_S': float(noise_components.get('gm_S', float('nan'))),
        'C_tot_F': float(noise_components.get('C_tot_F', float('nan'))),
        'detection_window_s': float(detection_window_s),
        'burst_shape': str(cfg.get('burst_shape', 'rect')),
        'T_release_ms': float(cfg.get('T_release_ms', 0.0)),
        'noise_sigma_da_measured': float('nan'),
        'noise_sigma_sero_measured': float('nan'),
        'noise_sigma_I_diff_measured': float('nan'),
        'noise_sigma_diff_charge_measured': float('nan'),
        'noise_sigma_single_measured': float('nan'),
        'noise_sigma_thermal_measured': float('nan'),
        'noise_sigma_flicker_measured': float('nan'),
        'noise_sigma_drift_measured': float('nan'),
        'noise_thermal_fraction_measured': float('nan'),
        'noise_flicker_fraction_measured': float('nan'),
        'noise_drift_fraction_measured': float('nan'),
        'noise_sigma_thermal_single_measured': float('nan'),
        'noise_sigma_flicker_single_measured': float('nan'),
        'noise_sigma_drift_single_measured': float('nan'),
        'noise_ctrl_reduction_fraction_thermal': float('nan'),
        'noise_ctrl_reduction_fraction_flicker': float('nan'),
        'noise_ctrl_reduction_fraction_drift': float('nan'),
        'noise_ctrl_reduction_fraction_mean': float('nan'),
        'noise_rho_measured': float('nan'),
        'noise_sigma_sample_size': float('nan'),
    }

    # Populate component defaults from noise_components if available (e.g., frozen noise payloads)
    if isinstance(noise_components, dict):
        val = noise_components.get('thermal_sigma_single')
        if val is not None:
            result['noise_sigma_thermal_single_measured'] = float(val)
        val = noise_components.get('flicker_sigma_single')
        if val is not None:
            result['noise_sigma_flicker_single_measured'] = float(val)
        val = noise_components.get('drift_sigma_single')
        if val is not None:
            result['noise_sigma_drift_single_measured'] = float(val)
        val = noise_components.get('ctrl_reduction_fraction_mean')
        if val is not None:
            result['noise_ctrl_reduction_fraction_mean'] = float(val)
        val = noise_components.get('ctrl_reduction_fraction_thermal')
        if val is not None:
            result['noise_ctrl_reduction_fraction_thermal'] = float(val)
        val = noise_components.get('ctrl_reduction_fraction_flicker')
        if val is not None:
            result['noise_ctrl_reduction_fraction_flicker'] = float(val)
        val = noise_components.get('ctrl_reduction_fraction_drift')
        if val is not None:
            result['noise_ctrl_reduction_fraction_drift'] = float(val)

    if psd_payload is not None:
        result['psd_payload'] = psd_payload

    if noise_only_mode:
        result['SER'] = float('nan')
        result['ser'] = float('nan')
        result['errors'] = 0
        result['noise_only'] = True
        if np.isfinite(sigma_da_measured):
            result['noise_sigma_da'] = float(sigma_da_measured)
            result['noise_sigma_da_measured'] = float(sigma_da_measured)
        if np.isfinite(sigma_sero_measured):
            result['noise_sigma_sero'] = float(sigma_sero_measured)
            result['noise_sigma_sero_measured'] = float(sigma_sero_measured)
        if np.isfinite(sigma_diff_charge_measured):
            result['noise_sigma_I_diff'] = float(sigma_diff_charge_measured)
            result['noise_sigma_diff_charge'] = float(sigma_diff_charge_measured)
            result['noise_sigma_I_diff_measured'] = float(sigma_diff_charge_measured)
            result['noise_sigma_diff_charge_measured'] = float(sigma_diff_charge_measured)
        if np.isfinite(sigma_single_measured):
            result['noise_sigma_single'] = float(sigma_single_measured)
            result['noise_sigma_single_measured'] = float(sigma_single_measured)
        sigma_thermal_diff = component_sigma_diff.get('thermal', float('nan'))
        sigma_flicker_diff = component_sigma_diff.get('flicker', float('nan'))
        sigma_drift_diff = component_sigma_diff.get('drift', float('nan'))

        result['noise_sigma_thermal'] = float(sigma_thermal_diff)
        result['noise_sigma_flicker'] = float(sigma_flicker_diff)
        result['noise_sigma_drift'] = float(sigma_drift_diff)
        result['noise_sigma_thermal_measured'] = float(sigma_thermal_diff)
        result['noise_sigma_flicker_measured'] = float(sigma_flicker_diff)
        result['noise_sigma_drift_measured'] = float(sigma_drift_diff)

        total_var_diff = (sigma_diff_charge_measured ** 2) if np.isfinite(sigma_diff_charge_measured) else float('nan')
        thermal_var = (sigma_thermal_diff ** 2) if np.isfinite(sigma_thermal_diff) else float('nan')
        flicker_var = (sigma_flicker_diff ** 2) if np.isfinite(sigma_flicker_diff) else float('nan')
        drift_var = (sigma_drift_diff ** 2) if np.isfinite(sigma_drift_diff) else float('nan')

        def _fraction(component_var: float) -> float:
            if not np.isfinite(component_var) or not np.isfinite(total_var_diff) or total_var_diff <= 0.0:
                return float('nan')
            return float(component_var / total_var_diff)

        result['noise_thermal_fraction'] = _fraction(thermal_var)
        result['noise_flicker_fraction'] = _fraction(flicker_var)
        result['noise_drift_fraction'] = _fraction(drift_var)
        result['noise_thermal_fraction_measured'] = result['noise_thermal_fraction']
        result['noise_flicker_fraction_measured'] = result['noise_flicker_fraction']
        result['noise_drift_fraction_measured'] = result['noise_drift_fraction']

        # Capture single-ended component sigmas and CTRL reduction where applicable
        single_thermal = component_sigma_single.get('thermal', float('nan'))
        single_flicker = component_sigma_single.get('flicker', float('nan'))
        single_drift = component_sigma_single.get('drift', float('nan'))

        result['noise_sigma_thermal_single_measured'] = float(single_thermal)
        result['noise_sigma_flicker_single_measured'] = float(single_flicker)
        result['noise_sigma_drift_single_measured'] = float(single_drift)

        def _ctrl_reduction(single_sigma: float, diff_sigma: float) -> float:
            if not np.isfinite(single_sigma) or single_sigma <= 0.0 or not np.isfinite(diff_sigma):
                return float('nan')
            return float(1.0 - (diff_sigma / single_sigma))

        thermal_reduction = _ctrl_reduction(single_thermal, sigma_thermal_diff)
        flicker_reduction = _ctrl_reduction(single_flicker, sigma_flicker_diff)
        drift_reduction = _ctrl_reduction(single_drift, sigma_drift_diff)

        result['noise_ctrl_reduction_fraction_thermal'] = thermal_reduction
        result['noise_ctrl_reduction_fraction_flicker'] = flicker_reduction
        result['noise_ctrl_reduction_fraction_drift'] = drift_reduction
        if np.isfinite(thermal_reduction) or np.isfinite(flicker_reduction) or np.isfinite(drift_reduction):
            reductions = [r for r in (thermal_reduction, flicker_reduction, drift_reduction) if np.isfinite(r)]
            if reductions:
                result['noise_ctrl_reduction_fraction_mean'] = float(np.mean(reductions))
            else:
                result['noise_ctrl_reduction_fraction_mean'] = float('nan')
        else:
            result['noise_ctrl_reduction_fraction_mean'] = float('nan')
        result['noise_rho_measured'] = float(rho_measured)
        result['noise_sigma_sample_size'] = float(noise_sample_size)
        result['noise_sigma_diff_current_measured'] = float(sigma_diff_current_measured)
        components_dict = result.get('noise_components')
        if isinstance(components_dict, dict):
            components_dict.update({
                'sigma_diff_charge': float(sigma_diff_charge_measured),
                'thermal_sigma': float(sigma_thermal_diff),
                'flicker_sigma': float(sigma_flicker_diff),
                'drift_sigma': float(sigma_drift_diff),
                'thermal_fraction': float(result['noise_thermal_fraction']),
                'flicker_fraction': float(result['noise_flicker_fraction']),
                'drift_fraction': float(result['noise_drift_fraction']),
                'thermal_sigma_single': float(single_thermal),
                'flicker_sigma_single': float(single_flicker),
                'drift_sigma_single': float(single_drift),
                'ctrl_reduction_fraction_thermal': float(thermal_reduction),
                'ctrl_reduction_fraction_flicker': float(flicker_reduction),
                'ctrl_reduction_fraction_drift': float(drift_reduction),
            })

    freeze_active = bool(cfg['pipeline'].get('_freeze_calibration_active', False))
    if freeze_active:
        result['calibration_frozen'] = True
        result['calibration_baseline_id'] = str(cfg['pipeline'].get('_freeze_baseline_id', ''))
    else:
        result['calibration_frozen'] = False
        result['calibration_baseline_id'] = ''

    # Add MoSK sigma metadata for Hybrid mode
    if mod == 'Hybrid':
        mosk_errors = int(subsymbol_errors.get('mosk', 0))
        csk_errors = int(subsymbol_errors.get('csk', 0))
        mosk_ser = (mosk_errors / L) if L > 0 else float('nan')
        csk_ser = (csk_errors / L) if L > 0 else float('nan')
        mosk_exposure_frac = (mosk_correct / L) if L > 0 else 0.0
        conditional_csk_ser = (csk_errors / mosk_correct) if mosk_correct > 0 else float('nan')
        M_levels = int(cfg['pipeline'].get('csk_levels', 4))
        bits_per_symbol_csk_branch = float(math.log2(max(M_levels, 2)))
        correct_csk = max(mosk_correct - csk_errors, 0)
        bits_csk_realized = (correct_csk / L * bits_per_symbol_csk_branch) if L > 0 else 0.0
        bits_mosk_realized = (1.0 - mosk_ser) if L > 0 else float('nan')
        if L == 0:
            bits_csk_realized = float('nan')

        result['mosk_sigma_diff_used'] = float(sigma_diff_mosk)
        if detector_mode == 'raw':
            result['mosk_stat_units'] = 'raw_charge'
        elif detector_mode == 'whitened':
            result['mosk_stat_units'] = 'whitened_sigma_diff'
        else:
            result['mosk_stat_units'] = 'normalized_sigma_diff'
        result['n_mosk_correct'] = int(mosk_correct)
        result['mosk_ser'] = float(mosk_ser)
        result['csk_ser'] = float(csk_ser)
        result['mosk_exposure_frac'] = float(mosk_exposure_frac)
        result['conditional_csk_ser'] = float(conditional_csk_ser) if not math.isnan(conditional_csk_ser) else float('nan')
        result['hybrid_bits_per_symbol_mosk'] = float(bits_mosk_realized) if not math.isnan(bits_mosk_realized) else float('nan')
        result['hybrid_bits_per_symbol_csk'] = float(bits_csk_realized) if not math.isnan(bits_csk_realized) else float('nan')
        bits_total_realized = bits_mosk_realized + bits_csk_realized
        if math.isnan(bits_mosk_realized) or math.isnan(bits_csk_realized):
            bits_total_realized = float('nan')
        result['hybrid_bits_per_symbol_total'] = float(bits_total_realized)

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
