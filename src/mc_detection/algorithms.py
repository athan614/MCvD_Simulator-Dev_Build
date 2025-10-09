# src/detection.py (updated with patches applied)
"""
Symbol-level detection and BER/SER evaluation for the tri-channel OECT receiver.

Implements detection algorithms from Section II-I of the manuscript:
- MoSK detection (Eq. 39-41)
- M-ary CSK detection (Eq. 42-46)
- SNR calculations (Eq. 48)
- Monte Carlo and analytical performance evaluation
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
from scipy.special import erfc # type: ignore[import]
from scipy.stats import norm # type: ignore[import]


# ------------------------------------------------------------------
#  Helper - integrate drain current over decision window
# ------------------------------------------------------------------
def integrate_current(i_t: np.ndarray, dt: float, win_s: float) -> float:
    """
    Integrate current over decision window to get charge.
    
    Parameters
    ----------
    i_t : np.ndarray
        Current time series [A]
    dt : float
        Time step [s]
    win_s : float
        Integration window [s]
        
    Returns
    -------
    float
        Integrated charge [C]
    """
    n_int = min(int(win_s / dt), len(i_t))
    return dt * np.sum(i_t[:n_int])


# ------------------------------------------------------------------
#  MoSK detector (DA vs SERO) - Implements Eq. (39)
# ------------------------------------------------------------------
def detect_mosk(
    i_da: np.ndarray,
    i_sero: np.ndarray,
    i_ctrl: np.ndarray,
    cfg: Dict[str, Any],
    use_differential: Optional[bool] = None,
    sigma_diff: Optional[float] = None,
    detector_mode: Optional[str] = None,
) -> int:
    """
    MoSK detector aligned with the end-to-end pipeline statistics.

    When ``use_differential`` is left as ``None`` it mirrors the simulator’s
    convention: MoSK statistics remain single-ended even if the control channel
    is enabled, while other modes default to CTRL-differential subtraction.
    """
    detection_cfg = cfg.get('detection', {})
    sim_cfg = cfg.get('sim', {})
    pipeline_cfg = cfg.get('pipeline', {})

    win = float(detection_cfg.get('decision_window_s', pipeline_cfg.get('symbol_period_s', 0.0)))
    dt = float(sim_cfg.get('dt_s', 1.0))

    detector_mode = detector_mode or str(pipeline_cfg.get('detector_mode', 'zscore')).lower()
    if detector_mode not in ('raw', 'zscore', 'whitened'):
        detector_mode = 'zscore'

    if use_differential is None:
        pipeline_mod = str(pipeline_cfg.get('modulation', 'MoSK')).upper()
        use_ctrl = bool(pipeline_cfg.get('use_control_channel', True))
        # Align with end-to-end pipeline: MoSK contrast stays single-ended.
        use_differential = use_ctrl and pipeline_mod != 'MOSK'

    if use_differential:
        i_da_use = i_da - i_ctrl
        i_sero_use = i_sero - i_ctrl
    else:
        i_da_use = i_da
        i_sero_use = i_sero

    n_int = min(int(win / dt), len(i_da_use))
    if n_int <= 0:
        return 1

    q_da = float(np.trapezoid(i_da_use[:n_int], dx=dt))
    q_sero = float(np.trapezoid(i_sero_use[:n_int], dx=dt))

    from ..constants import get_nt_params

    q_eff_da = get_nt_params(cfg, 'DA')['q_eff_e']
    q_eff_sero = get_nt_params(cfg, 'SERO')['q_eff_e']
    sign_da = 1.0 if q_eff_da >= 0 else -1.0
    sign_sero = 1.0 if q_eff_sero >= 0 else -1.0

    raw_stat = (sign_da * q_da) - (sign_sero * q_sero)

    if sigma_diff is None:
        diff_samples = (sign_da * i_da_use[:n_int]) - (sign_sero * i_sero_use[:n_int])
        sigma_est = float(np.std(diff_samples, ddof=1) * np.sqrt(dt))
        sigma_diff = max(sigma_est, 1e-15)

    sigma_norm = max(float(sigma_diff), 1e-15)
    stat_zscore = raw_stat / sigma_norm
    stat_whitened = stat_zscore

    if detector_mode == 'raw':
        decision_stat = raw_stat
    elif detector_mode == 'whitened':
        decision_stat = stat_whitened
    else:
        decision_stat = stat_zscore

    return int(decision_stat < 0)


# ------------------------------------------------------------------
#  ML threshold calculation for CSK - Implements Eq. (44)
# ------------------------------------------------------------------
def calculate_ml_threshold(mu_0: float, mu_1: float, 
                          sigma_0: float, sigma_1: float) -> float:
    """
    Calculate maximum likelihood threshold for binary detection.
    
    Implements Eq. (44) from the manuscript for non-equal noise variances.
    
    Parameters
    ----------
    mu_0, mu_1 : float
        Mean currents for symbols 0 and 1 [A]
    sigma_0, sigma_1 : float
        Noise standard deviations for symbols 0 and 1 [A]
        
    Returns
    -------
    float
        ML decision threshold [A]
    """
    if abs(sigma_0 - sigma_1) / max(sigma_0, sigma_1) < 1e-3:    # Treat noises as equal only if the *relative* difference is tiny
        # Equal noise case: threshold is midpoint
        return (mu_0 + mu_1) / 2
    else:
        # Unequal noise case: use full formula
        delta_sigma2 = sigma_0**2 - sigma_1**2
        
        # Quadratic formula coefficients
        a = 1
        b = -2 * (mu_1*sigma_0**2 - mu_0*sigma_1**2) / delta_sigma2
        c = ((mu_1**2 * sigma_0**2 - mu_0**2 * sigma_1**2) / delta_sigma2 -
             2 * sigma_0**2 * sigma_1**2 * np.log(sigma_1 / sigma_0) / delta_sigma2)
        
        # Choose root between mu_0 and mu_1
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            # Fallback to midpoint if no real solution
            return (mu_0 + mu_1) / 2
        
        root1 = (-b + np.sqrt(discriminant)) / (2*a)
        root2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Select root between the means
        if min(mu_0, mu_1) <= root1 <= max(mu_0, mu_1):
            return root1
        else:
            return root2


# ------------------------------------------------------------------
#  Binary CSK detector with ML threshold
# ------------------------------------------------------------------
def detect_csk_binary(i_ch: np.ndarray,
                     i_ctrl: np.ndarray,
                     threshold: float,
                     cfg: Dict[str, Any]) -> int:
    """
    Binary CSK detection with specified threshold.
    
    Parameters
    ----------
    i_ch : np.ndarray
        Channel current [A]
    i_ctrl : np.ndarray
        Control channel current [A]
    threshold : float
        Decision threshold [A]
    cfg : dict
        Configuration
        
    Returns
    -------
    int
        0 for low concentration, 1 for high concentration
    """
    win = cfg['detection']['decision_window_s']
    dt = cfg['sim']['dt_s']
    
    # Differential measurement
    i_diff = i_ch - i_ctrl
    
    # Average current over decision window
    q_avg = integrate_current(i_diff, dt, win) / win
    
    return int(q_avg < threshold)   # Flip to < if negative signals mean higher levels (depletion mode)


# ------------------------------------------------------------------
#  M-ary CSK detector - Implements Eq. (42)
# ------------------------------------------------------------------
def detect_csk_mary(i_ch: np.ndarray,
                   i_ctrl: np.ndarray,
                   thresholds: List[float],
                   cfg: Dict[str, Any]) -> int:
    """
    M-ary CSK detection with multiple thresholds.
    
    For M symbols, uses M-1 thresholds to divide the decision space.
    
    Parameters
    ----------
    i_ch : np.ndarray
        Channel current [A]
    i_ctrl : np.ndarray
        Control channel current [A]
    thresholds : List[float]
        M-1 ordered thresholds [A]
    cfg : dict
        Configuration
        
    Returns
    -------
    int
        Detected symbol (0 to M-1)
    """
    win = cfg['detection']['decision_window_s']
    dt = cfg['sim']['dt_s']
    
    # Differential measurement
    i_diff = i_ch - i_ctrl
    
    # Average current over decision window (signed—no abs; handles negative depletion currents)
    q_avg = integrate_current(i_diff, dt, win) / win    # Signed value (e.g., more negative = higher level if depletion)
    
    # Find symbol by comparing to thresholds (calibrated on signed stats)
    symbol = 0
    for threshold in thresholds:
        if q_avg < threshold:   # Flip to < if negative signals mean higher levels (depletion mode)
            symbol += 1
        else:
            break
    
    return symbol


# ------------------------------------------------------------------
#  Analytical BER/SEP calculations
# ------------------------------------------------------------------
def ber_mosk_analytic(snr_da: Union[float, np.ndarray], 
                     snr_sero: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Analytical BER for MoSK using Eq. (40).
    
    Pe = Q(|μ_I,diff,Glu - μ_I,diff,SERO| / sqrt(σ²_I,diff,Glu + σ²_I,diff,SERO))
    
    For equal noise powers and symmetric signals, this simplifies to standard form.
    
    Parameters
    ----------
    snr_da : float or array
        SNR for datamate channel (linear scale)
    snr_sero : float or array
        SNR for SERO channel (linear scale)
        
    Returns
    -------
    float or array
        Bit error rate
    """
    # For symmetric case with equal noise
    # SNR_effective = (SNR_DA + SNR_SERO) / 2
    snr_eff = (snr_da + snr_sero) / 2
    
    # Q-function = 0.5 * erfc(x/sqrt(2))
    return 0.5 * erfc(np.sqrt(snr_eff) / np.sqrt(2))


def sep_csk_binary(snr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Symbol error probability for binary CSK.
    
    Parameters
    ----------
    snr : float or array
        Signal-to-noise ratio (linear scale)
        
    Returns
    -------
    float or array
        Symbol error probability
    """
    return 0.5 * erfc(np.sqrt(snr) / np.sqrt(2))


# In src/mc_detection/algorithms.py
def sep_csk_mary(
    snr: Union[float, np.ndarray],
    M: int,
    level_scheme: str = 'linear'
) -> Union[float, np.ndarray]:
    """
    Symbol error probability for M-ary CSK using Eq. (46).

    NOTE: The analytical formula is only valid for 'linear' level spacing.
    
    Parameters
    ----------
    snr : float or array
        Signal-to-noise ratio (linear scale)
    M : int
        Number of symbols
    level_scheme : str, optional
        The spacing scheme, by default 'linear'. If not 'linear',
        returns NaN as the formula does not apply.
        
    Returns
    -------
    float or array
        Symbol error probability
    """
    if level_scheme != 'linear':
        # The analytical formula only applies to equally spaced levels.
        # Return NaN for other schemes to avoid incorrect results.
        if isinstance(snr, np.ndarray):
            return np.full_like(snr, np.nan)
        return np.nan

    return (M - 1) / M * erfc(np.sqrt(snr / (2 * (M**2 - 1))))


# ------------------------------------------------------------------
#  SNR calculation - Implements Eq. (48)
# ------------------------------------------------------------------
def calculate_snr(mu_signal: float,
                 mu_reference: float,
                 noise_variance: float,
                 gm: float,
                 q_eff: float,
                 C_tot: float,
                 N_e: int = 1) -> float:
    """
    Calculate SNR from bound aptamer counts using Eq. (48).
    
    SNR = (gm * (μ_NB,m - μ_NB,0) * q_eff * e * N_e / C_tot)² / σ²_I,diff
    
    Parameters
    ----------
    mu_signal : float
        Mean bound aptamers for signal symbol
    mu_reference : float
        Mean bound aptamers for reference symbol
    noise_variance : float
        Total noise variance after differential measurement [A²]
    gm : float
        Transconductance [S]
    q_eff : float
        Effective charge [elementary charges]
    C_tot : float
        Total capacitance [F]
    N_e : int
        Number of electrons per charge (typically 1)
        
    Returns
    -------
    float
        Signal-to-noise ratio (linear scale)
    """
    from ..constants import ELEMENTARY_CHARGE # type: ignore[import]
    
    # Signal amplitude
    delta_I = gm * (mu_signal - mu_reference) * q_eff * ELEMENTARY_CHARGE * N_e / C_tot
    
    # SNR = signal_power / noise_power
    snr = (delta_I ** 2) / noise_variance
    
    return float(snr)


# ------------------------------------------------------------------
#  Performance evaluation utilities
# ------------------------------------------------------------------
def snr_sweep(cfg: Dict[str, Any], 
              modulation: str = 'MoSK') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate analytical BER/SEP curves over SNR range.
    
    Parameters
    ----------
    cfg : dict
        Configuration with 'detection' section
    modulation : str
        'MoSK' or 'CSK-M' where M is the number of levels
        
    Returns
    -------
    snr_dB : np.ndarray
        SNR values in dB
    error_rate : np.ndarray
        BER or SEP values
    """
    # Get SNR range from config
    snr_range = cfg['detection'].get('snr_dB_range', [-10, 20])
    snr_dB = np.linspace(snr_range[0], snr_range[1], 61)
    snr_lin = 10**(snr_dB / 10)
    
    if modulation == 'MoSK':
        # Assume symmetric channels
        error_rate = ber_mosk_analytic(snr_lin, snr_lin)
    elif modulation.startswith('CSK'):
        # Extract M from 'CSK-M'
        if '-' in modulation:
            M = int(modulation.split('-')[1])
        else:
            M = 2  # Default binary
        
        if M == 2:
            error_rate = sep_csk_binary(snr_lin)
        else:
            error_rate = sep_csk_mary(snr_lin, M)
    else:
        raise ValueError(f"Unknown modulation: {modulation}")
    
    # Ensure error_rate is always an array
    error_rate = np.asarray(error_rate)
    
    return snr_dB, error_rate


def calculate_data_rate(symbol_period_s: float, 
                       modulation: str,
                       error_rate: float) -> Dict[str, float]:
    """
    Calculate achievable data rate for given modulation and error rate.
    
    Parameters
    ----------
    symbol_period_s : float
        Symbol period Ts [s]
    modulation : str
        Modulation scheme
    error_rate : float
        BER or SEP
        
    Returns
    -------
    dict
        Data rate metrics
    """
    # Symbol rate
    Rs = 1 / symbol_period_s
    
    # Bits per symbol
    if modulation == 'MoSK':
        bits_per_symbol = 1  # Binary
    elif modulation.startswith('CSK'):
        M = int(modulation.split('-')[1]) if '-' in modulation else 2
        bits_per_symbol = np.log2(M)
    else:
        bits_per_symbol = 1
    
    # Raw bit rate
    Rb = Rs * bits_per_symbol
    
    # Effective rate accounting for errors (approximate)
    Rb_eff = Rb * (1 - error_rate)
    
    return {
        'symbol_rate_Hz': Rs,
        'bits_per_symbol': bits_per_symbol,
        'bit_rate_bps': Rb,
        'effective_rate_bps': Rb_eff
    }


# ------------------------------------------------------------------
#  Monte Carlo simulation framework
# ------------------------------------------------------------------
def monte_carlo_detection(
    cfg: Dict[str, Any],
    generate_signal_fn,
    n_trials: int = 1000,
    modulation: str = 'MoSK',
    symbols: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Monte Carlo simulation for detection performance.
    
    This is a framework that needs to be integrated with the full
    signal generation pipeline (diffusion -> binding -> OECT).
    
    Parameters
    ----------
    cfg : dict
        Configuration
    generate_signal_fn : callable
        Function that generates channel currents for a given symbol
    n_trials : int
        Number of Monte Carlo trials
    modulation : str
        Modulation scheme
    symbols : List[int], optional
        Sequence of transmitted symbols (generated randomly if None)
        
    Returns
    -------
    dict
        Performance metrics including error rates and confusion matrix
    """
    # This is a placeholder for the full pipeline integration
    # Will be implemented when integrating all modules
    raise NotImplementedError(
        "Full Monte Carlo pipeline requires integration with "
        "diffusion, binding, and OECT modules"
    )
    
    
def default_params():
    return {
        'N_apt': 4e8,
        'DA': {'k_on_M_s': 5e4, 'k_off_s': 1.5, 'q_eff_e': 0.6},
        'SERO': {'k_on_M_s': 3e4, 'k_off_s': 0.9, 'q_eff_e': 0.2},
    }


# Module exports
__all__ = [
    # Detectors
    'detect_mosk',
    'detect_csk_binary',
    'detect_csk_mary',
    # Threshold calculation
    'calculate_ml_threshold',
    # Analytical performance
    'ber_mosk_analytic',
    'sep_csk_binary', 
    'sep_csk_mary',
    # SNR and metrics
    'calculate_snr',
    'calculate_data_rate',
    # Evaluation
    'snr_sweep',
    'monte_carlo_detection'
]
