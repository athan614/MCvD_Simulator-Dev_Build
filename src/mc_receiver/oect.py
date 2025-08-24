"""
OECT transduction and noise model for tri-channel biosensor.

VECTORIZED VERSION: Optimized noise generation and signal processing
with batch operations and efficient FFT-based methods.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple, Any
from scipy import signal    #type: ignore
from ..constants import get_nt_params, ELEMENTARY_CHARGE, BOLTZMANN


def _cholesky_matrix(rho: float) -> np.ndarray:
    """Return 3×3 lower-triangular L s.t. L @ L.T = Σ. (Unchanged)"""
    return np.linalg.cholesky(np.array([[1.0, rho, rho],
                                        [rho, 1.0, rho],
                                        [rho, rho, 1.0]]))


def _generate_correlated_triplet(base_vec: np.ndarray,
                                rho: float,
                                rng: np.random.Generator) -> np.ndarray:
    """
    FULLY VECTORIZED: Produce (3,n) traces with correlation using efficient FFT operations.
    Process all 3 channels simultaneously for better performance.
    """
    n = base_vec.size
    L = _cholesky_matrix(rho)

    # Get magnitude template from base vector
    env_mag = np.abs(np.fft.rfft(base_vec))
    
    # VECTORIZED: Generate all white noise at once
    white_noise = rng.normal(size=(3, n))
    
    # FULLY VECTORIZED: Apply FFT to all channels at once
    fft_white_all = np.fft.rfft(white_noise, axis=1)
    
    # Apply magnitude envelope while preserving phase
    shaped_fft_all = env_mag[np.newaxis, :] * np.exp(1j * np.angle(fft_white_all))
    
    # Transform back to time domain
    X = np.fft.irfft(shaped_fft_all, n=n, axis=1)
    
    # Apply correlation matrix
    return L @ X


def _generate_correlated_triplet_batch(base_vecs: np.ndarray,
                                      rho: float,
                                      rng: np.random.Generator) -> np.ndarray:
    """
    FULLY VECTORIZED: Generate multiple correlated triplets efficiently.
    Process all batches and channels simultaneously.
    
    Parameters
    ----------
    base_vecs : np.ndarray
        Shape (n_batch, n_time) base vectors
    rho : float
        Correlation coefficient
    rng : np.random.Generator
        Random generator
        
    Returns
    -------
    np.ndarray
        Shape (n_batch, 3, n_time) correlated traces
    """
    n_batch, n_time = base_vecs.shape
    L = _cholesky_matrix(rho)
    
    # Pre-allocate result
    result = np.zeros((n_batch, 3, n_time))
    
    # FULLY VECTORIZED: Process all batches at once if memory allows
    if n_batch * n_time < 1e6:  # If total size is reasonable
        # Get magnitude templates for all batches
        env_mags = np.abs(np.fft.rfft(base_vecs, axis=1))
        
        # Generate all white noise at once
        white_noise = rng.normal(size=(n_batch, 3, n_time))
        
        # FFT all channels and batches
        fft_white_all = np.fft.rfft(white_noise, axis=2)
        
        # Apply magnitude envelopes
        shaped_fft_all = env_mags[:, np.newaxis, :] * np.exp(1j * np.angle(fft_white_all))
        
        # Transform back
        X_all = np.fft.irfft(shaped_fft_all, n=n_time, axis=2)
        
        # Apply correlation matrix to each batch
        for i in range(n_batch):
            result[i] = L @ X_all[i]
    else:
        # Fall back to batch-wise processing for large arrays
        for i in range(n_batch):
            result[i] = _generate_correlated_triplet(base_vecs[i], rho, rng)
    
    return result


def oect_trio(bound_sites_trio: np.ndarray,
              nts: Tuple[str, str, str],
              cfg: Dict[str, Any],
              rng: np.random.Generator,
              rho: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Return correlated drain-current traces for GLU, GABA, CTRL.
    
    VECTORIZED: Optimized noise generation using batch FFT operations.
    """
    if rho is None:
        rho = cfg["noise"].get("rho_correlated", 0.9)

    assert rho is not None

    n = bound_sites_trio.shape[1]
    fs = 1 / cfg["sim"]["dt_s"]
    
    # VECTORIZED: Generate frequency vector once
    freqs = np.fft.rfftfreq(n, 1/fs)
    freqs[0] = freqs[1]  # avoid zero
    
    # Get parameters from nested config
    alpha_H = cfg["noise"]["alpha_H"]
    K_d = cfg["noise"]["K_d_Hz"]
    N_c = cfg["noise"]["N_c"]
    T = cfg["sim"]["temperature_K"]
    R_ch = cfg["oect"]["R_ch_Ohm"]
    gm = cfg["oect"]["gm_S"]
    C_tot = cfg["oect"]["C_tot_F"]
    V_g_bias = cfg.get('V_g_bias_V', -0.2)
    I_DC = gm * abs(V_g_bias)
    I_DC = max(I_DC, 1e-6)

    # VECTORIZED: Calculate PSD envelopes
    N_c = float(cfg["noise"]["N_c"]) if isinstance(cfg["noise"]["N_c"], str) else cfg["noise"]["N_c"]
    K_f = alpha_H / N_c
    psd_flick = K_f * I_DC**2 / freqs
    psd_drift = K_d * I_DC**2 / freqs**2
    
    if cfg.get('deterministic_mode', False):
        # Deterministic mode: No noise
        thermal = np.zeros((3, n))
        flick_corr = np.zeros((3, n))
        drift_corr = np.zeros((3, n))
    else:
        # VECTORIZED: Generate base noise spectra
        # Create complex noise for both flicker and drift at once
        noise_shape = (2, freqs.size)  # 2 for flicker and drift
        complex_noise = rng.normal(size=noise_shape) + 1j * rng.normal(size=noise_shape)
        
        # Apply PSD scaling
        flick_fft = complex_noise[0] * np.sqrt(psd_flick * fs / 2) / np.sqrt(n)
        drift_fft = complex_noise[1] * np.sqrt(psd_drift * fs / 2) / np.sqrt(n)
        
        # Transform to time domain
        flick_base = np.fft.irfft(flick_fft, n=n)
        drift_base = np.fft.irfft(drift_fft, n=n)
        
        # Generate correlated versions
        flick_corr = _generate_correlated_triplet(flick_base, rho, rng)
        drift_corr = _generate_correlated_triplet(drift_base, rho, rng)
        
        # VECTORIZED: Thermal noise generation
        B_det = cfg.get('detection_bandwidth_Hz', 100)
        psd_th = 4 * BOLTZMANN * T / R_ch
        effective_B = min(B_det, fs / 2)
        thermal_scale = np.sqrt(psd_th * effective_B)
        thermal = rng.normal(scale=thermal_scale, size=(3, n))

    # VECTORIZED: Signal current calculation (let q_eff handle sign; no forced -)
    q_eff_array = np.array([cfg['neurotransmitters'][nt]['q_eff_e'] for nt in nts])  # e.g., -1.0 for GLU, +1.0 for GABA
    signal = gm * q_eff_array[:, np.newaxis] * ELEMENTARY_CHARGE * bound_sites_trio / C_tot  # q_eff dictates sign
    
    # Total current
    total = signal + thermal + flick_corr + drift_corr
    
    # Pre-allocate result dictionary for better performance
    result = {}
    for i, nt in enumerate(nts):
        result[nt] = total[i]
    
    return result


def oect_current(
    bound_sites_t: np.ndarray,
    nt: str,
    cfg: Dict[str, Any],
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Convert bound aptamer trajectory to OECT drain current with noise.
    
    VECTORIZED: Optimized FFT-based noise generation.
    """
    # Set random seed if provided
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Get parameters
    nt_params = get_nt_params(cfg, nt)
    q_eff = get_nt_params(cfg, nt)["q_eff_e"] if nt != "CTRL" else 0.0
    
    # ✅ FIXED: Device parameters - support both nested and flat config
    oect = cfg.get('oect', cfg)  # Use nested if available, fallback to flat
    gm = oect['gm_S']
    C_tot = oect['C_tot_F']
    R_ch = oect['R_ch_Ohm']
    alpha_H = oect.get('alpha_H', cfg.get('alpha_H'))  # Handle noise params too
    N_c = oect.get('N_c', cfg.get('N_c'))
    K_d = oect.get('K_d_Hz', cfg.get('K_d_Hz'))
    T = cfg.get('temperature_K', 310)  # ✅ Add missing temperature
    
    # Time parameters
    dt = cfg['dt_s']
    fs = 1 / dt
    n_samples = len(bound_sites_t)
    duration = n_samples * dt
    
    # VECTORIZED: Signal current calculation
    i_signal = gm * q_eff * ELEMENTARY_CHARGE * bound_sites_t / C_tot
    
    # DC current for noise calculations
    V_g_bias = cfg.get('V_g_bias_V', -0.2)
    I_DC = gm * abs(V_g_bias)
    I_DC = max(I_DC, 1e-6)
    
    # VECTORIZED: Frequency vector generation
    freqs = np.fft.rfftfreq(n_samples, dt)
    freqs[0] = 1 / duration
    n_freqs = len(freqs)
    
    # VECTORIZED: Generate all noise components at once
    if cfg.get('deterministic_mode', False):
        # No noise in deterministic mode
        i_thermal = np.zeros(n_samples)
        i_flicker = np.zeros(n_samples)
        i_drift = np.zeros(n_samples)
    else:
        # Generate complex noise for all components
        complex_noise = rng.normal(size=(3, n_freqs)) + 1j * rng.normal(size=(3, n_freqs))
        
        # 1. Thermal noise
        B_det = cfg.get('detection_bandwidth_Hz', 100)
        psd_thermal = 4 * BOLTZMANN * T / R_ch
        effective_B = min(B_det, fs / 2)
        thermal_scale = np.sqrt(psd_thermal * effective_B / 2)
        i_thermal = np.fft.irfft(complex_noise[0] * thermal_scale, n=n_samples)
        
        # 2. Flicker noise
        K_f = alpha_H / N_c
        psd_flicker = K_f * I_DC**2 / freqs
        flicker_scale = np.sqrt(psd_flicker * fs / 2) / np.sqrt(n_samples)
        i_flicker = np.fft.irfft(complex_noise[1] * flicker_scale, n=n_samples)
        
        # 3. Drift noise
        psd_drift = K_d * I_DC**2 / (freqs**2)
        drift_scale = np.sqrt(psd_drift * fs / 2) / np.sqrt(n_samples)
        i_drift = np.fft.irfft(complex_noise[2] * drift_scale, n=n_samples)
    
    # Total current
    i_total = i_signal + i_thermal + i_flicker + i_drift
    
    return {
        'signal': i_signal,
        'thermal': i_thermal,
        'flicker': i_flicker,
        'drift': i_drift,
        'total': i_total
    }


def oect_current_batch(
    bound_sites_batch: np.ndarray,
    nt: str,
    cfg: Dict[str, Any],
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    FULLY VECTORIZED: Process multiple bound site trajectories in batch.
    All noise generation and signal processing done in parallel.
    """
    n_batch, n_time = bound_sites_batch.shape
    
    # Set random seed if provided
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Get parameters once for all batches
    nt_params = get_nt_params(cfg, nt)
    q_eff = nt_params["q_eff_e"] if nt != "CTRL" else 0.0
    
    # ✅ FIXED: Device parameters - support both nested and flat config
    oect = cfg.get('oect', cfg)  # Use nested if available, fallback to flat
    gm = oect['gm_S']
    C_tot = oect['C_tot_F']
    R_ch = oect['R_ch_Ohm']
    alpha_H = oect.get('alpha_H', cfg.get('alpha_H'))  # Handle noise params too
    N_c = oect.get('N_c', cfg.get('N_c'))
    K_d = oect.get('K_d_Hz', cfg.get('K_d_Hz'))
    T = cfg.get('temperature_K', 310)
    
    # Time parameters
    dt = cfg['dt_s']
    fs = 1 / dt
    duration = n_time * dt
    
    # FULLY VECTORIZED: Signal current for all batches at once
    i_signal_batch = gm * q_eff * ELEMENTARY_CHARGE * bound_sites_batch / C_tot
    
    # Check for deterministic mode
    if cfg.get('deterministic_mode', False):
        return {
            'signal': i_signal_batch,
            'thermal': np.zeros_like(i_signal_batch),
            'flicker': np.zeros_like(i_signal_batch),
            'drift': np.zeros_like(i_signal_batch),
            'total': i_signal_batch
        }
    
    # DC current for noise calculations (use mean of each batch)
    V_g_bias = cfg.get('V_g_bias_V', -0.2)
    I_DC = gm * abs(V_g_bias)
    I_DC = max(I_DC, 1e-6)
    
    # Frequency vector for single-sided spectrum
    freqs = np.fft.rfftfreq(n_time, dt)
    freqs[0] = 1 / duration
    n_freqs = len(freqs)
    
    # FULLY VECTORIZED: Generate noise for all batches at once
    # Shape: (n_batch, 3 noise types, n_freqs)
    complex_noise_all = rng.normal(size=(n_batch, 3, n_freqs)) + 1j * rng.normal(size=(n_batch, 3, n_freqs))
    
    # 1. Thermal noise (white) - same PSD for all batches
    B_det = cfg.get('detection_bandwidth_Hz', 100)
    psd_thermal = 4 * BOLTZMANN * T / R_ch
    effective_B = min(B_det, fs / 2)
    thermal_scale = np.sqrt(psd_thermal * effective_B / 2)
    
    # Apply thermal scaling and transform for all batches
    thermal_fft_batch = complex_noise_all[:, 0, :] * thermal_scale
    i_thermal_batch = np.fft.irfft(thermal_fft_batch, n=n_time, axis=1)
    
    # 2. Flicker (1/f) noise - same PSD shape for all batches
    K_f = alpha_H / N_c
    psd_flicker = K_f * I_DC**2 / freqs
    flicker_scale = np.sqrt(psd_flicker * fs / 2) / np.sqrt(n_time)
    
    # Apply flicker scaling and transform for all batches
    flicker_fft_batch = complex_noise_all[:, 1, :] * flicker_scale[np.newaxis, :]
    i_flicker_batch = np.fft.irfft(flicker_fft_batch, n=n_time, axis=1)
    
    # 3. Drift (1/f²) noise - same PSD shape for all batches
    psd_drift = K_d * I_DC**2 / (freqs**2)
    drift_scale = np.sqrt(psd_drift * fs / 2) / np.sqrt(n_time)
    
    # Apply drift scaling and transform for all batches
    drift_fft_batch = complex_noise_all[:, 2, :] * drift_scale[np.newaxis, :]
    i_drift_batch = np.fft.irfft(drift_fft_batch, n=n_time, axis=1)
    
    # Total current for all batches
    i_total_batch = i_signal_batch + i_thermal_batch + i_flicker_batch + i_drift_batch
    
    return {
        'signal': i_signal_batch,
        'thermal': i_thermal_batch,
        'flicker': i_flicker_batch,
        'drift': i_drift_batch,
        'total': i_total_batch
    }


def oect_static_gain(N_b: float, nt: str, cfg: Dict[str, Any]) -> float:
    """Calculate drain current change for deterministic bound sites. (Unchanged)"""
    nt_params = cfg['neurotransmitters'][nt]
    q_eff = nt_params['q_eff_e']
    gm = cfg['oect']['gm_S']
    C_tot = cfg['oect']['C_tot_F']
    
    delta_I = gm * q_eff * ELEMENTARY_CHARGE * N_b / C_tot * -1.0
    
    return delta_I


def oect_impulse_response(
    dt_s: float,
    n_samples: int,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Discrete-time impulse response of the OECT.
    
    VECTORIZED: Already uses numpy operations efficiently.
    """
    tau = cfg['oect']['tau_OECT_s']
    t = np.arange(n_samples) * dt_s
    return (1.0 / tau) * np.exp(-t / tau)


def differential_channels(
    i_glu: np.ndarray,
    i_gaba: np.ndarray,
    i_ctrl: np.ndarray,
    rho: float
) -> Dict[str, Any]:
    """
    Compute differential currents for tri-channel architecture.
    
    VECTORIZED: Already uses numpy operations efficiently.
    """
    # Vectorized differential calculation
    diff_glu = i_glu - i_ctrl
    diff_gaba = i_gaba - i_ctrl
    
    # Vectorized correlation calculation
    current_matrix = np.vstack([i_glu, i_gaba, i_ctrl])
    corr_matrix = np.corrcoef(current_matrix)
    rho_glu_ctrl = corr_matrix[0, 2]
    rho_gaba_ctrl = corr_matrix[1, 2]
    rho_achieved = np.mean([rho_glu_ctrl, rho_gaba_ctrl])
    
    return {
        'diff_glu': diff_glu,
        'diff_gaba': diff_gaba,
        'rho_achieved': rho_achieved
    }


def generate_correlated_noise(
    n_samples: int,
    rho: float,
    n_channels: int = 3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate correlated Gaussian noise for multiple channels.
    
    VECTORIZED: Optimized matrix operations.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # VECTORIZED: Generate all random numbers at once
    # Common mode component
    n_common = rng.normal(0, 1, n_samples)
    # Independent components
    n_independent = rng.normal(0, 1, (n_channels, n_samples))
    
    # Mix using broadcasting
    correlated_noise = np.sqrt(rho) * n_common + np.sqrt(1 - rho) * n_independent
    
    return correlated_noise


def calculate_noise_metrics(
    current_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate noise metrics from current components.
    
    VECTORIZED: Already uses numpy operations efficiently.
    """
    # Vectorized RMS calculations
    signal_rms = np.sqrt(np.mean(current_dict['signal']**2))
    thermal_rms = np.std(current_dict['thermal'])
    flicker_rms = np.std(current_dict['flicker'])
    drift_rms = np.std(current_dict['drift'])
    
    # Total noise (vectorized)
    noise_total = current_dict['total'] - current_dict['signal']
    total_noise_rms = np.std(noise_total)
    
    # SNR calculation
    snr_db = 20 * np.log10(signal_rms / total_noise_rms) if total_noise_rms > 0 else np.inf
    
    return {
        'signal_rms_A': float(signal_rms),
        'thermal_rms_A': float(thermal_rms),
        'flicker_rms_A': float(flicker_rms),
        'drift_rms_A': float(drift_rms),
        'total_noise_rms_A': float(total_noise_rms),
        'snr_db': float(snr_db)
    }


def calculate_noise_metrics_batch(
    current_dict_batch: Dict[str, np.ndarray],
    cfg: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    VECTORIZED: Calculate noise metrics for multiple trajectories.
    
    Parameters
    ----------
    current_dict_batch : dict
        Dictionary with arrays of shape (n_batch, n_time)
    cfg : dict
        Configuration
        
    Returns
    -------
    dict
        Dictionary with arrays of shape (n_batch,) for each metric
    """
    n_batch = current_dict_batch['signal'].shape[0]
    
    # Pre-allocate results
    results = {
        'signal_rms_A': np.zeros(n_batch),
        'thermal_rms_A': np.zeros(n_batch),
        'flicker_rms_A': np.zeros(n_batch),
        'drift_rms_A': np.zeros(n_batch),
        'total_noise_rms_A': np.zeros(n_batch),
        'snr_db': np.zeros(n_batch)
    }
    
    # Vectorized calculations across batch dimension
    results['signal_rms_A'] = np.sqrt(np.mean(current_dict_batch['signal']**2, axis=1))
    results['thermal_rms_A'] = np.std(current_dict_batch['thermal'], axis=1)
    results['flicker_rms_A'] = np.std(current_dict_batch['flicker'], axis=1)
    results['drift_rms_A'] = np.std(current_dict_batch['drift'], axis=1)
    
    # Total noise
    noise_total = current_dict_batch['total'] - current_dict_batch['signal']
    results['total_noise_rms_A'] = np.std(noise_total, axis=1)
    
    # SNR (handle division by zero)
    with np.errstate(divide='ignore'):
        results['snr_db'] = 20 * np.log10(
            results['signal_rms_A'] / results['total_noise_rms_A']
        )
    results['snr_db'][results['total_noise_rms_A'] == 0] = np.inf
    
    return results


def rms_in_band(x: np.ndarray, fs: float, f_cut: float = 5.0) -> float:
    """RMS after 4th-order Bessel LP. (Unchanged - already efficient)"""
    from scipy.signal import bessel, sosfiltfilt    #type: ignore
    sos = bessel(4, f_cut/(fs/2), btype="low", output="sos")
    x_filt = sosfiltfilt(sos, x)
    return np.sqrt(np.mean(x_filt**2))


def default_params():
    """Returns default OECT parameters. (Unchanged)"""
    return {
        'gm_S': 0.002,
        'C_tot_F': 18e-9,
        'R_ch_Ohm': 200,
        'alpha_H': 3.0e-3,
        'N_c': 3.0e14,
        'K_d_Hz': 1.3e-4,
    }