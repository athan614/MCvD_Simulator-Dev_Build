"""
OECT transduction and noise model for tri-channel biosensor.

This module implements the electrical transduction layer described in Section II-F and II-G
of the manuscript, converting bound aptamer counts to drain currents with realistic noise
characteristics including thermal, flicker (1/f), and drift components.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple, Any
from scipy import signal # type: ignore[import]
from ..constants import get_nt_params, ELEMENTARY_CHARGE, BOLTZMANN
from typing import Tuple

def _cholesky_matrix(rho: float) -> np.ndarray:
    """Return 3×3 lower-triangular L s.t. L @ L.T = Σ."""
    return np.linalg.cholesky(np.array([[1.0, rho, rho],
                                        [rho, 1.0, rho],
                                        [rho, rho, 1.0]]))

def _generate_correlated_triplet(base_vec: np.ndarray,
                                 rho: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Produce (3,n) traces:
        • each row has the same power-spectral envelope as `base_vec`
        • pair-wise Pearson correlation ≈ `rho`
    Strategy
    --------
    1.  Generate three iid white-Gaussian sequences.
    2.  Impose |FFT| of `base_vec` on each row (keeps PSD shape).
    3.  Apply Cholesky factor of Σ to inject correlation.
    """
    n = base_vec.size
    L = _cholesky_matrix(rho)

    # 1 & 2 – iid Gaussian  →  mould spectrum
    env_mag = np.abs(np.fft.rfft(base_vec))          # magnitude template
    X = np.empty((3, n))
    for k in range(3):
        fft_white = np.fft.rfft(rng.normal(size=n))
        X[k] = np.fft.irfft(env_mag * np.exp(1j * np.angle(fft_white)), n=n)

    # 3 – inject correlation
    return L @ X

def oect_trio(bound_sites_trio: np.ndarray,
              nts: Tuple[str, str, str],
              cfg: Dict[str, Any],
              rng: np.random.Generator,
              rho: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Return correlated drain-current traces for GLU, GABA, CTRL.
    Correlation matrix: [[1,ρ,ρ],[ρ,1,ρ],[ρ,ρ,1]]
    
    Parameters
    ----------
    bound_sites_trio : (3,n) int array
    nts : ("GLU","GABA","CTRL")
    cfg : nested dict with noise.*, oect.*, sim.*
    rng : random generator
    rho : correlation coefficient (default from config)
    
    Returns
    -------
    dict : {nt: current_array} for each channel
    """
    
    if rho is None:
        rho = cfg["noise"].get("rho_correlated", 0.9)

    # Now rho is guaranteed to be float, not None
    assert rho is not None  # Type checker hint

    n = bound_sites_trio.shape[1]
    fs = 1 / cfg["sim"]["dt_s"]
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
    # FIXED: Realistic baseline from bias (literature: 10-100 μA)
    V_g_bias = cfg.get('V_g_bias_V', -0.2)  # Paper/literature value
    I_DC = gm * abs(V_g_bias)  # Baseline current (positive for noise calc)
    I_DC = max(I_DC, 1e-6)  # Floor at 1 μA to avoid zero
    #I_DC = cfg["oect"].get("I_dc_A", gm * cfg["oect"].get("V_g_bias_V", 1e-3))

    # PSD envelopes
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
        # Generate base noise
        flick_fft = rng.normal(size=freqs.size) + 1j*rng.normal(size=freqs.size)
        flick_fft *= np.sqrt(psd_flick * fs /2) / np.sqrt(n)  # FIXED: Correct scaling
        drift_fft = rng.normal(size=freqs.size) + 1j*rng.normal(size=freqs.size)
        drift_fft *= np.sqrt(psd_drift * fs /2) / np.sqrt(n)  # FIXED: Correct scaling

        flick_base = np.fft.irfft(flick_fft, n=n)
        drift_base = np.fft.irfft(drift_fft, n=n)

        # Correlate the noise
        flick_corr = _generate_correlated_triplet(flick_base, rho, rng)
        drift_corr = _generate_correlated_triplet(drift_base, rho, rng)

        # Thermal noise (uncorrelated)
        B_det = cfg.get('detection_bandwidth_Hz', 100)  # Detection bandwidth
        psd_th = 4 * BOLTZMANN * T / R_ch
        effective_B = min(B_det, fs / 2)  # FIXED: Cap at Nyquist
        thermal = rng.normal(scale=np.sqrt(psd_th * effective_B), size=(3, n))

    # Signal current - USE RAW q_eff VALUES
    q_eff_tbl = {nt: cfg['neurotransmitters'][nt]['q_eff_e'] for nt in nts}
    signal = np.empty((3, n))
    boost_factor = 100.0  # Temp multiplier to boost signal magnitude for testing (adjust or remove later)
    signal = np.empty((3, n))
    for k, nt in enumerate(nts):
        signal[k] = -gm * ELEMENTARY_CHARGE * q_eff_tbl[nt] * bound_sites_trio[k] / C_tot  # Force negative, no boost
    #print(f"Mean signal for GLU: {np.mean(signal[0]):.3e} A")  # Debug, expect negative

    total = signal + thermal + flick_corr + drift_corr
    return {nt: total[i] for i, nt in enumerate(nts)}

def oect_current(
    bound_sites_t: np.ndarray,
    nt: str,
    cfg: Dict[str, Any],
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Convert bound aptamer trajectory to OECT drain current with noise.
    
    Implements the complete signal transduction chain including:
    - Signal current from bound aptamers
    - Thermal (Johnson-Nyquist) noise
    - Flicker (1/f) noise
    - Drift (1/f²) noise
    
    Parameters
    ----------
    bound_sites_t : np.ndarray
        Time series of bound aptamer counts (integer)
    nt : str
        Neurotransmitter type ('GLU' or 'GABA')
    cfg : dict
        Configuration dictionary with device parameters
    seed : int, optional
        Random seed for reproducible noise generation
        
    Returns
    -------
    dict
        Dictionary containing current components:
        - 'signal': Clean signal current [A]
        - 'thermal': Thermal noise component [A]
        - 'flicker': 1/f noise component [A]
        - 'drift': 1/f² drift component [A]
        - 'total': Total current with all noise [A]
    """
    # Set random seed if provided
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Get parameters
    nt_params = get_nt_params(cfg, nt)
    q_eff = get_nt_params(cfg, nt)["q_eff_e"] if nt != "CTRL" else 0.0  # Effective charge in units of e
    
    # Device parameters
    gm = cfg['gm_S']  # Transconductance
    C_tot = cfg['C_tot_F']  # Total capacitance
    R_ch = cfg['R_ch_Ohm']  # Channel resistance
    alpha_H = cfg['alpha_H']  # Hooge parameter
    N_c = cfg['N_c']  # Number of charge carriers
    K_d = cfg['K_d_Hz']  # Drift coefficient
    T = cfg.get('temperature_K', 310)  # Temperature
    
    # Time parameters
    dt = cfg['dt_s']
    fs = 1 / dt
    n_samples = len(bound_sites_t)
    duration = n_samples * dt
    
    # Calculate signal current: I_sig = gm * q_eff * N_b / C_tot
    i_signal = gm * q_eff * ELEMENTARY_CHARGE * bound_sites_t / C_tot
    
    # DC current for noise calculations (use mean of signal)
    #I_DC = np.mean(i_signal)
    # FIXED: Realistic baseline from bias (literature: 10-100 μA)
    V_g_bias = cfg.get('V_g_bias_V', -0.2)  # Paper/literature value
    I_DC = gm * abs(V_g_bias)  # Baseline current (positive for noise calc)
    I_DC = max(I_DC, 1e-6)  # Floor at 1 μA to avoid zero
    
    # Generate noise components using FFT method
    # Frequency vector for single-sided spectrum (positive frequencies only)
    freqs = np.fft.rfftfreq(n_samples, dt)
    freqs[0] = 1 / duration   # guard against f=0 in 1/f terms
    n_freqs = len(freqs)
    
    # 1. Thermal noise (white)
    B_det = cfg.get('detection_bandwidth_Hz', 100)
    psd_thermal = 4 * BOLTZMANN * T / R_ch
    noise_thermal_fft = rng.normal(0, 1, n_freqs) + 1j * rng.normal(0, 1, n_freqs)
    effective_B = min(B_det, fs / 2)  # FIXED: Cap at Nyquist to prevent overestimation
    noise_thermal_fft *= np.sqrt(psd_thermal * effective_B /2)
    i_thermal = np.fft.irfft(noise_thermal_fft, n=n_samples)

    
    # 2. Flicker (1/f) noise
    K_f = alpha_H / N_c  # Flicker coefficient
    psd_flicker = K_f * I_DC**2 / freqs  # 1/f spectrum
    noise_flicker_fft = rng.normal(0, 1, n_freqs) + 1j * rng.normal(0, 1, n_freqs)
    noise_flicker_fft *= np.sqrt(psd_flicker * fs / 2) / np.sqrt(n_samples)  # FIXED: Correct scaling for PSD generation (prevents var overestimation by ~n)
    i_flicker = np.fft.irfft(noise_flicker_fft, n=n_samples)
    
    # 3. Drift (1/f²) noise
    psd_drift = K_d * I_DC**2 / (freqs**2)  # 1/f² spectrum
    noise_drift_fft = rng.normal(0, 1, n_freqs) + 1j * rng.normal(0, 1, n_freqs)
    noise_drift_fft *= np.sqrt(psd_drift * fs /2) / np.sqrt(n_samples)  # FIXED: Same correction
    i_drift = np.fft.irfft(noise_drift_fft, n=n_samples)
    
    # Total current
    i_total = i_signal + i_thermal + i_flicker + i_drift
    
    return {
        'signal': i_signal,
        'thermal': i_thermal,
        'flicker': i_flicker,
        'drift': i_drift,
        'total': i_total
    }

def oect_static_gain(N_b: float, nt: str, cfg: Dict[str, Any]) -> float:
    """
    Calculate drain current change (ΔI_D) for deterministic bound sites.
    
    Pure static gain calculation without noise or dynamics, implementing Eq. 24:
    ΔI_D = (g_m · q_eff · e / C_tot) · N_b
    
    Parameters
    ----------
    N_b : float
        Number of bound aptamer sites
    nt : str
        Neurotransmitter type ('GLU' or 'GABA')
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    float
        Drain current change in amperes [A]
    """
    # --- parameter lookup --------------------------------------------
    #   cfg structure mirrors baseline.yaml
    #   cfg['neurotransmitters'][nt]['q_eff_e']   -> effective charge (e)
    #   cfg['oect']['gm_S']                       -> transconductance [S]
    #   cfg['oect']['C_tot_F']                    -> total capacitance [F]
    nt_params = cfg['neurotransmitters'][nt]
    q_eff = nt_params['q_eff_e']            # dimensionless, in units of e
    gm    = cfg['oect']['gm_S']             # [S]
    C_tot = cfg['oect']['C_tot_F']          # [F]
    
    # Calculate current: I = gm * q_eff * e * N_b / C_tot
    delta_I = gm * q_eff * ELEMENTARY_CHARGE * N_b / C_tot * -1.0  # negative for p-type OECTs
    
    return delta_I

def oect_impulse_response(
    dt_s: float,
    n_samples: int,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Discrete-time impulse response h_OECT of the single-pole transducer.

    The analogue model is a first-order low-pass with
        h(t) = (1/τ) · exp(-t/τ)   for  t ≥ 0
    where τ = cfg['oect']['tau_OECT_s'].

    We return the response sampled every `dt_s`, length = `n_samples`.
    The discrete sequence is h[k] = (1/τ) · exp(-k · dt_s / τ).

    Parameters
    ----------
    dt_s : float
        Time step in seconds
    n_samples : int
        Number of samples to generate
    cfg : dict
        Configuration dictionary with cfg['oect']['tau_OECT_s']

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Impulse response in A/A (dimensionless gain).
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
    Compute differential currents for tri-channel architecture with correlated noise.
    
    The control channel shares common-mode noise (temperature, pH, ionic strength
    variations) with the sensing channels, enabling noise cancellation through
    differential measurement.
    
    Parameters
    ----------
    i_glu : np.ndarray
        Total current from glutamate channel [A]
    i_gaba : np.ndarray
        Total current from GABA channel [A]
    i_ctrl : np.ndarray
        Total current from control channel [A]
    rho : float
        (Currently informational.) Correlation coefficient you aimed for when
        generating the pixel currents. The function simply subtracts CTRL from
        each sensing pixel; it does not alter correlation.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'diff_glu': Differential current GLU - CTRL [A]
        - 'diff_gaba': Differential current GABA - CTRL [A]
        - 'rho_achieved': Actual correlation coefficient achieved
    """
    # Simple differential measurement
    # In reality, the noise correlation is already built into the individual
    # channel currents through shared environmental factors
    diff_glu = i_glu - i_ctrl
    diff_gaba = i_gaba - i_ctrl
    
    # Calculate achieved correlation between channels
    # This is for validation/debugging purposes
    corr_matrix = np.corrcoef([i_glu, i_gaba, i_ctrl])
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
    
    Used to simulate common-mode environmental noise that affects
    all channels similarly (temperature, pH, ionic strength).
    
    Parameters
    ----------
    n_samples : int
        Number of time samples
    rho : float
        Correlation coefficient (0 to 1)
    n_channels : int
        Number of channels (default 3 for tri-channel)
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Shape (n_channels, n_samples) of correlated noise
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Generate common-mode and independent components
    n_common = rng.normal(0, 1, n_samples)
    n_independent = rng.normal(0, 1, (n_channels, n_samples))
    
    # Mix according to correlation coefficient
    # n_i = sqrt(rho) * n_common + sqrt(1-rho) * n_independent_i
    correlated_noise = np.sqrt(rho) * n_common + np.sqrt(1 - rho) * n_independent
    
    return correlated_noise


def calculate_noise_metrics(
    current_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate noise metrics from current components.
    
    Parameters
    ----------
    current_dict : dict
        Dictionary from oect_current() with noise components
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        Noise metrics including RMS values and SNR
    """
    signal_rms = np.sqrt(np.mean(current_dict['signal']**2))
    thermal_rms = np.std(current_dict['thermal'])
    flicker_rms = np.std(current_dict['flicker'])
    drift_rms = np.std(current_dict['drift'])
    total_noise_rms = np.std(current_dict['total'] - current_dict['signal'])
    
    # SNR in dB
    snr_db = 20 * np.log10(signal_rms / total_noise_rms) if total_noise_rms > 0 else np.inf
    
    return {
        'signal_rms_A': float(signal_rms),
        'thermal_rms_A': float(thermal_rms),
        'flicker_rms_A': float(flicker_rms),
        'drift_rms_A': float(drift_rms),
        'total_noise_rms_A': float(total_noise_rms),
        'snr_db': float(snr_db)
    }
    
def rms_in_band(x: np.ndarray, fs: float, f_cut: float = 5.0) -> float:
    """RMS after 4-th-order Bessel LP (zero-phase)."""
    from scipy.signal import bessel, sosfiltfilt # type: ignore[import]
    sos = bessel(4, f_cut/(fs/2), btype="low", output="sos")
    x_filt = sosfiltfilt(sos, x)
    return np.sqrt(np.mean(x_filt**2))

# In src/mc_receiver/oect.py

# --- ADD THIS FUNCTION AT THE BOTTOM OF THE FILE ---

def default_params():
    """Returns a dictionary of the default OECT electrical and noise parameters."""
    return {
        'gm_S': 0.002,
        'C_tot_F': 18e-9,  # CORRECTED: 18 nF (200 μm × 200 μm × 12 nm film)
        'R_ch_Ohm': 200,
        'alpha_H': 3.0e-3,
        'N_c': 3.0e14,
        'K_d_Hz': 1.3e-4,
    }