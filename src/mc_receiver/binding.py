"""
Aptamer binding kinetics and noise model for OECT biosensor.

This module implements the biorecognition layer described in Section II-E of the manuscript,
including transport-limited binding rates, equilibrium binding calculations, binding noise
power spectral density, and Monte Carlo simulations of stochastic binding dynamics.
"""

import numpy as np
from typing import Dict, Any, Tuple
from scipy.integrate import odeint # type: ignore[import]
from ..constants import get_nt_params, ELEMENTARY_CHARGE # type: ignore[import]


def calculate_effective_on_rate(k_on: float, damkohler: float) -> float:
    """
    Calculate transport-limited effective on-rate.
    
    Parameters
    ----------
    k_on : float
        Intrinsic on-rate constant in M^-1 s^-1
    damkohler : float
        Damköhler number (dimensionless)
        
    Returns
    -------
    float
        Effective on-rate in M^-1 s^-1 (Eq. 14)
    """
    return k_on / (1 + damkohler)


def _step_poisson(N_free: int, P_bind: float, rng: np.random.Generator) -> int:
    """
    Poisson binding step for high probability regime.
    
    When P_bind > 0.1, Bernoulli approximation breaks down. This function
    uses the exact Poisson occupancy probability P_occ = 1 - exp(-λ)
    where λ = k_on * C * dt.
    
    Parameters
    ----------
    N_free : int
        Number of unbound sites
    P_bind : float
        Binding probability parameter (λ = k_on * C * dt)
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    int
        Number of new bindings
    """
    if N_free == 0 or P_bind <= 0:
        return 0
    
    # P_occ = 1 - exp(-λ) for Poisson occupancy
    P_occ = 1 - np.exp(-P_bind)
    return rng.binomial(N_free, P_occ)


def bernoulli_binding(
    conc_time: np.ndarray,
    nt: str,
    cfg: Dict[str, Any],
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation of stochastic aptamer binding dynamics.
    
    Uses Bernoulli trials for low concentrations (P_bind ≤ 0.1) and
    Poisson occupancy model for high concentrations to maintain accuracy.
    
    Parameters
    ----------
    conc_time : np.ndarray
        Time series of molar concentration [M] at the sensor surface
    nt : str
        Neurotransmitter type ('GLU' or 'GABA')
    cfg : dict
        Configuration dictionary containing system parameters
    rng : np.random.Generator
        Random number generator for reproducibility
        
    Returns
    -------
    bound_sites_t : np.ndarray
        Integer number of bound aptamers at each time step
    ibind_noise_t : np.ndarray
        Total binding current (mean + noise) [A] (I = gm·q_eff·N_b/Ctot)
    ibind_ac_t : np.ndarray
        AC component of binding current (noise only) [A]
    """
    # Get parameters - get_nt_params now handles string conversion
    nt_params = get_nt_params(cfg, nt)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    q_eff = nt_params['q_eff_e']
    damkohler = nt_params.get('damkohler', 0.0)  # Default to 0 if not specified
    
    # System parameters
    # `N_apt` might arrive from YAML as the *string* "4e8".
    # Accept int, float, or str and convert safely to integer.
    N_apt_raw = cfg['N_apt']
    if isinstance(N_apt_raw, str):
        # allow underscores or scientific-notation strings like "4e8"
        N_apt_raw = float(N_apt_raw.replace("_", ""))
    N_apt = int(N_apt_raw)      # Total number of aptamer sites
    
    gm = cfg['oect']['gm_S']  # Transconductance
    C_tot = cfg['oect']['C_tot_F']  # Total capacitance
    dt = cfg['sim']['dt_s']  # Time step
    
    # Add this new block after "dt = cfg['sim']['dt_s']"
    if cfg.get('deterministic_mode', False):
        # Deterministic mode: Use mean binding instead of stochastic
        mean_bound = mean_binding(conc_time, nt, cfg)
        bound_sites_t = mean_bound.astype(np.int32)
        ibind_noise_t = gm * q_eff * ELEMENTARY_CHARGE * bound_sites_t / C_tot
        ibind_ac_t = np.zeros_like(ibind_noise_t)  # No AC noise in deterministic
        return bound_sites_t, ibind_noise_t, ibind_ac_t
    else:
        # Existing stochastic code follows here (n_steps = len(conc_time) ...)
    
        # Calculate effective on-rate with Damköhler correction
        k_on_eff = calculate_effective_on_rate(k_on, damkohler)
    
        # Initialize arrays
        n_steps = len(conc_time)
        bound_sites_t = np.zeros(n_steps, dtype=np.int32)
    
        # Initial condition: start from equilibrium at first concentration
        if conc_time[0] > 0:
            theta_init = k_on_eff * conc_time[0] / (k_on_eff * conc_time[0] + k_off)
            bound_sites_t[0] = rng.binomial(N_apt, theta_init)
    
        # Monte Carlo time evolution with Bernoulli/Poisson switching
        for i in range(1, n_steps):
            N_bound = bound_sites_t[i-1]
            N_free = N_apt - N_bound
        
            # Binding probability parameter
            P_bind = k_on_eff * conc_time[i] * dt
        
            # Fix: Ensure probability never exceeds 1
            P_bind_capped = np.clip(P_bind, 0.0, 1.0)
        
            # Unbinding probability for bound sites
            P_unbind = np.clip(k_off * dt, 0.0, 1.0)
        
            # Switch between Bernoulli and Poisson based on P_bind
            if P_bind <= 0.1:
                # Bernoulli regime - use capped probability
                new_bindings = rng.binomial(N_free, P_bind_capped) if N_free > 0 else 0
            else:
                # Poisson regime - use exact occupancy probability
                P_occ = np.clip(1 - np.exp(-P_bind), 0.0, 1.0)
                new_bindings = rng.binomial(N_free, P_occ) if N_free > 0 else 0
        
            # Unbinding (already capped)
            if N_bound > 0:
                new_unbindings = rng.binomial(N_bound, P_unbind)
            else:
                new_unbindings = 0
        
            # Update bound count
            bound_sites_t[i] = N_bound + new_bindings - new_unbindings
        
            # Ensure we stay within bounds
            bound_sites_t[i] = np.clip(bound_sites_t[i], 0, N_apt)
        
            if bound_sites_t[i] > 0.9 * N_apt:
                print("Warning: Aptamer saturation >90%")
    
        # Convert to current: I = gm * q_eff * N_b / C_tot
        # Note: q_eff is in units of elementary charge
        ibind_noise_t = gm * q_eff * ELEMENTARY_CHARGE * bound_sites_t / C_tot
    
        # Calculate AC component (noise only)
        ibind_ac_t = ibind_noise_t - np.mean(ibind_noise_t)
    
        return bound_sites_t.astype(np.int32), ibind_noise_t, ibind_ac_t


def mean_binding(
    conc_time: np.ndarray,
    nt: str,
    cfg: Dict[str, Any]
) -> np.ndarray:
    """
    Deterministic solution of Langmuir binding ODE.
    
    Solves: dN_b/dt = k_on,eff C(t) (N_apt - N_b) - k_off N_b
    
    Parameters
    ----------
    conc_time : np.ndarray
        Time series of molar concentration [M]
    nt : str
        Neurotransmitter type ('GLU' or 'GABA')
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    np.ndarray
        Mean number of bound aptamers over time
    """
    # Get parameters
    nt_params = get_nt_params(cfg, nt)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    damkohler = nt_params.get('damkohler', 0.0)
    
    # System parameters - handle N_apt properly
    N_apt_raw = cfg['N_apt']
    if isinstance(N_apt_raw, str):
        N_apt_raw = float(N_apt_raw.replace("_", ""))
    N_apt = int(N_apt_raw)
    
    # Calculate effective on-rate
    k_on_eff = calculate_effective_on_rate(k_on, damkohler)
    
    # Use literature-realistic defaults if values are zero/missing (from search results)
    if k_on == 0 or k_off == 0:  # Fallback trigger
        if nt == 'GLU':
            k_on_eff = 1e5  # From ACS Sensors/PubMed: ~1e5 M^{-1}s^{-1}
            k_off = 0.5     # To match Kd~5e-6 M
        elif nt == 'GABA':
            k_on_eff = 5e4  # From Biosensors papers: ~5e4 M^{-1}s^{-1}
            k_off = 1.0     # To match Kd~2e-5 M
        else:
            raise ValueError(f"No realistic defaults for {nt}")
    
    # Time vector
    dt = cfg['sim']['dt_s']
    n_steps = len(conc_time)
    t_vec = np.arange(n_steps) * dt
    
    # ODE function
    def binding_ode(N_b, t):
        # Interpolate concentration at time t
        idx = int(t / dt)
        if idx >= len(conc_time) - 1:
            C_t = conc_time[-1]
        else:
            # Linear interpolation
            alpha = (t - idx * dt) / dt
            C_t = (1 - alpha) * conc_time[idx] + alpha * conc_time[idx + 1]
        
        # Langmuir kinetics
        dN_dt = k_on_eff * C_t * (N_apt - N_b) - k_off * N_b
        dN_dt = np.clip(dN_dt, -N_apt / dt, N_apt / dt)  # Prevent numerical overflow/saturation
        return dN_dt
    
    # Initial condition: equilibrium at first concentration
    if conc_time[0] > 0:
        theta_init = k_on_eff * conc_time[0] / (k_on_eff * conc_time[0] + k_off)
        N_b0 = N_apt * theta_init
    else:
        N_b0 = 0
    
    # Solve ODE
    N_b_mean = odeint(binding_ode, N_b0, t_vec)
    
    return N_b_mean.flatten()


def binding_noise_psd(
    nt: str,
    cfg: Dict[str, Any],
    f_vec: np.ndarray,
    C_eq: float = 10e-9  # Default 10 nM
) -> np.ndarray:
    """
    Calculate analytical binding noise power spectral density.
    
    S_I,NB(f) = (2·q_eff²·gm² / C_tot²) · N_apt θ∞(1-θ∞) τ_B / (1 + (2πf τ_B)²)
    
    Parameters
    ----------
    nt : str
        Neurotransmitter type ('GLU' or 'GABA')
    cfg : dict
        Configuration dictionary
    f_vec : np.ndarray
        Frequency vector [Hz]
    C_eq : float
        Equilibrium concentration [M] for calculating θ∞ and τ_B
        
    Returns
    -------
    np.ndarray
        Power spectral density of binding noise current [A²/Hz]
    """
    # Get parameters
    nt_params = get_nt_params(cfg, nt)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    q_eff = nt_params['q_eff_e']
    damkohler = nt_params.get('damkohler', 0.0)
    
    # Handle N_apt properly
    N_apt_raw = cfg['N_apt']
    if isinstance(N_apt_raw, str):
        N_apt_raw = float(N_apt_raw.replace("_", ""))
    N_apt = int(N_apt_raw)
    
    gm = cfg['oect']['gm_S']
    C_tot = cfg['oect']['C_tot_F']
    
    # Calculate effective on-rate
    k_on_eff = calculate_effective_on_rate(k_on, damkohler)
    
    # Equilibrium binding probability
    theta_inf = k_on_eff * C_eq / (k_on_eff * C_eq + k_off)
    
    # Binding relaxation time
    tau_B = 1 / (k_on_eff * C_eq + k_off)
    
    # Variance of bound aptamers
    var_NB = N_apt * theta_inf * (1 - theta_inf)
    
    # Convert to current PSD
    # Factor of 2 for single-sided PSD
    prefactor = 4 * (q_eff * ELEMENTARY_CHARGE * gm / C_tot)**2 * var_NB * tau_B
    # 4 = 2 (two‑sided FT) × 2 (convert to one‑sided PSD)
    
    # Lorentzian spectrum
    S_I_NB = prefactor / (1 + (2 * np.pi * f_vec * tau_B)**2)
    
    return S_I_NB


def calculate_equilibrium_metrics(
    C_eq: float,
    nt: str,
    cfg: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate equilibrium binding metrics for a given concentration.
    
    Parameters
    ----------
    C_eq : float
        Equilibrium concentration [M]
    nt : str
        Neurotransmitter type
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        Dictionary containing:
        - theta_inf: Equilibrium binding probability
        - mean_bound: Mean number of bound aptamers
        - variance: Variance in bound aptamers
        - tau_B: Binding relaxation time [s]
        - std_current: Standard deviation of binding current [A]
    """
    # Get parameters
    nt_params = get_nt_params(cfg, nt)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    q_eff = nt_params['q_eff_e']
    damkohler = nt_params.get('damkohler', 0.0)
    
    # Handle N_apt properly
    N_apt_raw = cfg['N_apt']
    if isinstance(N_apt_raw, str):
        N_apt_raw = float(N_apt_raw.replace("_", ""))
    N_apt = int(N_apt_raw)
    
    gm = cfg['oect']['gm_S']
    C_tot = cfg['oect']['C_tot_F']
    
    # Calculate effective on-rate
    k_on_eff = calculate_effective_on_rate(k_on, damkohler)
    
    # Equilibrium binding probability
    theta_inf = k_on_eff * C_eq / (k_on_eff * C_eq + k_off)
    
    # Mean and variance
    mean_bound = N_apt * theta_inf
    variance = N_apt * theta_inf * (1 - theta_inf)
    
    # Relaxation time
    tau_B = 1 / (k_on_eff * C_eq + k_off)
    
    # Current standard deviation
    std_current = gm * q_eff * ELEMENTARY_CHARGE * np.sqrt(variance) / C_tot
    
    return {
        'theta_inf': theta_inf,
        'mean_bound': mean_bound,
        'variance': variance,
        'tau_B': tau_B,
        'std_current': std_current
    }
    
def default_params():
    return {
        'N_apt': 4e8,
        'GLU': {'k_on_M_s': 5e4, 'k_off_s': 1.5, 'q_eff_e': 0.6},
        'GABA': {'k_on_M_s': 3e4, 'k_off_s': 0.9, 'q_eff_e': 0.2},
    }