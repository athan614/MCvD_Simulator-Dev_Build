"""
Aptamer binding kinetics and noise model for OECT biosensor.

VECTORIZED VERSION: Optimized with batch processing and efficient computations
while maintaining exact stochastic behavior and reasonable memory usage.
"""

import numpy as np
from typing import Dict, Any, Tuple
from scipy.integrate import odeint  #type: ignore
from src.constants import get_nt_params, ELEMENTARY_CHARGE


def calculate_effective_on_rate(k_on: float, damkohler: float) -> float:
    """Calculate transport-limited effective on-rate. (Unchanged)"""
    return k_on / (1 + damkohler)


def _step_poisson(N_free: int, P_bind: float, rng: np.random.Generator) -> int:
    """Poisson binding step for high probability regime. (Unchanged)"""
    if N_free == 0 or P_bind <= 0:
        return 0
    
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
    
    MEMORY-EFFICIENT VERSION: Generates random numbers on-demand instead of
    pre-allocating massive arrays. Maintains exact stochastic behavior.
    """
    # Get parameters
    nt_params = get_nt_params(cfg, nt)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    q_eff = nt_params['q_eff_e']
    damkohler = nt_params.get('damkohler', 0.0)
    
    # System parameters
    N_apt_raw = cfg['N_apt']
    if isinstance(N_apt_raw, str):
        N_apt_raw = float(N_apt_raw.replace("_", ""))
    N_apt = int(N_apt_raw)
    
    gm = cfg['oect']['gm_S']
    C_tot = cfg['oect']['C_tot_F']
    dt = cfg['sim']['dt_s']
    
    # Check for deterministic mode
    if cfg.get('deterministic_mode', False):
        mean_bound = mean_binding(conc_time, nt, cfg)
        bound_sites_t = mean_bound.astype(np.int32)
        ibind_noise_t = gm * q_eff * ELEMENTARY_CHARGE * bound_sites_t / C_tot
        ibind_ac_t = np.zeros_like(ibind_noise_t)
        return bound_sites_t, ibind_noise_t, ibind_ac_t
    
    # Calculate effective on-rate with Damköhler correction
    k_on_eff = calculate_effective_on_rate(k_on, damkohler)
    
    # Initialize arrays
    n_steps = len(conc_time)
    bound_sites_t = np.zeros(n_steps, dtype=np.int32)
    
    # Initial condition: start from equilibrium at first concentration
    if conc_time[0] > 0:
        theta_init = k_on_eff * conc_time[0] / (k_on_eff * conc_time[0] + k_off)
        bound_sites_t[0] = rng.binomial(N_apt, theta_init)
    
    # Pre-compute binding probabilities (memory efficient)
    P_bind_all = k_on_eff * conc_time * dt
    P_unbind = np.clip(k_off * dt, 0.0, 1.0)
    
    # Monte Carlo time evolution - generate random numbers on demand
    for i in range(1, n_steps):
        N_bound = bound_sites_t[i-1]
        N_free = N_apt - N_bound
        
        # Binding probability for this time step
        P_bind = P_bind_all[i]
        
        # Ensure probability never exceeds 1
        P_bind_capped = np.clip(P_bind, 0.0, 1.0)
        
        # MEMORY EFFICIENT: Generate random numbers only as needed
        new_bindings = 0
        new_unbindings = 0
        
        # Switch between Bernoulli and Poisson based on P_bind
        if P_bind <= 0.1:
            # Bernoulli regime - generate N_free random numbers
            if N_free > 0:
                # For large N_free, use binomial directly (more efficient)
                if N_free > 1000:
                    new_bindings = rng.binomial(N_free, P_bind_capped)
                else:
                    # For small N_free, can still do explicit Bernoulli trials
                    randoms = rng.random(N_free)
                    new_bindings = np.sum(randoms < P_bind_capped)
        else:
            # Poisson regime - use exact occupancy probability
            if N_free > 0:
                P_occ = np.clip(1 - np.exp(-P_bind), 0.0, 1.0)
                new_bindings = rng.binomial(N_free, P_occ)
        
        # Unbinding - generate N_bound random numbers
        if N_bound > 0:
            # For large N_bound, use binomial directly
            if N_bound > 1000:
                new_unbindings = rng.binomial(N_bound, P_unbind)
            else:
                randoms = rng.random(N_bound)
                new_unbindings = np.sum(randoms < P_unbind)
        
        # Update bound count
        bound_sites_t[i] = N_bound + new_bindings - new_unbindings
        
        # Ensure we stay within bounds
        bound_sites_t[i] = np.clip(bound_sites_t[i], 0, N_apt)
    
    # Check for saturation warnings
    if np.any(bound_sites_t > 0.9 * N_apt):
        max_occupancy = np.max(bound_sites_t) / N_apt
        print(f"Warning: Aptamer saturation detected! Max occupancy: {max_occupancy:.2%}")
    
    # Convert to current
    ibind_noise_t = gm * q_eff * ELEMENTARY_CHARGE * bound_sites_t / C_tot
    
    # Calculate AC component (noise only)
    ibind_ac_t = ibind_noise_t - np.mean(ibind_noise_t)
    
    return bound_sites_t.astype(np.int32), ibind_noise_t, ibind_ac_t


def bernoulli_binding_batch(
    conc_batch: np.ndarray,
    nt: str,
    cfg: Dict[str, Any],
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process multiple concentration profiles efficiently.
    
    Parameters
    ----------
    conc_batch : np.ndarray
        Shape (n_batch, n_time) concentration profiles
    nt : str
        Neurotransmitter type
    cfg : dict
        Configuration
    rng : np.random.Generator
        Random generator
        
    Returns
    -------
    tuple of np.ndarray
        Each with shape (n_batch, n_time)
    """
    n_batch, n_time = conc_batch.shape
    
    # Pre-allocate results
    bound_sites_batch = np.zeros((n_batch, n_time), dtype=np.int32)
    ibind_noise_batch = np.zeros((n_batch, n_time))
    ibind_ac_batch = np.zeros((n_batch, n_time))
    
    # Process each profile
    for i in range(n_batch):
        bound, i_noise, i_ac = bernoulli_binding(conc_batch[i], nt, cfg, rng)
        bound_sites_batch[i] = bound
        ibind_noise_batch[i] = i_noise
        ibind_ac_batch[i] = i_ac
    
    return bound_sites_batch, ibind_noise_batch, ibind_ac_batch


def mean_binding(
    conc_time: np.ndarray,
    nt: str,
    cfg: Dict[str, Any]
) -> np.ndarray:
    """
    Deterministic solution of Langmuir binding ODE.
    
    VECTORIZED: Uses optimized ODE solver settings for better performance.
    """
    # Get parameters
    nt_params = get_nt_params(cfg, nt)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    damkohler = nt_params.get('damkohler', 0.0)
    
    # System parameters
    N_apt_raw = cfg['N_apt']
    if isinstance(N_apt_raw, str):
        N_apt_raw = float(N_apt_raw.replace("_", ""))
    N_apt = int(N_apt_raw)
    
    # Calculate effective on-rate
    k_on_eff = calculate_effective_on_rate(k_on, damkohler)
    
    # Use literature-realistic defaults if values are zero/missing
    if k_on == 0 or k_off == 0:
        if nt == 'DA':
            k_on_eff = 1e5
            k_off = 0.5
        elif nt == 'SERO':
            k_on_eff = 5e4
            k_off = 1.0
        else:
            raise ValueError(f"No realistic defaults for {nt}")
    
    # Time vector
    dt = cfg['sim']['dt_s']
    n_steps = len(conc_time)
    t_vec = np.arange(n_steps) * dt
    
    # VECTORIZED: Create interpolation function for concentration
    from scipy.interpolate import interp1d  #type: ignore
    conc_interp = interp1d(t_vec, conc_time, kind='linear', 
                          bounds_error=False, fill_value=conc_time[-1])
    
    # ODE function using interpolation
    def binding_ode(N_b, t):
        C_t = conc_interp(t)
        dN_dt = k_on_eff * C_t * (N_apt - N_b) - k_off * N_b
        dN_dt = np.clip(dN_dt, -N_apt / dt, N_apt / dt)
        return dN_dt
    
    # Initial condition
    if conc_time[0] > 0:
        theta_init = k_on_eff * conc_time[0] / (k_on_eff * conc_time[0] + k_off)
        N_b0 = N_apt * theta_init
    else:
        N_b0 = 0
    
    # OPTIMIZED: Use adaptive solver with larger initial step
    N_b_mean = odeint(binding_ode, N_b0, t_vec, 
                     rtol=1e-6, atol=1e-9, hmax=dt*10)
    
    return N_b_mean.flatten()


def binding_noise_psd(
    nt: str,
    cfg: Dict[str, Any],
    f_vec: np.ndarray,
    C_eq: float = 10e-9
) -> np.ndarray:
    """
    Calculate analytical binding noise power spectral density.
    
    VECTORIZED: Already uses numpy operations efficiently.
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
    
    # Convert to current PSD (already vectorized)
    prefactor = 4 * (q_eff * ELEMENTARY_CHARGE * gm / C_tot)**2 * var_NB * tau_B
    
    # Lorentzian spectrum (vectorized operation)
    S_I_NB = prefactor / (1 + (2 * np.pi * f_vec * tau_B)**2)
    
    return S_I_NB


def calculate_equilibrium_metrics(
    C_eq: float,
    nt: str,
    cfg: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate equilibrium binding metrics for a given concentration.
    (Already efficient - no vectorization needed)
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


def calculate_equilibrium_metrics_batch(
    C_eq_array: np.ndarray,
    nt: str,
    cfg: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    VECTORIZED: Calculate equilibrium metrics for multiple concentrations.
    
    Parameters
    ----------
    C_eq_array : np.ndarray
        Array of equilibrium concentrations [M]
    nt : str
        Neurotransmitter type
    cfg : dict
        Configuration
        
    Returns
    -------
    dict
        Dictionary with arrays for each metric
    """
    # Get parameters
    nt_params = get_nt_params(cfg, nt)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    q_eff = nt_params['q_eff_e']
    damkohler = nt_params.get('damkohler', 0.0)
    
    # Handle N_apt
    N_apt_raw = cfg['N_apt']
    if isinstance(N_apt_raw, str):
        N_apt_raw = float(N_apt_raw.replace("_", ""))
    N_apt = int(N_apt_raw)
    
    gm = cfg['oect']['gm_S']
    C_tot = cfg['oect']['C_tot_F']
    
    # Calculate effective on-rate
    k_on_eff = calculate_effective_on_rate(k_on, damkohler)
    
    # Vectorized calculations
    theta_inf = k_on_eff * C_eq_array / (k_on_eff * C_eq_array + k_off)
    mean_bound = N_apt * theta_inf
    variance = N_apt * theta_inf * (1 - theta_inf)
    tau_B = 1 / (k_on_eff * C_eq_array + k_off)
    std_current = gm * q_eff * ELEMENTARY_CHARGE * np.sqrt(variance) / C_tot
    
    return {
        'theta_inf': theta_inf,
        'mean_bound': mean_bound,
        'variance': variance,
        'tau_B': tau_B,
        'std_current': std_current
    }


def default_params():
    """Returns default parameters (unchanged)"""
    return {
        'N_apt': 4e8,
        'DA': {'k_on_M_s': 5e4, 'k_off_s': 1.5, 'q_eff_e': 0.6},
        'SERO': {'k_on_M_s': 3e4, 'k_off_s': 0.9, 'q_eff_e': 0.2},
    }
