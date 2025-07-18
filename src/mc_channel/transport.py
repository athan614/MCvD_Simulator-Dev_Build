"""
Molecular diffusion models for neurotransmitter transport in brain tissue.

This module implements the Green's function solution for 3D diffusion in
restricted extracellular space (ECS) with tortuosity and clearance, and
provides functions to compute concentration profiles for finite-duration
molecular release events.
"""

# At the top of src/mc_channel/transport.py
from typing import Dict, Any, Tuple, Union
import numpy as np
from scipy import integrate # type: ignore[import]
from scipy.stats import gamma as gamma_dist # type: ignore[import]
from scipy.special import erfc # type: ignore[import]

# --- THIS IS THE CORRECTED IMPORT LINE ---
from ..constants import get_nt_params, MS_TO_S, AVOGADRO

def greens_function_3d(r: float, t: float, D: float, lam: float, 
                      alpha: float, k_clear: float) -> float:
    """
    Calculate 3D Green's function for point source in restricted ECS.
    
    The Green's function represents the concentration at distance r and time t
    from an instantaneous release of a single molecule at t=0, accounting for:
    - Restricted volume fraction (α)
    - Tortuosity (λ) 
    - Clearance/uptake (k')
    
    G(r,t) = 1/[α(4πDt/λ²)^(3/2)] * exp(-r²λ²/4Dt) * exp(-k't)
    
    Parameters
    ----------
    r : float
        Distance from source in meters
    t : float
        Time since release in seconds
    D : float
        Free diffusion coefficient in m²/s
    lam : float
        Tortuosity factor (dimensionless)
    alpha : float
        ECS volume fraction (dimensionless)
    k_clear : float
        Clearance rate constant in s⁻¹
        
    Returns
    -------
    float
        Concentration normalized per molecule (m⁻³)
    """
    if t <= 0:
        return 0.0
    
    # Effective diffusion coefficient
    D_eff = D / (lam ** 2)
    
    # Prefactor includes volume fraction correction
    prefactor = 1.0 / (alpha * (4 * np.pi * D_eff * t) ** 1.5)
    
    # Exponential terms: diffusion and clearance
    exp_diffusion = np.exp(-(r ** 2) / (4 * D_eff * t))
    exp_clearance = np.exp(-k_clear * t)
    
    return prefactor * exp_diffusion * exp_clearance

def rectangular_release_rate(t: float, Nm: float, T_release: float) -> float:
    """
    Release rate for rectangular (constant) burst profile.
    
    Parameters
    ----------
    t : float
        Time in seconds
    Nm : float
        Total number of molecules to release
    T_release : float
        Duration of release in seconds
        
    Returns
    -------
    float
        Release rate in molecules/s
    """
    if 0 <= t <= T_release:
        return Nm / T_release
    else:
        return 0.0
    
def gamma_release_rate(t: float, Nm: float, k: float, theta: float) -> float:
    """
    Release rate for gamma-distributed burst profile.
    
    The gamma distribution provides a more biologically realistic release
    profile with a rise time and exponential decay.
    
    Parameters
    ----------
    t : float
        Time in seconds
    Nm : float
        Total number of molecules to release
    k : float
        Shape parameter (typically 2-3)
    theta : float
        Scale parameter in seconds
        
    Returns
    -------
    float
        Release rate in molecules/s
    """
    if t < 0:
        return 0.0
    
    # Gamma PDF normalized to release Nm molecules total
    pdf = gamma_dist.pdf(t, a=k, scale=theta)
    return float(Nm * pdf)

def finite_burst_concentration(
    Nm: float,
    r: float, 
    t_vec: np.ndarray,
    config: Dict[str, Any],
    nt_type: str
) -> np.ndarray:
    """Calculate concentration profile for finite-duration molecular release."""
    
    # Get neurotransmitter-specific parameters
    nt_params = get_nt_params(config, nt_type)
    D = nt_params['D_m2_s']
    lam = nt_params['lambda']
    
    # Get general parameters
    alpha = config['alpha']
    k_clear = config['clearance_rate']
    T_release = config['T_release_ms'] * MS_TO_S
    burst_shape = config['burst_shape']
    
    # Calculate effective diffusion coefficient (Task B)
    D_eff = D / (lam ** 2)
    
    # Initialize concentration array
    conc = np.zeros_like(t_vec)
    
    if burst_shape == 'rect':
        # Rectangular burst with adaptive erfc-skip
        T_rel = config['T_release_ms'] * MS_TO_S
        pref = Nm / (8 * np.pi * alpha * D_eff * r * T_rel)

        for i, t in enumerate(t_vec):
            if t <= 0.0:
                continue

            # Adaptive erfc-skip rule
            z1 = r / (2 * np.sqrt(D_eff * t))
            
            # First term: erfc(z1)
            if z1 > 4 and erfc(z1) < 1e-12:
                term1 = 0.0
            else:
                term1 = erfc(z1)
            
            # Second term: erfc(z2) if t > T_rel
            if t > T_rel:
                z2 = r / (2 * np.sqrt(D_eff * (t - T_rel)))
                if z2 > 4 and erfc(z2) < 1e-12:
                    term2 = 0.0
                else:
                    term2 = erfc(z2)
            else:
                term2 = 0.0
            
            diff = term1 - term2
            conc[i] = pref * diff * np.exp(-k_clear * t)
    
    elif burst_shape == 'gamma':
        # Gamma burst (unchanged)
        k = config['gamma_shape_k']
        theta = config['gamma_scale_theta']
        
        def release_rate(tau):
            return gamma_release_rate(tau, Nm, k, theta)
        
        for i, t in enumerate(t_vec):
            if t <= 0:
                continue
                
            def integrand(tau):
                R_tau = release_rate(tau)
                G = greens_function_3d(r, t - tau, D, lam, alpha, k_clear)
                return R_tau * G
            
            try:
                result, _ = integrate.quad(integrand, 0, t,
                                        limit=800,
                                        epsabs=1e-12, epsrel=1e-10)
                conc[i] = result
            except (integrate.IntegrationWarning, RuntimeError):
                tau = np.linspace(0.0, t, 1500, dtype=float)
                conc[i] = np.trapezoid(integrand(tau), tau)
    
    else:
        # Instantaneous (Dirac) burst - Task A: fix this branch
        for i, t in enumerate(t_vec):
            if t <= 0:
                continue
            
            # Adaptive erfc-skip rule (same as rect branch)
            z = r / np.sqrt(4 * D_eff * t)
            
            if z > 5 and erfc(z) < 1e-13:
                erfc_term = 0.0
            else:
                erfc_term = erfc(z)
            
            # Apply Green's function with adaptive erfc
            prefactor = Nm / (alpha * (4 * np.pi * D_eff * t) ** 1.5)
            exponential = np.exp(-(r ** 2) / (4 * D_eff * t) - k_clear * t)
            
            conc[i] = prefactor * exponential * erfc_term
    
    # Convert from molecules/m³ to molar
    # CORRECTED: 1 M = 6.022e23 molecules/L = 6.022e26 molecules/m³
    conc_molar = conc / (AVOGADRO * 1000)
    
    return conc_molar

def burst_profile(*a, **k):
    """Alias for finite_burst_concentration."""
    return finite_burst_concentration(*a, **k)

__all__ = [
    "greens_function_3d",
    "rectangular_release_rate",
    "gamma_release_rate",
    "finite_burst_concentration",
    "burst_profile",
]