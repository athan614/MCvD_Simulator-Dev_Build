"""
Physical constants and parameter helper functions for tri-channel OECT MC simulation.

This module provides physical constants and convenience functions to access
neurotransmitter-specific parameters from the configuration.
"""

"python -m pytest tests/test_oect.py -v"

from typing import Dict, Any
import numpy as np

# Physical constants
BOLTZMANN = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
AVOGADRO = 6.02214076e23  # mol^-1

# Unit conversions
UM_TO_M = 1e-6
MS_TO_S = 1e-3
NM_TO_M = 1e-9


def convert_to_numeric(value):
    """
    Convert a value to numeric type if it's a string.
    Handles scientific notation and underscores.
    """
    if isinstance(value, str):
        # Remove underscores and convert to float
        return float(value.replace("_", ""))
    return value


def get_nt_params(config: Dict[str, Any], nt_type: str) -> Dict[str, Any]:
    """
    Extract neurotransmitter-specific parameters from configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML
    nt_type : str
        Neurotransmitter type ('GLU' or 'GABA')
        
    Returns
    -------
    dict
        Dictionary containing parameters for the specified neurotransmitter
        
    Raises
    ------
    ValueError
        If nt_type is not 'GLU' or 'GABA'
    """
    if nt_type not in ['GLU', 'GABA', 'CTRL']:
        raise ValueError(f"Invalid neurotransmitter type: {nt_type}. Must be 'GLU', 'GABA', or 'CTRL'.")
    
    params = config['neurotransmitters'][nt_type].copy()
    
    # Apply tortuosity scaling **once**, here
    raw_q = convert_to_numeric(params['q_eff_e'])
    lam   = convert_to_numeric(params.get('lambda', 1.0))
    params['q_eff_e'] = raw_q / lam
    
    # Convert all string values to numeric
    for key, value in params.items():
        params[key] = convert_to_numeric(value)
    
    return params


def get_effective_diffusion_coefficient(config: Dict[str, Any], nt_type: str) -> float:
    """
    Calculate effective diffusion coefficient accounting for tortuosity.
    
    The effective diffusion coefficient in brain tissue is:
    D_eff = D / λ²
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    nt_type : str
        Neurotransmitter type ('GLU' or 'GABA')
        
    Returns
    -------
    float
        Effective diffusion coefficient in m²/s
    """
    nt_params = get_nt_params(config, nt_type)
    D = nt_params['D_m2_s']
    lambda_val = nt_params['lambda']
    
    return D / (lambda_val ** 2)


def calculate_damkohler_number(config: Dict[str, Any], nt_type: str, 
                              concentration_M: float) -> float:
    """
    Calculate Damköhler number to verify reaction-limited regime.
    
    Da = k_on * C * L_char² / (D/λ²)
    
    For Da << 1, the system is reaction-limited (desired).
    For Da >> 1, the system is transport-limited.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    nt_type : str
        Neurotransmitter type
    concentration_M : float
        Concentration in molar (M)
        
    Returns
    -------
    float
        Damköhler number (dimensionless)
    """
    nt_params = get_nt_params(config, nt_type)
    k_on = nt_params['k_on_M_s']
    D_eff = get_effective_diffusion_coefficient(config, nt_type)
    
    # Characteristic length from gate area
    L_char = np.sqrt(config['gate_area_m2'] / np.pi)
    
    Da = k_on * concentration_M * L_char**2 / D_eff
    
    return Da


def validate_system_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate key system parameters and check operating regime.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        Validation results including warnings if any parameters are out of range
    """
    results: Dict[str, Any] = {
        'valid': True,
        'warnings': [],
        'damkohler_numbers': {}
    }
    
    # Check Damköhler numbers for typical concentration (10 nM)
    typical_conc_M = 10e-9  # 10 nM
    
    for nt_type in ['GLU', 'GABA']:
        Da = calculate_damkohler_number(config, nt_type, typical_conc_M)
        results['damkohler_numbers'][nt_type] = Da
        
        if Da > 0.1:
            results['warnings'].append(
                f"Warning: {nt_type} Damköhler number = {Da:.3f} > 0.1. "
                "System may not be fully reaction-limited."
            )
            results['valid'] = False
    
    # Check if clearance rate is reasonable
    if config['clearance_rate'] > 0.1:
        results['warnings'].append(
            f"Warning: Clearance rate {config['clearance_rate']} s^-1 seems high. "
            "Typical values are 0.001-0.01 s^-1."
        )
    
    # Check temperature
    if not (293 <= config['temperature_K'] <= 320):
        results['warnings'].append(
            f"Warning: Temperature {config['temperature_K']} K is outside "
            "typical physiological range (293-320 K)."
        )
    
    return results


def molecules_to_molar(n_molecules: float, volume_m3: float) -> float:
    """
    Convert number of molecules to molar concentration.
    
    Parameters
    ----------
    n_molecules : float
        Number of molecules
    volume_m3 : float
        Volume in cubic meters
        
    Returns
    -------
    float
        Concentration in molar (M)
    """
    moles = n_molecules / AVOGADRO
    liters = volume_m3 * 1000  # m³ to L
    return moles / liters


def calculate_effective_volume(distance_m: float, alpha: float) -> float:
    """
    Calculate effective volume accounting for ECS volume fraction.
    
    For a spherical diffusion front at distance r, the effective volume
    is reduced by the volume fraction α.
    
    Parameters
    ----------
    distance_m : float
        Distance from source in meters
    alpha : float
        ECS volume fraction
        
    Returns
    -------
    float
        Effective volume in m³
    """
    geometric_volume = (4/3) * np.pi * distance_m**3
    return geometric_volume * alpha