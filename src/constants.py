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
        Neurotransmitter type ('DA', '5HT', 'SERO', or 'CTRL')
        
    Returns
    -------
    dict
        Dictionary containing parameters for the specified neurotransmitter
        
    Raises
    ------
    ValueError
        If nt_type is not found in configuration
    """
    if nt_type not in config['neurotransmitters']:
        available = list(config['neurotransmitters'].keys())
        raise ValueError(f"Invalid neurotransmitter type: {nt_type}. Available: {available}")
    
    params = config['neurotransmitters'][nt_type].copy()
    
    # Apply tortuosity scaling **once**, here
    raw_q = convert_to_numeric(params['q_eff_e'])
    lam   = convert_to_numeric(params.get('lambda', 1.0))
    params['q_eff_e'] = raw_q / lam
    
    # Convert numeric fields only (exclude descriptive fields like 'name')
    numeric_fields = ['D_m2_s', 'lambda', 'k_on_M_s', 'k_off_s', 'q_eff_e', 'damkohler']
    for key in numeric_fields:
        if key in params:
            params[key] = convert_to_numeric(params[key])
    
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
        Neurotransmitter type
        
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
    Calculate Damkohler number for a given configuration and concentration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    nt_type : str
        Neurotransmitter type
    concentration_M : float
        Concentration in M
        
    Returns
    -------
    float
        Damkohler number (dimensionless)
    """
    nt_params = get_nt_params(config, nt_type)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    
    # Characteristic binding time
    tau_bind = 1 / (k_on * concentration_M + k_off)
    
    # Characteristic diffusion time (approximate for sphere)
    D = nt_params['D_m2_s']
    # Use aptamer radius or default
    r_apt = config.get('aptamer_radius_m', 1e-9)  # 1 nm default
    tau_diff = r_apt**2 / D
    
    return tau_bind / tau_diff


def validate_system_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate system parameters and return validation report.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        Validation report with warnings and errors
    """
    warnings = []
    errors = []
    
    # Check temperature range
    if 'temperature_K' in config:
        T = config['temperature_K']
        if not (293 <= T <= 320):
            warnings.append(
                f"Temperature {T} K is outside the "
                "typical physiological range (293-320 K)."
            )
    
    # Check neurotransmitter parameters
    for nt_type in config.get('neurotransmitters', {}):
        try:
            params = get_nt_params(config, nt_type)
            
            # Check for reasonable values
            if params.get('D_m2_s', 0) <= 0:
                errors.append(f"Invalid diffusion coefficient for {nt_type}")
                
            if params.get('k_on_M_s', 0) <= 0:
                errors.append(f"Invalid k_on for {nt_type}")
                
            if params.get('k_off_s', 0) <= 0:
                errors.append(f"Invalid k_off for {nt_type}")
                
        except Exception as e:
            errors.append(f"Error processing {nt_type}: {e}")
    
    return {
        'warnings': warnings,
        'errors': errors,
        'valid': len(errors) == 0
    }


def molecules_to_molar(n_molecules: float, volume_m3: float) -> float:
    """
    Convert number of molecules to molar concentration.
    
    Parameters
    ----------
    n_molecules : float
        Number of molecules
    volume_m3 : float
        Volume in m³
        
    Returns
    -------
    float
        Concentration in M (mol/L)
    """
    moles = n_molecules / AVOGADRO
    volume_L = volume_m3 * 1000  # m³ to L
    return moles / volume_L


def calculate_effective_volume(distance_m: float, alpha: float) -> float:
    """
    Calculate effective volume for molecular communication.
    
    Parameters
    ----------
    distance_m : float
        Distance in meters
    alpha : float
        Geometry factor
        
    Returns
    -------
    float
        Effective volume in m³
    """
    return (4/3) * np.pi * (alpha * distance_m)**3