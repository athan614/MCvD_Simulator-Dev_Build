# At the top of src/analysis_utils.py
from typing import Dict, Any, Tuple
import numpy as np

# Import from our newly refactored module!
from .mc_channel.transport import finite_burst_concentration
from .constants import get_nt_params

def find_peak_concentration(c_profile: np.ndarray, 
                          t_vec: np.ndarray) -> Tuple[float, float]:
    """
    Find peak concentration and time-to-peak from concentration profile.
    
    This function is critical for determining propagation delays and
    peak signal amplitudes.
    
    Parameters
    ----------
    c_profile : np.ndarray
        Concentration profile in M
    t_vec : np.ndarray
        Time vector in seconds
        
    Returns
    -------
    tuple
        (peak_concentration_M, time_to_peak_s)
    """
    idx_peak = np.argmax(c_profile)
    return c_profile[idx_peak], t_vec[idx_peak]


def calculate_propagation_metrics(
    config: Dict[str, Any],
    Nm: float,
    distance_m: float,
    nt_type: str,
    t_max: float = 20.0,
    dt: float = 0.01
) -> Dict[str, float]:
    """
    Calculate key propagation metrics for a given configuration.
    
    This function computes the concentration profile and extracts:
    - Peak concentration
    - Time to peak (propagation delay)
    - Full width at half maximum (FWHM)
    - 10-90% rise time
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    Nm : float
        Number of molecules released
    distance_m : float
        Distance from source in meters
    nt_type : str
        Neurotransmitter type
    t_max : float
        Maximum simulation time in seconds
    dt : float
        Time step in seconds
        
    Returns
    -------
    dict
        Dictionary containing propagation metrics
    """
    # Create time vector
    t_vec = np.arange(0, t_max, dt)
    
    # Calculate concentration profile
    c_profile = finite_burst_concentration(Nm, distance_m, t_vec, config, nt_type)
    
    # Find peak
    c_peak, t_peak = find_peak_concentration(c_profile, t_vec)
    
    # Find FWHM
    half_max = c_peak / 2
    indices_above_half = np.where(c_profile >= half_max)[0]
    if len(indices_above_half) > 0:
        t_start_half = t_vec[indices_above_half[0]]
        t_end_half = t_vec[indices_above_half[-1]]
        fwhm = t_end_half - t_start_half
    else:
        fwhm = np.nan
    
    # Find 10-90% rise time
    indices_above_10 = np.where(c_profile >= 0.1 * c_peak)[0]
    indices_above_90 = np.where(c_profile >= 0.9 * c_peak)[0]
    
    if len(indices_above_10) > 0 and len(indices_above_90) > 0:
        t_10 = t_vec[indices_above_10[0]]
        t_90 = t_vec[indices_above_90[0]]
        rise_time = t_90 - t_10
    else:
        rise_time = np.nan
    
    # Calculate characteristic diffusion time for comparison
    nt_params = get_nt_params(config, nt_type)
    D = nt_params['D_m2_s']
    lam = nt_params['lambda']
    t_diff_characteristic = (distance_m ** 2) * (lam ** 2) / D
    
    return {
        'peak_concentration_M': float(c_peak),
        'time_to_peak_s': float(t_peak),
        'fwhm_s': float(fwhm) if not np.isnan(fwhm) else np.nan,
        'rise_time_10_90_s': float(rise_time) if not np.isnan(rise_time) else np.nan,
        't_diff_characteristic_s': float(t_diff_characteristic),
        'delay_factor': float(t_peak / t_diff_characteristic) if t_diff_characteristic > 0 else float('inf')
    }


def verify_propagation_delays(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify the propagation delays reported in the manuscript.
    
    The paper reports:
    - DA: 6.3s at 100 μm
    - SERO: 4.7s at 100 μm
    
    This function checks if these values are consistent with the model.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        Comparison of calculated vs reported delays
    """
    results = {}
    
    # Test parameters
    Nm = 1e4  # 10,000 molecules (typical vesicle)
    distance_m = 100e-6  # 100 μm
    
    # Calculate for both neurotransmitters
    for nt_type in ['DA', 'SERO']:
        metrics = calculate_propagation_metrics(
            config, Nm, distance_m, nt_type
        )
        
        results[nt_type] = {
            'calculated_delay_s': metrics['time_to_peak_s'],
            'reported_delay_s': 6.3 if nt_type == 'DA' else 4.7,
            'peak_concentration_M': metrics['peak_concentration_M'],
            'characteristic_diffusion_time_s': metrics['t_diff_characteristic_s'],
            'delay_factor': metrics['delay_factor']
        }
        
        # Calculate percent difference
        calc = metrics['time_to_peak_s']
        reported = results[nt_type]['reported_delay_s']
        results[nt_type]['percent_difference'] = abs(calc - reported) / reported * 100
    
    return results