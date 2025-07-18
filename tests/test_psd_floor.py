"""
Test F-2: Verify noise constants and PSD floor.
"""

"python -m pytest tests/test_psd_floor.py -v"

import numpy as np
import pytest
import yaml
from pathlib import Path
from src.config_utils import preprocess_config


def load_baseline_config():
    """Load and preprocess the baseline configuration."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return preprocess_config(cfg)


def test_yaml_noise_constants():
    """Verify YAML contains correct noise parameters per manuscript."""
    cfg = load_baseline_config()
    
    # Check Hooge parameter
    assert np.isclose(cfg['alpha_H'], 3e-3, rtol=0, atol=0), \
        f"alpha_H = {cfg['alpha_H']} should be 3e-3"
    
    # Check drift coefficient
    assert np.isclose(cfg['K_d_Hz'], 1.3e-4, rtol=0, atol=0), \
        f"K_d_Hz = {cfg['K_d_Hz']} should be 1.3e-4"
    
    # Verify N_c is present and reasonable
    assert 'N_c' in cfg, "N_c (number of carriers) missing from config"
    assert cfg['N_c'] > 1e10, f"N_c = {cfg['N_c']} seems too low"


def test_psd_floor_estimate():
    """Verify theoretical PSD floor ≤ 1×10⁻¹⁵ A²/Hz at 1 Hz for 1 µA current."""
    cfg = load_baseline_config()
    
    # Test conditions
    I_D = 1e-6  # 1 µA typical drain current
    f = 1.0     # 1 Hz
    TARGET = 1e-15  # A²/Hz - realistic target for OECT noise floor
    
    # Extract parameters
    alpha_H = cfg['alpha_H']  # 3e-3
    K_d = cfg['K_d_Hz']       # 1.3e-4
    N_c = cfg['N_c']          # 1e17
    
    # Calculate PSD components
    S_flicker = alpha_H * I_D**2 / (N_c * f)
    S_drift = K_d * I_D**2 / f**2  # Current-normalized coefficient
    S_total = S_flicker + S_drift
    
    # Log components for debugging
    print(f"\nPSD components at f=1 Hz, I_D=1 µA:")
    print(f"  Flicker (1/f):   {S_flicker:.2e} A²/Hz")
    print(f"  Drift (1/f²):    {S_drift:.2e} A²/Hz")
    print(f"  Total:           {S_total:.2e} A²/Hz")
    print(f"  Target:          ≤ {TARGET:.0e} A²/Hz")
    
    assert S_total <= TARGET, \
        f"PSD floor {S_total:.2e} A²/Hz exceeds {TARGET:.0e} A²/Hz at 1 Hz"


def test_psd_frequency_dependence():
    """Verify correct frequency scaling of noise components."""
    cfg = load_baseline_config()
    
    I_D = 1e-6
    frequencies = [0.1, 1.0, 10.0, 100.0]
    
    alpha_H = cfg['alpha_H']
    K_d = cfg['K_d_Hz']
    N_c = cfg['N_c']
    
    for i in range(len(frequencies) - 1):
        f1, f2 = frequencies[i], frequencies[i + 1]
        
        # Flicker noise should scale as 1/f
        S_flicker_1 = alpha_H * I_D**2 / (N_c * f1)
        S_flicker_2 = alpha_H * I_D**2 / (N_c * f2)
        ratio_flicker = S_flicker_1 / S_flicker_2
        assert np.isclose(ratio_flicker, f2/f1, rtol=1e-6), \
            f"Flicker noise not scaling as 1/f"
        
        # Drift noise should scale as 1/f²
        S_drift_1 = K_d * I_D**2 / f1**2  # Current-normalized
        S_drift_2 = K_d * I_D**2 / f2**2  # Current-normalized
        ratio_drift = S_drift_1 / S_drift_2
        assert np.isclose(ratio_drift, (f2/f1)**2, rtol=1e-6), \
            f"Drift noise not scaling as 1/f²"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])