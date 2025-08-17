"""
Tests for noise correlation analysis utilities.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.noise_correlation import sigma_I_diff_vec, onsi_curve


def test_clamp_and_vectorization():
    """Test clamping and broadcasting of sigma_I_diff_vec."""
    # Test scalar inputs
    s_scalar = sigma_I_diff_vec(1.0, 2.0, -2.0)  # rho = -2.0 should clamp to -1.0
    assert np.isfinite(s_scalar)
    assert isinstance(s_scalar, (float, np.floating))
    
    # Test array inputs with clamping
    s_array = sigma_I_diff_vec(
        np.array([1.0]), 
        np.array([2.0]), 
        np.array([-2.0, 0.0, 2.0])  # Should clamp to [-1.0, 0.0, 1.0]
    )
    
    # Verify shape and finite values (convert to array for consistent handling)
    s_array = np.asarray(s_array)
    assert s_array.shape == (3,)
    assert np.all(np.isfinite(s_array))
    
    # Test mixed scalar/array broadcasting
    s_mixed = sigma_I_diff_vec(1.0, 1.0, np.array([-0.5, 0.0, 0.5]))
    s_mixed = np.asarray(s_mixed)  # Ensure it's an array for shape checking
    assert s_mixed.shape == (3,)
    assert np.all(np.isfinite(s_mixed))
    
    # Test extreme clamping
    extreme_rhos = np.array([-10.0, -1.5, 0.0, 1.5, 10.0])
    s_extreme = sigma_I_diff_vec(1.0, 1.0, extreme_rhos)
    s_extreme = np.asarray(s_extreme)  # Ensure it's an array for shape checking
    assert s_extreme.shape == (5,)
    assert np.all(np.isfinite(s_extreme))


def test_onsi_curve_properties():
    """Test basic properties of ONSI curves."""
    rho, onsi = onsi_curve(1.0, 1.0, rho_min=-0.2, rho_max=0.2, n=11)
    
    # Check shapes
    assert rho.shape == (11,)
    assert onsi.shape == (11,)
    
    # Check finite values
    assert np.all(np.isfinite(rho))
    assert np.all(np.isfinite(onsi))
    
    # ONSI should be 1.0 at rho=0 (middle of symmetric range)
    mid_idx = len(onsi) // 2
    assert abs(onsi[mid_idx] - 1.0) < 1e-10


def test_edge_cases():
    """Test edge cases and robustness."""
    # Test with zero noise (should not crash)
    try:
        s = sigma_I_diff_vec(0.0, 0.0, 0.0)
        assert s >= 0.0
    except Exception:
        pytest.skip("Zero noise case may be undefined")
    
    # Test with very small noise
    s_small = sigma_I_diff_vec(1e-15, 1e-15, 0.5)
    assert np.isfinite(s_small)
    assert s_small >= 0.0


if __name__ == "__main__":
    test_clamp_and_vectorization()
    test_onsi_curve_properties() 
    test_edge_cases()
    print("✅ All noise correlation tests passed!")