"""
Test OECT gain linearity per manuscript Eq. 24.
Verifies ΔI_D = (g_m · q_eff · e / C_tot) · N_b is strictly linear.
"""

"python -m pytest tests/test_oect_gain.py -v"

import numpy as np
import pytest
from src.mc_receiver.oect import oect_static_gain
from src.constants import ELEMENTARY_CHARGE


@pytest.fixture
def test_config():
    """Test configuration for OECT gain validation."""
    return {
        'neurotransmitters': {
            'GLU':  {'q_eff_e': 0.6},
            'GABA': {'q_eff_e': 0.2},
        },
        'oect': {
            'gm_S':   0.002,
            'C_tot_F': 1.8e-8,
        },
    }


def test_oect_gain_linearity(test_config):
    """
    Verify OECT transduction is perfectly linear (R² > 0.999).
    Test both GLU and GABA channels.
    """
    for nt in ['GLU', 'GABA']:
        # Test points: 0 to 5000 bound sites
        N_b_values = np.array([0, 1e3, 2e3, 3e3, 4e3, 5e3])
        I_d_values = np.array([oect_static_gain(N, nt, test_config) for N in N_b_values])
        
        # Linear regression: I_d = m * N_b + b
        m, b = np.polyfit(N_b_values, I_d_values, 1)
        I_d_fit = m * N_b_values + b
        
        # Calculate R²
        ss_res = np.sum((I_d_values - I_d_fit)**2)
        ss_tot = np.sum((I_d_values - I_d_values.mean())**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 1.0
        
        # Test 1: R² should exceed 0.999
        assert R2 > 0.999, f"{nt}: R² = {R2:.6f} < 0.999"
        
        # Test 2: Slope matches theoretical value within 1%
        q_eff = test_config['neurotransmitters'][nt]['q_eff_e']
        gm    = test_config['oect']['gm_S']
        C_tot = test_config['oect']['C_tot_F']
        m_theory = gm * q_eff * ELEMENTARY_CHARGE / C_tot
        
        relative_error = abs(m - m_theory) / m_theory
        assert relative_error < 0.01, \
            f"{nt}: Slope error {relative_error:.1%} exceeds 1%"
        
        # Test 3: Intercept should be essentially zero
        assert abs(b) < 1e-12, f"{nt}: Non-zero intercept b = {b:.2e} A"


def test_oect_gain_values(test_config):
    """Verify specific gain values match expected calculations."""
    # GLU with 1 million bound sites
    N_b = 1e6
    I_glu = oect_static_gain(N_b, 'GLU', test_config)
    
    # Expected: gm * q_eff * e * N_b / C_tot
    # = 0.002 * 0.6 * 1.602e-19 * 1e6 / 1.8e-8
    expected_glu = 0.002 * 0.6 * ELEMENTARY_CHARGE * 1e6 / 1.8e-8
    
    assert np.isclose(I_glu, expected_glu, rtol=1e-10), \
        f"GLU current {I_glu:.3e} A != expected {expected_glu:.3e} A"
    
    # GABA should be 1/3 of GLU (q_eff ratio 0.2/0.6)
    I_gaba = oect_static_gain(N_b, 'GABA', test_config)
    assert np.isclose(I_gaba/I_glu, 0.2/0.6, rtol=1e-10), \
        "GABA/GLU current ratio incorrect"