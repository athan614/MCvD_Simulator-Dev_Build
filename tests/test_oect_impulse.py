"""
Test OECT dynamic response characteristics.
Validates τ_OECT = 1.6 ms impulse response properties.
"""

"python -m pytest tests/test_oect_impulse.py -v"

import numpy as np
from src.mc_receiver.oect import oect_impulse_response

# Minimal config with τ_OECT = 1.6 ms
cfg_test = {
    'oect': {
        'tau_OECT_s': 1.6e-3,
    }
}

def test_impulse_response_area_and_t90():
    """Impulse response must integrate to 1 and reach 90% area at ~2.3 τ."""
    tau = cfg_test['oect']['tau_OECT_s']
    dt = 2e-5               # 0.02 ms resolution (80 samples per τ)
    n  = int(0.025 / dt)    # simulate 25 ms (> 15 τ)

    h = oect_impulse_response(dt, n, cfg_test)

    # 1. Unit area check (use trapezoidal integration for better accuracy)
    area = np.trapezoid(h, dx=dt)        # silence deprecation warning
    assert np.isclose(area, 1.0, rtol=1e-2), f"Area = {area}"

    # 2. 90-percent settling time ~ 2.3026 τ  (within 5%)
    cumsum = np.cumsum(h) * dt
    idx_90 = np.searchsorted(cumsum, 0.9)
    t_90   = idx_90 * dt
    t_90_theory = 2.302585093 * tau    # ln(10) τ
    assert np.isclose(t_90, t_90_theory, rtol=0.05), \
        f"t90 {t_90:.4e} vs theory {t_90_theory:.4e}"


def test_impulse_response_shape():
    """Verify impulse response has correct shape and peak."""
    tau = cfg_test['oect']['tau_OECT_s']
    dt = 1e-5               # 10 μs fine sampling
    n = int(0.01 / dt)      # 10 ms duration
    
    h = oect_impulse_response(dt, n, cfg_test)
    
    # Peak should be at t=0 with value 1/τ
    assert h[0] == 1.0 / tau, f"h[0] = {h[0]}, expected {1.0/tau}"
    
    # Should decay monotonically
    assert all(h[i] >= h[i+1] for i in range(len(h)-1)), \
        "Impulse response should decay monotonically"
    
    # Check specific values at τ and 2τ
    idx_tau = int(tau / dt)
    idx_2tau = int(2 * tau / dt)
    
    # h(τ) = (1/τ) * exp(-1) ≈ 0.368/τ
    expected_at_tau = (1.0 / tau) * np.exp(-1)
    assert np.isclose(h[idx_tau], expected_at_tau, rtol=1e-2), \
        f"h(τ) = {h[idx_tau]}, expected {expected_at_tau}"
    
    # h(2τ) = (1/τ) * exp(-2) ≈ 0.135/τ
    expected_at_2tau = (1.0 / tau) * np.exp(-2)
    assert np.isclose(h[idx_2tau], expected_at_2tau, rtol=1e-2), \
        f"h(2τ) = {h[idx_2tau]}, expected {expected_at_2tau}"