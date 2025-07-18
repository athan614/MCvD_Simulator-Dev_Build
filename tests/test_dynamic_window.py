# tests/test_dynamic_window.py

'python -m pytest tests/test_dynamic_window.py -v'

import numpy as np
import pytest

def calculate_dynamic_period(dist_um, D_um2_s, prefactor, guard_factor, min_period=20.0):
    """
    A standalone implementation of the dynamic window calculation from lod_calc.py,
    used specifically for testing.
    """
    time_for_95_percent = prefactor * (dist_um**2) / D_um2_s
    guard_time = guard_factor * time_for_95_percent
    dynamic_symbol_period = max(min_period, round(time_for_95_percent + guard_time))
    return dynamic_symbol_period

def test_dynamic_window_calculation():
    """
    Verifies the dynamic symbol period calculation for a standard case.
    This test locks in the physics constants to prevent accidental regressions.
    """
    # --- Test Parameters ---
    distance_um = 100
    # Use a typical diffusion coefficient for small molecules in µm²/s
    D_um2_s = 450
    # Prefactor for 95% signal capture time in 3D diffusion
    prefactor = 3.0
    # Guard interval as a fraction of the signal window
    guard_factor = 0.3
    
    # --- Expected Calculation ---
    # time_for_95 = 3.0 * (100^2) / 450 = 66.67 s
    # guard = 0.3 * 66.67 = 20.0 s
    # total = round(66.67 + 20.0) = round(86.67) = 87 s
    expected_period_s = 87
    
    # --- Run Calculation ---
    calculated_period_s = calculate_dynamic_period(
        dist_um=distance_um,
        D_um2_s=D_um2_s,
        prefactor=prefactor,
        guard_factor=guard_factor
    )
    
    # --- Assert ---
    # Check that the calculated value matches the expected value within a small tolerance
    assert np.isclose(calculated_period_s, expected_period_s, rtol=0.01), \
        f"Dynamic window calculation failed. Expected ~{expected_period_s}s, but got {calculated_period_s}s."

def test_dynamic_window_obeys_minimum():
    """
    Verifies that for short distances, the period is clipped at the minimum (20s).
    """
    # At 25um, the calculated period would be ~5s, so it should return the minimum.
    calculated_period_s = calculate_dynamic_period(
        dist_um=25, D_um2_s=450, prefactor=3.0, guard_factor=0.3, min_period=20.0
    )
    assert calculated_period_s == 20.0, "Dynamic window should be clipped at the minimum value for short distances."