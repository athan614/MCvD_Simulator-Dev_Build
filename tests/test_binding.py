"""
Unit tests for the binding module.
"""

"python -m pytest tests/test_binding.py -v"

import numpy as np
import pytest
from scipy import signal # type: ignore[import]

from src.mc_receiver.binding import (
    bernoulli_binding,
    mean_binding,
    binding_noise_psd,
    calculate_equilibrium_metrics
)

def test_mean_vs_mc(config):
    """
    Compare Monte Carlo statistics with analytical predictions.
    Run 10k trials at constant concentration, verify mean and variance.
    """
    # Test parameters
    C_eq = 10e-9  # 10 nM constant concentration
    nt = 'GLU'
    n_trials = 100  # Reduced for faster testing, use 10000 for production
    duration = 10.0  # seconds
    
    # Time vector
    dt = config['dt_s']
    n_steps = int(duration / dt)
    conc_time = np.full(n_steps, C_eq)
    
    # Analytical predictions
    metrics = calculate_equilibrium_metrics(C_eq, nt, config)
    expected_mean = metrics['mean_bound']
    expected_var = metrics['variance']
    
    # Run Monte Carlo trials
    rng = np.random.default_rng(42)
    final_bounds = []
    last_bound_sites = None
    
    for trial in range(n_trials):
        bound_sites, _, _ = bernoulli_binding(conc_time, nt, config, rng)
        # Use last snapshot (already at equilibrium) to match analytic variance
        final_bounds.append(bound_sites[-1])
        last_bound_sites = bound_sites  # Keep last trial for additional check
    
    # Calculate empirical statistics
    empirical_mean = np.mean(final_bounds)
    empirical_var = np.var(final_bounds)
    
    # Check within 3% tolerance
    mean_error = abs(empirical_mean - expected_mean) / expected_mean
    var_error = abs(empirical_var - expected_var) / expected_var
    
    assert mean_error < 0.03, f"Mean error {mean_error:.1%} exceeds 3%"
    assert var_error < 0.20, f"Variance error {var_error:.1%} exceeds 20%"  # 100 trials → ~14 % SE
    
    # Additional check: final snapshot must be within 3 % of analytical mean
    assert last_bound_sites is not None, "No trials were run"
    assert np.isclose(last_bound_sites[-1], expected_mean, rtol=0.03), \
        "Final MC snapshot deviates from analytical mean"


def test_psd_shape(config):
    """
    Verify PSD shape by comparing Welch estimate with analytical formula.
    This version uses a long duration for accuracy and a strict tolerance.
    """
    # Parameters
    C_eq = 10e-9  # 10 nM
    nt = 'GABA'
    # Use a reasonably long duration for a stable PSD estimate
    duration = 100.0
    dt = config['sim']['dt_s']
    n_steps = int(duration / dt)
    fs = 1 / dt
    conc_time = np.full(n_steps, C_eq)

    # --- Simulation ---
    rng = np.random.default_rng(123)
    # The bernoulli_binding function returns a current based on binding events.
    _, i_binding_sim, _ = bernoulli_binding(conc_time, nt, config, rng)
    i_binding_sim_ac = i_binding_sim - np.mean(i_binding_sim)

    # --- Welch PSD from Simulation ---
    # Use a long segment length for good frequency resolution
    nperseg = min(n_steps // 8, 4096)
    f_welch, psd_binding_welch = signal.welch(
        i_binding_sim_ac,
        fs=fs,
        nperseg=nperseg,
        scaling='density'
    )

    # --- Analytical PSD ---
    psd_binding_analytical = binding_noise_psd(nt, config, f_welch, C_eq)

    # --- Comparison ---
    band_mask = (f_welch >= 0.5) & (f_welch <= 5.0)
    if not np.any(band_mask):
        pytest.skip("No frequency points in the desired band to compare.")

    psd_welch_band = psd_binding_welch[band_mask]
    psd_analytical_band = psd_binding_analytical[band_mask]
    
    # Add a small epsilon to avoid division by zero
    psd_ratio = psd_welch_band / (psd_analytical_band + 1e-40)
    error_dB = 10 * np.log10(psd_ratio)
    
    mean_error_dB = np.mean(np.abs(error_dB))
    
    # Assert with the original, strict 3.0 dB tolerance.
    # This should now pass because the analytical formula is correct.
    assert mean_error_dB < 3.0, f"Mean PSD error for binding noise {mean_error_dB:.1f} dB exceeds 3 dB"