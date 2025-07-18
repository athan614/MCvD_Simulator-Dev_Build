"""
Test Poisson binding branch for high concentration regime.
"""
"python -m pytest tests/test_binding_poisson.py -v"
import numpy as np
import pytest
from scipy import signal # type: ignore[import]
from src.mc_receiver.binding import bernoulli_binding
from src.constants import get_nt_params

def test_mean_occupancy(config):
    """Test 1: Verify mean occupancy matches theory."""
    # Parameters
    C = 100e-9  # 100 nM - high concentration
    nt = 'GLU'
    n_steps = 5000
    burn_in = 1000
    
    # Create concentration time series
    conc_time = np.full(n_steps, C)
    
    # Run simulation
    rng = np.random.default_rng(42)
    bound_sites, _, _ = bernoulli_binding(conc_time, nt, config, rng)
    
    # Calculate theoretical mean
    nt_params = get_nt_params(config, nt)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    N_apt = config['N_apt']
    
    mu_th = N_apt * k_on * C / (k_on * C + k_off)
    
    # Calculate simulated mean (after burn-in)
    mu_sim = np.mean(bound_sites[burn_in:])
    
    # Check relative error
    rel_error = abs(mu_sim - mu_th) / mu_th
    assert rel_error <= 0.03, f"Mean error {rel_error:.3f} exceeds 3%"

def test_psd_lorentzian(config):
    """Test 2: Verify PSD follows Lorentzian at high concentration."""
    # Parameters
    C = 100e-9  # 100 nM
    nt = 'GLU'
    n_steps = 20000        # 200 s trace
    burn_in = 4000
    dt = config['dt_s']
    
    # Run simulation
    rng = np.random.default_rng(42)
    conc_time = np.full(n_steps, C)
    bound_sites, _, _ = bernoulli_binding(conc_time, nt, config, rng)
    
    # Get parameters
    nt_params = get_nt_params(config, nt)
    k_on = nt_params['k_on_M_s']
    k_off = nt_params['k_off_s']
    
    # ----  Convert fluctuations to current units  ----
    from src.constants import ELEMENTARY_CHARGE
    gm   = config['gm_S']
    C_tot = config['C_tot_F']
    q_eff = nt_params['q_eff_e']

    mu_sim = np.mean(bound_sites[burn_in:])
    delta_N = bound_sites[burn_in:] - mu_sim
    delta_I = gm * q_eff * ELEMENTARY_CHARGE * delta_N / C_tot
    
    # Compute PSD
    fs = 1 / dt
    nper = min(4096, len(delta_I))
    f_welch, S_sim = signal.welch(delta_I, fs=fs, nperseg=nper)
    
    # Theoretical PSD at the Lorentzian knee f_c = 1/(2π τ_B)
    k_total = k_on * C + k_off
    f_test = k_total / (2 * np.pi)      # ≈ 0.032 Hz
    idx = np.argmin(np.abs(f_welch - f_test))
    S_sim_val = S_sim[idx]
    
    # ----   Analytic reference via library helper (current PSD) ----
    from src.binding import binding_noise_psd
    S_th = binding_noise_psd(nt, config, np.array([f_test]), C_eq=C)[0]
    
    # Compare in dB (variance now ≈ ±3 dB with 20 k samples)
    delta_db = 10 * np.log10(S_sim_val / S_th)
    assert np.abs(delta_db) <= 3.0, f"PSD mismatch {delta_db:.2f} dB (knee)"

def test_low_concentration_branch(config):
    """Test 3: Verify Bernoulli branch is used at low concentration."""
    # Low concentration parameters
    C = 1e-9  # 1 nM
    nt = 'GLU'
    n_steps = 100
    dt = config['dt_s']
    
    # Calculate expected P_bind
    nt_params = get_nt_params(config, nt)
    k_on = nt_params['k_on_M_s']
    damkohler = nt_params.get('damkohler', 0.0)
    k_on_eff = k_on / (1.0 + damkohler)
    P_bind = k_on_eff * C * dt
    
    # Verify we're in Bernoulli regime
    assert P_bind < 0.1, f"P_bind={P_bind:.3f} should be < 0.1 for Bernoulli"
    
    # Run simulation without errors
    conc_time = np.full(n_steps, C)
    rng = np.random.default_rng(1)
    bound_sites, _, _ = bernoulli_binding(conc_time, nt, config, rng)
    
    # Basic sanity check - should have some binding
    assert np.any(bound_sites > 0), "Should have some binding even at 1 nM"
    assert np.all(bound_sites <= config['N_apt']), "Bound sites exceed total"