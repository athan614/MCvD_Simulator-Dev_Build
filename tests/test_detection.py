# tests/test_detection.py (updated with patch)
"""
Unit tests for detection module.
"""

"python -m pytest tests/test_detection.py -v"

import numpy as np
import pytest
from pathlib import Path
import sys
import copy
from src.mc_detection import (
    detect_mosk, 
    detect_csk_binary,
    detect_csk_mary,
    calculate_ml_threshold,
    ber_mosk_analytic,
    sep_csk_binary,
    sep_csk_mary,
    calculate_snr,
    snr_sweep
)
from src.pipeline import run_sequence
from analysis.run_final_analysis import calculate_dynamic_symbol_period

def test_mosk_detection(config):
    """Test MoSK detector distinguishes DA from SERO (if standalone function exists)."""
    dt = config['dt_s']
    n_samples = int(0.5 / dt)  # 0.5s window
    
    # DA stronger than SERO
    i_da = np.full(n_samples, 2e-9)   # 2 nA
    i_sero = np.full(n_samples, 1e-9)  # 1 nA
    i_ctrl = np.full(n_samples, 0.1e-9)  # Small control current
    
    assert n_samples == len(i_da)  # sanity guard
    
    # Should detect DA (return 0)
    assert detect_mosk(i_da, i_sero, i_ctrl, config) == 0
    
    # Swap currents - should detect SERO (return 1)
    assert detect_mosk(i_sero, i_da, i_ctrl, config) == 1


def test_tri_channel_mosk_detection(config):
    """Test MoSK detection as implemented in the tri-channel pipeline."""
    cfg = copy.deepcopy(config)          # <- NEW: isolate this test
    cfg['pipeline']['modulation'] = 'MoSK'
    cfg['pipeline']['sequence_length'] = 100
    cfg['pipeline']['Nm_per_symbol'] = 1e4
    cfg['pipeline']['random_seed'] = 42
    cfg['pipeline']['enable_isi'] = True

    # --- NEW: physics‑based Ts so ISI is tolerable ---
    dist = cfg['pipeline']['distance_um']          # default 100 µm
    Ts   = calculate_dynamic_symbol_period(dist, cfg)
    cfg['pipeline']['symbol_period_s'] = Ts
    cfg['pipeline']['time_window_s']   = Ts
    # --------------------------------------------------
    
    # Run a sequence
    result = run_sequence(cfg)
    
    # Basic checks
    assert result['modulation'] == 'MoSK'
    assert len(result['symbols_tx']) == 100
    assert len(result['symbols_rx']) == 100
    assert 'SER' in result
    
    # With reasonable Nm, SER should be low
    assert result['SER'] < 0.2, f"SER {result['SER']:.2%} too high for MoSK"
    
    # Check that stats are collected
    assert 'stats_da' in result
    assert 'stats_sero' in result
    
    # Stats should be decision statistics (q_da - q_sero)
    # When DA is sent (tx=0), stats should be positive
    # When SERO is sent (tx=1), stats should be negative
    tx_symbols = result['symbols_tx']
    stats_da = result['stats_da']
    stats_sero = result['stats_sero']
    
    # Count of stats should match number of DA/SERO transmissions
    n_da_sent = sum(1 for s in tx_symbols if s == 0)
    n_sero_sent = sum(1 for s in tx_symbols if s == 1)
    
    assert len(stats_da) == n_da_sent, "Stats count mismatch for DA"
    assert len(stats_sero) == n_sero_sent, "Stats count mismatch for SERO"
    
    # Average stats should have correct sign
    if stats_da:  # If any DA was sent
        assert np.mean(stats_da) > 0, "DA stats should be positive on average"
    if stats_sero:  # If any SERO was sent
        assert np.mean(stats_sero) < 0, "SERO stats should be negative on average"


def test_csk_binary_detection(config):
    """Test binary CSK with ML threshold (if standalone function exists)."""
    n_samples = int(config['detection']['decision_window_s'] / config['sim']['dt_s'])
    i_ctrl = np.zeros(n_samples)
    threshold = 1.0e-9  # Threshold is 1.0 nA average current

    # Test Case 1: Signal is BELOW threshold
    i_low = np.random.normal(loc=0.5e-9, scale=0.1e-9, size=n_samples)
    assert detect_csk_binary(i_low, i_ctrl, threshold, config) == 0

    # Test Case 2: Signal is ABOVE threshold
    i_high = np.random.normal(loc=1.5e-9, scale=0.1e-9, size=n_samples)
    assert detect_csk_binary(i_high, i_ctrl, threshold, config) == 1


def test_tri_channel_csk_detection(config):
    """Test CSK detection as implemented in the tri-channel pipeline."""
    cfg = copy.deepcopy(config)
    cfg['pipeline']['modulation'] = 'CSK'
    cfg['pipeline']['sequence_length'] = 100
    cfg['pipeline']['Nm_per_symbol'] = 1e4
    cfg['pipeline']['random_seed'] = 42
    cfg['pipeline']['enable_isi'] = True        # keep ISI on
    cfg['pipeline']['csk_levels'] = 4  # 4-level CSK (0, 1, 2, 3)
    cfg['pipeline']['csk_target_channel'] = 'DA'  # Which channel to use for CSK
    cfg['pipeline']['csk_thresholds_da'] = [-1e-9, 0, 1e-9]  # 3 thresholds for 4 levels

    # <- new block ------------------------------------------------------------
    dist = cfg['pipeline']['distance_um']        # 100 µm by default
    Ts   = calculate_dynamic_symbol_period(dist, config)
    cfg['pipeline']['symbol_period_s'] = Ts
    cfg['pipeline']['time_window_s']   = Ts
    # -------------------------------------------------------------------------
    
    # Run a sequence
    result = run_sequence(cfg)
    
    # Basic checks
    assert result['modulation'] == 'CSK'
    assert len(result['symbols_tx']) == 100
    assert len(result['symbols_rx']) == 100
    
    # Check that symbols are in correct range
    assert all(0 <= s < 4 for s in result['symbols_tx']), "TX symbols out of range"
    assert all(0 <= s < 4 for s in result['symbols_rx']), "RX symbols out of range"
    
    # Stats should be collected for the target channel
    assert 'stats_da' in result
    assert len(result['stats_da']) == 100, "Should have stats for all symbols"


def test_tri_channel_hybrid_detection(config):
    """Test Hybrid detection as implemented in the tri-channel pipeline."""
    cfg = copy.deepcopy(config)
    # Configure for Hybrid
    cfg['pipeline']['modulation'] = 'Hybrid'
    cfg['pipeline']['sequence_length'] = 100
    cfg['pipeline']['Nm_per_symbol'] = 1e4
    cfg['pipeline']['random_seed'] = 42
    
    # Need to set thresholds for Hybrid
    cfg['pipeline']['hybrid_threshold_da'] = 0  # Example threshold
    cfg['pipeline']['hybrid_threshold_sero'] = 0  # Example threshold
    
    # Run a sequence
    result = run_sequence(cfg)
    
    # Basic checks
    assert result['modulation'] == 'Hybrid'
    assert len(result['symbols_tx']) == 100
    assert len(result['symbols_rx']) == 100
    
    # Hybrid uses 4 symbols (2 bits)
    assert all(0 <= s < 4 for s in result['symbols_tx']), "TX symbols out of range"
    assert all(0 <= s < 4 for s in result['symbols_rx']), "RX symbols out of range"
    
    # Check subsymbol errors
    assert 'subsymbol_errors' in result
    assert 'mosk' in result['subsymbol_errors']
    assert 'csk' in result['subsymbol_errors']
    
    # Total errors should be sum of subsymbol errors
    total_errors = result['errors']
    subsymbol_total = result['subsymbol_errors']['mosk'] + result['subsymbol_errors']['csk']
    assert total_errors >= subsymbol_total, "Total errors should include all subsymbol errors"


def test_ml_threshold_calculation():
    """Test ML threshold calculation for different noise scenarios (if function exists)."""
    # Equal noise case
    mu_0, mu_1 = 1e-9, 2e-9
    sigma_0 = sigma_1 = 0.1e-9
    threshold = calculate_ml_threshold(mu_0, mu_1, sigma_0, sigma_1)
    assert np.isclose(threshold, 1.5e-9), "Equal noise threshold should be midpoint"
    
    # Unequal noise case
    sigma_1 = 0.2e-9  # Higher noise for symbol 1
    threshold = calculate_ml_threshold(mu_0, mu_1, sigma_0, sigma_1)
    # Threshold should shift toward the noisier symbol
    assert mu_0 < threshold < 1.5e-9, "Threshold should shift toward lower noise symbol"


def test_analytical_ber_curves():
    """Test analytical BER calculations (if functions exist)."""
    # Test points
    snr_dB = np.array([0, 5, 10, 15])
    snr_lin = 10**(snr_dB / 10)
    
    # MoSK BER
    ber_mosk = ber_mosk_analytic(snr_lin, snr_lin)
    assert np.all(ber_mosk > 0) and np.all(ber_mosk < 0.5)
    assert np.all(np.diff(ber_mosk) < 0), "BER should decrease with SNR"
    
    # Binary CSK
    sep_binary = sep_csk_binary(snr_lin)
    assert np.allclose(sep_binary, ber_mosk), "Binary CSK should match MoSK for equal SNR"
    
    # 4-ary CSK
    sep_4ary = sep_csk_mary(snr_lin, M=4)
    assert np.all(sep_4ary > sep_binary), "M-ary should have higher SEP than binary"


def test_detection_performance_comparison(config):
    cfg = copy.deepcopy(config)
    """Compare detection performance across modulation schemes."""
    # Common parameters
    cfg['pipeline']['sequence_length'] = 200
    cfg['pipeline']['Nm_per_symbol'] = 1e4
    cfg['pipeline']['random_seed'] = 42
    
    results = {}
    
    # Test MoSK
    cfg['pipeline']['modulation'] = 'MoSK'
    results['MoSK'] = run_sequence(cfg)
    
    # Test CSK (binary)
    cfg['pipeline']['modulation'] = 'CSK'
    cfg['pipeline']['csk_levels'] = 2
    cfg['pipeline']['csk_target_channel'] = 'DA'
    cfg['pipeline']['csk_thresholds_da'] = [0]  # Single threshold for binary
    results['CSK-2'] = run_sequence(cfg)
    
    # Test CSK (4-level)
    cfg['pipeline']['csk_levels'] = 4
    cfg['pipeline']['csk_thresholds_da'] = [-1e-9, 0, 1e-9]  # 3 thresholds
    results['CSK-4'] = run_sequence(cfg)
    
    # Compare SER
    print("\nDetection Performance Comparison:")
    for scheme, result in results.items():
        print(f"{scheme}: SER = {result['SER']:.2%}")
    
    # Generally expect: MoSK < CSK-2 < CSK-4 in terms of SER
    # But this depends on SNR and other factors
    assert all(0 <= r['SER'] <= 1 for r in results.values()), "All SER should be valid"


def test_snr_calculation(config):
    """Test SNR calculation from bound aptamer counts (if function exists)."""
    # Parameters
    gm = config['gm_S']
    C_tot = config['C_tot_F']
    q_eff = config['neurotransmitters']['DA']['q_eff_e']
    
    # Bound aptamer counts
    mu_signal = 1e6    # 1 million bound
    mu_reference = 0.5e6  # 0.5 million bound
    noise_variance = 1e-20  # 10 pA RMS noise squared
    
    snr = calculate_snr(mu_signal, mu_reference, noise_variance, gm, q_eff, config['oect']['C_tot_F'])
    
    assert snr > 0, "SNR should be positive"
    # Check order of magnitude is reasonable
    assert 1 < snr < 1e6, f"SNR {snr} seems unrealistic"