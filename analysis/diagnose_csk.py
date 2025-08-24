# analysis/diagnose_csk.py
"""
Diagnostic script to identify why CSK is failing.
"""

import sys
from pathlib import Path
import numpy as np
import yaml
from copy import deepcopy
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline import _single_symbol_currents, calculate_proper_noise_sigma
from src.config_utils import preprocess_config
from src.constants import get_nt_params

def diagnose_csk_levels():
    """Diagnose CSK signal generation and detection."""
    
    # Load configuration
    with open(project_root / "config" / "default.yaml") as f:
        config_base = yaml.safe_load(f)
    
    # Preprocess
    from analysis.run_final_analysis import preprocess_config_full
    cfg = preprocess_config_full(config_base)
    
    # Configure for CSK
    cfg['pipeline']['modulation'] = 'CSK'
    cfg['pipeline']['csk_levels'] = 4
    cfg['pipeline']['csk_target_channel'] = 'GLU'
    cfg['pipeline']['csk_level_scheme'] = 'uniform'
    cfg['pipeline']['Nm_per_symbol'] = 10000
    cfg['pipeline']['distance_um'] = 50
    cfg['pipeline']['symbol_period_s'] = 30
    cfg['pipeline']['sequence_length'] = 100
    cfg['pipeline']['enable_molecular_noise'] = True
    cfg['pipeline']['enable_isi'] = False  # Disable ISI for clarity
    cfg['detection']['decision_window_s'] = 30
    cfg['sim']['dt_s'] = 0.01
    cfg['verbose'] = False
    cfg['disable_progress'] = True
    
    print("="*60)
    print("CSK DIAGNOSTIC TEST")
    print("="*60)
    print(f"Configuration:")
    print(f"  Modulation: CSK-4")
    print(f"  Target channel: {cfg['pipeline']['csk_target_channel']}")
    print(f"  Level scheme: {cfg['pipeline']['csk_level_scheme']}")
    print(f"  Nm_per_symbol: {cfg['pipeline']['Nm_per_symbol']}")
    print(f"  Distance: {cfg['pipeline']['distance_um']}μm")
    print()
    
    # Import the dual-channel helper
    from src.pipeline import _csk_dual_channel_Q, calculate_proper_noise_sigma
    
    # Calculate noise parameters once
    sigma_glu, sigma_gaba = calculate_proper_noise_sigma(cfg, cfg['detection']['decision_window_s'])
    rho_cc = float(cfg.get('noise', {}).get('rho_between_channels_after_ctrl', 0.0))
    rho_cc = max(-1.0, min(1.0, rho_cc))
    
    # Get CSK combiner settings
    combiner = cfg['pipeline'].get('csk_combiner', 'zscore')
    use_dual = bool(cfg['pipeline'].get('csk_dual_channel', True))
    leakage = float(cfg['pipeline'].get('csk_leakage_frac', 0.0))
    target_channel = cfg['pipeline']['csk_target_channel']
    
    print(f"  Combiner: {combiner} (dual={use_dual})")
    print(f"  Target channel: {target_channel}")
    print(f"  Noise: σ_GLU={sigma_glu:.2e}, σ_GABA={sigma_gaba:.2e}, ρ={rho_cc:+.3f}")
    if combiner == 'leakage':
        print(f"  Leakage fraction: {leakage:.3f}")
    print()
    
    # Test each symbol level
    rng = np.random.default_rng(42)
    tx_history = []
    
    results_per_symbol = {s: {'q_values': [], 'nm_actual': []} for s in range(4)}
    
    print("Testing 20 samples per symbol level...")
    print("-"*40)
    
    for symbol in range(4):
        print(f"\nSymbol {symbol}:")
        
        for trial in range(20):
            # Generate currents
            currents_glu, currents_gaba, currents_ctrl, nm_actual = _single_symbol_currents(
                symbol, tx_history, cfg, rng
            )
            
            # Integration
            q_glu = float(np.trapz(currents_glu, dx=cfg['sim']['dt_s']))
            q_gaba = float(np.trapz(currents_gaba, dx=cfg['sim']['dt_s']))
            
            # Compute dual-channel Q
            if use_dual:
                Q = _csk_dual_channel_Q(
                    q_glu=q_glu, q_gaba=q_gaba,
                    sigma_glu=sigma_glu, sigma_gaba=sigma_gaba,
                    rho_cc=rho_cc, combiner=combiner, leakage_frac=leakage,
                    target=target_channel
                )
            else:
                # Legacy single-channel
                Q = q_glu if target_channel == 'GLU' else q_gaba
                
            results_per_symbol[symbol]['q_values'].append(Q)
            results_per_symbol[symbol]['nm_actual'].append(nm_actual)
        
        # Statistics for this symbol
        q_values = results_per_symbol[symbol]['q_values']
        nm_values = results_per_symbol[symbol]['nm_actual']
        
        print(f"  Nm actual: mean={np.mean(nm_values):.0f}, std={np.std(nm_values):.0f}")
        print(f"  Q_comb:    mean={np.mean(q_values):.3e}, std={np.std(q_values):.3e}")
        print(f"  Range: [{np.min(q_values):.3e}, {np.max(q_values):.3e}]")
    
    # Check for overlaps
    print("\n" + "="*40)
    print("OVERLAP ANALYSIS:")
    print("-"*40)
    
    for i in range(3):
        q_low = results_per_symbol[i]['q_values']
        q_high = results_per_symbol[i+1]['q_values']
        
        overlap = (np.max(q_low) > np.min(q_high))
        
        print(f"Symbols {i} and {i+1}:")
        print(f"  Level {i}: max = {np.max(q_low):.3e}")
        print(f"  Level {i+1}: min = {np.min(q_high):.3e}")
        print(f"  Overlap: {'YES ❌' if overlap else 'NO ✓'}")
        
        if overlap:
            overlap_pct = sum(1 for q_l in q_low for q_h in q_high if q_l > q_h) / (len(q_low) * len(q_high)) * 100
            print(f"  Overlap %: {overlap_pct:.1f}%")
    
    # Visualize
    print("\n" + "="*40)
    print("VISUALIZATION:")
    
    plt.figure(figsize=(10, 6))
    
    for symbol in range(4):
        q_values = results_per_symbol[symbol]['q_values']
        plt.scatter([symbol]*len(q_values), q_values, alpha=0.6, s=50, label=f'Symbol {symbol}')
        
        # Add mean and std lines
        mean_q = float(np.mean(q_values))
        std_q = float(np.std(q_values))
        plt.hlines(mean_q, symbol-0.3, symbol+0.3, colors='red', linewidth=2)
        plt.hlines([mean_q - std_q, mean_q + std_q], symbol-0.2, symbol+0.2, colors='red', linewidth=1, linestyles='dashed')
    
    plt.xlabel('Symbol')
    plt.ylabel('Charge Q (C)')
    plt.title('CSK Symbol Detection - Charge Distribution per Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = project_root / "results" / "figures" / "csk_diagnostic.png"
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.close()  # Close figure to free memory
    
    # Calculate expected thresholds FIRST
    print("\n" + "="*40)
    print("SUGGESTED THRESHOLDS:")
    print("-"*40)
    
    try:
        from src.detection import calculate_ml_threshold
    except ImportError:
        print("Error importing calculate_ml_threshold")
        from src.mc_detection.algorithms import calculate_ml_threshold
    
    thresholds = []
    for i in range(3):
        q_low = results_per_symbol[i]['q_values']
        q_high = results_per_symbol[i+1]['q_values']
        
        mean_low = float(np.mean(q_low))
        mean_high = float(np.mean(q_high))
        std_low = float(np.std(q_low))
        std_high = float(np.std(q_high))
        
        try:
            threshold = calculate_ml_threshold(mean_low, mean_high, std_low, std_high)
            thresholds.append(threshold)
            print(f"Threshold {i}→{i+1}: {threshold:.3e}")
        except Exception as e:
            print(f"Error calculating threshold {i}→{i+1}: {e}")
            # Fallback to midpoint
            threshold = (mean_low + mean_high) / 2
            thresholds.append(threshold)
            print(f"Threshold {i}→{i+1} (fallback): {threshold:.3e}")
    
    # Stage 14: Save threshold and overlap tables (AFTER thresholds are calculated)
    out_dir = project_root / "results"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    
    # 1) Thresholds CSV
    with open(out_dir / "tables" / "csk_thresholds_ml.csv", "w", newline='', encoding='utf-8') as f:  # Add encoding='utf-8'
        writer = csv.writer(f)
        writer.writerow(["pair", "threshold"])
        for i, thr in enumerate(thresholds, start=1):
            writer.writerow([f"{i-1}-{i}", f"{thr:.6e}"])
    
    # 2) Overlap CSV (adjacent levels; Gaussian approx from samples)
    overlaps = []
    for i in range(3):  # 4 levels = 3 pairs
        q_low = results_per_symbol[i]['q_values']
        q_high = results_per_symbol[i+1]['q_values']
        mu0, s0 = np.mean(q_low), max(float(np.std(q_low)), 1e-15)
        mu1, s1 = np.mean(q_high), max(float(np.std(q_high)), 1e-15)
        # Bhattacharyya overlap proxy
        bc = 0.25*np.log(0.25*((s0/s1)+(s1/s0)+2)) + 0.25*((mu0-mu1)**2)/(s0+s1)
        overlap = float(np.exp(-bc))
        overlaps.append((i, i+1, overlap))
    
    with open(out_dir / "tables" / "csk_overlap.csv", "w", newline='', encoding='utf-8') as f:  # Add encoding='utf-8'
        writer = csv.writer(f)
        writer.writerow(["pair", "overlap_proxy"])
        for i, j, ov in overlaps:
            writer.writerow([f"{i}-{j}", f"{ov:.6f}"])
    
    print(f"Tables saved:")
    print(f"  - {out_dir / 'tables' / 'csk_thresholds_ml.csv'}")
    print(f"  - {out_dir / 'tables' / 'csk_overlap.csv'}")
    
    # Check polarity
    print("\n" + "="*40)
    print("POLARITY CHECK:")
    print("-"*40)
    
    try:
        q_eff = get_nt_params(cfg, cfg['pipeline']['csk_target_channel'])['q_eff_e']
        print(f"Channel polarity (q_eff): {q_eff}")
        
        if q_eff > 0:
            print("Expected: Thresholds should increase (positive polarity)")
            if len(thresholds) > 1 and all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1)):
                print("✓ Thresholds are correctly ordered")
            else:
                print("❌ Thresholds are INCORRECTLY ordered!")
        else:
            print("Expected: Thresholds should decrease (negative polarity)")
            if len(thresholds) > 1 and all(thresholds[i] > thresholds[i+1] for i in range(len(thresholds)-1)):
                print("✓ Thresholds are correctly ordered")
            else:
                print("❌ Thresholds are INCORRECTLY ordered!")
    except Exception as e:
        print(f"Error checking polarity: {e}")
    
    # Test detection
    print("\n" + "="*40)
    print("DETECTION TEST:")
    print("-"*40)
    
    try:
        confusion_matrix = np.zeros((4, 4), dtype=int)
        
        # Get polarity once before the detection loop
        q_eff = get_nt_params(cfg, cfg['pipeline']['csk_target_channel'])['q_eff_e']
        
        for true_symbol in range(4):
            for q in results_per_symbol[true_symbol]['q_values']:
                # Detect symbol with polarity-aware threshold ordering
                detected_symbol = 0
                thr = sorted(thresholds, reverse=(q_eff < 0))
                for t in thr:
                    if (q_eff > 0 and q > t) or (q_eff < 0 and q < t):
                        detected_symbol += 1
                    else:
                        break
                
                confusion_matrix[true_symbol, detected_symbol] += 1
        
        print("Confusion Matrix (rows=true, cols=detected):")
        print(confusion_matrix)
        
        # Calculate SER
        correct = np.diag(confusion_matrix).sum()
        total = confusion_matrix.sum()
        ser = 1 - (correct / total)
        
        print(f"\nSymbol Error Rate: {ser:.3f}")
        print(f"Correct detections: {correct}/{total}")
        
    except Exception as e:
        print(f"Error in detection test: {e}")
        import traceback
        traceback.print_exc()
    
    return results_per_symbol, thresholds

if __name__ == "__main__":
    try:
        results, thresholds = diagnose_csk_levels()
        print("\n" + "="*40)
        print("DIAGNOSTIC COMPLETE")
        print("="*40)
    except Exception as e:
        print(f"\nError running diagnostic: {e}")
        import traceback
        traceback.print_exc()