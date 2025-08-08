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
            ig, ia, ic, Nm_actual = _single_symbol_currents(
                symbol, tx_history, cfg, rng
            )
            
            # Calculate charge
            dt = cfg['sim']['dt_s']
            n_detect = int(cfg['detection']['decision_window_s'] / dt)
            q_glu = np.trapezoid((ig - ic)[:n_detect], dx=dt)
            q_gaba = np.trapezoid((ia - ic)[:n_detect], dx=dt)
            
            # Store results (using GLU channel as configured)
            results_per_symbol[symbol]['q_values'].append(q_glu)
            results_per_symbol[symbol]['nm_actual'].append(Nm_actual)
        
        # Statistics for this symbol
        q_values = results_per_symbol[symbol]['q_values']
        nm_values = results_per_symbol[symbol]['nm_actual']
        
        print(f"  Nm actual: mean={np.mean(nm_values):.0f}, std={np.std(nm_values):.0f}")
        print(f"  Charge Q:  mean={np.mean(q_values):.3e}, std={np.std(q_values):.3e}")
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
    # plt.show()  # Comment out to prevent blocking
    plt.close()  # Close figure to free memory
    
    # Calculate expected thresholds
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
        
        print(f"Threshold {i}→{i+1}: {threshold:.3e}")
    
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
        
        for true_symbol in range(4):
            for q in results_per_symbol[true_symbol]['q_values']:
                # Detect symbol
                detected_symbol = 0
                for thresh in sorted(thresholds):  # Ascending for positive q_eff
                    if q > thresh:
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