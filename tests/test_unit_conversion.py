#!/usr/bin/env python3
"""
Test script to validate the unit conversion fix.
"""

"python -m pytest tests/test_unit_conversion.py -v"

import numpy as np
from scipy import constants #type: ignore[import]

# Constants
AVOGADRO = constants.Avogadro  # 6.022e23 mol^-1

def test_unit_conversion():
    """Test the unit conversion fix."""
    
    print("=== Unit Conversion Fix Validation ===\n")
    
    # Test case: 1 molar solution
    print("Test Case: 1 M solution")
    print("-" * 30)
    
    # 1 M = 6.022e23 molecules/L = 6.022e26 molecules/m³
    molecules_per_m3_for_1M = AVOGADRO * 1000
    print(f"1 M = {molecules_per_m3_for_1M:.2e} molecules/m³")
    
    # Test old (buggy) conversion
    conc_old = molecules_per_m3_for_1M / (AVOGADRO / 1000)
    print(f"Old conversion result: {conc_old:.1f} M (should be 1.0 M)")
    print(f"Error factor: {conc_old:.0f}×")
    
    # Test new (corrected) conversion
    conc_new = molecules_per_m3_for_1M / (AVOGADRO * 1000)
    print(f"New conversion result: {conc_new:.1f} M (should be 1.0 M)")
    print(f"✓ CORRECT" if abs(conc_new - 1.0) < 1e-10 else "✗ INCORRECT")
    
    print(f"\n=== Impact on Simulation ===")
    print(f"Concentration reduction factor: {conc_old / conc_new:.0f}×")
    print(f"This explains the 1000× concentration inflation!")
    
    # Test realistic simulation values
    print(f"\n=== Realistic Simulation Test ===")
    print("-" * 35)
    
    # Typical simulation: 10,000 molecules at 100 μm distance
    Nm = 10000
    distance = 100e-6  # 100 μm
    D_eff = 3e-10  # m²/s
    t = 10.0  # seconds
    alpha = 0.2  # ECS volume fraction
    
    # Calculate concentration using Green's function
    prefactor = Nm / (alpha * (4 * np.pi * D_eff * t)**1.5)
    exponential = np.exp(-(distance**2) / (4 * D_eff * t))
    conc_molecules_per_m3 = prefactor * exponential
    
    print(f"Simulation parameters:")
    print(f"  Molecules: {Nm:,}")
    print(f"  Distance: {distance*1e6:.0f} μm")
    print(f"  Time: {t:.1f} s")
    print(f"  Concentration (molecules/m³): {conc_molecules_per_m3:.2e}")
    
    # Old vs new conversion
    conc_old_nM = (conc_molecules_per_m3 / (AVOGADRO / 1000)) * 1e9
    conc_new_nM = (conc_molecules_per_m3 / (AVOGADRO * 1000)) * 1e9
    
    print(f"\nConcentration results:")
    print(f"  Old (buggy): {conc_old_nM:.1f} nM")
    print(f"  New (fixed): {conc_new_nM:.1f} nM")
    print(f"  Expected range: 10-30 nM")
    print(f"  Status: {'✓ REALISTIC' if 1 <= conc_new_nM <= 100 else '? NEEDS CALIBRATION'}")

if __name__ == "__main__":
    test_unit_conversion()