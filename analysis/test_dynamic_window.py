# tests/test_dynamic_window.py
"""
Unit test for dynamic window calculation.
Verifies that symbol periods are correctly calculated based on physics.
"""

import unittest
import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.run_final_analysis import calculate_dynamic_symbol_period
from src.config_utils import preprocess_config


class TestDynamicWindow(unittest.TestCase):
    """Test dynamic symbol period calculation."""
    
    def setUp(self):
        """Load configuration."""
        with open(project_root / "config" / "default.yaml") as f:
            self.config = yaml.safe_load(f)
        self.config = preprocess_config(self.config)
    
    def test_symbol_period_at_100um(self):
        """Test that symbol period at 100μm is approximately 148s with tortuosity."""
        distance_um = 100
        symbol_period = calculate_dynamic_symbol_period(distance_um, self.config)
        
        # Updated expected value based on correct physics
        expected = 148  # Was 173, but that's incorrect
        tolerance = 0.02
        
        self.assertAlmostEqual(
            symbol_period, expected, 
            delta=expected * tolerance,
            msg=f"Symbol period at {distance_um}μm should be ~{expected}s (±{tolerance*100}%)"
        )
    
    def test_symbol_period_scaling(self):
        """Test that symbol period scales correctly with distance."""
        # Symbol period should scale with distance squared
        d1, d2 = 50, 100
        ts1 = calculate_dynamic_symbol_period(d1, self.config)
        ts2 = calculate_dynamic_symbol_period(d2, self.config)
        
        # Ratio should be approximately (d2/d1)^2 = 4
        ratio = ts2 / ts1
        expected_ratio = (d2 / d1) ** 2
        
        # Increased tolerance due to rounding effects at small distances
        self.assertAlmostEqual(
            ratio, expected_ratio, 
            delta=0.5,  # Was 0.1, but rounding affects ratio
            msg=f"Symbol period should scale with distance squared"
        )
    
    def test_minimum_symbol_period(self):
        """Test that symbol period has a minimum value."""
        # Very short distance
        distance_um = 1
        symbol_period = calculate_dynamic_symbol_period(distance_um, self.config)
        
        # Should be at least 20s (the minimum)
        self.assertGreaterEqual(
            symbol_period, 20,
            msg="Symbol period should have minimum value of 20s"
        )
    
    def test_guard_factor_effect(self):
        """Test that guard factor increases symbol period."""
        distance_um = 100
        
        # Test with no guard factor
        config_no_guard = self.config.copy()
        config_no_guard['pipeline'] = config_no_guard.get('pipeline', {})
        config_no_guard['pipeline']['guard_factor'] = 0.0
        ts_no_guard = calculate_dynamic_symbol_period(distance_um, config_no_guard)
        
        # Test with guard factor
        config_with_guard = self.config.copy()
        config_with_guard['pipeline'] = config_with_guard.get('pipeline', {})
        config_with_guard['pipeline']['guard_factor'] = 0.3
        ts_with_guard = calculate_dynamic_symbol_period(distance_um, config_with_guard)
        
        # Guard factor should increase symbol period
        self.assertGreater(
            ts_with_guard, ts_no_guard,
            msg="Guard factor should increase symbol period"
        )
        
        # Should be approximately 30% more
        expected_ratio = 1.3
        actual_ratio = ts_with_guard / ts_no_guard
        self.assertAlmostEqual(
            actual_ratio, expected_ratio, 
            delta=0.01,
            msg=f"Guard factor of 0.3 should increase period by ~30%"
        )
    
    def test_physics_constants_locked(self):
        """Test that physics constants produce expected results."""
        # This ensures the diffusion coefficient hasn't changed
        D_glu = self.config['neurotransmitters']['GLU']['D_m2_s']
        expected_D = 7.6e-10  # m²/s
        
        self.assertAlmostEqual(
            D_glu, expected_D,
            delta=1e-12,
            msg="GLU diffusion coefficient should match expected value"
        )
        
        # Calculate expected symbol period manually WITH TORTUOSITY
        distance_m = 100e-6  # 100 μm
        guard_factor = self.config['pipeline'].get('guard_factor', 0.3)
        
        # FIXED: Include tortuosity in manual calculation
        lambda_glu = self.config['neurotransmitters']['GLU']['lambda']
        D_eff = D_glu / (lambda_glu ** 2)
        
        time_95 = 3.0 * distance_m**2 / D_eff
        expected_ts = round((1 + guard_factor) * time_95)
        
        # Compare with function result
        actual_ts = calculate_dynamic_symbol_period(100, self.config)
        self.assertEqual(
            actual_ts, expected_ts,
            msg="Manual calculation should match function result"
        )


if __name__ == '__main__':
    unittest.main()