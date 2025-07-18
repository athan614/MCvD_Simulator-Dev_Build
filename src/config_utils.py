# src/config_utils.py (new file)
"""
Configuration utilities for handling YAML string-to-numeric conversions.
"""

from typing import Dict, Any, Union


def convert_numeric_strings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert string representations of numbers to actual numeric types.
    
    Handles scientific notation (e.g., '1e6', '4e8') and underscores.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from YAML
        
    Returns
    -------
    dict
        Configuration with numeric strings converted to float/int
    """
    def convert_value(value: Any) -> Any:
        if isinstance(value, str):
            # Try to convert to float
            try:
                # Remove underscores and convert
                cleaned = value.replace('_', '')
                return float(cleaned)
            except ValueError:
                # Not a numeric string, return as-is
                return value
        elif isinstance(value, dict):
            # Recursively convert dictionary values
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Convert list items
            return [convert_value(item) for item in value]
        else:
            # Return other types unchanged
            return value
    
    return convert_value(config)


def preprocess_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess configuration to ensure all numeric values are properly typed.
    
    Parameters
    ----------
    config : dict
        Raw configuration from YAML
        
    Returns
    -------
    dict
        Processed configuration
    """
    # Convert all numeric strings
    config = convert_numeric_strings(config)
    
    # Special handling for N_apt - ensure it's an integer
    if 'N_apt' in config:
        config['N_apt'] = int(config['N_apt'])
    
    # Ensure N_c is numeric if present
    if 'N_c' in config:
        config['N_c'] = float(config['N_c'])
        
    return config