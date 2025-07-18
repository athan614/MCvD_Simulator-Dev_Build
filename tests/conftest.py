# tests/conftest.py
import pytest
import yaml
from pathlib import Path
from copy import deepcopy

# This is a direct copy of the robust preprocessor from our analysis scripts.
# It ensures tests use the exact same config structure as the main analysis.
def comprehensive_preprocess_config(config):
    """Prepares the flat YAML config into the nested structure our code expects."""
    cfg = deepcopy(config)
    def to_float(key):
        if key in cfg and cfg[key] is not None and not isinstance(cfg[key], (int, float)):
            try:
                cfg[key] = float(cfg[key])
            except (ValueError, TypeError):
                pass # Keep as is if conversion fails
    
    # This list must include ALL keys that need to be numeric
    numeric_keys = [
        'temperature_K', 'alpha', 'clearance_rate', 'T_release_ms', 
        'gm_S', 'C_tot_F', 'R_ch_Ohm', 'alpha_H', 'N_c', 'K_d_Hz', 'dt_s',
        'N_apt' # This was the main missing key
    ]
    for key in numeric_keys:
        to_float(key)

    # Also handle keys inside nested dictionaries
    if 'pipeline' in cfg:
        for pkey in ['distance_um', 'Nm_per_symbol', 'guard_factor', 'lod_nm_min']:
            if pkey in cfg['pipeline'] and cfg['pipeline'][pkey] is not None:
                cfg['pipeline'][pkey] = float(cfg['pipeline'][pkey])
    
    # Handle lists of numbers
    if 'Nm_range' in cfg and cfg['Nm_range'] is not None:
        cfg['Nm_range'] = [float(x) for x in cfg['Nm_range']]
    if 'distances_um' in cfg and cfg['distances_um'] is not None:
        cfg['distances_um'] = [float(x) for x in cfg['distances_um']]
    
    # Create the final nested structure
    cfg['oect'] = {'gm_S': cfg.get('gm_S'), 'C_tot_F': cfg.get('C_tot_F'), 'R_ch_Ohm': cfg.get('R_ch_Ohm')}
    cfg['noise'] = {'alpha_H': cfg.get('alpha_H'), 'N_c': cfg.get('N_c'), 'K_d_Hz': cfg.get('K_d_Hz')}
    cfg['sim'] = {'dt_s': cfg.get('dt_s'), 'temperature_K': cfg.get('temperature_K')}
    return cfg

@pytest.fixture(scope="session")
def config():
    """A master fixture that loads, processes, and provides a correct config for all tests."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        cfg_base = yaml.safe_load(f)
    
    # Process the config once for the entire test session
    processed_cfg = comprehensive_preprocess_config(cfg_base)
    return processed_cfg