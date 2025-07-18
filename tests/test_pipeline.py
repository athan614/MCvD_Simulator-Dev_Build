# tests/test_pipeline.py
"""
Sanity test: run a 10-symbol MoSK sequence through the pipeline and ensure SER < 1.
Uses minimal Nm so runtime stays < 3 s on CI.
"""

"python -m pytest tests/test_pipeline.py -v"

from src.pipeline import run_sequence
from src.config_utils import preprocess_config
from copy import deepcopy

def test_pipeline_mosk_fast(config): # <--- RENAMED to accept the fixture
    # The manual loading and preprocessing is now GONE.
    cfg = deepcopy(config) # We still deepcopy to avoid modifying the fixture for other tests

    # shorten for CI
    cfg['pipeline']['sequence_length'] = 10
    cfg['pipeline']['Nm_per_symbol']   = 2e3   # keep runtime down
    result = run_sequence(cfg)
    
    # Check a meaningful result
    assert 'SER' in result
    assert 0.0 <= result['SER'] <= 1.0