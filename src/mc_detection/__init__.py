# In src/mc_detection/__init__.py
from .algorithms import (
    detect_mosk,
    detect_csk_binary,
    detect_csk_mary,
    calculate_ml_threshold,
    ber_mosk_analytic,
    sep_csk_binary,
    sep_csk_mary,
    calculate_snr,
    calculate_data_rate,
    snr_sweep,
    monte_carlo_detection,
)

__all__ = [
    "detect_mosk",
    "detect_csk_binary",
    "detect_csk_mary",
    "calculate_ml_threshold",
    "ber_mosk_analytic",
    "sep_csk_binary",
    "sep_csk_mary",
    "calculate_snr",
    "calculate_data_rate",
    "snr_sweep",
    "monte_carlo_detection",
]