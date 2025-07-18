# In src/oect.py (this is now a shim)

# Import everything from the new, refactored location
from .mc_receiver.oect import (
    oect_trio,
    oect_current,
    oect_static_gain,
    oect_impulse_response,
    differential_channels,
    generate_correlated_noise,
    calculate_noise_metrics,
    rms_in_band,
)

# Make them available for any file that still imports from here
__all__ = [
    "oect_trio",
    "oect_current",
    "oect_static_gain",
    "oect_impulse_response",
    "differential_channels",
    "generate_correlated_noise",
    "calculate_noise_metrics",
    "rms_in_band",
]