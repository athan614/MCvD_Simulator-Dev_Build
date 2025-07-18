from .binding import (
    bernoulli_binding,
    mean_binding,
    binding_noise_psd,
    calculate_equilibrium_metrics,
    calculate_effective_on_rate
)

from .oect import (
    oect_current,
    oect_static_gain,
    oect_impulse_response,
    differential_channels,
    generate_correlated_noise,
    calculate_noise_metrics,
    rms_in_band  # Add this
)

# src/__init__.py (append these imports to existing ones)
from .detection import (
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
    monte_carlo_detection
)

from .pipeline import run_sequence

from .config_utils import convert_numeric_strings, preprocess_config