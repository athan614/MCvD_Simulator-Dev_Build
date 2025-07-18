# In src/binding.py (this is now a shim)

# Import the functions from their new home
from .mc_receiver.binding import (
    calculate_effective_on_rate,
    bernoulli_binding,
    mean_binding,
    binding_noise_psd,
    calculate_equilibrium_metrics,
)

# Make them available for any file that still imports from here
__all__ = [
    "calculate_effective_on_rate",
    "bernoulli_binding",
    "mean_binding",
    "binding_noise_psd",
    "calculate_equilibrium_metrics",
]