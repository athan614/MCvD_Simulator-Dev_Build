# test_oect_noise.py
import numpy as np
import yaml
from src.mc_receiver.oect import oect_trio

# Load and setup config (same as notebook)
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

# Setup nested structure
cfg['oect'] = {
    'gm_S': 0.002,
    'C_tot_F': 1.8e-8,
    'R_ch_Ohm': 200,
    'I_dc_A': 1e-6  # Important!
}
cfg['noise'] = {
    'alpha_H': 3e-3,
    'N_c': 3e14,
    'K_d_Hz': 1.3e-4,
    'rho_correlated': 0.9
}
cfg['sim'] = {
    'dt_s': 0.001,
    'temperature_K': 310
}
cfg['neurotransmitters'] = {
    'GLU': {'q_eff_e': 0.6},
    'GABA': {'q_eff_e': 0.2}
}

# Test with constant bound sites
n_samples = 1000
bound_sites = np.ones((3, n_samples), dtype=int) * 1000  # 1000 bound sites

# Run multiple times with different seeds
results = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    currents = oect_trio(bound_sites, ("GLU", "GABA", "CTRL"), cfg, rng)
    results.append(currents["GLU"].mean())

results = np.array(results)
print(f"Mean across trials: {results.mean():.3e} A")
print(f"Std across trials: {results.std():.3e} A")
print(f"CV: {results.std()/abs(results.mean()):.1%}")