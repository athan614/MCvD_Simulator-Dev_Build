# Tri-Channel OECT Molecular Communication Simulator

This is a scientific simulation framework for modeling **Organic Electrochemical Transistor (OECT)** based molecular communication systems with **tri-channel** architecture (GLU/GABA/CTRL) supporting three modulation schemes: **MoSK**, **CSK**, and **Hybrid**.

## Architecture Overview

### Core Pipeline Flow
The simulation follows a **4-stage pipeline** from molecular release to bit error detection:

1. **Transport** (`src/mc_channel/transport.py`) - Molecular diffusion via finite burst concentration profiles
2. **Binding** (`src/mc_receiver/binding.py`) - Stochastic aptamer-neurotransmitter binding kinetics  
3. **Transduction** (`src/mc_receiver/oect.py`) - OECT current generation with correlated noise
4. **Detection** (`src/mc_detection/algorithms.py`) - Symbol error rate (SER) calculation with ML thresholds

All stages are **vectorized** for batch processing and use consistent `Dict[str, Any]` config patterns.

### Configuration System
- **Single source of truth**: `config/default.yaml` (nested structure required by code)
- **Preprocessing**: Flat YAML → nested dicts via `src/config_utils.py:preprocess_config()`
- **Test configs**: Use `tests/conftest.py:comprehensive_preprocess_config()` for consistency

```python
# Required nested structure (auto-created by preprocessor):
cfg = {
    'oect': {'gm_S': float, 'C_tot_F': float, 'R_ch_Ohm': float},
    'noise': {'alpha_H': float, 'N_c': float, 'K_d_Hz': float, 'rho_correlated': float},
    'sim': {'dt_s': float, 'temperature_K': float},
    'neurotransmitters': {nt: {'k_on_M_s', 'k_off_s', 'q_eff_e', 'lambda'}}
}
```

## Development Workflows

### Running Simulations
**Single mode (fast sanity check)**:
```bash
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --resume --progress tqdm
```

**Full publication suite**:
```bash
python analysis/run_master.py --modes all --resume --progress rich --parallel-modes 3
```

### Testing
- Run with `python -m pytest tests/ -v` 
- **Key pattern**: Tests use `conftest.py` fixture for config preprocessing
- **Performance tests**: Target <0.2s for 20s trace generation (`test_oect.py:test_performance`)
- **Validation tests**: Compare Monte Carlo vs analytical predictions

### Resume & Checkpointing
- **Crash-safe**: Each `(param, seed)` job writes `results/cache/<mode>/<sweep>_<ctrl>/<value>/seed_<id>.json`
- **Resume flag**: `--resume` skips completed combinations, safe to `Ctrl+C`
- **Atomic writes**: CSVs use `*.tmp → .csv` renames

## Key Patterns & Conventions

### Vectorization Requirements
All simulation functions must support **batch processing**:
```python
# Single trace (legacy)
bound_sites = binding.bernoulli_binding(conc_time, nt, cfg, rng)  # shape: (n_time,)

# Batch processing (required)
bound_sites_batch = binding.bernoulli_binding_batch(conc_batch, nt, cfg, rng)  # shape: (n_batch, n_time)
```

### Tri-Channel Architecture
- **Fixed channels**: GLU (glutamate), GABA, CTRL (control for differential subtraction)
- **Channel positions**: Spatially separated by `channel_spacing_um: 200` 
- **OECT function**: Always use `oect_trio()` (not single-channel functions)

### Random Number Generation
- **Reproducible RNG**: Use `numpy.random.default_rng(seed)` with `SeedSequence` for parallel workers
- **Per-worker isolation**: Each Monte Carlo seed gets independent RNG stream
- **Deterministic mode**: Set `cfg['deterministic_mode'] = True` for mean-field (no stochastic noise)

### Inter-Symbol Interference (ISI)
- **Toggle**: `cfg['pipeline']['enable_isi']` (master switch)
- **Memory**: `isi_memory` symbols of exponential decay
- **Guard factor**: `guard_factor` fraction of symbol period Ts for guard time

### Performance Optimization
```python
# CRITICAL: Disable BLAS oversubscription for process-level parallelism
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") 
os.environ.setdefault("OMP_NUM_THREADS", "1")
```

## Scientific Metrics

### Modulation Schemes
- **MoSK**: Binary On-Off Keying with GLU=1, GABA=0 (or combined neurotransmitter pairs)
- **CSK**: 4-level Concentration Shift Keying on single channel (DA by default)
- **Hybrid**: 2 bits/symbol combining MoSK amplitude + CSK concentration levels

### Key Metrics Computed
- **SER**: Symbol Error Rate from Monte Carlo trials
- **LoD**: Limit of Detection (maximum distance achieving target SER)
- **ONSI**: OECT-Normalized Sensitivity Index = (ΔI_diff/σ_I) / (g_m/C_tot)
- **ISI-Robust Throughput**: Effective data rate = (bits/symbol)/T_s × (1-SER)

### Noise Model
- **Correlated tri-channel**: `rho_corr = 0.9` between channels before CTRL subtraction
- **1/f noise**: `alpha_H = 1e-3`, **drift**: `K_d_Hz`, **thermal**: Johnson noise
- **CTRL differential**: Subtracts correlated noise, leaves `rho_between_channels_after_ctrl`

## Output Structure
```
results/
├── data/ser_vs_nm_{mode}.csv           # Main SER sweeps  
├── data/lod_vs_distance_{mode}.csv     # Limit of detection
├── figures/fig7_comparative_ser.png    # IEEE-style figures (≥300dpi)
├── figures/notebook_replicas/          # Mechanism validation plots
└── cache/<mode>/<sweep>/               # Checkpoint files for resume
```

## Common Debugging Patterns

### Configuration Issues
- **Missing numeric conversion**: Use `src/config_utils.py` preprocessor, not raw YAML
- **Nested structure**: Code expects `cfg['oect']['gm_S']`, not flat `cfg['gm_S']`

### Performance Issues  
- **Slow LoD search**: Use `--analytic-lod-bracket --max-ts-for-lod 180`
- **Memory issues**: Check vectorization - avoid pre-allocating massive arrays in binding

### Result Validation
- **SER sanity**: MoSK should achieve <1e-3 SER at high SNR (Nm≥10k)
- **Correlation check**: Verify `rho_corr ≈ 0.9` in tri-channel noise before CTRL subtraction
- **ISI impact**: With ISI enabled, SER should increase at short symbol periods

## Dependencies
- **Python ≥3.11** (required for SciPy ≥1.16, NumPy ≥2.x)
- **Core**: numpy, scipy, pandas, matplotlib, pyyaml, tqdm, rich
- **Optional**: jupyter, seaborn (supplementary plots), pyarrow (large CSVs)

See `pyproject.toml` for version constraints and `setup_env.py` for environment setup.