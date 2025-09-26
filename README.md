# Tri-Channel OECT Molecular Communication Simulator

This repository targets IEEE TMBMC practitioners who need reproducible end-to-end studies of a tri-channel organic electrochemical transistor (OECT) receiver operated over a diffusion-based molecular communication link. The implementation couples analytical channel models with device-level electrical models, supports hybrid multi-molecule modulation, and automates the publication workflow used in the accompanying manuscript.

## System Overview
- Transmitter: finite-duration point releases with configurable Nm, burst shape, and guard policies.
- Propagation: 3-D diffusion with tortuosity, extracellular volume fraction, and first-order clearance.
- Receiver interface: selective aptamer binding kinetics driving an OECT-based readout for GLU, GABA, and a differential CTRL channel.
- Detection: modulation-aware thresholding and SER estimation for MoSK, CSK-4, and a 2-bit hybrid mode.
- Automation: parallel Monte Carlo engine with crash-safe resume, calibration cache management, and rich logging/visualisation utilities.

The top-level entry points (`src/pipeline.py` and `analysis/run_final_analysis.py`) expose these components as a composable simulator with deterministic configuration, while analysis scripts deliver the figures and tables aligned with IEEE submission requirements.

## Physical and Algorithmic Models

### Transport (`src/mc_channel/transport.py`)
The propagator implements the restricted extracellular space Green's function:
\[
G(r,t) = \frac{1}{\alpha\,(4\pi D_\text{eff} t)^{3/2}}\,\exp\!\left(-\frac{r^2}{4D_\text{eff} t}\right)\exp(-k_\text{clear} t),
\]
where \(D_\text{eff} = D/\lambda^2\). Vectorised helpers (`finite_burst_concentration`, `finite_burst_concentration_batch`, `finite_burst_concentration_batch_time`) integrate this kernel against arbitrary burst profiles and time grids, enabling simultaneous evaluation of multiple molecules, channels, and decision windows.

### Binding layer (`src/mc_receiver/binding.py`)
A stochastic birth-death process models aptamer occupancy with:
\[
\frac{dN_b}{dt} = k_\text{on} C(t) (N_\text{sites}-N_b) - k_\text{off} N_b.
\]
The solver supports both ODE integration (via `scipy.integrate.odeint`) and Bernoulli thinning for Monte Carlo seeds, allowing the simulator to switch between deterministic previews and full stochastic sampling. Dual-channel CSK operation relies on `N_sites_*` asymmetry and configurable leakage factors.

### OECT transduction (`src/mc_receiver/oect.py`)
The electrical frontend maps bound charge to differential drain current using the small-signal relationship \(\Delta I = g_m \Delta V_g\). The code parameterises gate capacitance, thermal and 1/f noise (`alpha_H`), and correlated noise between GLU/GABA/CTRL. A Bessel post-filter approximates the front-end bandwidth and maintains causal filtering when discretised.

### Noise synthesis
Shot noise (Poisson), thermal noise (Johnson), and correlated 1/f components are synthesised per time step. Residual correlation between CTRL-subtracted channels is controlled via `rho_between_channels_after_ctrl`, supporting sensitivity studies of imperfect differential cancellation.

### Detection and modulation (`src/mc_detection/algorithms.py`)
Detection functions compute maximum-likelihood thresholds for MoSK and CSK, analytic BER curves (`ber_mosk_analytic`, `sep_csk_mary`), and decision window policies. Hybrid modulation combines molecule presence (MoSK) with amplitude discrimination (CSK) under `csk_dual_channel` combiner strategies (`zscore`, `whitened`, `leakage`).

## Simulation Workflow
1. **Configuration ingestion**: YAML files are parsed via `src/config_utils.py` to normalise numeric literals and expand shorthand keys.
2. **Calibration**: `analysis/run_final_analysis.py` precomputes detection thresholds, caches LoD bracket states, and harmonises RNG seeds using `numpy.random.SeedSequence` to guarantee reproducibility across workers.
3. **Monte Carlo execution**: Each (sweep parameter, seed) pair runs through the pipeline to generate symbol traces, binding trajectories, and OECT currents. Process pools leverage `ProcessPoolExecutor` with oversubscription guards via environment variables.
4. **Aggregation & resume**: Job outputs are atomically written to `results/cache/.../seed_<id>.json`. Aggregators consolidate SER/LoD metrics, append provenance metadata, and drop `.tmp` suffixes only after successful writes, making repeated runs safe under interruption.
5. **Analysis generation**: Scripts in `analysis/` consume cached CSVs to build IEEE-style figures (Matplotlib with custom cyclers) and LaTeX tables.

`analysis/run_master.py` orchestrates multi-mode sweeps, CTRL ablations, and supplementary figure builds. It exposes pass-through arguments for LoD search heuristics (`--lod-num-seeds`, `--analytic-lod-bracket`) and parallelises modulation modes, CTRL states, and seeds to saturate heterogeneous CPU architectures.

## Repository Layout
```
analysis/          publication workflows, figure generation, diagnostics, and validation scripts
config/            canonical and scenario-specific YAML configurations (default.yaml reflects the baseline study)
results/           created on demand; hosts cache/, data/, figures/, tables/, logs/
src/               reusable simulator modules (transport, binding, OECT, detection, pipeline glue)
tests/             regression tests for transport, decision windows, and analytic validators
setup_env.py       virtual environment bootstrap
setup_project.py   dependency checker and results/ tree initialiser
Makefile           shortcuts for linting, tests, and figure rebuilds
```

## Requirements and Environments
- Python 3.11 or newer (SciPy 1.16 targets Python >=3.11).
- 64-bit Linux, macOS, or Windows (validated on Ubuntu 22.04, macOS 14, Windows 11).
- Optional GPU kernels are not required; everything runs on CPU-class hardware.

### Runtime dependencies (latest tested versions)
- numpy 2.3.3
- scipy 1.16.2
- pandas 2.3.2
- matplotlib 3.10.6
- PyYAML 6.0.3
- tqdm 4.67.1
- rich 14.1.0
- psutil 7.1.0
- joblib 1.5.2
- statsmodels 0.14.5 (confidence intervals for SER estimates)
- cycler 0.12.1 (IEEE plot palettes)

Optional analysis extras (install with `pip install .[analysis]` or `setup_env.py --extras analysis`):
- seaborn 0.13.2 (supplementary figure palettes)
- pyarrow 21.0.0 (high-throughput CSV/Feather IO)

Developer tooling is exposed via `pip install .[dev]` (pytest 8.4.2, pytest-cov 7.0.0, black 25.9.0, ruff 0.13.2, mypy 1.18.2, pip-tools 7.5.0, jupyter 1.1.1).

## Installation and Environment Helpers
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -U pip wheel setuptools
pip install -e .[dev]
python setup_project.py
```

`setup_env.py` automates these steps and supports frozen installs:
```bash
# Latest packages + dev extras
python setup_env.py --extras dev --editable

# Reproduce requirements.latest.txt
python setup_env.py --use-freeze --editable
```
After installation, rerun `python setup_project.py` to verify dependencies, create `results/` folders, or reset caches (`--reset cache` or `--reset all`).

## Running Primary Experiments
```bash
# Fast SER sanity for CSK (4 seeds)
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 \
    --resume --recalibrate --progress tqdm

# Full publication sweep (MoSK + CSK + Hybrid with CTRL ablations)
python analysis/run_master.py --modes all --resume --progress rich --parallel-modes 3
```

Entry points installed through `pip` provide CLI aliases:
```
mcvd-simulate  # analysis/run_final_analysis.py
mcvd-master    # analysis/run_master.py
```

### Useful arguments
- `--distance-grid`, `--nt-pairs`, and `--lod-seq-len` expose LoD/NT trade-offs without editing YAML.
- `--progress {tqdm,rich,gui,none}` selects the progress backend implemented in `analysis/ui_progress.py`.
- `--parallel-modes` and `--ablation-parallel` enable simultaneous modulation-mode and CTRL-state execution on multi-core hosts.

## Configuration and Extensibility
`config/default.yaml` is the single source of truth for baseline studies. Key sections:
- `neurotransmitters`: diffusion tensors, binding constants, and effective charge assignments for GLU, GABA, CTRL.
- `pipeline`: symbol period, ISI horizon (`isi_memory`, `guard_factor`), LoD bounds, modulation selection, and dual-channel combiners.
- `detection`: window policies (`fraction_of_Ts`, `full_Ts`), per-channel thresholds, Monte Carlo sequence lengths.
- `analysis`: adaptive CI termination, LoD retry budgets, ISI enable/disable toggles.

`src/config_utils.py` normalises these values and guards against invalid combinations. For scenario sweeps, copy `default.yaml`, adjust the relevant subsections, and pass the file via `--config` to the analysis scripts.

## Data Products and Reproducibility
- `results/data/ser_vs_nm_<mode>.csv`: SER vs molecules per symbol across modulation modes (Hybrid includes decomposed MoSK/CSK SER columns).
- `results/data/lod_vs_distance_<mode>.csv`: limit-of-detection tables with symbol period, data rate, and CI metadata.
- `results/data/isi_tradeoff_<mode>.csv`: guard factor sweeps for ISI robustness.
- `results/figures/*.png`: IEEE-ready 300 dpi figures with consistent typography and colour ordering.
- `results/tables/*.tex`: LaTeX tables (`table1.tex`, `table_ii_performance.tex`) ready for manuscript inclusion.
- `results/logs/*.log`: tee-d runtime logs capturing CLI arguments, Git revisions, and timing.

Each Monte Carlo job records its RNG seed and configuration hash inside the cache JSON, enabling exact replay. Atomic writes (`*.tmp` -> `.csv`) guarantee that partially completed runs can be resumed with `--resume` without manual cleanup.

## Development and Testing
```bash
pytest                               # physics regression tests
ruff check src analysis tests        # linting / style
black src analysis tests             # formatting
mypy src analysis                    # static typing
```
The `tests/` directory currently includes transport-versus-Fick validators, dynamic decision-window checks, and analytic SER regression tests. Extend these when adding new physics features to preserve reproducibility claims.

## Citation
If you use this simulator, please cite the IEEE Transactions on Molecular, Biological, and Multi-Scale Communications manuscript associated with this repository. A machine-readable entry is available in `CITATION.cff`.

## License
Released under the MIT License. See `LICENSE` for details.
