# Tri-Channel OECT Molecular Communication Simulator

## Introduction
This repository provides a research-grade simulation and analysis framework for the tri-channel organic electrochemical transistor (OECT) receiver documented in the associated IEEE Transactions on Molecular, Biological, and Multi-Scale Communications (TMBMC) manuscript. The implementation delivers reproducible, end-to-end evaluations of diffusion-based molecular communication links that employ glutamate (GLU), gamma-aminobutyric acid (GABA), and a differential control (CTRL) channel. Every stage of the physical pipeline, from molecular release through detection, is modeled with closed-form physics, stochastic sampling, and device-level electrical abstractions.

## Feature Summary
- **Complete transceiver stack**: parameterized transmitter, three-dimensional diffusive channel, receptor binding layer, OECT front-end, and digital detection logic for MoSK, CSK-4, and hybrid (2 bit/symbol) modulation.
- **Hybrid analytic/Monte Carlo workflow**: vectorized solvers alongside calibrated Monte Carlo sampling with deterministic seeding, resume support, and full provenance tracking.
- **IEEE publication assets**: automated generation of SER, LoD, ISI, sensitivity, and supplementary figures/tables in IEEE-compliant formats.
- **Comprehensive configurability**: YAML-driven scenario definition covering tissue transport, binding kinetics, device parameters, modulation controls, and analysis heuristics.
- **Scalable execution**: crash-safe multiprocessing, optional GUI/CLI progress front-ends, and balanced workload scheduling for heterogeneous CPU architectures.

## System Architecture
The simulator decomposes the tri-channel link into modular components, each exposed through `src/` packages and controlled by the configuration in `config/default.yaml`.

### Transmitter
- Finite-duration molecular bursts with configurable molecule budgets (`Nm_per_symbol`), shaping functions (`rect`, `gamma`), and guard factors.
- Deterministic symbol schedules for analytic previews and stochastic generation for Monte Carlo runs.

### Propagation (`src/mc_channel/transport.py`)
The diffusion channel employs the restricted extracellular-space Green's function
\[
G(r,t) = \frac{1}{\alpha (4 \pi D_\text{eff} t)^{3/2}} \exp\Big(-\frac{r^2}{4 D_\text{eff} t}\Big) \exp(-k_\text{clear} t),
\]
with \(D_\text{eff} = D / \lambda^2\). Vectorized helpers (`finite_burst_concentration`, `finite_burst_concentration_batch`, `finite_burst_concentration_batch_time`) evaluate this kernel across time grids, molecule species, and channel geometries.

### Binding Layer (`src/mc_receiver/binding.py`)
A stochastic birth-death process captures aptamer occupancy:
\[
\frac{dN_b}{dt} = k_\text{on} C(t) (N_\text{sites} - N_b) - k_\text{off} N_b.
\]
The module supports ordinary differential equation integration (`scipy.integrate.odeint`) and Monte Carlo Bernoulli updating, enabling both deterministic previews and sample-driven statistics. Dual-channel CSK operation is modeled through channel-specific site counts and leakage factors.

### OECT Transduction (`src/mc_receiver/oect.py`)
Bound charge perturbs the gate potential and modulates drain current according to \(\Delta I = g_m \Delta V_g\). The implementation parameterizes gate capacitance, transconductance, thermal and 1/f noise (`alpha_H`), and channel correlation. A causal Bessel post-filter approximates the OECT bandwidth prior to digitization.

### Noise Synthesis
Shot, thermal, and correlated 1/f noise components are synthesized per time step. Residual correlation between CTRL-subtracted currents is captured through `rho_between_channels_after_ctrl`, enabling studies of imperfect differential rejection.

### Detection and Modulation (`src/mc_detection/algorithms.py`)
The detection layer provides maximum-likelihood thresholds, analytic BER expressions (`ber_mosk_analytic`, `sep_csk_mary`), and decision-window policies. Hybrid modulation combines molecular presence and amplitude discrimination via `csk_dual_channel` combiners (`zscore`, `whitened`, `leakage`).

## Simulation Workflow
1. **Configuration ingestion** - `src/config_utils.py` normalizes numeric literals, validates parameter consistency, and expands shorthand configuration keys.
2. **Calibration** - `analysis/run_final_analysis.py` derives detection thresholds, maintains limit-of-detection (LoD) bracket caches, and harmonizes RNG seeds via `numpy.random.SeedSequence` to ensure replicable worker execution.
3. **Monte Carlo execution** - Each (sweep value, seed) pair traverses the end-to-end pipeline to produce symbol traces, binding trajectories, and OECT currents. Workloads are dispatched through `ProcessPoolExecutor` with oversubscription guards (`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, etc.).
4. **Aggregation and resume** - Intermediate results are written atomically to `results/cache/.../seed_<id>.json`; aggregation stages emit SER/LoD tables only after successful completion, permitting interruption and subsequent restart via `--resume`.
5. **Analysis generation** - Scripts under `analysis/` transform cached data into IEEE-style figures (Matplotlib with custom cyclers), LaTeX tables, and diagnostic reports.

`analysis/run_master.py` orchestrates complete study pipelines (mode sweeps, CTRL ablations, supplementary figures) and exposes pass-through controls for LoD heuristics (`--lod-num-seeds`, `--analytic-lod-bracket`), neurotransmitter-pair exploration, and parallel execution.

## Software Layout
```
analysis/          publication workflows, figure generation, diagnostics, validation harnesses
config/            canonical and scenario-specific YAML configurations (default.yaml is the baseline study)
results/           created on demand; hosts cache/, data/, figures/, tables/, logs/
src/               reusable simulator modules (transport, binding, OECT, detection, pipeline integration)
tests/             regression tests for transport, decision windows, and analytic validators
setup_env.py       virtual environment bootstrapper
setup_project.py   dependency checker and results/ tree initializer
Makefile           convenience targets for linting, testing, and figure regeneration
```

## Key Metrics
- **Symbol Error Rate (SER)** for MoSK, CSK, and hybrid modulation, including CTRL on/off ablations.
- **Limit of Detection (LoD)** versus link distance with adaptive seed scheduling and analytic bracketing.
- **Inter-symbol Interference (ISI) throughput** as a function of guard factor and symbol period.
- **Hybrid Decision Surface (HDS)**, **OECT-Normalized Sensitivity Index (ONSI)**, and **ISI-Robust Throughput (IRT)** generated by the analysis suite.

## Requirements and Dependencies
- Python 3.11 or newer (SciPy >= 1.16 requires Python 3.11+).
- 64-bit Linux, macOS, or Windows (validated on Ubuntu 22.04, macOS 14, Windows 11).
- CPU execution only; no GPU backends are required.

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
- statsmodels 0.14.5 (Wilson/Wald confidence intervals)
- cycler 0.12.1 (IEEE color palettes)

Optional analysis extras (install via `pip install .[analysis]` or `setup_env.py --extras analysis`):
- seaborn 0.13.2 for supplementary visualizations
- pyarrow 21.0.0 for high-throughput CSV/Feather interchange

Developer tooling (`pip install .[dev]`): pytest 8.4.2, pytest-cov 7.0.0, black 25.9.0, ruff 0.13.2, mypy 1.18.2, pip-tools 7.5.0, jupyter 1.1.1.

## Environment Provisioning
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -U pip wheel setuptools
pip install -e .[dev]
python setup_project.py
```

`setup_env.py` automates these steps and supports frozen installations:
```bash
# Latest packages + developer extras
python setup_env.py --extras dev --editable

# Reproduce requirements.latest.txt
python setup_env.py --use-freeze --editable
```
Re-run `python setup_project.py` after installation to verify dependencies, create the `results/` hierarchy, or reset cached artifacts via `--reset cache` or `--reset all`.

## Primary Experiment Entrypoints
```bash
# Short validation run (CSK mode, 4 seeds, resumable)
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 \
    --sequence-length 200 --resume --recalibrate --progress tqdm

# Full publication workflow (MoSK + CSK + Hybrid with CTRL ablations)
python analysis/run_master.py --modes all --resume --progress rich --parallel-modes 3
```
Installed entry points offer concise aliases:
```
mcvd-simulate  # analysis/run_final_analysis.py
mcvd-master    # analysis/run_master.py
```

### Frequently Used Arguments
- `--distance-grid`, `--nt-pairs`, `--lod-seq-len` - scenario sweeps without editing YAML files.
- `--progress {tqdm,rich,gui,none}` - select progress backends implemented in `analysis/ui_progress.py`.
- `--parallel-modes`, `--ablation-parallel` - control concurrency across modulation modes and CTRL states.
- `--preset {ieee,verify}` - load manuscript-aligned or quick-verification parameter sets.

## Configuration Guidance
`config/default.yaml` is the baseline scenario. Notable sections include:
- `neurotransmitters` - diffusion coefficients, binding rates, and effective charges for GLU, GABA, CTRL.
- `pipeline` - symbol period, ISI memory (`isi_memory`, `guard_factor`), LoD bounds, modulation selection, CSK combiner settings.
- `detection` - decision-window policies (`fraction_of_Ts`, `full_Ts`), per-channel thresholds, Monte Carlo sequence lengths.
- `analysis` - adaptive CI termination, LoD retry budget, ISI toggles, and timeout safeguards.

Use `src/config_utils.py` to validate custom YAML files. Scenario-specific studies can supply an alternate configuration with `--config path/to/custom.yaml` when invoking `analysis/run_final_analysis.py` or `analysis/run_master.py`.

## Data Products and Reproducibility
- `results/data/ser_vs_nm_<mode>.csv` - SER versus molecules per symbol (hybrid mode includes MoSK/CSK components).
- `results/data/lod_vs_distance_<mode>.csv` - LoD metrics with symbol period, data rate, and confidence intervals.
- `results/data/isi_tradeoff_<mode>.csv` - guard-factor sweeps for ISI throughput.
- `results/figures/*.png` - IEEE-ready 300 dpi figures with consistent typography and color ordering.
- `results/tables/*.tex` - LaTeX tables (`table1.tex`, `table_ii_performance.tex`) for manuscript integration.
- `results/logs/*.log` - mirrored runtime logs capturing CLI arguments, Git revision, and execution timing.

Every Monte Carlo job records its RNG seed and configuration hash in the cache JSON payload. Atomic renames (`*.tmp` to `.csv`) ensure that interrupted runs can safely resume once work is restarted with `--resume`.

## Development and Verification
```bash
pytest                               # Core physics regression tests
ruff check src analysis tests        # Static linting
black src analysis tests             # Formatting
mypy src analysis                    # Static typing validation
```
The `tests/` suite presently covers transport versus Fickian diffusion, adaptive decision-window logic, and analytic SER checks. Extend the suite alongside new physics or algorithmic contributions to maintain reproducibility claims.

## Citation
If this simulator supports your research, please cite the corresponding IEEE TMBMC article. A machine-readable entry (`CITATION.cff`) is included at the repository root.

## License
The project is distributed under the MIT License; see `LICENSE` for full terms.
