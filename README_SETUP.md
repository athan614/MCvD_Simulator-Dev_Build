# Setup Guide

Use this checklist when provisioning an environment for the Tri-Channel OECT molecular communication simulator.

## Requirements
- Python 3.11 or newer
- Recent `pip`, `wheel`, and `setuptools`
- C/C++ build tools supplied by your OS package manager (SciPy wheels cover common platforms, but keeping compilers available helps with niche dependencies)

## Standard Installation
```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
pip install -e ".[dev]"
python setup_project.py
```
`setup_project.py` validates the core Python packages and creates the expected `results/` subdirectories. Re-run it anytime; use `--reset cache` or `--reset all` to clear cached data.

## Guided Environment Creation
`setup_env.py` bootstraps a dedicated virtual environment and installs the manifests shipped with the repository:
```bash
# Latest versions from requirements.in + optional extras + editable install
python setup_env.py --extras dev --editable

# Frozen snapshot defined in requirements.latest.txt
python setup_env.py --use-freeze --editable
```
Supply `--env <path>` to customize the virtual environment location.

## Post-Install Smoke Test
```bash
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --resume --progress tqdm
```
Expect the run to create `results/cache/`, emit progress via `tqdm`, and write a log file under `results/logs/`.

## Troubleshooting
- Confirm `python --version` reports 3.11.x. Earlier interpreters lack ABI support for SciPy 1.16 and NumPy 2.x.
- On macOS and Linux, ensure `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and related environment variables are set to one when running large Monte Carlo batches (the analysis scripts enforce this by default).
- If `pip` cannot fetch wheels for SciPy or NumPy, upgrade `pip` first; fall back to conda (`environment.yml`) if your platform is exotic.
