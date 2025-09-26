# Setup Guide

Use this guide when provisioning a fresh environment for the Tri-Channel OECT molecular communication simulator.

## Prerequisites
- Python 3.11 or newer
- pip, wheel, and setuptools (latest versions recommended)

## Fast path (editable install)
```bash
pip install -U pip wheel setuptools
pip install -e .[dev]
python setup_project.py
```

## Managed virtual environment
`setup_env.py` can create an isolated virtual environment, install dependencies, and optionally add extras.
```bash
# Latest package set (requirements.in) + dev extras + editable install
python setup_env.py --extras dev --editable

# Reproduce the frozen snapshot in requirements.latest.txt
python setup_env.py --use-freeze --editable
```

Activate the environment before running simulations:
- Linux/macOS: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

## Project bootstrap
After installing the dependencies, run:
```bash
python setup_project.py
```
This verifies core packages, creates the `results/` directory structure, and offers reset options via `--reset cache` or `--reset all`.

## Handy commands
- `python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --resume --progress tqdm`
- `python analysis/run_master.py --modes all --resume --progress rich --parallel-modes 3`

Logging is mirrored to `results/logs/`, and cached jobs allow safe interruption and resumption with `--resume`.
