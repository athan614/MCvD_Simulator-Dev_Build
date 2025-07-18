# Tri-Channel OECT Molecular Communication Simulator

This repository implements a simulation framework for a tri-channel Organic Electrochemical 
Transistor (OECT) based molecular communication receiver designed to detect neurotransmitter 
signals from brain organoids.

## Overview

The system models:
- Molecular diffusion in brain tissue (with tortuosity and clearance)
- Stochastic aptamer binding kinetics
- OECT transduction with realistic noise models
- Differential measurement for noise cancellation
- Detection algorithms for MoSK, CSK, and PPM modulation schemes

## Installation

```bash
pip install -e .
```

## Usage

Run parameter sweeps:
```bash
python -m src.runner run_parameter_sweep --cfg config/default.yaml --sweep base_distance_sweep
```

## Project Structure

- `src/`: Core simulation modules
- `config/`: Configuration files
- `tests/`: Unit tests for all modules
- `notebooks/`: Jupyter notebooks for analysis
- `results/`: Output data and figures
