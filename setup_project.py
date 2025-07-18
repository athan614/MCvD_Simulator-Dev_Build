#!/usr/bin/env python3
"""
Setup script to create the tri_channel_OECT_MC project structure
Run this to initialize the project directory
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete folder structure for tri_channel_OECT_MC"""
    
    # Define the directory structure
    base_dir = Path("tri_channel_OECT_MC")
    
    directories = [
        base_dir / "config",
        base_dir / "src",
        base_dir / "notebooks",
        base_dir / "tests",
        base_dir / "results" / "data",
        base_dir / "results" / "figures",
    ]
    
    # Create all directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    # Create __init__.py files
    init_files = [
        base_dir / "src" / "__init__.py",
        base_dir / "tests" / "__init__.py",
    ]
    
    for init_file in init_files:
        init_file.touch(exist_ok=True)
        print(f"Created: {init_file}")
    
    # Create README.md
    readme_content = """# Tri-Channel OECT Molecular Communication Simulator

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
"""
    
    readme_file = base_dir / "README.md"
    readme_file.write_text(readme_content)
    print(f"Created: {readme_file}")
    
    # Create pyproject.toml
    pyproject_content = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tri-channel-oect-mc"
version = "0.1.0"
description = "Tri-channel OECT molecular communication simulator"
requires-python = ">=3.10"
dependencies = [
    "numpy==1.26.4",
    "scipy==1.13.0",
    "pandas==2.2.1",
    "matplotlib==3.8.3",
    "tqdm==4.66.2",
    "pyyaml==6.0.1",
    "pyarrow==15.0.1",
    "pytest==8.1.1",
    "jupyter==1.0.0",
]

[project.optional-dependencies]
dev = [
    "black==24.2.0",
    "mypy==1.9.0",
    "pytest-cov==4.1.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src"]
"""
    
    pyproject_file = base_dir / "pyproject.toml"
    pyproject_file.write_text(pyproject_content)
    print(f"Created: {pyproject_file}")
    
    print("\nProject structure created successfully!")
    print(f"Navigate to {base_dir} and run: pip install -e .")

if __name__ == "__main__":
    create_project_structure()