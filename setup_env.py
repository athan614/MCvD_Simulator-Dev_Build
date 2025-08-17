#!/usr/bin/env python3
"""
Bootstrap a local virtual environment and install project dependencies.

Default behavior installs the **latest** versions from `requirements.in`.
If you want the exact (frozen) set we used on 2025‑08‑17, pass `--use-freeze`.

Examples
--------
# install latest + dev tools, then editable package
python setup_env.py --extras dev

# install the frozen set we validated on 2025‑08‑17
python setup_env.py --use-freeze

# specify a custom env location
python setup_env.py --env .venv-paper
"""

from __future__ import annotations
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _bin(env: Path) -> Path:
    return env / ("Scripts" if os.name == "nt" else "bin")

def _exe(env: Path, name: str) -> str:
    return str(_bin(env) / (name + (".exe" if os.name == "nt" and not name.endswith(".exe") else "")))

def run(cmd: list[str], **kwargs) -> None:
    print(f"→ {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)

def main() -> None:
    p = argparse.ArgumentParser(description="Create venv and install dependencies")
    p.add_argument("--env", default=".venv", help="Virtualenv directory (default: .venv)")
    p.add_argument("--use-freeze", action="store_true",
                   help="Install from requirements.latest.txt instead of requirements.in")
    p.add_argument("--extras", default="", help="Optional extras to install, e.g. 'dev'")
    p.add_argument("--editable", action="store_true",
                   help="Also install the package in editable mode (-e .)")
    args = p.parse_args()

    # SciPy>=1.16 requires Python 3.11+; fail fast if older
    if sys.version_info < (3, 11):
        sys.exit("ERROR: Python >= 3.11 is required for latest SciPy. Please upgrade Python.")

    env = HERE / args.env
    if not env.exists():
        print(f"Creating virtual environment at {env} ...")
        run([sys.executable, "-m", "venv", str(env)])
    else:
        print(f"Using existing virtual environment at {env}")

    pip = _exe(env, "pip")
    python = _exe(env, "python")

    # Upgrade pip tooling
    run([pip, "install", "-U", "pip", "wheel", "setuptools"])

    # Install dependencies (latest by default)
    req_file = HERE / ("requirements.latest.txt" if args.use_freeze else "requirements.in")
    if not req_file.exists():
        sys.exit(f"Missing {req_file.name} next to this script.")

    run([pip, "install", "-U", "-r", str(req_file)])

    # Install the package itself (+ optional extras)
    if args.editable:
        pkg_spec = ".[{}]".format(args.extras) if args.extras else "."
        run([pip, "install", "-U", "-e", pkg_spec])
    else:
        pkg_spec = ".[{}]".format(args.extras) if args.extras else "."
        run([pip, "install", "-U", pkg_spec])

    # Final message
    act = _bin(env) / ("activate.bat" if os.name == "nt" else "activate")
    print("
✓ Environment ready.")
    if os.name == "nt":
        print(f"Activate: {env}\Scripts\activate")
    else:
        print(f"Activate: source {act}")

if __name__ == "__main__":
    main()
