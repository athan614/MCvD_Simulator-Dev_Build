#!/usr/bin/env python3
"""
Create a local virtual environment and install project dependencies.

By default the script installs the latest versions declared in `requirements.in`.
Use `--use-freeze` to install the pinned snapshot stored in `requirements.latest.txt`.

Examples
--------
# install latest + dev extras, then editable package
python setup_env.py --extras dev --editable

# install the frozen snapshot
python setup_env.py --use-freeze --editable

# specify a custom environment location
python setup_env.py --env .venv-paper
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _bin(env: Path) -> Path:
    return env / ("Scripts" if os.name == "nt" else "bin")


def _exe(env: Path, name: str) -> str:
    suffix = ".exe" if os.name == "nt" and not name.endswith(".exe") else ""
    return str(_bin(env) / f"{name}{suffix}")


def run(cmd: list[str], **kwargs) -> None:
    print(f"> {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create venv and install dependencies")
    parser.add_argument("--env", default=".venv", help="Virtualenv directory (default: .venv)")
    parser.add_argument(
        "--use-freeze",
        action="store_true",
        help="Install from requirements.latest.txt instead of requirements.in",
    )
    parser.add_argument("--extras", default="", help="Optional extras to install, e.g. 'dev'")
    parser.add_argument("--editable", action="store_true", help="Install the package in editable mode (-e .)")
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        sys.exit("ERROR: Python >= 3.11 is required. Please upgrade Python before continuing.")

    env = HERE / args.env
    if not env.exists():
        print(f"Creating virtual environment at {env} ...")
        run([sys.executable, "-m", "venv", str(env)])
    else:
        print(f"Using existing virtual environment at {env}")

    pip = _exe(env, "pip")

    run([pip, "install", "-U", "pip", "wheel", "setuptools"])

    req_file = HERE / ("requirements.latest.txt" if args.use_freeze else "requirements.in")
    if not req_file.exists():
        sys.exit(f"Missing {req_file.name} next to this script.")

    run([pip, "install", "-U", "-r", str(req_file)])

    extras_suffix = f"[{args.extras}]" if args.extras else ""
    pkg_spec = f".{extras_suffix}"
    install_cmd = [pip, "install", "-U"]
    if args.editable:
        install_cmd.extend(["-e", pkg_spec])
    else:
        install_cmd.append(pkg_spec)
    run(install_cmd)

    act = _bin(env) / ("Activate.ps1" if os.name == "nt" else "activate")
    print("\nEnvironment ready.")
    if os.name == "nt":
        print(f"Activate: {env}\\Scripts\\Activate.ps1")
    else:
        print(f"Activate: source {act}")


if __name__ == "__main__":
    main()
