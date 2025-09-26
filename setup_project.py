#!/usr/bin/env python3
"""
Project bootstrapper for the Tri-Channel OECT molecular communication simulator.

What it does
------------
- Verifies Python and required third-party packages
- Creates the expected results/ tree (data/, figures/, tables/, cache/, logs/, figures/notebook_replicas/)
- Optionally resets cached results via --reset {cache|all}

Usage
-----
    python setup_project.py              # check environment & create folders
    python setup_project.py --reset all  # wipe results/ completely (careful)
    python setup_project.py --reset cache  # just clear caches/state

This script is safe to run multiple times.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REQUIRED = [
    ("numpy", None),
    ("scipy", None),
    ("pandas", None),
    ("matplotlib", None),
    ("yaml", "PyYAML"),
    ("tqdm", None),
    ("rich", None),
    ("psutil", None),
    ("joblib", None),
    ("statsmodels", None),
]


def _check_python() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit(
            "Python 3.11+ is required (SciPy>=1.16 and NumPy>=2.x). "
            f"Found {sys.version.split()[0]}"
        )


def _check_packages() -> None:
    missing: list[str] = []
    versions: dict[str, str] = {}
    for module_name, pip_name in REQUIRED:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            missing.append(pip_name or module_name)
            continue
        version = getattr(module, "__version__", "unknown")
        versions[module_name] = version

    if missing:
        print("Missing packages:", ", ".join(sorted(missing)))
        print("Install them with:  pip install -e .[dev]")
        raise SystemExit(1)

    print("Detected package versions:")
    for name in sorted(versions):
        print(f"  {name:>10s} : {versions[name]}")


def _mkdirs(root: Path) -> None:
    for sub in (
        "results/data",
        "results/figures",
        "results/figures/notebook_replicas",
        "results/tables",
        "results/cache",
        "results/logs",
    ):
        path = root / sub
        path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured results/ tree under {root.resolve()}")


def _safe_rmtree(path: Path) -> None:
    import os
    import shutil
    import stat

    root = (Path(__file__).parent / "results").resolve()
    target = path.resolve()
    if not str(target).startswith(str(root)):
        raise RuntimeError(f"Refusing to delete outside results/: {target}")
    if target.exists():
        def _on_rm_error(func, p, exc_info):
            try:
                os.chmod(p, stat.S_IWUSR)
                func(p)
            except Exception:
                pass
        shutil.rmtree(target, onerror=_on_rm_error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap environment and folders")
    parser.add_argument("--reset", choices=["cache", "all"], help="Delete results cache or full results/*")
    args = parser.parse_args()

    _check_python()
    _check_packages()

    root = Path(__file__).parent
    results = root / "results"
    results.mkdir(exist_ok=True)

    if args.reset:
        if args.reset == "all":
            _safe_rmtree(results)
        else:
            _safe_rmtree(results / "cache")
        print(f"Reset completed: {args.reset}")

    _mkdirs(root)
    print("All good. Try a quick sanity run, for example:")
    print("  python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --resume --progress tqdm")
    print("\nOr test parallel execution:")
    print("  python analysis/run_master.py --modes all --parallel-modes 3 --resume --progress rich")


if __name__ == "__main__":
    main()
