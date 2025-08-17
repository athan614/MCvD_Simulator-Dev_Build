#!/usr/bin/env python3
"""
Project bootstrapper for the Tri‑Channel OECT MC simulator.

What it does
------------
- Verifies Python and core dependencies
- Creates the expected results/ tree (data/, figures/, tables/, cache/)
- (Optional) Resets results with --reset {cache|all}

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
]

def _check_python() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required (SciPy>=1.16 & NumPy>=2.x). "
                         f"Found {sys.version.split()[0]}")

def _check_packages() -> None:
    missing = []
    versions = {}
    for mod, pip_name in REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "unknown")
            versions[mod] = ver
        except Exception:
            missing.append(pip_name or mod)
    if missing:
        print("✗ Missing packages:", ", ".join(missing))
        print("   Install with:  pip install -e .[viz,dev]")
        raise SystemExit(1)
    print("✓ Package versions:")
    for k, v in versions.items():
        print(f"   {k:>10s} : {v}")

def _mkdirs(root: Path) -> None:
    for sub in ("results/data", "results/figures", "results/tables", "results/cache"):
        p = root / sub
        p.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created/verified results/ tree under {root.resolve()}")

def _safe_rmtree(path: Path) -> None:
    import shutil, os, stat
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
    p = argparse.ArgumentParser(description="Bootstrap environment and folders")
    p.add_argument("--reset", choices=["cache", "all"], help="Delete results cache or full results/*")
    args = p.parse_args()

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
        print(f"✓ Reset completed: {args.reset}")

    _mkdirs(root)
    print("All good. Try a quick sanity run, e.g.:")
    print("  python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --resume --progress tqdm")

if __name__ == "__main__":
    main()
