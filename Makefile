.PHONY: setup verify run-all supplementary clean cache

setup:
\tpip install -e .[dev]
\tpython setup_project.py

verify:
\tpython analysis/run_master.py --preset verify --resume --progress rich

run-all:
\tpython analysis/run_master.py --modes all --resume --progress rich

supplementary:
\tpython analysis/run_master.py --modes all --resume --progress rich --supplementary

clean:
\tpython setup_project.py --reset all

cache:
\tpython setup_project.py --reset cache
