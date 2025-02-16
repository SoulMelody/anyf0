#!/bin/bash
mamba install conda-lock -y
# conda-lock -f pyproject.toml --virtual-package-spec virtual-packages.yml
conda-lock render -p win-64
mamba env create -n anyf0 --file conda-win-64.lock --solver libmamba
mamba activate anyf0