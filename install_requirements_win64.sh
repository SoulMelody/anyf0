#!/bin/bash
mamba install conda-lock -y
conda-lock render -p win-64
mamba env create -n anyf0 --file conda-win-64.lock --solver libmamba
mamba activate anyf0