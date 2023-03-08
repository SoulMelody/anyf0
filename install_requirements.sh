#!/bin/bash
pip install pdm
pdm config set pypi.url https://mirrors.bfsu.edu.cn/pypi/web/simple
pdm plugin install pdm-plugin-torch
pdm install
pdm torch install cu117