#!/bin/bash
pip install pdm
pdm config pypi.url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pdm sync
pdm run python -m pip install \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/torch_stable.html \
    --no-cache-dir \
    torchaudio===2.1.1+cu121