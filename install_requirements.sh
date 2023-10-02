#!/bin/bash
pip install pdm
pdm config pypi.url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pdm sync
pip install \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/torch_stable.html \
    --no-cache-dir \
    torch===2.0.1+cu118