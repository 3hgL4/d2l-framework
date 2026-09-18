"""随机种子管理：覆盖 python / numpy / torch / cuda 四个随机源。

deterministic=True 时启用 cudnn 确定性算法（完全复现，牺牲速度）；
False 时开启 cudnn.benchmark（更快，结果可能有微小浮点差异）。
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
