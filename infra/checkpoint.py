"""断点：全状态保存 / 恢复。

一个"等价续训"的 checkpoint 除模型权重外还必须包含：
optimizer / scheduler / AMP scaler / 当前 epoch / 四路 RNG 状态 /
callbacks 的 extras（早停计数、最佳值等跨轮状态）。

写入用 tmp + os.replace 原子替换：任何时刻中断都不会留下半个文件。
load 的 weights_only=False：本仓库 checkpoint 含 RNG 状态等非张量对象，
只加载自己 run 目录下的文件，勿加载来路不明的 .pt。
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    t = state["torch"]
    torch.set_rng_state(t.cpu() if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save(path, payload: dict) -> None:
    """原子写入一个 checkpoint（payload 由 trainer/_state_dict 组装）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, p)  # Windows 上原子替换


def load(path, map_location="cpu") -> dict:
    return torch.load(str(path), map_location=map_location, weights_only=False)
