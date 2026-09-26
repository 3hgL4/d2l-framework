"""数据集装载（工程件）：IDX 二进制格式的手写读取器 + FashionMNIST 装载。

定位：替代 torchvision.datasets 的数据转换层。IDX 解析纯工程零算法决策；
FashionMNIST 装载为数据集级复用件（2026-09-23 用户决策下沉：路径选择、
/255 归一化、形状与 dtype 随 load_data 一并收编，算法层不再各自维护）。
回归/合成等其它数据集仍由各算法的 load_data(cfg) 自行实现。

IDX 格式（LeCun 1998, MNIST database）：
  图像文件：magic(0x00000803) + n + rows + cols（大端 int32）+ n×rows×cols uint8
  标签文件：magic(0x00000801) + n + n 个 uint8
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def read_idx_images(path) -> np.ndarray:
    """读 IDX3 图像文件 -> uint8 ndarray (n, rows, cols)。"""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} 不是 IDX3 图像文件（magic={magic}，应为 2051）")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def read_idx_labels(path) -> np.ndarray:
    """读 IDX1 标签文件 -> uint8 ndarray (n,)。"""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} 不是 IDX1 标签文件（magic={magic}，应为 2049）")
        return np.frombuffer(f.read(), dtype=np.uint8)[:n]


def load_fashion_mnist(cfg) -> tuple:
    """FashionMNIST 装载 -> (train_ds, val_ds)。

    图像 (N,1,28,28) float32 已 /255（等价 ToTensor），标签 int64。
    一次性预转换 TensorDataset（2026-09-18 优化：逐样本 ToTensor 5.9s/epoch
    -> 整块转换 0.5s）。cfg.data_root 可覆盖数据根，默认 d2l/data。
    """
    root = Path(cfg.data_root) if getattr(cfg, "data_root", "") else \
        Path(__file__).resolve().parents[1] / "data"
    raw = root / "FashionMNIST" / "raw"
    out = []
    for prefix in ("train", "t10k"):
        X = read_idx_images(raw / f"{prefix}-images-idx3-ubyte")
        y = read_idx_labels(raw / f"{prefix}-labels-idx1-ubyte")
        X = torch.from_numpy(np.array(X)).unsqueeze(1).float().div(255)
        y = torch.from_numpy(y.astype(np.int64))
        out.append(TensorDataset(X, y))
    return tuple(out)
