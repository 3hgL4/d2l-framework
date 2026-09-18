"""冒烟测试用合成算法：20 维特征、3 类的线性可分问题 + 小线性网络。

不依赖任何真实数据集； FakeDataset 固定生成规则（模块级类，可 pickle，
num_workers>0 的 Windows spawn 路径也可验证）。

本文件同时是 MiniSpec 声明式接入的活样例：只声明 model/loss/datasets/
metrics/config 五组算法知识，经 infra.minispec 适配成完整 AlgoSpec 契约
—— 冒烟 2/3 因此同时回归"声明式接入"与"infra 冻结面"。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from infra.minispec import MiniSpec


class FakeDataset(Dataset):
    """y = argmax(X @ W + b)，W/b 固定种子生成 -> 线性可分。"""

    def __init__(self, n: int = 2048, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 20, generator=g)
        W = torch.randn(20, 3, generator=g)
        b = torch.randn(3, generator=g)
        self.y = (self.X @ W + b).argmax(dim=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class FakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 3)

    def forward(self, X):
        return self.fc(X)


def _acc(y_hat, y):
    return (y_hat.argmax(1) == y).float().mean().item()


def _load_data(cfg):
    return FakeDataset(seed=0), FakeDataset(n=512, seed=1)


class FakeAlgo(MiniSpec):
    name = "fake"
    model = FakeNet
    loss = F.cross_entropy
    datasets = _load_data
    optimizer = torch.optim.SGD      # lr 走 cfg.lr（默认 0.1，--override 可调）
    metrics = {"acc": _acc}
    config = {"epochs": 12, "patience": 0}


SPEC = FakeAlgo()
