"""冒烟测试用合成算法：20 维特征、3 类的线性可分问题 + 小线性网络。

不依赖任何真实数据集； FakeDataset 固定生成规则（模块级类，可 pickle，
num_workers>0 的 Windows spawn 路径也可验证）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from infra.data import make_dataloaders


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


class FakeAlgo:
    name = "fake"

    def default_config(self):
        return {"epochs": 12, "lr": 0.1, "batch_size": 128, "patience": 0}

    def build_model(self, cfg):
        return FakeNet()

    def build_loss(self):
        return F.cross_entropy

    def build_optimizer(self, params, cfg):
        return torch.optim.SGD(params, lr=cfg.lr)

    def build_dataloaders(self, cfg):
        return make_dataloaders(FakeDataset(seed=0), FakeDataset(n=512, seed=1), cfg)

    def unpack_batch(self, batch):
        return batch

    def compute_metrics(self, y_hat, y):
        return {"acc": (y_hat.argmax(1) == y).float().mean().item()}


SPEC = FakeAlgo()
