
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]   # d2l 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.house_price.algo import (AdamScratch, KaggleSubmit,   # 复用基线件
                                         build_features, load_data, log_rmse)
from algorithms.mlp.algo import dropout_layer
from infra.minispec import MiniSpec



class HouseMLPScratch(nn.Module):
    def __init__(self, num_inputs: int, hidden=(256, 64), dropouts=(0.0, 0.0)):
        super().__init__()

        sizes = [num_inputs] + list(hidden) + [1]
        self.W = nn.ParameterList([nn.Parameter(torch.randn(a, b) * 0.01)
                                   for a, b in zip(sizes[:-1], sizes[1:])])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(h)) for h in sizes[1:]])
        self.do = list(dropouts) + [0.0] * (len(hidden) - len(dropouts))

    def forward(self, X):
        H, last = X, len(self.W) - 1
        for i in range(last):                       # 隐藏层：线性+ReLU+dropout
            H = torch.relu(H @ self.W[i] + self.b[i])
            if self.training and self.do[i] > 0:
                H = dropout_layer(H, self.do[i])
        return H @ self.W[last] + self.b[last]      # 输出层：纯线性，原价空间


class HouseMLPSpec(MiniSpec):
    name = "house_mlp"
    model = HouseMLPScratch
    loss = nn.MSELoss                     # 与线性基线相同：原价空间 MSE
    datasets = load_data                  # 复用 5 折切分（fold=0..4 / -1 全量）
    metrics = {"log_rmse": log_rmse}
    optimizer = (AdamScratch, {"betas": (0.9, 0.999), "eps": 1e-8,
                               "weight_decay": 0.0})
    config = {"epochs": 100, "lr": 5, "batch_size": 64,
              "k": 5, "fold": 0,
              "num_hiddens1": 256, "num_hiddens2": 64,
              "dropout1": 0.0, "dropout2": 0.0,
              "monitor": "val_log_rmse", "patience": 0, "amp": False}

    def build_model(self, cfg):
        X, _, _, _ = build_features()
        hidden = [int(cfg.num_hiddens1)]
        if int(cfg.num_hiddens2) > 0:               # 0 = 只用单隐藏层
            hidden.append(int(cfg.num_hiddens2))
        return HouseMLPScratch(num_inputs=X.shape[1], hidden=hidden,
                               dropouts=(float(cfg.dropout1), float(cfg.dropout2)))

    def get_callbacks(self, cfg):
        return [KaggleSubmit()]


SPEC = HouseMLPSpec()
