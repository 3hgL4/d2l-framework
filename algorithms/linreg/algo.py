"""线性回归（d2l 3.2 节）· 从零实现，接入 infra 契约。

七步对应关系（d2l 3.2 -> 本框架契约）：
    生成数据集 / 读取数据集  -> build_dataloaders（合成数据 + TensorDataset）
    初始化模型参数          -> LinearRegressionScratch.__init__（w 小随机、b 全 0）
    定义模型               -> LinearRegressionScratch.forward（X@w + b）
    定义损失函数           -> squared_loss（均方误差的一半，d2l 写法）
    定义优化算法           -> SGDScratch（手写小批量 SGD，d2l 3.2 节）
    训练                   -> python run.py --algo linreg（infra 通用循环完成）

注意 amp=False：手写 SGD 无 param_groups，GradScaler 需要参数组。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]   # d2l 项目根
if str(ROOT) not in sys.path:                # 允许从任意位置 import 本模块
    sys.path.insert(0, str(ROOT))

from infra.data import make_loader


# ---------------------------------------------------------------------------
# 从零实现部分（与 d2l 3.2 节保持一致的写法）
# ---------------------------------------------------------------------------
class LinearRegressionScratch(nn.Module):
    """y = X @ w + b；w/b 均为显式 Parameter（d2l 3.2 节：w~N(0,0.01)，b=0）。"""

    def __init__(self, num_inputs: int = 2, init_std: float = 0.01):
        super().__init__()
        self.w = nn.Parameter(torch.normal(0, init_std, size=(num_inputs, 1)))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b


def squared_loss(y_hat, y):
    """均方误差的一半（d2l 3.2 节）；y 需 reshape 成与 y_hat 同形。"""
    return ((y_hat - y.reshape(y_hat.shape)) ** 2 / 2).mean()


class SGDScratch:
    """从零实现的小批量 SGD（d2l 3.2 节）+ 最小状态接口（支持断点续训）。"""

    def __init__(self, params, lr: float):
        self.params, self.lr = list(params), lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def state_dict(self):
        return {"lr": self.lr}

    def load_state_dict(self, state):
        self.lr = state["lr"]


def synthetic_data(w, b, num_examples):
    """y = Xw + b + 噪声（d2l 3.2 节生成数据集）。w 先 reshape 成列向量，
    保证 matmul 产生 (n,1)，与噪声 (n,1) 同形，避免广播扩张。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    w = w.reshape(-1, 1)
    y = torch.matmul(X, w) + b + torch.normal(0, 0.01, (num_examples, 1))
    return X, y.reshape((-1, 1))


# ---------------------------------------------------------------------------
# 契约实现（infra 通过它注入本算法）
# ---------------------------------------------------------------------------
class LinregAlgo:
    name = "linreg"

    def default_config(self):
        return {"epochs": 5, "lr": 0.03, "batch_size": 32, "num_workers": 0,
                "num_examples": 1000, "init_std": 0.01,
                "true_w": [2.0, -3.4], "true_b": 4.2, "noise_std": 0.01,
                "amp": False, "patience": 0}

    def build_model(self, cfg):
        return LinearRegressionScratch(num_inputs=len(cfg.true_w), init_std=cfg.init_std)

    def build_loss(self):
        return squared_loss

    def build_optimizer(self, params, cfg):
        return SGDScratch(params, lr=cfg.lr)

    def build_dataloaders(self, cfg):
        # 生成数据集 + 读取数据集：合成数据的划分决策（算法知识）在此完成
        w = torch.tensor(cfg.true_w)
        X, y = synthetic_data(w, cfg.true_b, cfg.num_examples)
        n_train = int(cfg.num_examples * 0.8)
        train_ds = torch.utils.data.TensorDataset(X[:n_train], y[:n_train])
        val_ds = torch.utils.data.TensorDataset(X[n_train:], y[n_train:])
        return (make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, seed=cfg.seed),
                make_loader(val_ds, cfg.batch_size, False, cfg.num_workers))

    def unpack_batch(self, batch):
        return batch  # (X, y) 张量对，无需解包

    def compute_metrics(self, y_hat, y):
        return {}  # 回归无 acc 类指标；loss 本身即 MSE，曲线看 loss/val_loss 即可


SPEC = LinregAlgo()
