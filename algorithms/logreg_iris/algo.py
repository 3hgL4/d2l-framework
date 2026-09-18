"""logistic 回归（iris 二分类）—— MiniSpec 声明式接入 + 机器学习课程第二次作业对照。

手写 sigmoid 与从零 BCE 损失（与作业手写 numpy 版同一数学）；
数据与作业一致：iris 取 y<2 两类、前两特征（花萼长/宽）、
train_test_split(random_state=666)（sklearn 默认 test_size=0.25）。
运行: python run.py --algo logreg_iris
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.minispec import MiniSpec


# ── ② 模型：手写 sigmoid 的 logistic 回归（零初始化对齐手写 numpy 版）──
def sigmoid(z):
    """数值稳定的手写 sigmoid：z>=0 走 1/(1+e^-z)，z<0 走 e^z/(1+e^z)，防 exp 溢出。"""
    return torch.where(z >= 0, 1.0 / (1.0 + torch.exp(-z)),
                       torch.exp(z) / (1.0 + torch.exp(z)))


class LogRegScratch(nn.Module):
    def __init__(self, num_inputs: int = 2):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(num_inputs, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        z = (X @ self.w + self.b).squeeze(1)   # (n,)
        return sigmoid(z)                       # 正类概率 p ∈ (0,1)


def bce_loss(p, y):
    """从零二元交叉熵：-(y·log p + (1-y)·log(1-p)) 均值；eps 防 log(0)。"""
    eps = 1e-12
    return -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean()


def accuracy(p, y):
    return ((p >= 0.5).float() == y).float().mean().item()


# ── ③ 数据：iris 二分类，与作业同一 split ───────────────────────────
def load_data(cfg):
    iris = datasets.load_iris()
    X = iris.data[iris.target < 2, :2]         # 花萼长度/宽度，两类 100 样本
    y = iris.target[iris.target < 2].astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=666)
    to_ds = lambda a, b: TensorDataset(torch.tensor(a, dtype=torch.float32),
                                       torch.tensor(b, dtype=torch.float32))
    return to_ds(X_tr, y_tr), to_ds(X_te, y_te)


# ── ⑥ 声明注册 ──────────────────────────────────────────────────────
class LogRegIris(MiniSpec):
    name = "logreg_iris"
    model = LogRegScratch
    loss = bce_loss
    datasets = load_data
    metrics = {"acc": accuracy}
    optimizer = torch.optim.SGD
    config = {"epochs": 200, "lr": 0.1, "batch_size": 32}


SPEC = LogRegIris()
