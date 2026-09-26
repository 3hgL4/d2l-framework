"""lenet_simple —— 《动手学深度学习》PyTorch版 第6章（卷积神经网络）简洁实现链路。

来源：d2l第6章-卷积神经网络-代码.md（PDF p258–p262，6.6 卷积神经网络（LeNet））。
颗粒度定位：与 algorithms/lenet（全手写算子版）相对的"书·简洁版"——
    模型 = nn.Conv2d/Sigmoid/AvgPool2d/Linear 组网（书 p258）
    初始化 = train_ch6 的 xavier_uniform_ 应用到 Linear/Conv2d（书 p261）
    损失 = nn.CrossEntropyLoss（书 p261）
    优化器 = 书用 torch.optim.SGD；按本框架纪律改为手写 SGDScratch（无动量，等价）
差异（相对原书）：训练循环/Animator/Accumulator 由 infra Trainer 承担；
    超参取书默认（batch=256, lr=0.9, epochs=10），不承诺复现书上数字。

运行：python run.py --algo lenet_simple
      快速验证：python run.py --algo lenet_simple --override epochs=2 num_workers=0
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]   # d2l 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.datasets import load_fashion_mnist
from infra.minispec import MiniSpec


# ── ② 模型：书 p258 的简洁 LeNet（nn 层组网，非手写算子）──────────────
def lenet_simple() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))


def init_weights(m):
    """书 p261 train_ch6 的初始化：Linear/Conv2d 一律 xavier_uniform_。"""
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


# ── ④ 损失与指标 ──────────────────────────────────────────────────────
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


class SGDScratch:
    """手写 SGD（书 p261 用 torch.optim.SGD，此处按框架纪律手写等价实现）。

    契约：step()/zero_grad()/state_dict()/load_state_dict() + param_groups；
    签名兼容 MiniSpec 的 cls(params, lr=cfg.lr)；无动量（与书一致）。
    """

    def __init__(self, params, lr=0.05):
        self.params = list(params)
        self.lr = float(lr)
        self.param_groups = [{"lr": self.lr, "params": self.params}]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p -= self.lr * p.grad

    def state_dict(self):
        return {"lr": self.lr}

    def load_state_dict(self, state):
        self.lr = float(state["lr"])
        self.param_groups[0]["lr"] = self.lr


# ── ⑥ 声明注册 ────────────────────────────────────────────────────────
class LeNetSimpleSpec(MiniSpec):
    name = "lenet_simple"
    model = lenet_simple
    loss = nn.CrossEntropyLoss
    datasets = load_fashion_mnist
    optimizer = SGDScratch
    metrics = {"acc": accuracy}
    config = {
        "epochs": 10,
        "lr": 0.9,               # 书 train_ch6 默认值（p262）
        "batch_size": 256,       # 书 p260
        "data_root": str(ROOT / "data"),
        "patience": 0,
        "amp": False,            # 手写优化器无 param_groups 兼容 AMP，关
    }

    def build_model(self, cfg):
        net = lenet_simple()
        net.apply(init_weights)
        return net

    def build_optimizer(self, params, cfg):
        return SGDScratch(params, lr=float(cfg.lr))


SPEC = LeNetSimpleSpec()
