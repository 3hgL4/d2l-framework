"""Softmax 回归（FashionMNIST）· 从零实现，接入 infra 契约。

模型/损失/优化器沿用 d2l 第 3 章从零实现写法：
- 参数化: W 小随机初始化 + b 全 0（d2l 3.4 节），保留手写形式而非 nn.Linear；
- 损失:   logsumexp 稳定交叉熵（d2l 3.6 节写法）；
- 优化器: 手写 SGD（d2l 3.2 节），补了 state_dict/load_state_dict 以支持断点。

注意 amp=False：手写 SGD 无 param_groups，GradScaler 需要 param_groups，
因此本算法关闭混合精度（infra 对 nn 优化器无此限制）。
原交互式预测展示保留在 softmax/softmax_scratch.py，此处专注训练流程。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parents[2]   # d2l 项目根
if str(ROOT) not in sys.path:                # 允许从任意位置 import 本模块
    sys.path.insert(0, str(ROOT))

from infra.data import make_loader

DATA_ROOT = ROOT.parent / "data"             # 与既有脚本共用同一数据目录


# ---------------------------------------------------------------------------
# 从零实现部分（与 softmax_scratch.py 保持一致的写法）
# ---------------------------------------------------------------------------
class SoftmaxScratch(nn.Module):
    """y = X.reshape(n,-1) @ W + b，W/b 均为显式 Parameter。"""

    def __init__(self, num_inputs: int = 784, num_outputs: int = 10,
                 init_std: float = 0.01):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, init_std, size=(num_inputs, num_outputs)))
        self.b = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, X):
        return X.reshape((-1, self.W.shape[0])) @ self.W + self.b


def cross_entropy(logits, y):
    """log-sumexp 稳定交叉熵，等价 -log softmax(logits)[y]（d2l 3.6 节）。"""
    rows = torch.arange(len(logits), device=logits.device)
    return -(logits[rows, y] - logits.logsumexp(dim=1)).mean()


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return (y_hat.type(y.dtype) == y).float().mean().item()


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


# ---------------------------------------------------------------------------
# 契约实现（infra 通过它注入本算法）
# ---------------------------------------------------------------------------
class SoftmaxAlgo:
    name = "softmax"

    def default_config(self):
        return {"epochs": 50, "lr": 0.1, "batch_size": 256, "num_workers": 4,
                "init_std": 0.01, "amp": False, "patience": 8,
                "dataset": "FashionMNIST", "data_root": "",
                "predict_demo": True, "predict_rows": 3}

    def build_model(self, cfg):
        return SoftmaxScratch(init_std=cfg.init_std)

    def build_loss(self):
        return cross_entropy

    def build_optimizer(self, params, cfg):
        return SGDScratch(params, lr=cfg.lr)

    def build_dataloaders(self, cfg):
        root = Path(cfg.data_root) if cfg.data_root else DATA_ROOT
        tfm = transforms.ToTensor()  # [0,255] HWC uint8 -> [0,1] CHW float32
        ds = getattr(torchvision.datasets, cfg.dataset)
        train_ds = ds(root=str(root), train=True, download=False, transform=tfm)
        test_ds = ds(root=str(root), train=False, download=False, transform=tfm)
        return (make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, seed=cfg.seed),
                make_loader(test_ds, cfg.batch_size, False, cfg.num_workers))

    def unpack_batch(self, batch):
        return batch  # 模型内部自行 reshape

    def compute_metrics(self, y_hat, y):
        return {"acc": accuracy(y_hat, y)}

    def get_callbacks(self, cfg):
        """算法专属回调：预测展示图（predictions.png）。经此注入，infra 零感知。"""
        if not cfg.get("predict_demo", True):
            return []
        from .predict import PredictionPlotter
        return [PredictionPlotter(dataset=cfg.dataset, rows=cfg.get("predict_rows", 3))]


SPEC = SoftmaxAlgo()


if __name__ == "__main__":
    # 直接运行本文件时的自检：合成数据走一遍前向/反向，不碰真实数据集。
    # 完整训练请从项目根执行: python run.py --algo softmax
    from infra.config import AttrDict
    cfg = AttrDict(SPEC.default_config())
    cfg["seed"] = 0
    model = SPEC.build_model(cfg)
    loss_fn = SPEC.build_loss()
    opt = SPEC.build_optimizer(model.parameters(), cfg)
    X = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    l = loss_fn(model(X), y)
    l.backward()
    opt.step()
    print(f"self-check ok: model={type(model).__name__}, loss={l.item():.4f}")
    print("训练请从项目根运行: python run.py --algo softmax")
