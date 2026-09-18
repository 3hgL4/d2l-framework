"""softmax 回归从零实现（d2l 3.4-3.6）—— MiniSpec 声明式接入范例。

本文件演示"算法研究者只写算法，不写工程"：
    写  log_softmax / 交叉熵 / 手写 W,b 的模型 / 数据集 / 指标 —— 全是算法知识；
    不写 DataLoader 装配、batch 解包、指标聚合、断点续训、早停、曲线 —— infra 代劳。
运行: python run.py --algo softmax
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parents[2]   # d2l 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.callbacks import Callback
from infra.minispec import MiniSpec

DATA_ROOT = ROOT.parent / "data"


# ---------------------------------------------------------------------------
# ① 模型部分 1：softmax 的数值稳定实现（d2l 3.4.2，log-sum-exp 技巧）
#    直接 exp 会溢出，先减去行最大值再归一化，最终以 log 概率形式输出
# ---------------------------------------------------------------------------
def log_softmax(X):
    X_max = X.max(dim=1, keepdim=True).values
    return X - X_max - (X - X_max).exp().sum(dim=1, keepdim=True).log()


# ---------------------------------------------------------------------------
# ② 模型部分 2：从零交叉熵（d2l 3.4.3）
#    y_hat 是 log 概率，取真实类别那一列的负 log 概率求均值（契约要求返回已 mean 标量）
# ---------------------------------------------------------------------------
def cross_entropy(y_hat, y):
    return -y_hat[range(len(y_hat)), y].mean()


# ---------------------------------------------------------------------------
# ③ 模型部分 3：手写参数的 softmax 回归（d2l 3.4.1）
#    W(784x10), b(10) 为 nn.Parameter 从零维护，不用 nn.Linear；
#    forward = 展平 + 线性 + log_softmax。契约要求返回 nn.Module（infra 依赖
#    其 parameters()/state_dict()/to(device)），从零实现也必须包成 Module
# ---------------------------------------------------------------------------
class SoftmaxRegressionScratch(nn.Module):
    def __init__(self, num_inputs: int = 784, num_outputs: int = 10):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_inputs, num_outputs) * 0.01)
        self.b = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, X):
        return log_softmax(X.reshape((-1, self.W.shape[0])) @ self.W + self.b)


# ---------------------------------------------------------------------------
# ④ 数据：FashionMNIST（d2l 3.5）。返回 (train_ds, val_ds) 即可，
#    DataLoader 装配（batch_size/种子/pin_memory/workers）由 MiniSpec 代劳。
#    归一化是算法决策点：d2l 原书仅 ToTensor；要加 Normalize 就改这一行
# ---------------------------------------------------------------------------
def load_data(cfg):
    tfm = transforms.ToTensor()
    ds = torchvision.datasets.FashionMNIST
    return (ds(root=str(DATA_ROOT), train=True, download=False, transform=tfm),
            ds(root=str(DATA_ROOT), train=False, download=False, transform=tfm))


# ---------------------------------------------------------------------------
# ⑤ 指标：准确率（d2l 3.4.4）。键名 "acc" 会出现在 history.csv 与 curves.png
# ---------------------------------------------------------------------------
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# ---------------------------------------------------------------------------
# ⑥ 算法专属可视化（可选扩展点）：预测对照图，绿=预测对，红=错，副行为真实标签。
#    经 get_callbacks 注入，infra 完全不感知本算法（红线不破）
# ---------------------------------------------------------------------------
_CLASSES = ["t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle boot"]


class PredictionPlotter(Callback):
    def __init__(self, n: int = 8):
        self.n = n

    def on_train_end(self, ctx):
        import matplotlib.pyplot as plt

        _, val_ds = load_data(ctx.cfg)
        X = torch.stack([val_ds[i][0] for i in range(self.n)])
        y = torch.tensor([int(val_ds[i][1]) for i in range(self.n)])
        device = next(ctx.model.parameters()).device
        with torch.no_grad():
            pred = ctx.model(X.to(device)).argmax(dim=1).cpu()
        cols = self.n // 2
        fig, axes = plt.subplots(2, cols, figsize=(cols * 1.8, 4.4))
        for ax, img, p, t in zip(axes.ravel(), X[:, 0], pred.tolist(), y.tolist()):
            good = p == t
            title = _CLASSES[p] if good else f"{_CLASSES[p]}\n[{_CLASSES[t]}]"
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(title, fontsize=9,
                         color="#0F6E56" if good else "#A32D2D")
        fig.suptitle("predictions (green = correct, red = wrong)")
        fig.tight_layout()
        out = ctx.run_dir / "predictions.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        ctx.logger.info(f"[预测图] 已保存 {out.name}（绿=预测正确，红=错误，副行为真实标签）")


# ---------------------------------------------------------------------------
# ⑦ 声明注册：MiniSpec 五组算法知识，其余全默认
#    loss 直接赋函数、datasets 直接赋函数 —— 声明经类级读取，无需 staticmethod
# ---------------------------------------------------------------------------
class SoftmaxScratch(MiniSpec):
    name = "softmax"
    model = SoftmaxRegressionScratch
    loss = cross_entropy
    datasets = load_data
    metrics = {"acc": accuracy}
    optimizer = torch.optim.SGD            # lr 走 cfg.lr，--override lr=0.3 可调
    config = {"epochs": 20, "lr": 0.1, "batch_size": 256, "patience": 5}

    def get_callbacks(self, cfg):
        return [PredictionPlotter()]


SPEC = SoftmaxScratch()
