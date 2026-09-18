"""softmax 专属可视化：预测展示图 predictions.png。

算法层回调示范： predictions.png 依赖"图片渲染 + 类别名 + 逐格正确/错误
着色"，这些全是算法知识，不进 infra；通过可选契约 get_callbacks(cfg)
注入，infra 不感知其存在。

on_train_end 时加载 best.pt（缺省）或 last.pt 权重，取测试集首个 batch
画 rows*8 网格图（绿=正确，红=错误），存入当前 run 目录。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from infra.callbacks import Callback
from infra.checkpoint import load

ROOT = Path(__file__).resolve().parents[2]   # d2l 项目根
DATA_ROOT = ROOT.parent / "data"
COLS = 8  # 每行 8 张，与原 softmax_scratch.py 保持一致


class PredictionPlotter(Callback):
    def __init__(self, dataset: str = "FashionMNIST", rows: int = 3,
                 use_best: bool = True, batch_size: int = 64):
        self.dataset, self.rows = dataset, rows
        self.use_best, self.batch_size = use_best, batch_size

    def on_train_end(self, ctx):
        ckpt_dir = Path(ctx.run_dir) / "ckpt"
        p_best, p_last = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
        p = p_best if (self.use_best and p_best.exists()) else p_last
        if not p.exists():
            ctx.logger.warning("[预测图] 无可用 checkpoint，跳过")
            return
        payload = load(p, map_location="cpu")

        model = ctx.model
        model.load_state_dict(payload["model"])
        model.eval()
        device = next(model.parameters()).device

        tfm = transforms.ToTensor()
        ds = getattr(torchvision.datasets, self.dataset)(
            root=str(DATA_ROOT), train=False, download=False, transform=tfm)
        X, y = next(iter(DataLoader(ds, batch_size=self.batch_size, shuffle=False)))
        with torch.no_grad():
            preds = model(X.to(device)).argmax(dim=1).cpu()
        classes = ds.classes

        n_show = min(len(y), self.rows * COLS)
        n_rows = (n_show + COLS - 1) // COLS
        fig, axes = plt.subplots(n_rows, COLS, figsize=(12, 1.5 * n_rows))
        axes = axes.flatten()
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        for i in range(n_show):
            ax = axes[i]
            ax.imshow(X[i].squeeze(), cmap="gray")
            ax.axis("off")
            ok = preds[i] == y[i]
            ax.set_title(f"{classes[preds[i]]}\n(真实:{classes[y[i]]})",
                         fontsize=8, color=("green" if ok else "red"))
        for ax in axes[n_show:]:
            ax.axis("off")
        fig.suptitle(f"预测结果（绿=正确，红=错误） · epoch {payload['epoch']} 权重")
        plt.tight_layout()
        fig.savefig(Path(ctx.run_dir) / "predictions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        ctx.logger.info(f"[预测图] predictions.png 已存（基于 epoch {payload['epoch']} 权重）")
