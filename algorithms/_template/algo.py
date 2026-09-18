"""算法模板：学新算法时复制本目录为 algorithms/<算法名>/，替换 TODO 即可。

三步接入（infra 零改动）：
  1. 复制:   整个 _template/ 目录改名为 algorithms/<算法名>/（目录名即 --algo 值）
  2. 填空:   替换下方所有 TODO；模块级必须保留 SPEC = XxxAlgo()
  3. 运行:   项目根执行 python run.py --algo <算法名>
             试跑建议先 --override epochs=3 num_workers=0 快速验证

要点回顾：
  - 本文件是被 run.py 注入加载的库模块，不是入口，不要直接 python algo.py 训练；
  - 只实现契约方法（duck-typing，无需继承），缺方法会在启动时立即报错；
  - 归一化/变换/batch_size 等数据决策写在 build_dataloaders；
  - 自定义优化器需带 step()/zero_grad()，支持断点需补 state_dict/load_state_dict；
  - 用 nn 优化器可用 amp=True 提速；手写优化器没有 param_groups，必须 amp=False；
  - 算法专属可视化/逻辑走可选 get_callbacks(cfg)（参考 algorithms/softmax/predict.py）。
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

DATA_ROOT = ROOT.parent / "data"             # 与其他算法共用同一数据目录


# ---------------------------------------------------------------------------
# 从零实现部分（按 d2l 对应章节写在这里）
# ---------------------------------------------------------------------------
class TODOModel(nn.Module):
    """TODO: 换成你正在学的模型（d2l 对应章节的从零实现）。"""

    def __init__(self):
        super().__init__()
        # 例: self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
        raise NotImplementedError


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return (y_hat.type(y.dtype) == y).float().mean().item()


# ---------------------------------------------------------------------------
# 契约实现（infra 通过它注入本算法）
# ---------------------------------------------------------------------------
class TODOAlgo:
    name = "TODO"  # 与目录名一致

    def default_config(self):
        # TODO: 本算法默认超参；命令行 --override 可临时覆盖
        return {"epochs": 20, "lr": 0.05, "batch_size": 256, "num_workers": 4,
                "amp": True, "patience": 5,
                "dataset": "FashionMNIST", "data_root": ""}

    def build_model(self, cfg):
        return TODOModel()

    def build_loss(self):
        # TODO: 分类常用 nn.CrossEntropyLoss()；回归用 nn.MSELoss()
        return nn.CrossEntropyLoss()

    def build_optimizer(self, params, cfg):
        # TODO: torch.optim.SGD(params, lr=cfg.lr) 等
        raise NotImplementedError

    def build_dataloaders(self, cfg):
        root = Path(cfg.data_root) if cfg.data_root else DATA_ROOT
        # TODO: 归一化/变换是算法决策，写在这里（如 Normalize(mean, std)）
        tfm = transforms.ToTensor()
        ds = getattr(torchvision.datasets, cfg.dataset)
        train_ds = ds(root=str(root), train=True, download=False, transform=tfm)
        test_ds = ds(root=str(root), train=False, download=False, transform=tfm)
        return (make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, seed=cfg.seed),
                make_loader(test_ds, cfg.batch_size, False, cfg.num_workers))

    def unpack_batch(self, batch):
        return batch  # TODO: 若 batch 不是 (X, y) 形式（如 one-hot）在此解包

    def compute_metrics(self, y_hat, y):
        # TODO: 分类=acc，回归=MSE/MAE；键名会出现在曲线图和 history.csv
        return {"acc": accuracy(y_hat, y)}

    # 可选：算法专属回调（不需要就整段删掉）
    # def get_callbacks(self, cfg):
    #     return [MyCallback(...)]


SPEC = TODOAlgo()
