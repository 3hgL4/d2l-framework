"""算法模板（MiniSpec 声明式）：复制本目录为 algorithms/<算法名>/，只填算法，不写工程。

三步接入（infra 零改动）：
  1. 复制:   整个 _template/ 目录改名为 algorithms/<算法名>/（目录名 = --algo 值 = name）
  2. 填空:   下方 ①~⑤ 处 TODO —— 全部是算法知识（模型/数据/损失/指标/超参）
  3. 运行:   项目根执行 python run.py --algo <算法名>
             试跑建议先 --override epochs=3 num_workers=0 快速验证

你写什么（纯算法）：模型 nn.Module 从零实现 / 数据集构造与归一化 / 损失函数 /
指标函数 / 默认超参。
你不用写什么（工程，MiniSpec 代劳，见 infra/minispec.py）：DataLoader 装配
（pin_memory/种子生成器/persistent_workers）、batch 解包、指标按批加权聚合、
优化器样板、断点续训、早停与曲线。

优化器三种写法（可选，默认 SGD）：
    torch.optim.SGD                    lr 走 cfg.lr（默认 0.1，--override lr=... 可调）
    (torch.optim.Adam, {"lr": 1e-3})   显式给定则固定
    "sgd" / "adam" / "adamw"           字符串快捷方式

进阶（全部可选，完整契约 AlgoSpec 仍可用，见 infra/contract.py）：
  - 需按 cfg 组网（如超参定宽度）：覆写 build_model(self, cfg)
  - batch 非 (X, y) 形式：声明 unpack = fn(batch) -> (X, y)
  - 学习率调度：scheduler = fn(optimizer, cfg)，按 epoch 步进
  - 算法专属可视化：定义 get_callbacks(self, cfg) -> list[Callback]
  - 手写优化器：传工厂 fn(params, **kw)；无 param_groups 时须 --override amp=False，
    且建议实现 state_dict()/load_state_dict() 否则续训退化为仅恢复 lr
  - 本文件是被 run.py 注入加载的库模块，不要直接 python algo.py 训练
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parents[2]   # d2l 项目根
if str(ROOT) not in sys.path:                # 允许从任意位置 import 本模块
    sys.path.insert(0, str(ROOT))

from infra.minispec import MiniSpec

DATA_ROOT = ROOT.parent / "data"             # 与其他算法共用同一数据目录


# ---------------------------------------------------------------------------
# ① 模型：d2l 对应章节的从零实现
# ---------------------------------------------------------------------------
class MyNet(nn.Module):
    """TODO: 换成你正在学的模型。"""

    def __init__(self):
        super().__init__()
        # 例: self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ② 数据：返回 (train_ds, val_ds)；归一化/变换等数据决策写在这里
# ---------------------------------------------------------------------------
def load_data(cfg):
    root = Path(cfg.data_root) if cfg.data_root else DATA_ROOT
    tfm = transforms.ToTensor()              # TODO: 需要 Normalize 在此拼接
    ds = getattr(torchvision.datasets, cfg.dataset)
    return (ds(root=str(root), train=True, download=False, transform=tfm),
            ds(root=str(root), train=False, download=False, transform=tfm))


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return (y_hat.type(y.dtype) == y).float().mean().item()


# ---------------------------------------------------------------------------
# ③ 声明：五处 TODO 之外零工程代码
# ---------------------------------------------------------------------------
class MyAlgo(MiniSpec):
    name = "TODO"                            # ④ 与目录名一致
    model = MyNet                            # ① 模型
    datasets = load_data                     # ② 数据
    loss = nn.CrossEntropyLoss               # ⑤ 损失（回归用 nn.MSELoss）
    metrics = {"acc": accuracy}              # 键名会出现在曲线图和 history.csv
    optimizer = torch.optim.SGD              # 可选；lr 默认 0.1，--override 可调
    config = {"epochs": 20, "patience": 5,
              "dataset": "FashionMNIST", "data_root": ""}   # 默认超参


SPEC = MyAlgo()
