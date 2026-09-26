"""算法模板（MiniSpec 声明式）：复制本目录为 algorithms/<算法名>/，只填算法，不写工程。

三步接入（infra 零改动）：
  1. 复制:   整个 _template/ 目录改名为 algorithms/<算法名>/（目录名 = --algo 值 = name）
  2. 填空:   ②③④⑥ 标 TODO 处 —— 全部是算法知识；⑤ 按需选配
  3. 运行:   项目根执行 python run.py --algo <算法名>
             试跑建议先 --override epochs=3 num_workers=0 快速验证

═══════════════════════════════════════════════════════════════════
固定骨架：任何 ML 算法的 algo.py 都是同一段式，只有 ②③④ 内容不同
    ① import 与路径引导（原样保留，永不改）
    ② 模型：从零实现的 nn.Module          ← 每个算法唯一的大块
    ③ 数据：load_data(cfg) -> (train_ds, val_ds)
    ④ 损失 + 指标函数
    ⑤ 可选扩展：scheduler / unpack / callbacks（按需，默认全删）
    ⑥ 声明注册：五行 MiniSpec
═══════════════════════════════════════════════════════════════════

算法家族差异速查（换算法 = 换 ②③④⑥ 里的内容，骨架不动）：
  ┌──────────┬────────────────────┬─────────────────┬────────────────────┐
  │ 家族     │ 损失(④)            │ 指标(④)         │ 注意               │
  ├──────────┼────────────────────┼─────────────────┼────────────────────┤
  │ 分类     │ nn.CrossEntropyLoss│ acc             │ y 必须 int64 类别  │
  │ 回归     │ nn.MSELoss         │ mse / mae       │ y 形状 (n,1)，见③  │
  │ 手写损失 │ 自定义 fn(y_hat,y) │ 同上            │ 返回已 mean 标量   │
  └──────────┴────────────────────┴─────────────────┴────────────────────┘
  对应 d2l 章节：linreg 3.2 / softmax 3.4-3.6 / MLP 4.1-4.3 / 权重衰减 4.5 /
  dropout 4.6 / CNN 6.x（参考 algorithms/softmax/algo.py 完整范例）

优化器三种写法（可选，默认 SGD）：
    torch.optim.SGD                    lr 走 cfg.lr（默认 0.1，--override 可调）
    (torch.optim.Adam, {"lr": 1e-3})   显式给定则固定
    "sgd" / "adam" / "adamw"           字符串快捷方式
  手写优化器：传工厂 fn(params, **kw)；无 param_groups 时须 --override amp=False，
  建议实现 state_dict()/load_state_dict() 否则续训退化（见 infra/contract.py）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ── ① import 与路径引导（固定，永不改）──────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # d2l 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.minispec import MiniSpec

DATA_ROOT = ROOT / "data"                    # 数据目录 d2l/data（2026-09-22 统一）


# ── ② 模型：从零实现（每个算法唯一的大块，d2l 对应章节抄这里）──────
class MyNet(nn.Module):
    """TODO: 换成你正在学的模型（分类默认骨架，784 输入 / 10 类可改）。"""

    def __init__(self, num_inputs: int = 784, num_outputs: int = 10):
        super().__init__()
        # 例（从零）: self.W = nn.Parameter(torch.randn(784, 10) * 0.01)
        # 例（速成）: self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError


# 回归变体（linreg 3.2）：W/b 手写 + 前向输出连续值
# class LinearRegressionScratch(nn.Module):
#     def __init__(self, num_inputs: int = 2, lr: float = 0.03):
#         super().__init__()
#         self.w = nn.Parameter(torch.randn(num_inputs, 1) * 0.01)   # 列向量 (n,1)！
#         self.b = nn.Parameter(torch.zeros(1))
#
#     def forward(self, X):
#         return X @ self.w + self.b


# ── ③ 数据：返回 (train_ds, val_ds)；归一化/变换是算法决策点 ───────
def load_data(cfg):
    # TODO: 分类（FashionMNIST）默认骨架；回归/合成数据见下方注释
    root = Path(cfg.data_root) if cfg.data_root else DATA_ROOT
    tfm = transforms.ToTensor()              # 需要 Normalize 在此拼接
    ds = getattr(torchvision.datasets, cfg.dataset)
    return (ds(root=str(root), train=True, download=False, transform=tfm),
            ds(root=str(root), train=False, download=False, transform=tfm))


# 回归/合成数据变体：注意 y 必须是列向量 (n,1)，(n,)+(n,1) 会广播成 (n,n) 炸掉
# def load_data(cfg):
#     n, d = 1000, 2
#     X = torch.randn(n, d)
#     w_true = torch.tensor([2.0, -3.4]).reshape(-1, 1)
#     y = X @ w_true + 4.2 + torch.randn(n, 1) * 0.01
#     ds = torch.utils.data.TensorDataset(X, y)          # 整块当 val 亦可切分
#     return ds, ds


# ── ④ 损失 + 指标 ──────────────────────────────────────────────────
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return (y_hat.type(y.dtype) == y).float().mean().item()


# 回归指标变体（y 形状须与 y_hat 一致，都是 (n,1)）
# def mse(y_hat, y):
#     return ((y_hat - y) ** 2).mean().item()


# ── ⑤ 可选扩展（不需要就把本节整段删掉）────────────────────────────
# 学习率调度（按 epoch 步进，声明为 fn(optimizer, cfg)）：
#     from torch.optim.lr_scheduler import StepLR
#     scheduler = lambda opt, cfg: StepLR(opt, step_size=5, gamma=0.5)
#
# batch 非 (X, y) 形式时解包（如 one-hot 标签转回类别）：
#     unpack = lambda batch: (batch[0], batch[1].argmax(dim=1))
#
# 算法专属可视化回调（参考 algorithms/softmax/algo.py 的 PredictionPlotter）：
#     def get_callbacks(self, cfg):
#         return [MyPlotter()]
#
# 需按 cfg 组网（如超参定隐藏层宽度）：在声明类里覆写
#     def build_model(self, cfg):
#         return MLPScratch(num_hiddens=cfg.num_hiddens)


# ── ⑥ 声明注册：五组算法知识，其余全默认 ───────────────────────────
class MyAlgo(MiniSpec):
    name = "TODO"                            # 与目录名一致
    model = MyNet                            # ② 模型
    datasets = load_data                     # ③ 数据
    loss = nn.CrossEntropyLoss               # ④ 损失（回归用 nn.MSELoss）
    metrics = {"acc": accuracy}              # ④ 指标（回归用 {"mse": mse}）
    optimizer = torch.optim.SGD              # 可选；lr 默认 0.1，--override 可调
    config = {"epochs": 20, "patience": 5,
              "dataset": "FashionMNIST", "data_root": ""}   # 默认超参


SPEC = MyAlgo()
