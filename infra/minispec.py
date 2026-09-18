"""MiniSpec：声明式算法层 —— 让算法研究者只写算法，不写工程。

理念：算法研究者的全部产出 = 模型 + 损失 + 数据 + 指标 + 超参（纯算法知识）；
DataLoader 装配（pin_memory/种子生成器/persistent_workers）、batch 解包、
指标按批聚合、优化器样板等工程环节由本适配器代劳。

算法侧只声明类属性（全部是算法知识）：

    from infra.minispec import MiniSpec

    class MyNet(nn.Module):
        ...                                # d2l 对应章节的从零实现

    def load_data(cfg):
        ...                                # 下载/变换/归一化等数据决策
        return train_dataset, val_dataset

    class MyAlgo(MiniSpec):
        name      = "my_algo"              # 必填，与 algorithms/ 目录名一致
        model     = MyNet                  # 必填，类或零参工厂；需 cfg 时覆写 build_model
        loss      = nn.CrossEntropyLoss    # 必填，损失本身；类会自动实例化
        datasets  = load_data              # 必填，(cfg) -> (train_ds, val_ds)
        optimizer = torch.optim.SGD        # 可选，默认 SGD；lr 走 cfg.lr（默认 0.1）
        metrics   = {"acc": acc_fn}        # 可选，fn(y_hat, y) -> float，键名入曲线/history
        config    = {"epochs": 20}         # 可选，算法超参默认值（--override 可调）

optimizer 三种写法：
    torch.optim.SGD                       lr 用 cfg.lr，命令行 --override lr=... 可调
    (torch.optim.Adam, {"lr": 1e-3})      显式给定则固定（不走 cfg.lr）
    "sgd" / "adam" / "adamw"              字符串快捷方式
自定义/手写优化器：传工厂 fn(params, **kw)；无 param_groups 时须 amp=False，
且建议实现 state_dict()/load_state_dict() 否则续训退化（见 contract.py）。

进阶（全部可选）：
    build_model(self, cfg)                覆写后可用 cfg 组网（如按超参定宽度）
    unpack = fn(batch) -> (X, y)          batch 非 (X, y) 形式时声明
    scheduler = fn(optimizer, cfg)        按 epoch 步进的学习率调度
    get_callbacks(self, cfg)              算法专属回调（追加在默认回调之后）

本类是 contract.AlgoSpec 的声明式糖衣：实例自动满足完整契约并通过
validate_spec；trainer/checkpoint/callbacks 等冻结面对此无感知 —— 本模块
属于边缘面（新增可选能力），不改变既有算法的任何接入方式。

实现注记：所有声明一律经类级读取（type(self).xxx，绕开实例访问的描述符
绑定），因此 loss/datasets 等可直接赋普通函数，无需 staticmethod 包装。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .contract import validate_spec
from .data import make_dataloaders

_OPTIMIZERS = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam,
               "adamw": torch.optim.AdamW}


class MiniSpec:
    """声明式算法基类：子类只声明算法知识，工程细节由本类适配成 AlgoSpec 契约。"""

    name = None
    model = None
    loss = None
    datasets = None
    optimizer = torch.optim.SGD
    metrics = None
    config = None
    unpack = None
    scheduler = None

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        missing = [a for a in ("name", "model", "loss", "datasets")
                   if getattr(cls, a, None) is None]
        if missing:
            raise TypeError(f"{cls.__name__} 缺少必填声明: {missing}"
                            f"（name/model/loss/datasets 四项必填）")

    def __init__(self):
        validate_spec(self)  # 双保险：子类覆写破坏契约时，实例化即报错

    # ------------------------------------------------------------------
    # 声明解析
    # ------------------------------------------------------------------
    def _optimizer_spec(self) -> tuple:
        o = type(self).optimizer
        if isinstance(o, str):
            key = o.lower()
            if key not in _OPTIMIZERS:
                raise ValueError(f"未知优化器别名 {o!r}，可选: {sorted(_OPTIMIZERS)}"
                                 f"（自定义请传优化器类或工厂函数）")
            o = _OPTIMIZERS[key]
        if isinstance(o, tuple):
            cls, kw = o
        else:
            cls, kw = o, None
        return cls, dict(kw or {})

    # ------------------------------------------------------------------
    # AlgoSpec 契约实现（infra 只看得到这组方法）
    # ------------------------------------------------------------------
    def default_config(self) -> dict:
        # lr/batch_size 是算法域超参（infra 默认集不含），给安全默认值
        cfg = {"batch_size": 128, "lr": 0.1}
        cfg.update(type(self).config or {})
        return cfg

    def build_model(self, cfg) -> nn.Module:
        return type(self).model()

    def build_loss(self):
        L = type(self).loss
        if isinstance(L, type) and issubclass(L, nn.Module):
            return L()  # 传类则实例化；传函数/实例则原样使用
        return L

    def build_optimizer(self, params, cfg):
        cls, kw = self._optimizer_spec()
        kw.setdefault("lr", cfg.lr)  # 显式给 lr 则固定，否则走 cfg.lr 可 override
        return cls(params, **kw)

    def build_dataloaders(self, cfg) -> tuple:
        train_ds, val_ds = type(self).datasets(cfg)
        return make_dataloaders(train_ds, val_ds, cfg)  # 工程细节：种子/worker/pin

    def unpack_batch(self, batch) -> tuple:
        unpack = type(self).unpack
        return batch if unpack is None else unpack(batch)

    def compute_metrics(self, y_hat, y) -> dict:
        return {k: float(fn(y_hat, y))
                for k, fn in (type(self).metrics or {}).items()}

    def build_scheduler(self, optimizer, cfg):
        sched = type(self).scheduler
        return sched(optimizer, cfg) if callable(sched) else None

    def get_callbacks(self, cfg) -> list:
        return list(getattr(type(self), "callbacks", None) or [])
