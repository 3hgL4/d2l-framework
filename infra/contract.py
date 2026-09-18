"""算法契约：infra 与算法层之间唯一的接口定义。

infra 只依赖本文件描述的能力面，绝不反向 import 任何算法实现。
算法层按 duck-typing 实现即可（无需继承）；build_scheduler 可选。

契约方法签名（cfg 为合成后的全局配置 AttrDict）：
    name: str
    default_config() -> dict                    算法超参默认值
    build_model(cfg) -> nn.Module               必须是 nn.Module：infra 依赖其
                                                parameters()/state_dict()/to(device)
                                                枚举与搬迁张量，裸张量列表不满足契约
    build_loss() -> Callable                    loss(y_hat, y) -> 标量(已 mean)
    build_optimizer(params, cfg) -> Any         任意带 step()/zero_grad() 的优化器
    build_scheduler(optimizer, cfg) -> Any|None 可选，按 epoch 步进
    build_dataloaders(cfg) -> (train_dl, val_dl) 归一化等数据决策在此
    unpack_batch(batch) -> (X, y)               batch 解包
    compute_metrics(y_hat, y) -> dict           {指标名: batch 内均值}

可选扩展（算法专属能力走回调注入，infra 同样不感知具体算法）：
    get_callbacks(cfg) -> list[Callback]        算法专属回调（如预测展示图），
                                                追加在默认回调之后执行

声明式接入（推荐给算法学习者）：算法层可继承 infra.minispec.MiniSpec，只声明
model/loss/datasets/metrics/config 等纯算法属性，适配器自动生成下述完整契约。
本文件仍是唯一注入点——MiniSpec 产物同样通过 validate_spec；需要全部控制权
（多优化器/非常规 batch/自定义装配）时直接实现本契约。

优化器序列化协议（断点续训的正确性前提）：
    自定义优化器必须实现 state_dict()/load_state_dict()；
    缺失时 trainer 退化为仅恢复 lr 并显式告警——若优化器含可变状态
    （动量/二阶矩），续训将与完整训练不等价。

变更纪律（核心冻结 + 边缘扩展，替代"infra 永零改动"的不可达目标）：
    冻结面（改动需走冒烟全量回归）：trainer 主循环、checkpoint 语义、
        契约必选方法（REQUIRED）、回调事件名；
    边缘面（允许增量演进，不破坏既有算法）：新增契约可选方法（getattr
        探测）、新内置回调、INFRA_DEFAULTS 新增键、viz 样式参数。

审计：冻结面改动必须在项目根 CHANGELOG.md 记一行（日期 | 改了什么 | 为什么）。
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class AlgoSpec(Protocol):
    name: str

    def default_config(self) -> dict: ...
    def build_model(self, cfg) -> nn.Module: ...
    def build_loss(self) -> Callable: ...
    def build_optimizer(self, params, cfg) -> Any: ...
    def build_dataloaders(self, cfg) -> tuple: ...
    def unpack_batch(self, batch) -> tuple: ...
    def compute_metrics(self, y_hat, y) -> dict: ...


REQUIRED = ("default_config", "build_model", "build_loss", "build_optimizer",
            "build_dataloaders", "unpack_batch", "compute_metrics")


def validate_spec(spec) -> None:
    """启动前快速失败：缺契约方法立即报错，而不是训练中途才炸。"""
    missing = [m for m in REQUIRED if not callable(getattr(spec, m, None))]
    if missing:
        raise TypeError(f"算法 {spec!r} 缺少契约方法: {missing}")
