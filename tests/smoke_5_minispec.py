"""冒烟测试 批5：MiniSpec 声明式算法层（适配器正确性，不训练）。

运行: python tests/smoke_5_minispec.py
覆盖: 声明校验 / 契约兼容 / 配置合成 / 优化器三形态 / 损失两形态 /
      DataLoader 装配 / 指标聚合 / unpack / scheduler / callbacks
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from infra.config import AttrDict
from infra.contract import validate_spec
from infra.minispec import MiniSpec
from tests.fake_algo import FakeAlgo


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        sys.exit(1)


def main():
    cfg = AttrDict({"batch_size": 256, "lr": 0.5})

    # 1. 声明校验：缺必填项在类定义时报错（早于训练，早于 run.py）
    try:
        class BadAlgo(MiniSpec):
            name = "bad"
            model = nn.Linear
            # loss / datasets 故意缺失
    except TypeError as e:
        check("缺必填声明 -> 类定义即报错", "loss" in str(e) and "datasets" in str(e), str(e))
    else:
        check("缺必填声明 -> 类定义即报错", False)

    # 2. FakeAlgo（声明式实现）满足完整契约
    spec = FakeAlgo()
    validate_spec(spec)
    check("MiniSpec 实例通过 validate_spec", True)
    check("default_config 合成正确",
          spec.default_config() == {"batch_size": 128, "lr": 0.1, "epochs": 12, "patience": 0},
          str(spec.default_config()))

    # 3. 优化器三形态
    opt = spec.build_optimizer(torch.nn.ParameterList([nn.Parameter(torch.zeros(1))]), cfg)
    check("类形态: SGD + lr 走 cfg.lr",
          isinstance(opt, torch.optim.SGD) and opt.param_groups[0]["lr"] == 0.5)

    class TupleOpt(MiniSpec):
        name, model, loss, datasets = "t", nn.Linear, nn.MSELoss, lambda c: (None, None)
        optimizer = (torch.optim.SGD, {"momentum": 0.9, "lr": 0.01})

    o2 = TupleOpt().build_optimizer([nn.Parameter(torch.zeros(1))], cfg)
    check("元组形态: kwargs 显式给定则固定",
          o2.param_groups[0]["lr"] == 0.01 and o2.param_groups[0].get("momentum") == 0.9)

    class StrOpt(MiniSpec):
        name, model, loss, datasets = "s", nn.Linear, nn.MSELoss, lambda c: (None, None)
        optimizer = "adam"

    o3 = StrOpt().build_optimizer([nn.Parameter(torch.zeros(1))], cfg)
    check("字符串形态: adam 解析正确", isinstance(o3, torch.optim.Adam) and o3.param_groups[0]["lr"] == 0.5)

    class BadOpt(MiniSpec):
        name, model, loss, datasets = "b", nn.Linear, nn.MSELoss, lambda c: (None, None)
        optimizer = "nope"

    try:
        BadOpt().build_optimizer([nn.Parameter(torch.zeros(1))], cfg)
    except ValueError as e:
        check("未知优化器别名报错带提示", "sgd" in str(e), str(e))
    else:
        check("未知优化器别名报错带提示", False)

    # 4. 损失两形态：类 -> 自动实例化；函数/实例 -> 原样使用
    class ClassLoss(MiniSpec):
        name, model, datasets = "cl", nn.Linear, lambda c: (None, None)
        loss = nn.MSELoss

    check("损失为类: 自动实例化", isinstance(ClassLoss().build_loss(), nn.MSELoss))
    check("损失为函数: 原样使用", FakeAlgo().build_loss() is F.cross_entropy)

    class InstLoss(MiniSpec):
        name, model, datasets = "i", nn.Linear, lambda c: (None, None)
        loss = nn.CrossEntropyLoss()

    check("损失为实例: 原样使用", InstLoss().build_loss() is InstLoss.loss)

    # 5. DataLoader 装配（工程细节由适配器代劳）
    tr, va = FakeAlgo().build_dataloaders(AttrDict({"batch_size": 256, "lr": 0.1, "num_workers": 0, "seed": 42}))
    batch = next(iter(tr))
    check("datasets 声明 -> DataLoader 装配", len(batch) == 2 and batch[0].shape[0] == 256)
    check("val_loader 不 shuffle", va.sampler.__class__.__name__ == "SequentialSampler")

    # 6. 指标聚合与 batch 解包
    m = FakeAlgo().compute_metrics(torch.tensor([[2.0, 0.0, 0.0]]), torch.tensor([0]))
    check("metrics 声明 -> compute_metrics", list(m) == ["acc"] and m["acc"] == 1.0, str(m))
    X, y = FakeAlgo().unpack_batch((torch.zeros(2), torch.ones(2)))
    check("unpack 默认恒等解包", X.shape == (2,) and y.shape == (2,))

    class UnpackAlgo(MiniSpec):
        name, model, loss, datasets = "u", nn.Linear, nn.MSELoss, lambda c: (None, None)
        unpack = lambda b: (b["X"], b["y"])

    Xu, yu = UnpackAlgo().unpack_batch({"X": 1, "y": 2})
    check("unpack 自定义生效", (Xu, yu) == (1, 2))

    # 7. scheduler / callbacks 可选扩展
    check("scheduler 未声明 -> None", FakeAlgo().build_scheduler(opt, cfg) is None)
    check("get_callbacks 默认空", FakeAlgo().get_callbacks(cfg) == [])

    class CbAlgo(MiniSpec):
        name, model, loss, datasets = "c", nn.Linear, nn.MSELoss, lambda c: (None, None)
        callbacks = ["cb1"]

    check("callbacks 声明透传", CbAlgo().get_callbacks(cfg) == ["cb1"])

    # 8. build_model 默认零参实例化
    check("model 声明 -> 零参实例化", isinstance(FakeAlgo().build_model(cfg), nn.Module))

    print("批5 冒烟测试全部通过 —— MiniSpec 声明式接入与完整契约等价")


if __name__ == "__main__":
    main()
