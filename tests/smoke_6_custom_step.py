"""冒烟测试 批6：可选契约 training_step（算法全权接管单批优化）+ 契约版本化。

运行: python tests/smoke_6_custom_step.py
覆盖: training_step 接管默认循环 / ctx.device 搬运 / 手写参数更新（绕开 optimizer）/
      (loss, None, None) 跳过指标聚合 / checkpoint 含 contract_version /
      续训保留自定义步 / 跨版本续训告警
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

import infra
from infra import checkpoint as ckpt_mod
from infra.callbacks import BestTracker, Checkpointer, HistoryLogger
from infra.config import build_config, new_run_dir
from infra.contract import validate_spec
from infra.minispec import MiniSpec
from infra.trainer import Trainer
from tests.fake_algo import FakeDataset, FakeNet


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        sys.exit(1)


def _acc(y_hat, y):
    return (y_hat.argmax(1) == y).float().mean().item()


def _load_data(cfg):
    return FakeDataset(seed=0), FakeDataset(n=512, seed=1)


def _callbacks():
    return [BestTracker("val_loss", "min", 0.0), HistoryLogger(), Checkpointer()]


class ManualStepAlgo(MiniSpec):
    """training_step 演示：手写 SGD 绕开 optimizer；每 3 批跳过指标聚合。

    声明了 loss 但自定义步不用它 —— 证明训练循环被完全接管，
    infra 只负责聚合 loss/指标与回调。
    """

    name = "custom_step"
    model = FakeNet
    loss = F.cross_entropy
    datasets = _load_data
    metrics = {"acc": _acc}
    config = {"batch_size": 256, "lr": 0.5}

    calls = 0      # training_step 总调用数（跨实例累计，验证续训仍走自定义步）
    with_yhat = 0  # 返回了 y_hat/y 的批数（非跳过批）

    def training_step(self, batch, ctx):
        ManualStepAlgo.calls += 1
        X, y = batch
        X, y = X.to(ctx.device), y.to(ctx.device)   # 契约承诺：ctx.device 可用
        y_hat = ctx.model(X)                        # 训练态经 ctx 拿回（model/optimizer/scaler）
        loss = F.cross_entropy(y_hat, y)
        ctx.model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():                       # 手写 SGD，完全绕开 optimizer
            for p in ctx.model.parameters():
                p -= ctx.cfg.lr * p.grad
        if ManualStepAlgo.calls % 3 == 0:           # 每 3 批：回归跳指标分支
            return loss, None, None
        ManualStepAlgo.with_yhat += 1
        return loss, y_hat.detach(), y


def _history_rows(run_dir: Path):
    import csv
    with open(run_dir / "history.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    tmp = Path(tempfile.mkdtemp())
    spec = ManualStepAlgo()
    validate_spec(spec)
    check("含 training_step 仍通过 validate_spec", True)

    batches_per_epoch = 2048 // 256   # FakeDataset 2048 样本 / batch 256 = 8

    # 1. 完整训练 1 轮：自定义步接管、ctx.device 搬运、跳指标分支
    rd = new_run_dir(tmp, "custom_step")
    cfg1 = build_config(spec.default_config(), {"epochs": 1, "amp": False})
    Trainer(spec, cfg1, rd, callbacks=_callbacks()).fit()
    check("training_step 被调用（每批一次）", ManualStepAlgo.calls == batches_per_epoch,
          str(ManualStepAlgo.calls))
    n_skip = sum(1 for i in range(1, ManualStepAlgo.calls + 1) if i % 3 == 0)
    check("(loss, None, None) 批数正确", ManualStepAlgo.with_yhat == ManualStepAlgo.calls - n_skip,
          f"{ManualStepAlgo.with_yhat} vs {ManualStepAlgo.calls - n_skip}")
    rows = _history_rows(rd)
    check("history 有 train_loss/val_loss/acc 列",
          "train_loss" in rows[0] and "val_loss" in rows[0] and "acc" in rows[0],
          str(rows[0].keys()))

    # 2. checkpoint 落盘 contract_version
    payload = ckpt_mod.load(str(rd / "ckpt" / "last.pt"))
    check("checkpoint 含 contract_version", payload.get("contract_version") == infra.__version__,
          str(payload.get("contract_version")))

    # 3. 续训：自定义步保留、历史衔接、手写更新确实在优化
    calls_before = ManualStepAlgo.calls
    cfg2 = build_config(spec.default_config(),
                        {"epochs": 4, "amp": False, "viz": {"live": False, "save": False}})
    Trainer(ManualStepAlgo(), cfg2, rd, callbacks=_callbacks()).fit(resume=str(rd / "ckpt" / "last.pt"))
    rows2 = _history_rows(rd)
    check("续训历史衔接 1..4", [int(r["epoch"]) for r in rows2] == [1, 2, 3, 4],
          str([r["epoch"] for r in rows2]))
    check("续训仍走自定义步", ManualStepAlgo.calls - calls_before == batches_per_epoch * 3,
          str(ManualStepAlgo.calls - calls_before))
    check("手写更新使 train_loss 下降", float(rows2[0]["train_loss"]) > float(rows2[-1]["train_loss"]),
          f"{rows2[0]['train_loss']} -> {rows2[-1]['train_loss']}")
    # 注意：FakeDataset 的 W 随 seed 变化，train/val 是不同线性映射，val 本就不可学
    # （批3 早停测试正依赖此特性）；手写更新是否真的在优化，以 train_acc 为准。
    check("手写更新使 train_acc 上升 (>=0.90)", float(rows2[-1]["acc"]) >= 0.90,
          str(rows2[-1].get("acc")))

    # 4. 跨版本续训告警
    old = dict(payload)
    old["contract_version"] = "0.0.1"
    old_pt = tmp / "old_version.pt"
    ckpt_mod.save(old_pt, old)
    rd3 = new_run_dir(tmp, "custom_step")
    Trainer(ManualStepAlgo(), cfg2, rd3).fit(resume=str(old_pt))
    log = (rd3 / "train.log").read_text(encoding="utf-8")
    check("跨版本续训触发告警", "契约版本" in log and "0.0.1" in log, log[-400:])

    print("批6 冒烟测试全部通过 —— training_step 接管 + 契约版本化生效")


if __name__ == "__main__":
    main()
