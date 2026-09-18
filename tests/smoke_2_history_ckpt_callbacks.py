"""冒烟测试 批2：训练历史 / 断点全状态 / 回调链（早停、最佳追踪、存档）。

运行: python tests/smoke_2_history_ckpt_callbacks.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from infra.callbacks import (BestTracker, Checkpointer, EarlyStopping,
                             HistoryLogger)
from infra.checkpoint import capture_rng, load, restore_rng, save
from infra.history import History
from infra.logger import get_logger
from infra.seed import set_seed


def check(name, cond):
    print(("PASS" if cond else "FAIL") + f"  {name}")
    if not cond:
        sys.exit(1)


class Ctx:
    """最小上下文替身（trainer 尚未登场的批2 使用）。"""

    def __init__(self, **kw):
        self.stop_training = False
        self.is_best = False
        self.val_metrics = {}
        self.train_metrics = {}
        self.train_loss = 0.0
        self.train_time_s = 0.1
        self.epoch = 0
        self.__dict__.update(kw)


def main():
    tmp = Path(tempfile.mkdtemp())
    logger = get_logger(None)

    # 1. History：字段可增长 + resume 续读
    hp = tmp / "history.csv"
    h = History(hp)
    h.log(1, train_loss=0.9, val_loss=0.8)
    h.log(2, train_loss=0.7, val_acc=0.6)  # 新字段 val_acc
    check("History 字段增长后整表完整", len(list(hp.read_text(encoding="utf-8").splitlines())) == 3)
    h2 = History(hp, resume=True)
    check("History resume 读回 2 行", len(h2.rows) == 2)
    h2.log(3, val_loss=0.7)
    check("History resume 后续写", len(History(hp, resume=True).rows) == 3)

    # 2. RNG 捕获/恢复
    set_seed(0)
    a1 = torch.rand(3)
    st = capture_rng()
    b1 = torch.rand(3)
    restore_rng(st)
    b2 = torch.rand(3)
    check("RNG 快照恢复后随机流等价", torch.equal(b1, b2))
    check("RNG 快照不影响已生成的 a1", a1.shape == (3,))

    # 3. checkpoint 原子保存/回读
    m = nn.Linear(4, 2)
    payload = {"epoch": 3, "model": m.state_dict(),
               "optimizer": {"lr": 0.1}, "extras": {"best.val_loss": 0.8, "es.val_loss.count": 1},
               "rng": capture_rng()}
    p = tmp / "ckpt" / "last.pt"
    save(p, payload)
    back = load(p)
    check("checkpoint epoch/extras 回读", back["epoch"] == 3 and back["extras"]["es.val_loss.count"] == 1)
    check("checkpoint 权重张量一致", torch.equal(back["model"]["weight"], m.state_dict()["weight"]))

    # 4. 回调链：BestTracker -> EarlyStopping(patience=2) -> HistoryLogger -> Checkpointer
    extras = {}
    ctx = Ctx(run_dir=tmp, logger=logger, history=History(tmp / "cb_history.csv"),
              extras=extras, optimizer=None,
              state_fn=lambda e: {"epoch": e, "model": m.state_dict(), "extras": extras})
    cbs = [BestTracker("val_loss", "min"), EarlyStopping("val_loss", "min", patience=2),
           HistoryLogger(), Checkpointer()]
    for cb in cbs:
        cb.on_train_start(ctx)
    series = [0.9, 0.8, 0.85, 0.86, 0.87]  # 0.8 为最优；此后连续 3 轮无改善
    stopped_at = None
    for i, v in enumerate(series):
        ctx.epoch, ctx.val_loss = i + 1, v
        for cb in cbs:
            cb.on_epoch_end(ctx)
        if ctx.stop_training and stopped_at is None:
            stopped_at = ctx.epoch
            break  # 与 trainer 语义一致：stop_training 后立即跳出 epoch 循环
    check("早停在第 4 轮触发(p=2)", stopped_at == 4)
    check("最佳值追踪正确", extras["best.val_loss"] == 0.8)
    check("早停计数持久化", extras["es.val_loss.count"] == 2)
    check("history 记录到停止轮", len(History(tmp / "cb_history.csv", resume=True).rows) == 4)
    check("last.pt 已存", (tmp / "ckpt" / "last.pt").exists())
    check("best.pt 已存(is_best 轮)", (tmp / "ckpt" / "best.pt").exists())

    # 5. 续训续接：用 checkpoint 里的 extras 重建回调，状态无缝衔接
    back = load(tmp / "ckpt" / "last.pt")
    extras2 = dict(back["extras"])
    ctx2 = Ctx(run_dir=tmp, logger=logger, history=History(tmp / "cb2.csv"),
              extras=extras2, optimizer=None, state_fn=lambda e: {})
    cbs2 = [BestTracker("val_loss", "min"), EarlyStopping("val_loss", "min", patience=2)]
    for cb in cbs2:
        cb.on_train_start(ctx2)
    ctx2.epoch, ctx2.val_loss = 5, 0.88   # 继续恶化 -> count=3? 先看是否继承 count
    check("续训恢复早停计数", extras2["es.val_loss.count"] == 2)
    for cb in cbs2:
        cb.on_epoch_end(ctx2)
    check("恢复后第 1 轮即再触发早停", ctx2.stop_training is True)
    ctx3 = Ctx(run_dir=tmp, logger=logger, history=History(tmp / "cb3.csv"),
              extras=extras2, optimizer=None, state_fn=lambda e: {})
    for cb in cbs2:
        cb.on_train_start(ctx3)
    ctx3.epoch, ctx3.val_loss = 5, 0.79   # 新低 -> best 更新、计数清零
    for cb in cbs2:
        cb.on_epoch_end(ctx3)
    check("恢复后改善会刷新 best 并清零计数",
          extras2["best.val_loss"] == 0.79 and extras2["es.val_loss.count"] == 0)

    print("批2 冒烟测试全部通过")


if __name__ == "__main__":
    main()
