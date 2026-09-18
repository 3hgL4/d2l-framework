"""生命周期钩子系统与内置回调。

trainer 在固定事件点按列表顺序调用回调：
    on_train_start -> (on_epoch_start -> on_batch_end* -> on_eval_end ->
    on_epoch_end)* -> on_train_end

约定：
- ctx.extras 是跨回调、跨续训共享的状态口袋，trainer 原样存入 checkpoint；
- EarlyStopping 依赖 ctx.is_best（由 BestTracker 判定），
  因此 BestTracker 必须排在 EarlyStopping 之前（默认回调集已保证）。
"""
from __future__ import annotations

from . import checkpoint as ckpt_mod
from .history import History


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------
class TrainContext:
    """训练循环传给回调的上下文。"""

    def __init__(self, *, model, optimizer, scheduler, scaler, cfg, run_dir,
                 logger, history: History, extras: dict, spec_name: str,
                 state_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.cfg = cfg
        self.run_dir = run_dir
        self.logger = logger
        self.history = history
        self.extras = extras          # 跨回调/跨续训共享状态
        self.spec_name = spec_name
        self.state_fn = state_fn      # state_fn(epoch) -> checkpoint payload
        # 每 epoch 由 trainer 刷新
        self.epoch = 0
        self.epochs = 0
        self.train_loss = None
        self.train_metrics: dict = {}
        self.val_loss = None
        self.val_metrics: dict = {}
        self.train_time_s = 0.0
        # 回调产物
        self.is_best = False
        self.stop_training = False


class Callback:
    """钩子基类：按需覆写感兴趣的事件即可。"""

    def on_train_start(self, ctx): ...
    def on_epoch_start(self, ctx): ...
    def on_batch_end(self, ctx, batch_idx: int, loss: float): ...
    def on_eval_end(self, ctx): ...
    def on_epoch_end(self, ctx): ...
    def on_train_end(self, ctx): ...


# ---------------------------------------------------------------------------
# 内置回调
# ---------------------------------------------------------------------------
def _get_metric(ctx, monitor: str):
    v = getattr(ctx, monitor, None)
    if v is None and ctx.val_metrics:
        v = ctx.val_metrics.get(monitor)
    return v


def _better(a, b, mode: str, min_delta: float) -> bool:
    return (a < b - min_delta) if mode == "min" else (a > b + min_delta)


class BestTracker(Callback):
    """追踪 monitor 最优值，产出 ctx.is_best。必须排在其他回调之前。"""

    def __init__(self, monitor: str = "val_loss", mode: str = "min",
                 min_delta: float = 0.0):
        self.monitor, self.mode, self.min_delta = monitor, mode, min_delta
        self.key = f"best.{monitor}"

    def on_train_start(self, ctx):
        ctx.extras.setdefault(self.key, None)

    def on_epoch_end(self, ctx):
        v = _get_metric(ctx, self.monitor)
        best = ctx.extras.get(self.key)
        ctx.is_best = v is not None and (best is None or _better(v, best, self.mode, self.min_delta))
        if ctx.is_best:
            ctx.extras[self.key] = v


class EarlyStopping(Callback):
    """monitor 连续 patience 轮无改善则置 ctx.stop_training（trainer 据此跳出）。"""

    def __init__(self, monitor: str, mode: str, patience: int, min_delta: float = 0.0):
        self.monitor, self.mode, self.patience = monitor, mode, patience
        self.min_delta = min_delta
        self.key = f"es.{monitor}.count"

    def on_train_start(self, ctx):
        ctx.extras.setdefault(self.key, 0)

    def on_epoch_end(self, ctx):
        if ctx.stop_training:
            return
        cnt = 0 if ctx.is_best else ctx.extras.get(self.key, 0) + 1
        ctx.extras[self.key] = cnt
        if cnt >= self.patience:
            ctx.stop_training = True
            ctx.logger.info(f"[早停] {self.monitor} 连续 {self.patience} 轮无改善，第 {ctx.epoch} 轮停止")


def _current_lr(optimizer):
    if hasattr(optimizer, "param_groups"):
        return optimizer.param_groups[0].get("lr")
    return getattr(optimizer, "lr", "")


class HistoryLogger(Callback):
    """把当前 epoch 的指标行写入 history.csv。"""

    def on_epoch_end(self, ctx):
        row = {"lr": _current_lr(ctx.optimizer),
               "train_loss": round(ctx.train_loss, 6),
               "train_time_s": round(ctx.train_time_s, 3)}
        row.update({k: round(v, 6) for k, v in ctx.train_metrics.items()})
        if ctx.val_loss is not None:
            row["val_loss"] = round(ctx.val_loss, 6)
        row.update({k: round(v, 6) for k, v in ctx.val_metrics.items()})
        ctx.history.log(ctx.epoch, **row)


class Checkpointer(Callback):
    """每轮存 last.pt；is_best 时另存 best.pt（依赖 ctx.state_fn）。"""

    def __init__(self, save_last: bool = True, save_best: bool = True):
        self.save_last, self.save_best = save_last, save_best

    def on_epoch_end(self, ctx):
        if ctx.state_fn is None:
            return
        if self.save_last:
            ckpt_mod.save(ctx.run_dir / "ckpt" / "last.pt", ctx.state_fn(ctx.epoch))
        if self.save_best and ctx.is_best:
            ckpt_mod.save(ctx.run_dir / "ckpt" / "best.pt", ctx.state_fn(ctx.epoch))


class CurvePlotter(Callback):
    """训练曲线：epoch 级实时刷新（可选），训练结束存盘。样式经 style 注入。

    style: dict -> rcParams 更新；callable(fig, ax) -> 完全自定义（见 viz.py）。
    """

    def __init__(self, live: bool = True, save: bool = True, style=None):
        self.live, self.save, self.style = live, save, style
        self._plot = None

    def on_train_start(self, ctx):
        if self.live or self.save:
            from .viz import CurvePlot  # 延迟导入：Agg 环境自动退化为仅存盘
            self._plot = CurvePlot(ctx.run_dir / "curves.png",
                                   title=f"{ctx.spec_name} training curves",
                                   live=self.live, style=self.style)

    def on_epoch_end(self, ctx):
        if self._plot is not None:
            self._plot.update(ctx.history.rows)

    def on_train_end(self, ctx):
        if self._plot is None:
            return
        if self.save:
            self._plot.save()
        self._plot.close()
        self._plot = None
