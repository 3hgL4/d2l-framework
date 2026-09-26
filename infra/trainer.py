"""通用训练循环：只依赖 AlgoSpec 契约与 infra 各模块，不认识任何具体算法。

串联能力：日志(2) / 历史(2) / 断点续训(3) / 早停与最佳追踪(4) / 数据加载(5) /
曲线可视化(7)。默认回调集按 cfg 自动装配；传 callbacks 参数可完全接管。
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from . import checkpoint as ckpt_mod
from . import __version__ as infra_version
from .callbacks import (BestTracker, Checkpointer, CurvePlotter,
                        EarlyStopping, HistoryLogger, TrainContext)
from .contract import validate_spec
from .evaluator import evaluate
from .history import History
from .logger import get_logger
from .seed import set_seed


def pick_device(which: str = "auto") -> torch.device:
    if which == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(which)


def _load_optimizer_state(opt, state, logger=None) -> bool:
    """优先标准 state_dict；无状态接口的极简优化器（如手写 SGD）退化为恢复 lr。

    返回是否发生退化。退化路径必须显式告警并写入 extras 元数据：
    若优化器带可变状态（如动量），仅恢复 lr 会使续训不等价——宁可吵，不可静默错。
    """
    if state is None:
        return False
    if hasattr(opt, "load_state_dict"):
        opt.load_state_dict(state)
        return False
    if isinstance(state, dict) and "lr" in state and hasattr(opt, "lr"):
        opt.lr = state["lr"]
    if logger is not None:
        logger.warning(f"[续训] 优化器 {type(opt).__name__} 无 state_dict 接口，"
                       f"仅恢复 lr={getattr(opt, 'lr', '?')}；若其含可变状态，"
                       f"续训与完整训练可能不等价")
    return True


class Trainer:
    def __init__(self, spec, cfg, run_dir, callbacks=None, logger=None):
        validate_spec(spec)
        self.spec, self.cfg = spec, cfg
        self.run_dir = Path(run_dir)
        self.logger = logger or get_logger(self.run_dir / "train.log")
        self.callbacks = callbacks  # None -> fit 时按 cfg 自动装配

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def _default_callbacks(self):
        cfg = self.cfg
        cbs = [
            BestTracker(cfg.monitor, cfg.mode, cfg.min_delta),
            HistoryLogger(),
        ]
        if cfg.patience > 0:
            cbs.append(EarlyStopping(cfg.monitor, cfg.mode, cfg.patience, cfg.min_delta))
        cbs.append(CurvePlotter(live=cfg.viz.live, save=cfg.viz.save,
                                wait_close=cfg.viz.get("wait_close", False)))
        cbs.append(Checkpointer(cfg.ckpt.save_last, cfg.ckpt.save_best))
        return cbs

    def _spec_callbacks(self) -> list:
        """算法层可选扩展点：spec.get_callbacks(cfg) -> list[Callback]。

        infra 不 import 任何算法，只调用该可选方法；返回的回调追加在
        默认回调之后（on_train_end 晚于曲线存盘等收尾动作）。
        """
        fn = getattr(self.spec, "get_callbacks", None)
        if not callable(fn):
            return []
        return list(fn(self.cfg) or [])

    def _state_dict(self, epoch: int) -> dict:
        d = {
            "spec": self.spec.name,
            "contract_version": infra_version,
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": (self.optimizer.state_dict()
                          if hasattr(self.optimizer, "state_dict")
                          else {"lr": getattr(self.optimizer, "lr", None)}),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "rng": ckpt_mod.capture_rng(),
            "extras": self.extras,
        }
        if self.scheduler is not None:
            d["scheduler"] = self.scheduler.state_dict()
        return d

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def fit(self, resume=None):
        cfg, spec = self.cfg, self.spec
        device = pick_device(cfg.device)
        set_seed(cfg.seed, cfg.deterministic)
        self.device = device

        self.model = spec.build_model(cfg).to(device)
        self.loss_fn = spec.build_loss()
        self.optimizer = spec.build_optimizer(self.model.parameters(), cfg)
        b = getattr(spec, "build_scheduler", None)
        self.scheduler = b(self.optimizer, cfg) if callable(b) else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
        train_dl, val_dl = spec.build_dataloaders(cfg)

        # 数据与优化器落盘（INFO 双写 train.log）：报告/审计可溯源，不必翻代码
        def _n(ds):
            try:
                return len(ds.dataset)
            except (TypeError, AttributeError):
                return "?"
        opt_lr = getattr(self.optimizer, "lr", None)
        if opt_lr is None and getattr(self.optimizer, "param_groups", None):
            opt_lr = self.optimizer.param_groups[0].get("lr")
        self.logger.info(f"[优化器] {type(self.optimizer).__name__}"
                         + (f" | lr={opt_lr}" if opt_lr is not None else ""))
        self.logger.info(f"[数据] train={_n(train_dl)} val={_n(val_dl) if val_dl else 0} "
                         f"| batch_size={getattr(train_dl, 'batch_size', '?')} "
                         f"| batches/epoch={len(train_dl)}")

        resume_epoch, extras = 0, {}
        if resume:
            payload = ckpt_mod.load(resume, map_location=str(device))
            self.model.load_state_dict(payload["model"])
            if _load_optimizer_state(self.optimizer, payload.get("optimizer"), self.logger):
                extras["resumed_degraded_optimizer"] = True  # 随后续 checkpoint 落盘，可审计
            if payload.get("scheduler") is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(payload["scheduler"])
            if payload.get("scaler") is not None and self.scaler.is_enabled():
                self.scaler.load_state_dict(payload["scaler"])
            resume_epoch = payload["epoch"]
            extras = dict(payload.get("extras") or {})
            ckpt_mod.restore_rng(payload.get("rng"))
            cv = payload.get("contract_version")
            if cv is not None and cv != infra_version:
                self.logger.warning(f"[续训] checkpoint 契约版本 {cv} ≠ 当前 "
                                    f"{infra_version}，跨版本兼容未验证")
            self.logger.info(f"[续训] 从 epoch {resume_epoch} 恢复: {resume}")
        self.extras = extras

        self.history = History(self.run_dir / "history.csv", resume=bool(resume))
        cbs = (self.callbacks if self.callbacks is not None
               else self._default_callbacks() + self._spec_callbacks())
        ctx = TrainContext(
            model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            scaler=self.scaler, cfg=cfg, run_dir=self.run_dir, logger=self.logger,
            history=self.history, extras=extras, spec_name=spec.name,
            state_fn=self._state_dict, device=device,
        )
        ctx.epochs = cfg.epochs

        self.logger.info(f"[运行] {spec.name} | device={device} | epochs={cfg.epochs} "
                         f"| monitor={cfg.monitor}({cfg.mode}) | run_dir={self.run_dir}")
        self.logger.debug(f"[配置] {dict(cfg)}")
        for cb in cbs:
            cb.on_train_start(ctx)

        amp_on = cfg.amp and device.type == "cuda"
        for epoch in range(resume_epoch + 1, cfg.epochs + 1):
            ctx.epoch = epoch
            for cb in cbs:
                cb.on_epoch_start(ctx)

            t0 = time.perf_counter()
            ctx.train_loss, ctx.train_metrics = self._train_one_epoch(train_dl, ctx, cbs)
            val_out = (evaluate(self.model, val_dl, self.loss_fn, spec.unpack_batch,
                                spec.compute_metrics, device, amp=amp_on)
                       if val_dl is not None else {})
            ctx.val_loss = val_out.get("val_loss")
            ctx.val_metrics = {k: v for k, v in val_out.items() if k != "val_loss"}
            ctx.train_time_s = time.perf_counter() - t0

            for cb in cbs:
                cb.on_eval_end(ctx)
            for cb in cbs:
                cb.on_epoch_end(ctx)
            if self.scheduler is not None:
                self.scheduler.step()
            self._log_epoch(ctx, val_out)

            if ctx.stop_training:
                break

        for cb in cbs:
            cb.on_train_end(ctx)
        best = extras.get(f"best.{cfg.monitor}")
        self.logger.info(f"[完成] 共 {ctx.epoch} 轮 | best {cfg.monitor}={best} | 产物: {self.run_dir}")
        return ctx

    # ------------------------------------------------------------------
    # 单轮训练
    # ------------------------------------------------------------------
    def _train_one_epoch(self, loader, ctx, cbs):
        cfg, spec = self.cfg, self.spec
        self.model.train()
        n, loss_sum = 0, 0.0
        metric_sums: dict = {}
        custom_step = getattr(spec, "training_step", None)
        amp_on = cfg.amp and self.device.type == "cuda"
        for i, batch in enumerate(loader):
            if custom_step is not None:
                # 可选契约：算法全权接管单批优化（含 AMP/裁剪决策），infra 只聚合
                loss, y_hat, y = custom_step(batch, ctx)
                loss = loss if torch.is_tensor(loss) else torch.as_tensor(float(loss))
                loss = loss.detach()
                bs = int(y.shape[0]) if y is not None and hasattr(y, "shape") else 1
            else:
                X, y = spec.unpack_batch(batch)
                # non_blocking：pin_memory 批次走异步 H2D（非 pinned 时等价同步，无害）
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=amp_on):
                    y_hat = self.model(X)
                    loss = self.loss_fn(y_hat, y)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                bs = y.shape[0] if hasattr(y, "shape") else len(y)
            loss_val = float(loss)  # 每批仅此一次 D2H 同步，下文复用（重复 .item() 即重复同步）
            loss_sum += loss_val * bs
            n += bs
            if y_hat is not None and y is not None:
                for k, v in (spec.compute_metrics(y_hat.detach().float(), y) or {}).items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v) * bs
            if cfg.verbose:
                ctx.logger.debug(f"  epoch {ctx.epoch} batch {i} loss {loss_val:.4f}")
            for cb in cbs:
                cb.on_batch_end(ctx, i, loss_val)
        return loss_sum / max(n, 1), {k: s / max(n, 1) for k, s in metric_sums.items()}

    def _log_epoch(self, ctx, val_out):
        if (ctx.epoch - 1) % max(int(self.cfg.log_every), 1) != 0 and ctx.epoch != self.cfg.epochs:
            return
        parts = [f"epoch {ctx.epoch}/{ctx.epochs}", f"loss {ctx.train_loss:.4f}"]
        parts += [f"{k} {v:.4f}" for k, v in ctx.train_metrics.items()]
        parts += [f"{k} {v:.4f}" for k, v in val_out.items()]
        parts.append(f"{ctx.train_time_s:.1f}s")
        if ctx.is_best:
            parts.append("*best")
        self.logger.info(" | ".join(parts))
