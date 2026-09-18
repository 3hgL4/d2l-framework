"""评估执行器：验证集上的 loss + 指标聚合。

指标计算由算法层经 compute_metrics 注入，本模块只做聚合。
聚合方式：按 batch 内样本数加权平均（compute_metrics 返回 batch 均值时，
对最后一个不满 batch 有可忽略的舍入差；如需精确计数请返回计数并在算法层换算）。

性能注记：inference_mode 替代 no_grad（验证输出从不进 autograd，少版本计数
开销）；autocast 上下文提到循环外（每批进出无意义）；pinned 批次走
non_blocking H2D 拷贝。聚合语义与回调契约零变化。
"""
from __future__ import annotations

import torch


@torch.inference_mode()
def evaluate(model, loader, loss_fn, unpack_batch, compute_metrics,
             device, amp: bool = False, prefix: str = "val_") -> dict:
    model.eval()
    n, loss_sum = 0, 0.0
    metric_sums: dict = {}
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        for batch in loader:
            X, y = unpack_batch(batch)
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            bs = y.shape[0] if hasattr(y, "shape") else len(y)
            loss_sum += float(loss) * bs
            n += bs
            for k, v in (compute_metrics(y_hat.detach().float(), y) or {}).items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v) * bs
    out = {prefix + "loss": loss_sum / max(n, 1)}
    out.update({prefix + k: s / max(n, 1) for k, s in metric_sums.items()})
    return out
