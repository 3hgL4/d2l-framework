"""训练历史：逐 epoch 指标行，CSV 落盘。

整表重写策略：每轮全量重写 CSV。轮数量级（几十~几百）下开销可忽略，
换来的是"字段可增长 + 中断后已存数据始终完整可读"，比增量追加更稳。
resume=True 时读回已有行继续追加（断点续训的历史不断档）。
"""
from __future__ import annotations

import csv
from pathlib import Path


class History:
    def __init__(self, path, resume: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        if resume and self.path.exists():
            with open(self.path, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    self.rows.append(dict(r))
        # resume=False 时不清文件：首次 _flush 全量重写即覆盖，构造保持无副作用

    def log(self, epoch: int, **vals) -> None:
        self.rows.append({"epoch": epoch, **vals})
        self._flush()

    def last(self) -> dict | None:
        return self.rows[-1] if self.rows else None

    def _flush(self) -> None:
        fields: list[str] = []
        for r in self.rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)
