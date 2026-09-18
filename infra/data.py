"""DataLoader 工厂（Windows 显式考虑）。

Windows 下 DataLoader 多进程为 spawn 模式，num_workers > 0 必须满足：
1. 入口脚本有 `if __name__ == "__main__":` 保护（run.py / 各测试均已保证）；
2. dataset 与 collate 可被 pickle（dataset 不得含 lambda / 局部类）；
3. 开启 persistent_workers，避免每个 epoch 重启子进程。
shuffle 用固定种子的 Generator，保证可复现。
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0,
                drop_last: bool = False, seed: int | None = None) -> DataLoader:
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, generator=g,
    )


def make_dataloaders(train_ds, val_ds, cfg) -> tuple[DataLoader, DataLoader]:
    """按全局配置批量构造 (train_loader, val_loader)。"""
    return (make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, seed=cfg.seed),
            make_loader(val_ds, cfg.batch_size, False, cfg.num_workers))
