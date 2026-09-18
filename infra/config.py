"""配置合成与运行档案。

优先级（低 -> 高）两层，刻意不设文件覆盖层：
    infra 默认 <- 算法 default_config() <- 命令行 --override
默认值只活在 default_config()（单一事实源）；临时调参走 --override。

所有文件 IO 强制 utf-8（Windows 控制台默认 GBK，编码兜底见 logger.py）。
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# infra 级默认配置（算法层只写与算法相关的键）
# ---------------------------------------------------------------------------
INFRA_DEFAULTS: dict = {
    # 复现
    "seed": 42,
    "deterministic": False,   # True = cudnn 确定性（复现优先）
    # 设备与性能
    "device": "auto",         # auto / cuda / cpu
    "amp": True,              # 混合精度（仅 cuda 生效；手写优化器需关闭）
    "num_workers": 0,         # Windows spawn 模式下 0 最稳；>0 需 dataset 可 pickle
    # 训练
    "epochs": 10,
    "grad_clip": 0.0,         # >0 启用梯度范数裁剪
    "log_every": 1,           # 每 N 轮打印一行摘要
    # 停止策略
    "monitor": "val_loss",    # 监控指标名
    "mode": "min",            # min=越小越好 / max=越大越好
    "patience": 0,            # 早停耐心轮数；0 = 关闭早停
    "min_delta": 0.0,
    # 记录
    "verbose": False,         # True 时记录 batch 级细节（仅进文件）
    # 可视化
    "viz": {"live": True, "save": True},
    # 断点
    "ckpt": {"save_last": True, "save_best": True},
}


class AttrDict(dict):
    """dict 的属性访问封装：cfg.epochs 等价 cfg["epochs"]（嵌套递归生效）。"""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return AttrDict(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 的嵌套 dict 只覆盖对应子键，不丢同级键。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _coerce(raw: str):
    s = raw.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_overrides(items) -> dict:
    """把 ["lr=0.1", "viz.live=False"] 解析成嵌套 dict（点号表嵌套）。"""
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"覆盖项需形如 key=value，收到: {it}")
        k, _, raw = it.partition("=")
        node = out
        parts = k.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw)
    return out


def build_config(algo_defaults: dict, cli: dict | None) -> AttrDict:
    cfg = deep_merge(INFRA_DEFAULTS, algo_defaults or {})
    cfg = deep_merge(cfg, cli or {})
    return AttrDict(cfg)


def _plain(d):
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items()}
    return d


def save_config(cfg: dict, path) -> None:
    """最终生效配置存档（可复现三件套之一）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(_plain(cfg), f, allow_unicode=True, sort_keys=False)


def new_run_dir(runs_root, algo: str) -> Path:
    """runs/<algo>/<YYYYMMDD-HHMMSS>/，内含 ckpt/ 子目录；同秒冲突自动加后缀。"""
    root = Path(runs_root) / algo
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = root / stamp
    i = 1
    while d.exists():
        d = root / f"{stamp}-{i}"
        i += 1
    d.mkdir(parents=True)
    (d / "ckpt").mkdir()
    return d


def snapshot_env(path, extra: dict | None = None) -> None:
    """环境快照 -> env.txt（可复现三件套之一）。"""
    lines = [f"time = {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    lines.append(f"python = {sys.version.split()[0]} ({sys.executable})")
    lines.append(f"os = {platform.platform()}")
    try:
        import torch
        lines.append(f"torch = {torch.__version__}")
        lines.append(f"cuda_available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu = {torch.cuda.get_device_name(0)}")
            lines.append(f"cudnn = {torch.backends.cudnn.version()}")
    except ImportError:
        lines.append("torch = 未安装")
    for mod in ("numpy", "torchvision", "matplotlib"):
        try:
            m = __import__(mod)
            lines.append(f"{mod} = {getattr(m, '__version__', '?')}")
        except ImportError:
            pass
    for k, v in (extra or {}).items():
        lines.append(f"{k} = {v}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
