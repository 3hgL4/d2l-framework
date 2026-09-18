"""统一训练入口。用法：

  python run.py --algo softmax
  python run.py --algo fake --override epochs=20 patience=3
  python run.py --algo softmax --resume runs/softmax/20260918-101500/ckpt/last.pt --override epochs=60
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.config import (build_config, new_run_dir,
                          parse_overrides, save_config, snapshot_env)
from infra.logger import get_logger
from infra.trainer import Trainer


def load_spec(algo: str):
    try:
        mod = importlib.import_module(f"algorithms.{algo}.algo")
    except ModuleNotFoundError as e:
        raise SystemExit(f"找不到算法 algorithms/{algo}/algo.py ({e})") from e
    spec = getattr(mod, "SPEC", None)
    if spec is None:
        raise SystemExit(f"algorithms/{algo}/algo.py 未暴露模块级 SPEC 实例")
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(description="d2l 通用训练入口")
    ap.add_argument("--algo", required=True, help="algorithms/ 下的算法目录名")
    ap.add_argument("--resume", default=None, help="last.pt 路径，从断点续训")
    ap.add_argument("--override", nargs="*", default=[],
                    help="配置覆盖 key=value，嵌套用点号（如 viz.live=False）")
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    args = ap.parse_args(argv)

    spec = load_spec(args.algo)
    cfg = build_config(spec.default_config(), parse_overrides(args.override))

    if args.resume:
        run_dir = Path(args.resume).resolve().parent.parent  # ckpt/last.pt -> run 目录
        logger = get_logger(run_dir / "train.log")
    else:
        run_dir = new_run_dir(args.runs_root, args.algo)
        logger = get_logger(run_dir / "train.log")

    snapshot_env(run_dir / "env.txt")
    save_config(cfg, run_dir / "config.yaml")
    logger.info(f"[配置] epochs={cfg.epochs} seed={cfg.seed} device={cfg.device} "
                f"amp={cfg.amp} workers={cfg.num_workers} patience={cfg.patience}")

    Trainer(spec, cfg, run_dir, logger=logger).fit(resume=args.resume)


if __name__ == "__main__":
    main()
