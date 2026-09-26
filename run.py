"""统一训练入口。用法：

  python run.py --algo softmax
  python run.py --algo fake --override epochs=20 patience=3
  python run.py --algo softmax --resume runs/softmax/20260918-101500/ckpt/last.pt --override epochs=60
  python run.py --algo mlp --sweep lr=0.01,0.1,0.3 --seeds 3
  python run.py --list
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.config import (_coerce, build_config, new_run_dir,
                          parse_overrides, save_config, snapshot_env)
from infra.logger import get_logger
from infra.trainer import Trainer


def list_algos() -> list:
    """可发现性：扫描 algorithms/ 下的目录约定（含 algo.py 即算法，下划线开头跳过）。"""
    base = ROOT / "algorithms"
    return sorted(p.parent.name for p in base.glob("*/algo.py")
                  if not p.parent.name.startswith("_"))


def load_spec(algo: str):
    try:
        mod = importlib.import_module(f"algorithms.{algo}.algo")
    except ModuleNotFoundError as e:
        raise SystemExit(f"找不到算法 algorithms/{algo}/algo.py ({e})；"
                         f"可用: {', '.join(list_algos() or ['<无>'])}") from e
    spec = getattr(mod, "SPEC", None)
    if spec is None:
        raise SystemExit(f"algorithms/{algo}/algo.py 未暴露模块级 SPEC 实例")
    if getattr(spec, "name", None) != algo:
        import warnings
        warnings.warn(f"spec.name={spec.name!r} 与目录名 {algo!r} 不一致"
                      f"（约定应相同；不一致会导致 checkpoint 元数据/日志与目录对不上）")
    return spec


def parse_sweep(items) -> list:
    """把 ["lr=0.01,0.1", "batch_size=128,256"] 解析成 [(key, [v1, v2]), ...]（保序）。"""
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--sweep 项需形如 key=v1,v2,...，收到: {it}")
        k, _, raw = it.partition("=")
        vals = [_coerce(v) for v in raw.split(",") if v.strip() != ""]
        if not vals:
            raise SystemExit(f"--sweep 项 {it} 缺少取值")
        out.append((k.strip(), vals))
    return out


def run_once(spec, algo: str, overrides: dict, runs_root, resume=None):
    """单次训练：合成配置、建 run 目录、落三件套存档、训练。

    单跑与 sweep 共用本路径；返回 (run_dir, ctx, cfg)。
    """
    cfg = build_config(spec.default_config(), overrides)
    if resume:
        run_dir = Path(resume).resolve().parent.parent  # ckpt/last.pt -> run 目录
        logger = get_logger(run_dir / "train.log")
    else:
        run_dir = new_run_dir(runs_root, algo)
        logger = get_logger(run_dir / "train.log")
    snapshot_env(run_dir / "env.json")
    save_config(cfg, run_dir / "config.yaml")
    logger.info(f"[配置] epochs={cfg.epochs} seed={cfg.seed} device={cfg.device} "
                f"amp={cfg.amp} workers={cfg.num_workers} patience={cfg.patience}")
    ctx = Trainer(spec, cfg, run_dir, logger=logger).fit(resume=resume)
    return run_dir, ctx, cfg


def run_sweep(spec, algo: str, base: dict, sweep_items: list, n_seeds: int,
              runs_root) -> None:
    """批量实验：sweep 组合(笛卡尔积) × seed 逐个训练。

    产物：runs/<algo>/sweeps/<时间戳>.csv 索引（组合、seed、run_dir、最优值、
    实际轮数、耗时），结束后控制台按最优值排名。脚本纪律（见 USAGE §3）：
    扫描模式强制 viz.live=False、viz.wait_close=False，避免 N 个窗口弹出；
    需要看曲线请对感兴趣的 run_dir 单跑或续训。
    """
    import csv
    import itertools
    import time

    keys = [k for k, _ in sweep_items]
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*[vals for _, vals in sweep_items])]
    if not combos:
        combos = [{}]  # 仅 --seeds>1：单组合多 seed
    total = len(combos) * n_seeds
    out = Path(runs_root) / algo / "sweeps" / f"{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, done = [], 0
    for combo in combos:
        merged = {**base, **combo}  # 扫描值优先于 --override
        seed0 = int(build_config(spec.default_config(), merged).seed)
        for i in range(n_seeds):
            seed = seed0 + i
            overrides = {**merged, "seed": seed,
                         "viz": {"live": False, "wait_close": False}}
            done += 1
            label = " ".join([f"{k}={v}" for k, v in combo.items()]
                             + [f"seed={seed}"])
            print(f"[sweep {done}/{total}] {label} 运行中...")
            t0 = time.perf_counter()
            run_dir, ctx, cfg = run_once(spec, algo, overrides, runs_root)
            wall = time.perf_counter() - t0
            best = ctx.extras.get(f"best.{cfg.monitor}")
            rows.append({"run_dir": str(run_dir), "seed": seed,
                         **combo, f"best_{cfg.monitor}": best,
                         "epochs_done": ctx.epoch, "wall_s": round(wall, 1)})
            print(f"[sweep {done}/{total}] {label} -> best {cfg.monitor}={best}"
                  f"（{wall:.1f}s, {ctx.epoch} 轮）")

    fields: list = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    monitor, mode = cfg.monitor, cfg.mode
    key = f"best_{monitor}"
    ranked = sorted(rows, key=lambda r: (float("inf") if mode == "min" else float("-inf"))
                    if r[key] is None else r[key], reverse=(mode == "max"))
    print(f"\n[扫描完成] 共 {total} 次运行 | 索引: {out} | 按 best {monitor} 排名:")
    for i, r in enumerate(ranked, 1):
        cfg_txt = " ".join([f"{k}={r[k]}" for k in keys] + [f"seed={r['seed']}"])
        print(f"  {i}. {key}={r[key]}  {cfg_txt}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="d2l 通用训练入口")
    ap.add_argument("--algo", default=None, help="algorithms/ 下的算法目录名")
    ap.add_argument("--list", action="store_true", dest="list_algos",
                    help="列出可用算法后退出")
    ap.add_argument("--resume", default=None, help="last.pt 路径，从断点续训")
    ap.add_argument("--override", nargs="*", default=[],
                    help="配置覆盖 key=value，嵌套用点号（如 viz.live=False）")
    ap.add_argument("--sweep", nargs="*", default=[],
                    help="参数扫描 key=v1,v2,...（多项做笛卡尔积；扫描值优先于 --override）")
    ap.add_argument("--seeds", type=int, default=None, metavar="N",
                    help="每个组合连跑 N 个 seed（从 seed 起连续递增；>1 即进入扫描模式）")
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    args = ap.parse_args(argv)

    if args.list_algos:
        algos = list_algos()
        print("\n".join(algos) if algos else "(algorithms/ 下无算法)")
        return
    if not args.algo:
        ap.error(f"必须指定 --algo（可用: {', '.join(list_algos() or ['<无>'])}），或 --list 查看全部")
    if args.resume and (args.sweep or (args.seeds or 1) > 1):
        ap.error("--sweep/--seeds 与 --resume 互斥：扫描面向全新实验，续训针对单次运行")

    spec = load_spec(args.algo)
    base = parse_overrides(args.override)

    if not args.sweep and (args.seeds or 1) <= 1:
        run_once(spec, args.algo, base, args.runs_root, resume=args.resume)
        return
    run_sweep(spec, args.algo, base, parse_sweep(args.sweep),
              args.seeds or 1, args.runs_root)


if __name__ == "__main__":
    main()
