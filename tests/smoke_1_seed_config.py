"""冒烟测试 批1：种子复现 / 配置合成 / 运行目录 / 日志双写。

运行: python tests/smoke_1_seed_config.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from infra.config import (build_config, new_run_dir, parse_overrides,
                          save_config, snapshot_env)
from infra.logger import get_logger
from infra.seed import set_seed


def check(name, cond):
    print(("PASS" if cond else "FAIL") + f"  {name}")
    if not cond:
        sys.exit(1)


def main():
    # 1. 种子复现
    set_seed(0)
    a = torch.rand(1000)
    set_seed(0)
    b = torch.rand(1000)
    check("种子复现: 同种子同结果", torch.equal(a, b))
    set_seed(0, deterministic=True)
    check("cudnn deterministic 开关生效", torch.backends.cudnn.deterministic is True)

    # 2. 配置两层合并（默认值单一事实源 = default_config；调参走 cli）
    algo_defaults = {"epochs": 30, "lr": 0.05, "viz": {"live": False}}
    cli = parse_overrides(["lr=0.9", "viz.save=False", "seed=7"])
    cfg = build_config(algo_defaults, cli)
    check("cli 覆盖默认", cfg.lr == 0.9 and cfg.epochs == 30)
    check("嵌套覆盖不丢同级键", cfg.viz.live is False and cfg.viz.save is False)
    check("infra 默认值仍在", cfg.patience == 0 and cfg.num_workers == 0)
    check("AttrDict 与 dict 访问等价", cfg["viz"]["live"] is False)
    check("嵌套 override 解析", parse_overrides(["a.b.c=1"])["a"]["b"]["c"] == 1)

    # 3. 运行目录 + 存档
    tmp = Path(tempfile.mkdtemp())
    rd = new_run_dir(tmp, "demo")
    check("run 目录结构 runs/demo/<stamp>/ckpt", (rd / "ckpt").is_dir() and rd.parent.name == "demo")
    rd2 = new_run_dir(tmp, "demo")
    check("同秒目录自动避让", rd != rd2)
    save_config(cfg, rd / "config.yaml")
    back = yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8"))
    check("config.yaml 落盘可回读", back["epochs"] == 30 and back["lr"] == 0.9)
    snapshot_env(rd / "env.json")
    import json as _json
    env = _json.loads((rd / "env.json").read_text(encoding="utf-8"))
    check("env.json 机器可读且含 torch/GPU", "torch" in env and "cuda_available" in env)

    # 4. 日志双写（含中文）
    log = get_logger(rd / "train.log")
    log.info("中文日志 行1")
    log.debug("debug 细节只进文件")
    content = (rd / "train.log").read_text(encoding="utf-8")
    check("文件含 debug 细节", "debug 细节只进文件" in content)
    check("文件含时间戳格式", " | INFO    | " in content)

    print("批1 冒烟测试全部通过")


if __name__ == "__main__":
    main()
