"""冒烟测试 批3：FakeAlgo 全流程（训练->曲线存盘->断点续训->多进程）+ 依赖红线检查。

运行: python tests/smoke_3_full_flow.py
说明: 经由 run.py 以子进程方式跑，MPLBACKEND=Agg 下验证"仅存盘"退化路径。
"""
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        sys.exit(1)


def run_cli(args, env_extra=None):
    import os
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    env.update(env_extra or {})
    return subprocess.run([PY, str(ROOT / "run.py"), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(ROOT), env=env, timeout=600)


def history_rows(run_dir: Path):
    with open(run_dir / "history.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    tmp = Path(tempfile.mkdtemp())

    # 0. 红线：infra 不得 import algorithms
    bad = []
    for py in (ROOT / "infra").glob("*.py"):
        if re.search(r"^\s*(from|import)\s+algorithms", py.read_text(encoding="utf-8"), re.M):
            bad.append(py.name)
    check("依赖红线: infra 零反向依赖", not bad, str(bad))

    # 1. 完整流程（含早停：线性可分问题给大 lr 让 val_loss 停滞，patience=2）
    r = run_cli(["--algo", "fake", "--runs-root", str(tmp),
                 "--override", "epochs=30", "patience=2", "lr=1.5", "viz.live=False"])
    check("全流程退出码 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
    run_dirs = sorted((tmp / "fake").iterdir())
    check("run 目录已创建", len(run_dirs) == 1)
    rd = run_dirs[0]
    for f in ("config.yaml", "env.json", "train.log", "history.csv", "curves.png",
              "ckpt/last.pt", "ckpt/best.pt"):
        check(f"产物 {f} 存在", (rd / f).exists())

    rows = history_rows(rd)
    check("早停使轮数 < 30", 0 < len(rows) < 30, str(len(rows)))
    check("history 含 train/val 列", "train_loss" in rows[0] and "val_loss" in rows[0])
    check("曲线非空文件", (rd / "curves.png").stat().st_size > 5000)
    check("日志含 epoch 摘要", "epoch 1/30" in (rd / "train.log").read_text(encoding="utf-8"))

    # 2. 断点续训：epochs 提高后续跑到新上限，历史衔接
    r2 = run_cli(["--algo", "fake", "--resume", str(rd / "ckpt" / "last.pt"),
                  "--override", "epochs=40", "patience=0", "lr=0.1", "viz.live=False"])
    check("续训退出码 0", r2.returncode == 0, r2.stderr[-800:] if r2.returncode else "")
    rows2 = history_rows(rd)
    check("续训后历史衔接无缝隙", [int(x["epoch"]) for x in rows2] ==
          list(range(1, len(rows2) + 1)), str([x['epoch'] for x in rows2]))
    ck = len(rows2)
    check("续训轮数超过原停止点", ck > len(rows), f"{ck} vs {len(rows)}")
    check("续训后仍每轮有 val 指标", all(x.get("val_loss") for x in rows2[1:]))

    # 3. Windows spawn 多进程路径（num_workers=2）
    r3 = run_cli(["--algo", "fake", "--runs-root", str(tmp),
                  "--override", "epochs=1", "num_workers=2", "viz.live=False"])
    check("num_workers=2 spawn 训练通过", r3.returncode == 0, r3.stderr[-800:] if r3.returncode else "")

    # 4. 控制台输出保持简洁（单行/epoch 摘要，无 tqdm 刷屏）
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("epoch ")]
    check("控制台仅 epoch 级摘要", len(lines) == len(rows), f"{len(lines)} vs {len(rows)}")

    print("批3 冒烟测试全部通过")


if __name__ == "__main__":
    main()
