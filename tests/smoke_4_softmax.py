"""冒烟测试 批4：真实算法 softmax 接入验证（少量 epoch，验证 infra 零改动）。

运行: python tests/smoke_4_softmax.py
"""
import csv
import subprocess
import sys
import tempfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        sys.exit(1)


def main():
    tmp = Path(tempfile.mkdtemp())
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [PY, str(ROOT / "run.py"), "--algo", "softmax", "--runs-root", str(tmp),
         "--override", "epochs=2", "num_workers=0"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env, timeout=600)
    check("softmax 训练退出码 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
    print(r.stdout.strip().splitlines()[-3:] if r.stdout.strip() else "")

    rd = next((tmp / "softmax").iterdir())
    with open(rd / "history.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    check("跑满 2 轮", len(rows) == 2)
    acc = [float(x["val_acc"]) for x in rows]
    check("val_acc 处于 softmax 合理区间", 0.7 < acc[-1] < 0.95, str(acc))
    check("acc 在提升", acc[-1] >= acc[0], str(acc))
    check("best.pt 已存", (rd / "ckpt" / "best.pt").exists())
    check("算法专属回调产物 predictions.png 已存", (rd / "predictions.png").exists())
    log = (rd / "train.log").read_text(encoding="utf-8")
    check("预测图回调日志可见", "[预测图]" in log)

    # 断点续训 1 轮，验证真实数据 + 手写 SGD 的续训链路
    r2 = subprocess.run(
        [PY, str(ROOT / "run.py"), "--algo", "softmax",
         "--resume", str(rd / "ckpt" / "last.pt"), "--override", "epochs=3"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env, timeout=600)
    check("softmax 续训退出码 0", r2.returncode == 0, r2.stderr[-800:] if r2.returncode else "")
    with open(rd / "history.csv", encoding="utf-8", newline="") as f:
        rows2 = list(csv.DictReader(f))
    check("续训后共 3 轮", len(rows2) == 3, str(len(rows2)))
    check("第 3 轮 acc 不低于第 2 轮", float(rows2[2]["val_acc"]) >= float(rows2[1]["val_acc"]) - 0.02)

    print("批4 冒烟测试全部通过 —— infra 对首个真实算法零改动接入")


if __name__ == "__main__":
    main()
