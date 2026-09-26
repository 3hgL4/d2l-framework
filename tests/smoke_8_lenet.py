"""冒烟测试 批8：LeNet 全组件从零实现验证（手写层数值正确性 + 真实训练）。

覆盖：形状推演 / 手写 conv・maxpool・avgpool 的 gradcheck（双精度有限差分，
比标量对拍更强）/ 真实训练 1 轮 + 续训 1 轮（手写 SGD 状态等价）。

运行: python tests/smoke_8_lenet.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.lenet.algo import (LeNetScratch, _AvgPoolFunc, _Conv2dFunc,
                                   _MaxPoolFunc)


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        sys.exit(1)


def main():
    tmp = Path(tempfile.mkdtemp())
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")

    # ---- ① 形状推演（d2l 6.6：28→28→14→10→5→400→120→84→10） ----
    torch.manual_seed(0)   # 仅测试进程内固定种子，不影响训练路径
    net = LeNetScratch()
    y = net(torch.randn(2, 1, 28, 28))
    check("LeNet 前向输出形状 (2,10)", tuple(y.shape) == (2, 10), str(tuple(y.shape)))
    check("pool=max 变体前向正常",
          tuple(LeNetScratch(pool="max")(torch.randn(2, 1, 28, 28)).shape) == (2, 10))

    # ---- ② 手写 conv：gradcheck 双精度数值对拍（覆盖解析梯度 dW/db/dx 全部） ----
    x = torch.randn(2, 2, 8, 8, dtype=torch.float64, requires_grad=True)
    W = torch.randn(3, 2, 3, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, dtype=torch.float64, requires_grad=True)
    check("conv gradcheck（stride=2, padding=1）",
          torch.autograd.gradcheck(lambda *a: _Conv2dFunc.apply(*a, 2, 1), (x, W, b)))
    check("conv gradcheck（stride=1, padding=0）",
          torch.autograd.gradcheck(lambda *a: _Conv2dFunc.apply(*a, 1, 0), (x, W, b)))

    # ---- ③ 手写汇聚层：gradcheck ----
    xp = torch.randn(2, 2, 8, 8, dtype=torch.float64, requires_grad=True)
    check("maxpool gradcheck（k2,s2）",
          torch.autograd.gradcheck(lambda t: _MaxPoolFunc.apply(t, 2, 2), (xp,)))
    check("avgpool gradcheck（k2,s2）",
          torch.autograd.gradcheck(lambda t: _AvgPoolFunc.apply(t, 2, 2), (xp,)))

    # ---- ④ 真实训练冒烟：手写 IDX 解析 + 全手写层 + 手写 SGD 全链路 ----
    r = subprocess.run(
        [PY, str(ROOT / "run.py"), "--algo", "lenet", "--runs-root", str(tmp),
         "--override", "epochs=1", "num_workers=0"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env, timeout=600)
    check("lenet 训练退出码 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
    rd = next((tmp / "lenet").iterdir())
    log = (rd / "train.log").read_text(encoding="utf-8")
    check("数据经手写 IDX 解析加载（无 torchvision）",
          "torchvision" not in log and "val_acc" in log)
    last = [l for l in r.stdout.splitlines() if l.startswith("epoch")][-1]
    val_acc = float(last.split("val_acc ")[1].split(" |")[0])
    check("1 轮 val_acc 处于 LeNet 合理区间", 0.1 < val_acc < 0.85, str(val_acc))

    # ---- ⑤ 续训：手写 SGDScratch 的 state_dict 恢复（无退化告警） ----
    r2 = subprocess.run(
        [PY, str(ROOT / "run.py"), "--algo", "lenet", "--runs-root", str(tmp),
         "--resume", str(rd / "ckpt" / "last.pt"), "--override", "epochs=2"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env, timeout=600)
    check("lenet 续训退出码 0", r2.returncode == 0, r2.stderr[-800:] if r2.returncode else "")
    check("续训无优化器退化告警", "仅恢复" not in (rd / "train.log").read_text(encoding="utf-8"))

    print("批8 冒烟测试全部通过 —— LeNet 全组件从零实现数值正确、链路完整")


if __name__ == "__main__":
    main()
