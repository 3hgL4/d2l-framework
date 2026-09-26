"""algo_report.py — 按算法生成训练报告（边缘工具：不属 infra，不属算法层）

用法（在 d2l 根目录）:
  python tools/algo_report.py --algo mlp                  # 该算法最新一次 run
  python tools/algo_report.py --algo mlp --run 20260922-200605
  python tools/algo_report.py --algo mlp --compare        # 汇总对比该算法全部 runs

输出:
  reports/<algo>/<stamp>.md          单次 run 报告
  reports/<algo>/compare.md          多 run 对比表（--compare 时覆盖更新）

设计:
  - "我的笔记"取自 algorithms/<algo>/algo.py 文件头部的连续 # 注释块，
    在那里写目标/假设/结论，报告自动收录。
  - 数据全部来自 runs/<algo>/<stamp>/ 既有产物（config/log/history/env），
    本工具不 import torch、不触碰 infra。
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
ALGOS = ROOT / "algorithms"
REPORTS = ROOT / "reports"

# infra 默认键（config 里这些原样列出，不做解读）；其余键视为算法自定义超参
INFRA_KEYS = {
    "seed", "deterministic", "device", "amp", "num_workers", "epochs",
    "grad_clip", "log_every", "monitor", "mode", "patience", "min_delta",
    "verbose", "workers",
}


def header_notes(algo: str) -> str:
    """取算法文件头部连续的 # 注释块（跳过空行）。"""
    path = ALGOS / algo / "algo.py"
    if not path.exists():
        return f"(找不到 {path.relative_to(ROOT)})"
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("#"):
            lines.append(raw.strip().lstrip("#").strip())
        elif raw.strip() == "" and (lines or not lines):
            continue
        else:
            break
    return "\n".join(lines) if lines else "(algo.py 顶部暂无注释——把目标/假设/结论写在那里，报告会自动收录)"


def load_config_from_log(log_path: Path) -> dict:
    """train.log 的 [配置] 行是完整 config 的 dict 字面量，直接取。"""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\[配置\]\s*(\{.*\})", text)
    if m:
        try:
            return ast.literal_eval(m.group(1))
        except Exception:
            pass
    return {}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def load_history(csv_path: Path) -> tuple[list[str], list[dict]]:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], []
    cols = [c for c in rows[0] if not c.startswith("_")]
    clean = [{c: r.get(c, "") for c in cols} for r in rows]
    return cols, clean


def fmt_val(x):
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else f"{f:.4f}"
    except (TypeError, ValueError):
        return str(x)


def history_table(cols: list[str], rows: list[dict], max_show: int = 14) -> str:
    if not rows:
        return "(无 history)"
    def render(rs):
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        body = ["| " + " | ".join(fmt_val(r[c]) for c in cols) + " |" for r in rs]
        return "\n".join([head, sep, *body])
    if len(rows) <= max_show:
        return render(rows)
    show = rows[:5] + [None] + rows[-5:]
    lines = render([r for r in show if r]).splitlines()
    dots = "|" + "|".join([" … "] * len(cols)) + "|"
    return "\n".join(lines[:3] + [dots] + lines[3:])


def run_paths(algo: str, stamp: str) -> Path:
    p = RUNS / algo / stamp
    if not p.is_dir():
        sys.exit(f"[错误] run 不存在: {p}")
    return p


def pick_cols(cols: list[str]) -> list[str]:
    """history 列过多时优先展示核心列。"""
    priority = ["epoch", "lr", "train_loss", "loss", "acc", "train_acc",
                "val_loss", "val_acc", "train_time_s"]
    chosen = [c for c in priority if c in cols]
    extra = [c for c in cols if c not in chosen]
    return (chosen + extra) if chosen else cols


def make_run_report(algo: str, stamp: str) -> Path:
    d = run_paths(algo, stamp)
    cfg_log = load_config_from_log(d / "train.log")
    cols, rows = load_history(d / "history.csv")
    show_cols = pick_cols(cols)

    env = {}
    if (d / "env.json").exists():
        env = json.loads((d / "env.json").read_text(encoding="utf-8"))

    final = rows[-1] if rows else {}
    best_loss = min(rows, key=lambda r: float(r["val_loss"])) if rows and "val_loss" in cols else None
    best_acc = max(rows, key=lambda r: float(r["val_acc"])) if rows and "val_acc" in cols else None
    total_s = sum(float(r.get("train_time_s") or 0) for r in rows)

    flat = flatten(cfg_log)
    algo_keys = {k: v for k, v in flat.items()
                 if k not in INFRA_KEYS and not k.startswith(("viz.", "ckpt."))}
    infra_line = ", ".join(f"{k}={flat[k]}" for k in
                           ("seed", "device", "epochs", "monitor", "patience") if k in flat)

    log_tail = "(无 train.log)"
    if (d / "train.log").exists():
        tail = (d / "train.log").read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        log_tail = "\n".join(f"    {line}" for line in tail)

    rel = f"../../runs/{algo}/{stamp}"
    lines = [
        f"# {algo} · 训练报告 {stamp}",
        "",
        "## 我的笔记（algo.py 头部）",
        "",
        header_notes(algo),
        "",
        "## 关键指标",
        "",
        f"- 轮数: {len(rows)} | 总训练时长: {total_s:.1f}s | 环境: {env.get('torch', '?')} @ {env.get('gpu', '?')}",
    ]
    if final:
        lines.append(f"- **最终**: " + ", ".join(f"{c}={fmt_val(final[c])}" for c in show_cols if c not in ("epoch", "lr", "train_time_s")))
    if best_loss:
        lines.append(f"- **best val_loss**: {fmt_val(best_loss['val_loss'])} (epoch {best_loss.get('epoch', '?')})")
    if best_acc:
        lines.append(f"- **best val_acc**: {fmt_val(best_acc['val_acc'])} (epoch {best_acc.get('epoch', '?')})")
    if "train_loss" in cols and "val_loss" in cols and rows:
        gap = float(final["val_loss"]) - float(final["train_loss"])
        tag = "过拟合信号" if gap > 0.1 else ("欠拟合信号" if float(final["train_loss"]) > float(final["val_loss"]) else "拟合均衡")
        lines.append(f"- train/val loss 差: {gap:+.4f}（{tag}，仅供参考）")

    lines += [
        "",
        "## 超参",
        "",
        f"- infra: {infra_line}",
        "- 算法自定义: " + (", ".join(f"{k}={v}" for k, v in algo_keys.items()) or "(无)"),
        "",
        "## 训练历史",
        "",
        history_table(show_cols, rows),
        "",
        "## 日志尾部",
        "",
        "```",
        log_tail,
        "```",
        "",
        "## 曲线与环境",
        "",
        f"![curves]({rel}/curves.png)",
        "",
        f"- 完整产物: `runs/{algo}/{stamp}/`（config.yaml · env.json · history.csv · train.log · ckpt/）",
        "- 本报告由 `tools/algo_report.py` 生成，可再生，不入库。",
    ]

    out_dir = REPORTS / algo
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stamp}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def make_compare_report(algo: str, stamps: list[str]) -> Path:
    rows_out = []
    for s in sorted(stamps):
        d = run_paths(algo, s)
        cols, rows = load_history(d / "history.csv")
        if not rows:
            continue
        cfg = load_config_from_log(d / "train.log")
        final = rows[-1]
        bl = min(float(r["val_loss"]) for r in rows) if "val_loss" in cols else float("nan")
        ba = max(float(r["val_acc"]) for r in rows) if "val_acc" in cols else float("nan")
        fl = float(final.get("val_loss", "nan"))
        fa = float(final.get("val_acc", "nan"))
        ts = sum(float(r.get("train_time_s") or 0) for r in rows)
        rows_out.append({
            "run": s, "lr": cfg.get("lr", "?"), "batch": cfg.get("batch_size", "?"),
            "epochs": len(rows), "final_val_loss": fl, "final_val_acc": fa,
            "best_val_loss": bl, "best_val_acc": ba, "time_s": round(ts, 1),
        })
    if not rows_out:
        sys.exit(f"[错误] {algo} 没有可对比的 run")

    valid = [r for r in rows_out if r["best_val_loss"] == r["best_val_loss"]]
    best_stamp = min(valid, key=lambda r: r["best_val_loss"])["run"] if valid else None

    head = "| run | lr | batch | epochs | final_val_loss | final_val_acc | best_val_loss | best_val_acc | time_s |"
    sep = "|" + "|".join(["---"] * 9) + "|"
    body = []
    for r in rows_out:
        mark = " **← best**" if r["run"] == best_stamp else ""
        body.append(
            f"| {r['run']}{mark} | {r['lr']} | {r['batch']} | {r['epochs']} "
            f"| {r['final_val_loss']:.4f} | {r['final_val_acc']:.4f} "
            f"| {r['best_val_loss']:.4f} | {r['best_val_acc']:.4f} | {r['time_s']} |"
        )
    notes = header_notes(algo)
    text = "\n".join([
        f"# {algo} · runs 对比",
        "",
        "## 我的笔记（algo.py 头部）",
        "",
        notes,
        "",
        "## 汇总（按时间排序）",
        "",
        head, sep, *body, "",
        "- 对比读法：best_val_loss 越低越好；final 与 best 差距大 = 训练尾段在退化（lr 太大或该早停）；",
        "  同 best_val_loss 下 time_s 小者配置更划算。单次明细见同目录 `<stamp>.md`。",
        "",
    ])
    out_dir = REPORTS / algo
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "compare.md"
    out.write_text(text, encoding="utf-8")
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--run", help="指定 run 时间戳（默认取最新）")
    ap.add_argument("--compare", action="store_true", help="汇总对比该算法全部 runs")
    args = ap.parse_args()

    algo_dir = RUNS / args.algo
    if not algo_dir.is_dir():
        sys.exit(f"[错误] 算法无训练记录: {algo_dir}")

    if args.compare:
        stamps = [p.name for p in sorted(algo_dir.iterdir()) if p.is_dir()]
        out = make_compare_report(args.algo, stamps)
    else:
        stamp = args.run or max(p.name for p in algo_dir.iterdir() if p.is_dir())
        out = make_run_report(args.algo, stamp)
    print(f"[完成] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
