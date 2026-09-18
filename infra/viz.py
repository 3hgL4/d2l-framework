"""训练曲线：实时刷新 + 存盘（matplotlib）。

- style 注入点：dict -> 更新 rcParams；callable(fig, ax) -> 算法层完全自定义；
- 后端为 Agg（无显示器/测试环境）时自动退化为仅存盘；
- 中文字体按 Windows 常见字体降级（Microsoft YaHei -> SimHei -> DejaVu）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# lr 与耗时间和 loss/acc 量纲差太大，默认不画
SKIP_KEYS = {"lr", "train_time_s", "time_s", "epoch"}


def _live_ok() -> bool:
    return matplotlib.get_backend().lower() != "agg"


def _setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


class CurvePlot:
    def __init__(self, save_path, title="training curves", live: bool = True, style=None):
        self.save_path = Path(save_path)
        self.title = title
        self.live = live and _live_ok()
        _setup_fonts()
        self.fig, self.ax = plt.subplots(figsize=(7, 4.5))
        if isinstance(style, dict):
            matplotlib.rcParams.update(style)
        elif callable(style):
            style(self.fig, self.ax)
        if self.live:
            plt.ion()
            self.fig.show()

    def update(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.ax.clear()
        xs = [r.get("epoch", i + 1) for i, r in enumerate(rows)]
        keys = [k for k in rows[-1] if k not in SKIP_KEYS
                and isinstance(rows[-1][k], (int, float))]
        for k in keys:
            pts = [(r.get("epoch", i + 1), r[k]) for i, r in enumerate(rows)
                   if isinstance(r.get(k), (int, float))]
            self.ax.plot([p[0] for p in pts], [p[1] for p in pts],
                         marker="o", ms=3, label=k)
        self.ax.set_xlabel("epoch")
        self.ax.set_title(self.title)
        self.ax.grid(alpha=0.3)
        self.ax.legend(fontsize=8)
        if self.live:
            self.fig.canvas.draw_idle()
            plt.pause(0.01)

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.save_path, dpi=150, bbox_inches="tight")

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)

    def wait_for_close(self) -> None:
        """阻塞直到用户手动关闭窗口（Agg 后端无窗口，立即返回）。"""
        if self.live:
            plt.ioff()
            plt.show(block=True)
