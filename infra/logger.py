"""日志：控制台 + 文件双写。

- 文件一律 utf-8；控制台在 Windows GBK 下遇到不可编码字符时替换而非崩溃。
- 输出保持简洁：控制台只到 INFO（epoch 级摘要）；DEBUG 细节只进文件。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


class _SafeConsoleHandler(logging.StreamHandler):
    """Windows 控制台(GBK)下不可编码字符替换输出，不抛 UnicodeEncodeError。"""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            enc = getattr(stream, "encoding", None) or "utf-8"
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                stream.write(msg.encode(enc, "replace").decode(enc) + self.terminator)
            stream.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)


def get_logger(log_file=None, name: str = "d2l") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    console = _SafeConsoleHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        logger.addHandler(fh)
    return logger
