"""冒烟测试 批7：config 模块单元级边界 + 多源 RNG 恢复。

定位：批1-6 验证"流程正确"，本批验证"纯函数边界"——不跑训练，
只打 infra 最基础的地基（合并/解析/告警/RNG），失败即中止。

运行: python tests/smoke_7_config_units.py
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infra.config import (AttrDict, _coerce, _unknown_paths, build_config,
                          deep_merge, parse_overrides)


def check(name, cond):
    print(("PASS" if cond else "FAIL") + f"  {name}")
    if not cond:
        sys.exit(1)


def main():
    # 1. deep_merge 边界
    check("空覆盖返回等值", deep_merge({"a": 1, "v": {"x": 1}}, None) == {"a": 1, "v": {"x": 1}})
    check("嵌套覆盖只动子键", deep_merge({"v": {"x": 1, "y": 2}}, {"v": {"y": 3}}) == {"v": {"x": 1, "y": 3}})
    check("dict 覆盖标量为替换语义", deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}})
    check("标量覆盖 dict 为替换语义", deep_merge({"v": {"x": 1}}, {"v": 5}) == {"v": 5})
    b = {"a": 1}
    deep_merge(b, {"a": 9})
    check("不修改入参 base", b["a"] == 1)

    # 2. _coerce / parse_overrides
    check("布尔/空值/整型/浮点/字符串",
          (_coerce("True"), _coerce("none"), _coerce("3"), _coerce("0.5"),
           _coerce("abc")) == (True, None, 3, 0.5, "abc"))
    check("负数与科学计数", parse_overrides(["a=-1.5e-3"])["a"] == -1.5e-3)
    try:
        parse_overrides(["bad"])
        check("缺 = 即报错", False)
    except ValueError:
        check("缺 = 即报错", True)

    # 3. 未知键告警（2026-09-19 新增路径）
    known = {"epochs": 1, "viz": {"live": True}}
    check("未知顶层键", _unknown_paths(known, {"epohs": 5}) == ["epohs"])
    check("未知嵌套键", _unknown_paths(known, {"viz": {"lve": 0}}) == ["viz.lve"])
    check("dict 覆盖标量结构冲突", _unknown_paths(known, {"epochs": {"a": 1}}) == ["epochs"])
    check("已知键不误报", _unknown_paths(known, {"viz": {"live": False}, "epochs": 2}) == [])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = build_config({"epochs": 30}, {"epohs": 9, "epochs": 2})
        check("告警触发且指名道姓", any("epohs" in str(x.message) for x in w))
    check("告警不阻断、正确键仍生效", cfg.epochs == 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_config({}, {"epochs": 3, "viz": {"save": False}})
        check("正常键零告警", all("未知配置键" not in str(x.message) for x in w))

    # 4. AttrDict
    check("嵌套属性访问", AttrDict({"a": {"b": 1}}).a.b == 1)
    try:
        AttrDict({"a": 1}).missing_key
        check("缺键 AttributeError", False)
    except AttributeError:
        check("缺键 AttributeError", True)

    # 5. 多源 RNG 恢复（torch 源批2已测，此处补 python/numpy 源）
    import random

    import numpy as np
    from infra.checkpoint import capture_rng, restore_rng
    random.seed(1)
    np.random.seed(1)
    ref = [random.random() for _ in range(5)], list(np.random.rand(5))
    random.seed(1)
    np.random.seed(1)
    st = capture_rng()
    [random.random() for _ in range(5)]
    np.random.rand(5)
    restore_rng(st)
    got = [random.random() for _ in range(5)], list(np.random.rand(5))
    check("python/numpy RNG 恢复后流等价", ref == got)
    check("restore 空状态安全", restore_rng(None) is None)

    print("批7 冒烟测试全部通过")


if __name__ == "__main__":
    main()
