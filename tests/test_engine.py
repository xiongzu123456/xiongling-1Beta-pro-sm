# -*- coding: utf-8 -*-
"""模型引擎单元测试：python3 -m pytest tests/ -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_model_loaded():
    e = Engine(os.path.join(ROOT, "model"))
    assert e.backend != "unavailable"
    assert e.info()["id"] == "xiongling-1Beta-pro-sm"


def test_generate_nonempty():
    e = Engine(os.path.join(ROOT, "model"))
    out = e.generate("雄灵一号", 40, 0.7, 5)
    assert isinstance(out, str) and len(out) > 40
    assert out.startswith("雄灵一号")


def test_generate_is_deterministic_at_zero_temp():
    e = Engine(os.path.join(ROOT, "model"))
    a = e.generate("训练", 20, 0.0, 10)
    b = e.generate("训练", 20, 0.0, 10)
    assert a == b
