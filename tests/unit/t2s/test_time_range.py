import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NORM = ROOT / "paddlespeech" / "t2s" / "frontend" / "zh_normalization"


def load_chronology():
    pkg_name = "ps_zh_norm_under_test"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(NORM)]
    sys.modules[pkg_name] = pkg

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(
            name, NORM / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load(pkg_name + ".num", "num.py")
    return load(pkg_name + ".chronology", "chronology.py")


def spoken(text):
    chronology = load_chronology()
    match = chronology.RE_TIME_RANGE.fullmatch(text)
    if match is None:
        match = chronology.RE_TIME.fullmatch(text)
    return chronology.replace_time(match)


def test_range_end_uses_its_own_minutes():
    assert spoken("8:30-9:15") == "八点半至九点十五分"
    assert spoken("8:30-9:45") == "八点半至九点四十五分"
    assert spoken("8:30~9:15") == "八点半至九点十五分"
    assert spoken("8:15-9:30") == "八点十五分至九点半"
    assert spoken("8:30-9:30") == "八点半至九点半"
    assert spoken("9:30") == "九点半"
    assert spoken("9:15") == "九点十五分"
