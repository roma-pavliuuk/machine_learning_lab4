import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[2]
LAB3_DIR = Path(os.environ.get("LAB3_DIR", _HERE / "lab_03"))
_LOADER = LAB3_DIR / "src" / "preprocessor.py"


def _load():
    name = "lab03_preprocessor"
    if name in sys.modules:
        return sys.modules[name]
    if not _LOADER.exists():
        raise FileNotFoundError(f"Не знайдено {_LOADER}: покладіть lab_01–lab_03 поруч із lab_04 або задайте LAB3_DIR")
    spec = importlib.util.spec_from_file_location(name, _LOADER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lab3 = _load()
DynamicPipelineBuilder = lab3.DynamicPipelineBuilder
AdaptiveImputer = lab3.AdaptiveImputer
