import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def cfg():
    from src.data import load_config

    return load_config()


@pytest.fixture(scope="session")
def reference(cfg):
    from src.data import load_reference

    return load_reference(cfg)
