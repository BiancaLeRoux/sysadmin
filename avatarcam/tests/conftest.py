import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = os.environ.get("AVATARCAM_MODELS_DIR")


def pytest_configure(config):
    config.addinivalue_line("markers", "models: needs downloaded models (set AVATARCAM_MODELS_DIR)")


def pytest_collection_modifyitems(config, items):
    if MODELS and Path(MODELS).exists():
        return
    skip = pytest.mark.skip(reason="set AVATARCAM_MODELS_DIR to a directory with the models")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)
