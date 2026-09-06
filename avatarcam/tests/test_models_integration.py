"""End-to-end checks with the real models. Skipped unless AVATARCAM_MODELS_DIR is set."""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from avatarcam.config import Config
    from avatarcam.pipeline import Pipeline

    cfg = Config.load()
    cfg.runtime.providers = ["cpu"]
    cfg.runtime.models_dir = os.environ["AVATARCAM_MODELS_DIR"]
    cfg.runtime.context_cache = False
    cfg.body.enabled = True
    cfg.enhance.enabled = True
    return Pipeline(cfg)


def _sample(name):
    p = Path(os.environ.get("AVATARCAM_SAMPLES", "")) / name
    if not p.exists():
        pytest.skip(f"sample {name} missing")
    import cv2

    return cv2.imread(str(p))


def test_detector_finds_faces(pipeline):
    img = _sample("zidane.jpg")
    faces = pipeline.detector(img)
    assert len(faces) >= 2
    assert all(f.kps.shape == (5, 2) for f in faces)


def test_swap_changes_face_region_only(pipeline):
    import cv2

    src = _sample("bus.jpg")
    tgt = _sample("zidane.jpg")
    pipeline.identity_from_image(src)
    pipeline.cfg.body.enabled = False
    out = None
    for _ in range(12):
        out, stats = pipeline.process(tgt.copy(), 1 / 30)
    assert stats["presence"] == 1.0
    diff = np.abs(out.astype(int) - tgt.astype(int)).mean(-1)
    face = pipeline.tracker.face
    x0, y0, x1, y1 = [int(v) for v in face.bbox]
    assert diff[y0:y1, x0:x1].mean() > 3.0
    assert diff[:50, :50].mean() < 0.5


def test_body_stage_runs(pipeline):
    img = _sample("bus.jpg")
    pipeline.cfg.body.enabled = True
    pipeline.cfg.body.waist = -0.5
    pipeline.cfg.body.background = "blur"
    out, stats = pipeline.process(img.copy(), 1 / 30)
    assert out.shape == img.shape
    assert "matte" in stats["stages"]
