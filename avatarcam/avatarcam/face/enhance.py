"""Face enhancers (GPEN-BFR-256, GFPGAN 1.4) on FFHQ-aligned crops."""

from __future__ import annotations

import cv2
import numpy as np

from ..runtime import SessionInfo

ENHANCER_SIZES = {"gpen_bfr_256": 256, "gfpgan_1.4": 512}


class Enhancer:
    def __init__(self, session: SessionInfo, model: str):
        self.sess = session
        self.model = model
        self.size = ENHANCER_SIZES[model]
        self._input = "input"

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Enhance an aligned ``size``x``size`` BGR crop; returns BGR uint8."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = self.sess.run({self._input: blob})[0][0]
        img = np.clip((out.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
