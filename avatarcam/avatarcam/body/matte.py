"""Robust Video Matting (MobileNetV3) with recurrent state."""

from __future__ import annotations

import cv2
import numpy as np

from ..runtime import SessionInfo


class Matter:
    """Returns a float32 alpha (HxW, 0..1) at frame resolution."""

    def __init__(self, session: SessionInfo, input_width: int = 384, downsample_ratio: float = 1.0):
        self.sess = session
        self.input_width = int(input_width)
        self.downsample_ratio = np.array([downsample_ratio], np.float32)
        self.reset()

    def reset(self) -> None:
        self.rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        iw = self.input_width
        ih = int(round(h * iw / w / 8)) * 8
        small = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
        src = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        feeds = {"src": src, "r1i": self.rec[0], "r2i": self.rec[1], "r3i": self.rec[2], "r4i": self.rec[3],
                 "downsample_ratio": self.downsample_ratio}
        fgr, pha, *rec = self.sess.run(feeds)
        self.rec = rec
        alpha = pha[0, 0]
        return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
