"""Landmark tracking between detector runs and temporal smoothing.

Detection runs every N frames. In between, the 5 landmarks are carried forward with
sparse optical flow, then smoothed with a one-euro filter so the swapped face neither
jitters nor lags on fast head turns.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .detect import Face


class OneEuro:
    """One-euro filter for vectors (Casiez et al. 2012)."""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, x: np.ndarray, dt: float | None = None) -> np.ndarray:
        x = np.asarray(x, np.float32)
        freq = 1.0 / dt if dt and dt > 0 else self.freq
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * freq
        a_d = self._alpha(self.d_cutoff, freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        speed = float(np.linalg.norm(dx_hat) / max(x.size, 1))
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class FaceTracker:
    def __init__(self, detect_every: int = 3, smoothing: float = 0.6, fps: float = 30.0, fade_frames: int = 8):
        self.detect_every = max(1, int(detect_every))
        self.fade_frames = max(1, int(fade_frames))
        self.set_smoothing(smoothing, fps)
        self.face: Face | None = None
        self.prev_gray: np.ndarray | None = None
        self.frame_index = 0
        self.lost = 0
        self.presence = 0.0  # 0..1, ramps for fade in/out

    def set_smoothing(self, smoothing: float, fps: float = 30.0) -> None:
        s = float(np.clip(smoothing, 0.0, 1.0))
        # smoothing 0 -> almost raw; 1 -> heavy. min_cutoff decreases, beta decreases.
        self.filter = OneEuro(freq=fps, min_cutoff=3.0 - 2.7 * s, beta=0.05 - 0.045 * s)

    def should_detect(self) -> bool:
        return self.face is None or self.frame_index % self.detect_every == 0

    def update(self, gray: np.ndarray, detected: Face | None, ran_detector: bool, dt: float | None = None) -> Face | None:
        """Advance one frame. ``detected`` is the detector's pick when it ran."""
        self.frame_index += 1
        face = self.face
        if ran_detector:
            if detected is not None:
                if face is not None and np.hypot(*(detected.center - face.center)) > face.size * 1.5:
                    self.filter.reset()  # jumped to a different face
                face = detected
                self.lost = 0
            else:
                self.lost += 1
                if self.lost >= 2:
                    face = None
        elif face is not None and self.prev_gray is not None:
            face = self._flow(self.prev_gray, gray, face)
        self.prev_gray = gray
        if face is not None:
            kps = self.filter(face.kps, dt)
            face = Face(face.bbox, kps.reshape(5, 2), face.score)
            self.presence = min(1.0, self.presence + 1.0 / self.fade_frames)
        else:
            self.filter.reset()
            self.presence = max(0.0, self.presence - 1.0 / self.fade_frames)
        self.face = face
        return face

    @staticmethod
    def _flow(prev: np.ndarray, cur: np.ndarray, face: Face) -> Face | None:
        p0 = face.kps.reshape(-1, 1, 2).astype(np.float32)
        win = int(np.clip(face.size / 8, 11, 41)) | 1
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(win, win), maxLevel=3)
        if p1 is None or st is None or st.sum() < 3:
            return face
        ok = st.reshape(-1) == 1
        shift = (p1.reshape(-1, 2) - p0.reshape(-1, 2))[ok].mean(0)
        kps = face.kps.copy()
        kps[ok] = p1.reshape(-1, 2)[ok]
        kps[~ok] += shift
        bbox = face.bbox.copy()
        bbox[:2] += shift
        bbox[2:] += shift
        return Face(bbox, kps, face.score)
