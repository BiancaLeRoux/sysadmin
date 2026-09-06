"""Background replacement, hair recolour, and skin smoothing."""

from __future__ import annotations

import cv2
import numpy as np


class Background:
    def __init__(self, mode: str = "none", color=(24, 24, 24), image_path: str = "", blur: int = 25):
        self.mode = mode
        self.color = tuple(int(c) for c in color)
        self.blur = int(blur)
        self.image: np.ndarray | None = None
        self._scaled: np.ndarray | None = None
        if image_path:
            self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    def render(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        if self.mode == "blur":
            k = max(1, self.blur) | 1
            small = cv2.resize(frame_bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (k, k), 0)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.mode == "color":
            return np.full_like(frame_bgr, self.color[::-1])  # RGB config -> BGR
        if self.mode == "image" and self.image is not None:
            if self._scaled is None or self._scaled.shape[:2] != (h, w):
                self._scaled = _cover_resize(self.image, w, h)
            return self._scaled
        return None


def _cover_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = int(np.ceil(iw * scale)), int(np.ceil(ih * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return np.ascontiguousarray(r[y0:y0 + h, x0:x0 + w])


def composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    out = fg.astype(np.float32) * a + bg.astype(np.float32) * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def recolor_hair(frame_bgr: np.ndarray, hair_mask: np.ndarray, target_rgb, strength: float) -> np.ndarray:
    """Shift hue/saturation toward ``target_rgb`` inside ``hair_mask`` (HxW 0..1)."""
    if strength <= 0 or hair_mask.max() <= 0:
        return frame_bgr
    target = np.uint8([[list(target_rgb)[::-1]]])
    t_h, t_s, t_v = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    m = (hair_mask * strength)[..., None]
    cur_v_mean = float((hsv[..., 2] * hair_mask).sum() / max(hair_mask.sum(), 1))
    v_gain = t_v / max(cur_v_mean, 1.0)
    new = hsv.copy()
    new[..., 0] = t_h
    new[..., 1] = t_s
    new[..., 2] = np.clip(hsv[..., 2] * v_gain, 0, 255)
    out = hsv * (1 - m) + new * m
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def smooth_skin(frame_bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return frame_bgr
    d = int(5 + amount * 6) | 1
    soft = cv2.bilateralFilter(frame_bgr, d, 30 + 50 * amount, 5 + 10 * amount)
    m = (mask * amount)[..., None].astype(np.float32)
    return np.clip(frame_bgr * (1 - m) + soft * m, 0, 255).astype(np.uint8)
