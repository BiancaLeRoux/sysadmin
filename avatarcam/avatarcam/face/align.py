"""Five-point face alignment: templates, affine estimation, crop and paste-back."""

from __future__ import annotations

import cv2
import numpy as np

# Normalised 5-point templates (left eye, right eye, nose, left mouth, right mouth).
# arcface_112_v2 is InsightFace's standard; ffhq_512 is the FFHQ alignment used by the
# GFPGAN/GPEN family.
TEMPLATES: dict[str, np.ndarray] = {
    "arcface_112_v2": np.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089],
        ],
        dtype=np.float32,
    ),
    "ffhq_512": np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46864664],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.60654937, 0.72541100],
        ],
        dtype=np.float32,
    ),
}


def template_points(name: str, size: int) -> np.ndarray:
    return TEMPLATES[name] * np.float32(size)


def estimate_affine(kps: np.ndarray, template: str, size: int) -> np.ndarray:
    """Similarity transform (2x3) mapping ``kps`` (5x2) onto the template at ``size``."""
    dst = template_points(template, size)
    src = np.asarray(kps, np.float32)
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=100)
    if m is None:  # degenerate landmarks; fall back to least squares
        m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return m.astype(np.float32)


def warp_crop(frame: np.ndarray, kps: np.ndarray, template: str, size: int):
    """Return (crop, affine) where crop is ``size``x``size``."""
    m = estimate_affine(kps, template, size)
    crop = cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return crop, m


def invert_affine(m: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(m).astype(np.float32)


def crop_bbox_in_frame(m: np.ndarray, size: int, frame_shape) -> tuple[int, int, int, int]:
    """Axis-aligned bbox (x0, y0, x1, y1) of the crop square mapped back into the frame."""
    inv = invert_affine(m)
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    pts = cv2.transform(corners[None], inv)[0]
    h, w = frame_shape[:2]
    x0 = int(max(0, np.floor(pts[:, 0].min())))
    y0 = int(max(0, np.floor(pts[:, 1].min())))
    x1 = int(min(w, np.ceil(pts[:, 0].max())))
    y1 = int(min(h, np.ceil(pts[:, 1].max())))
    return x0, y0, x1, y1


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, m: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Composite ``crop`` (aligned space) into ``frame`` using ``mask`` (HxW float 0..1).

    Work is restricted to the crop's bounding box in frame space so cost scales with the
    face, not the frame. ``frame`` is modified in place and returned.
    """
    size = crop.shape[0]
    x0, y0, x1, y1 = crop_bbox_in_frame(m, size, frame.shape)
    if x1 <= x0 or y1 <= y0 or strength <= 0:
        return frame
    inv = invert_affine(m).copy()
    inv[0, 2] -= x0
    inv[1, 2] -= y0
    w, h = x1 - x0, y1 - y0
    warped = cv2.warpAffine(crop, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = cv2.warpAffine(mask.astype(np.float32), inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if strength < 1.0:
        alpha *= np.float32(strength)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    roi = frame[y0:y1, x0:x1]
    blended = roi.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
    frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(frame.dtype)
    return frame


def box_mask(size: int, blur: float = 0.3, padding=(0.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    """Static feathered box mask for an aligned crop (facefusion-style).

    ``padding`` is (top, right, bottom, left) as a fraction of ``size``; positive values
    shrink the box.
    """
    blur_px = int(size * 0.5 * float(blur))
    pad_t, pad_r, pad_b, pad_l = [max(int(size * p), 0) for p in padding]
    mask = np.zeros((size, size), np.float32)
    inset = max(blur_px // 2, 1)
    y0, y1 = inset + pad_t, size - inset - pad_b
    x0, x1 = inset + pad_l, size - inset - pad_r
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = 1.0
    if blur_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_px * 0.25)
    return mask
