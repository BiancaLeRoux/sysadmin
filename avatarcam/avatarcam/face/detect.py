"""SCRFD face detection with 5 landmarks (InsightFace decode, no insightface dependency)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..runtime import SessionInfo

STRIDES = (8, 16, 32)
NUM_ANCHORS = 2


@dataclass
class Face:
    bbox: np.ndarray        # x0, y0, x1, y1 in frame coordinates
    kps: np.ndarray         # 5x2 landmarks in frame coordinates
    score: float

    @property
    def size(self) -> float:
        return float(max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))

    @property
    def center(self) -> np.ndarray:
        return (self.bbox[:2] + self.bbox[2:]) / 2


class Detector:
    def __init__(self, session: SessionInfo, size: int = 320, threshold: float = 0.5, nms: float = 0.4):
        self.sess = session
        self.size = int(size)
        self.threshold = float(threshold)
        self.nms = float(nms)
        self._anchor_cache: dict[tuple[int, int], np.ndarray] = {}
        self._input = session.inputs[0]

    def _anchors(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w)
        if key not in self._anchor_cache:
            ys, xs = np.mgrid[:h, :w]
            centers = np.stack([xs, ys], -1).astype(np.float32).reshape(-1, 2) * stride
            centers = np.repeat(centers, NUM_ANCHORS, axis=0)
            self._anchor_cache[key] = centers
        return self._anchor_cache[key]

    def __call__(self, frame_bgr: np.ndarray) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:nh, :nw] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run({self._input: blob})

        boxes, scores, kpss = [], [], []
        for i, stride in enumerate(STRIDES):
            sc = outs[i].reshape(-1)
            bb = outs[i + 3].reshape(-1, 4) * stride
            kp = outs[i + 6].reshape(-1, 10) * stride
            fh = fw = self.size // stride
            anchors = self._anchors(fh, fw, stride)
            keep = np.where(sc >= self.threshold)[0]
            if keep.size == 0:
                continue
            a = anchors[keep]
            d = bb[keep]
            box = np.stack([a[:, 0] - d[:, 0], a[:, 1] - d[:, 1], a[:, 0] + d[:, 2], a[:, 1] + d[:, 3]], -1)
            k = kp[keep].reshape(-1, 5, 2) + a[:, None, :]
            boxes.append(box)
            scores.append(sc[keep])
            kpss.append(k)
        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale
        scores = np.concatenate(scores)
        kpss = np.concatenate(kpss) / scale
        order = nms(boxes, scores, self.nms)
        return [Face(boxes[i].astype(np.float32), kpss[i].astype(np.float32), float(scores[i])) for i in order]


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep


def pick_face(faces: list[Face], previous: Face | None = None) -> Face | None:
    """Choose the face to swap: nearest to the previous one, else the largest."""
    if not faces:
        return None
    if previous is not None:
        c = previous.center
        nearest = min(faces, key=lambda f: float(np.hypot(*(f.center - c))))
        if np.hypot(*(nearest.center - c)) < previous.size * 1.5:
            return nearest
    return max(faces, key=lambda f: f.size)
