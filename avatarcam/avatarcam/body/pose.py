"""YOLOv8-pose decode (17 COCO keypoints)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..face.detect import nms
from ..runtime import SessionInfo

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = range(17)


@dataclass
class Pose:
    bbox: np.ndarray       # x0,y0,x1,y1
    kps: np.ndarray        # 17x3 (x, y, confidence) in frame coordinates
    score: float

    def pt(self, i: int, min_conf: float = 0.3):
        x, y, c = self.kps[i]
        return (float(x), float(y)) if c >= min_conf else None


class PoseEstimator:
    def __init__(self, session: SessionInfo, size: int = 320, threshold: float = 0.4):
        self.sess = session
        self.size = int(size)
        self.threshold = float(threshold)
        self._input = session.inputs[0]

    def __call__(self, frame_bgr: np.ndarray) -> list[Pose]:
        h, w = frame_bgr.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (self.size, self.size), (0, 0, 0), swapRB=True)
        out = self.sess.run({self._input: blob})[0]  # 1 x 56 x N
        pred = out[0].T  # N x 56: cx, cy, w, h, conf, 17*(x,y,c)
        keep = pred[:, 4] >= self.threshold
        pred = pred[keep]
        if pred.shape[0] == 0:
            return []
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
        order = nms(boxes, pred[:, 4], 0.5)
        poses = []
        for i in order:
            kps = pred[i, 5:].reshape(17, 3).copy()
            kps[:, :2] /= scale
            poses.append(Pose((boxes[i] / scale).astype(np.float32), kps.astype(np.float32), float(pred[i, 4])))
        return poses


def pick_pose(poses: list[Pose], face_center=None) -> Pose | None:
    if not poses:
        return None
    if face_center is not None:
        def dist(p: Pose) -> float:
            n = p.pt(NOSE, 0.1)
            if n is None:
                return 1e9
            return float(np.hypot(n[0] - face_center[0], n[1] - face_center[1]))
        return min(poses, key=dist)
    return max(poses, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))
