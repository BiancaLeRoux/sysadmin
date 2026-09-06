"""Face parsing (BiSeNet, 19 classes) and XSeg occlusion masks on aligned crops."""

from __future__ import annotations

import cv2
import numpy as np

from ..runtime import SessionInfo

# CelebAMask-HQ label ids used by face-parsing.PyTorch
BG, SKIN, L_BROW, R_BROW, L_EYE, R_EYE, EYE_G, L_EAR, R_EAR, EAR_R, NOSE, MOUTH, U_LIP, L_LIP, NECK, NECK_L, CLOTH, HAIR, HAT = range(19)
FACE_REGION = (SKIN, L_BROW, R_BROW, L_EYE, R_EYE, EYE_G, NOSE, MOUTH, U_LIP, L_LIP)
MOUTH_REGION = (MOUTH, U_LIP, L_LIP)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class FaceParser:
    """BiSeNet ResNet-34 at 512x512. Returns an HxW int label map for the crop."""

    def __init__(self, session: SessionInfo):
        self.sess = session
        self._input = "input"

    def labels(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (512, 512)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]
        out = self.sess.run({self._input: blob})[0][0]  # 19x512x512
        lab = out.argmax(0).astype(np.uint8)
        if (h, w) != (512, 512):
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        return lab

    @staticmethod
    def region_mask(labels: np.ndarray, regions=FACE_REGION, blur: int = 9) -> np.ndarray:
        m = np.isin(labels, regions).astype(np.float32)
        if blur > 0:
            m = cv2.GaussianBlur(m, (0, 0), blur)
        return np.clip(m, 0, 1)


class OcclusionMasker:
    """XSeg: 1 where the face is visible, 0 where something covers it."""

    def __init__(self, session: SessionInfo):
        self.sess = session
        self._input = "input"

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        img = cv2.resize(crop_bgr, (256, 256)).astype(np.float32) / 255.0
        out = self.sess.run({self._input: img[None]})[0][0, :, :, 0]
        mask = np.clip(out, 0, 1).astype(np.float32)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.GaussianBlur(mask, (0, 0), 5)
        return np.clip((mask - 0.5) * 2.0, 0, 1)
