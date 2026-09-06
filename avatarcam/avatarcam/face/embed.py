"""ArcFace identity embeddings and the avatar identity profile."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..runtime import SessionInfo
from .align import warp_crop


class Embedder:
    def __init__(self, session: SessionInfo):
        self.sess = session
        self._input = session.inputs[0]

    def embed_aligned(self, crop112_bgr: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(crop112_bgr, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        return self.sess.run({self._input: blob})[0][0].astype(np.float32)

    def embed(self, frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        crop, _ = warp_crop(frame_bgr, kps, "arcface_112_v2", 112)
        return self.embed_aligned(crop)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def average_embedding(embeddings: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted mean of L2-normalised embeddings, re-normalised."""
    if not embeddings:
        raise ValueError("no embeddings")
    w = np.ones(len(embeddings), np.float32) if weights is None else np.asarray(weights, np.float32)
    stack = np.stack([normalize(e) for e in embeddings])
    mean = (stack * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
    return normalize(mean).astype(np.float32)


def save_profile(path: Path, embedding: np.ndarray, meta: dict | None = None, thumbs: np.ndarray | None = None) -> None:
    payload = {"embedding": embedding.astype(np.float32)}
    if thumbs is not None:
        payload["thumbs"] = thumbs
    for k, v in (meta or {}).items():
        payload["meta_" + k] = np.asarray(v)
    np.savez_compressed(path, **payload)


def load_profile(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    out = {"embedding": normalize(data["embedding"].astype(np.float32))}
    for k in data.files:
        if k.startswith("meta_"):
            out[k[5:]] = data[k]
        elif k == "thumbs":
            out["thumbs"] = data[k]
    return out
