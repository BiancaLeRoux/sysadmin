"""inswapper_128 wrapper: identity latent, swap, and colour transfer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..runtime import SessionInfo
from .embed import normalize


def load_emap(model_path: Path) -> np.ndarray:
    """Read the 512x512 identity projection matrix stored as an initializer.

    Uses the ``onnx`` package when present, otherwise a small protobuf scan so the runtime
    does not depend on ``onnx`` on platforms where it is hard to install.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return _load_emap_raw(model_path)
    model = onnx.load(str(model_path), load_external_data=False)
    for init in model.graph.initializer:
        if list(init.dims) == [512, 512]:
            return numpy_helper.to_array(init).astype(np.float32)
    raise RuntimeError("emap initializer not found in inswapper model")


def _load_emap_raw(path: Path) -> np.ndarray:
    """Minimal protobuf scan for a float32 [512,512] TensorProto in ``graph.initializer``."""
    data = path.read_bytes()
    graph = _pb_find(data, 7)  # ModelProto.graph
    if graph is None:
        raise RuntimeError("no graph in model")
    for init in _pb_iter(graph, 5):  # GraphProto.initializer
        dims = list(_pb_iter_varints(init, 1))
        if dims == [512, 512]:
            raw = _pb_find(init, 9)  # TensorProto.raw_data
            if raw is not None and len(raw) == 512 * 512 * 4:
                return np.frombuffer(raw, np.float32).reshape(512, 512).copy()
            floats = _pb_find(init, 4)  # packed float_data
            if floats is not None and len(floats) == 512 * 512 * 4:
                return np.frombuffer(floats, np.float32).reshape(512, 512).copy()
    raise RuntimeError("emap initializer not found")


def _pb_read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _pb_fields(buf: bytes):
    pos, n = 0, len(buf)
    while pos < n:
        key, pos = _pb_read_varint(buf, pos)
        field, wt = key >> 3, key & 7
        if wt == 0:
            val, pos = _pb_read_varint(buf, pos)
            yield field, wt, val
        elif wt == 2:
            ln, pos = _pb_read_varint(buf, pos)
            yield field, wt, buf[pos:pos + ln]
            pos += ln
        elif wt == 1:
            yield field, wt, buf[pos:pos + 8]
            pos += 8
        elif wt == 5:
            yield field, wt, buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError("unsupported wire type")


def _pb_find(buf: bytes, field: int):
    for f, wt, v in _pb_fields(buf):
        if f == field and wt == 2:
            return v
    return None


def _pb_iter(buf: bytes, field: int):
    for f, wt, v in _pb_fields(buf):
        if f == field and wt == 2:
            yield v


def _pb_iter_varints(buf: bytes, field: int):
    for f, wt, v in _pb_fields(buf):
        if f == field and wt == 0:
            yield v
        elif f == field and wt == 2:  # packed
            pos = 0
            while pos < len(v):
                val, pos = _pb_read_varint(v, pos)
                yield val


class Swapper:
    """Swap identity into an aligned 128x128 crop."""

    def __init__(self, session: SessionInfo, emap: np.ndarray):
        self.sess = session
        self.emap = emap.astype(np.float32)
        self.latent: np.ndarray | None = None

    def set_identity(self, embedding: np.ndarray) -> None:
        latent = normalize(embedding.astype(np.float32)).reshape(1, -1) @ self.emap
        self.latent = (latent / np.linalg.norm(latent)).astype(np.float32)

    def __call__(self, crop128_bgr: np.ndarray) -> np.ndarray:
        if self.latent is None:
            raise RuntimeError("identity not set")
        blob = cv2.dnn.blobFromImage(crop128_bgr, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        out = self.sess.run({"target": blob, "source": self.latent})[0]
        img = np.clip(out[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img[:, :, ::-1])


def match_color(src_bgr: np.ndarray, ref_bgr: np.ndarray, mask: np.ndarray | None = None, amount: float = 1.0) -> np.ndarray:
    """Match LAB mean/std of ``src`` to ``ref`` (optionally within ``mask``)."""
    if amount <= 0:
        return src_bgr
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is not None and (mask > 0.5).sum() > 16:
        w = mask > 0.5
        s_mean, s_std = src[w].mean(0), src[w].std(0) + 1e-3
        r_mean, r_std = ref[w].mean(0), ref[w].std(0) + 1e-3
    else:
        s_mean, s_std = src.reshape(-1, 3).mean(0), src.reshape(-1, 3).std(0) + 1e-3
        r_mean, r_std = ref.reshape(-1, 3).mean(0), ref.reshape(-1, 3).std(0) + 1e-3
    out = (src - s_mean) / s_std * r_std + r_mean
    out = src + (out - src) * np.float32(amount)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
