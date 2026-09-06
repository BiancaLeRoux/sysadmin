"""Model catalogue: download URLs, sizes, sha256, and licence notes.

All files are fetched from GitHub release assets (no Hugging Face dependency) and verified
against the pinned sha256 before use. Files that ship inside the repository are copied
from ``assets/models``.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "models"
_FF = "https://github.com/facefusion/facefusion-assets/releases/download/"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str | None
    sha256: str
    size: int
    licence: str
    note: str = ""

    @property
    def bundled(self) -> bool:
        return self.url is None


MODELS: dict[str, ModelSpec] = {
    "scrfd": ModelSpec(
        "scrfd", "scrfd_2.5g.onnx", _FF + "models-3.0.0/scrfd_2.5g.onnx",
        "2c07342347cef21a619c49dd5664fb8c09570ae9eda5bff3e385c11eafc45ada", 3295067,
        "InsightFace (non-commercial research)", "face detector + 5 landmarks",
    ),
    "arcface": ModelSpec(
        "arcface", "arcface_w600k_r50.onnx", _FF + "models-3.0.0/arcface_w600k_r50.onnx",
        "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1", 174388474,
        "InsightFace (non-commercial research)", "identity embedding",
    ),
    "inswapper": ModelSpec(
        "inswapper", "inswapper_128_fp16.onnx", _FF + "models-3.0.0/inswapper_128_fp16.onnx",
        "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420", 277680829,
        "InsightFace (non-commercial research)", "face swapper",
    ),
    "gpen_bfr_256": ModelSpec(
        "gpen_bfr_256", "gpen_bfr_256.onnx", _FF + "models-3.0.0/gpen_bfr_256.onnx",
        "bad8bf0426873828df2dbf4e3b3d9ababba9da7965b8b72426569486f7ae5c25", 75792988,
        "GPEN (Apache-2.0 code, weights by Tencent/Alibaba)", "fast face enhancer",
    ),
    "gfpgan_1.4": ModelSpec(
        "gfpgan_1.4", "gfpgan_1.4.onnx", _FF + "models-3.0.0/gfpgan_1.4.onnx",
        "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385", 340299087,
        "GFPGAN (Apache-2.0 with NVIDIA StyleGAN2 terms)", "quality face enhancer",
    ),
    "bisenet": ModelSpec(
        "bisenet", "bisenet_resnet_34.onnx", _FF + "models-3.0.0/bisenet_resnet_34.onnx",
        "4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96", 93632546,
        "face-parsing.PyTorch (MIT)", "face parsing (19 classes)",
    ),
    "xseg": ModelSpec(
        "xseg", "xseg_1.onnx", _FF + "models-3.1.0/xseg_1.onnx",
        "c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3", 70324286,
        "DeepFaceLab XSeg (GPL-3.0)", "face occlusion mask",
    ),
    "rvm": ModelSpec(
        "rvm", "rvm_mobilenetv3_fp32.onnx",
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828", 14975696,
        "RobustVideoMatting (GPL-3.0)", "person matting",
    ),
    "yolov8n_pose": ModelSpec(
        "yolov8n_pose", "yolov8n-pose-320.onnx", None,
        "", 0,  # filled in by assets/models/MANIFEST at load time
        "Ultralytics YOLOv8 (AGPL-3.0)", "17-keypoint body pose, 320x320",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bundled_spec(spec: ModelSpec) -> ModelSpec:
    """Resolve sha256/size for models shipped in assets/models via MANIFEST."""
    manifest = ASSETS_DIR / "MANIFEST"
    if not manifest.exists():
        return spec
    for line in manifest.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == spec.filename:
            return ModelSpec(spec.key, spec.filename, None, parts[0], int(parts[1]), spec.licence, spec.note)
    return spec


def ensure_model(key: str, models_dir: Path, progress=None) -> Path:
    """Return the local path of model ``key``, downloading and verifying if needed."""
    spec = MODELS[key]
    if spec.bundled:
        spec = _bundled_spec(spec)
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / spec.filename
    if dest.exists() and (not spec.sha256 or _sha256(dest) == spec.sha256):
        return dest
    if spec.bundled:
        src = ASSETS_DIR / spec.filename
        if not src.exists():
            raise FileNotFoundError(f"bundled model missing: {src}")
        shutil.copyfile(src, dest)
    else:
        _download(spec, dest, progress)
    if spec.sha256:
        got = _sha256(dest)
        if got != spec.sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"{spec.filename}: sha256 mismatch ({got} != {spec.sha256})")
    return dest


def _download(spec: ModelSpec, dest: Path, progress=None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s (%.0f MB)", spec.filename, spec.size / 1e6)
    req = urllib.request.Request(spec.url, headers={"User-Agent": "avatarcam/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or spec.size or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(spec.filename, done, total)
            elif total and sys.stderr.isatty():
                sys.stderr.write(f"\r  {spec.filename}: {done * 100 // total:3d}%")
    if sys.stderr.isatty() and not progress:
        sys.stderr.write("\n")
    tmp.replace(dest)


def ensure_all(keys, models_dir: Path, progress=None) -> dict[str, Path]:
    return {k: ensure_model(k, models_dir, progress) for k in keys}


def licence_report() -> str:
    lines = ["Model licences (you are responsible for complying with each):"]
    for spec in MODELS.values():
        lines.append(f"  {spec.filename:32s} {spec.licence}")
    return "\n".join(lines)
