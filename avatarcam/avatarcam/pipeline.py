"""Per-frame orchestration of the body and face stages.

Order per frame: body (matting, pose, reshape, background) first, then face (detect or
track, swap, mask, paste, enhance), then overlays. Doing the body warp first means the
face stage sees the final geometry, so a height stretch or shoulder change never leaves
the swapped face misaligned.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .body.matte import Matter
from .body.pose import PoseEstimator, pick_pose
from .body.reshape import BodyReshaper, ReshapeParams, build_control_points
from .body.restyle import Background, composite, recolor_hair, smooth_skin
from .config import Config
from .face.align import box_mask, invert_affine, paste_back, warp_crop
from .face.detect import Detector, pick_face
from .face.embed import Embedder, load_profile
from .face.enhance import ENHANCER_SIZES, Enhancer
from .face.parse import HAIR, MOUTH_REGION, FaceParser, OcclusionMasker
from .face.swap import Swapper, load_emap, match_color
from .face.track import FaceTracker
from .models.registry import ensure_model
from .overlay import draw_label
from .runtime import SessionFactory

log = logging.getLogger(__name__)


class Stopwatch:
    def __init__(self):
        self.times: dict[str, float] = {}
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.times[name] = self.times.get(name, 0.0) + (now - self._t) * 1000
        self._t = now


class Pipeline:
    def __init__(self, cfg: Config, factory: SessionFactory | None = None, progress=None):
        self.cfg = cfg
        self.models_dir = cfg.models_dir()
        self.factory = factory or SessionFactory(
            list(cfg.runtime.providers),
            overrides=cfg.runtime.get("overrides", {}),
            cache_dir=self.models_dir / "qnn-cache" if cfg.runtime.context_cache else None,
            threads=int(cfg.runtime.threads),
        )
        self._progress = progress
        self.frame_index = 0
        self.identity: np.ndarray | None = None
        self.last_stats: dict = {}
        self._fps_hist: list[float] = []

        # Face stage -------------------------------------------------------------
        self.detector: Detector | None = None
        self.embedder: Embedder | None = None
        self.swapper: Swapper | None = None
        self.enhancer: Enhancer | None = None
        self.parser: FaceParser | None = None
        self.occluder: OcclusionMasker | None = None
        self.tracker = FaceTracker(cfg.face.detect_every, cfg.face.smoothing, cfg.camera.fps, cfg.face.fade_frames)
        self._mask_cache: dict[tuple, np.ndarray] = {}
        self._enh_cache: np.ndarray | None = None
        self._parse_cache: tuple[int, np.ndarray] | None = None

        # Body stage -------------------------------------------------------------
        self.matter: Matter | None = None
        self.pose_est: PoseEstimator | None = None
        self.reshaper: BodyReshaper | None = None
        self.background: Background | None = None
        self.last_pose = None
        self._hair_mask: np.ndarray | None = None
        self._bg_key = None

        self.load_models()
        if cfg.avatar.profile:
            self.load_identity(Path(cfg.avatar.profile))

    # -- setup --------------------------------------------------------------------

    def _model(self, key: str) -> Path:
        return ensure_model(key, self.models_dir, self._progress)

    def load_models(self) -> None:
        cfg = self.cfg
        if cfg.face.enabled:
            self.detector = Detector(
                self.factory.create("scrfd", self._model("scrfd"), static_shapes={"?": int(cfg.face.detect_size)}),
                size=int(cfg.face.detect_size), threshold=float(cfg.face.detect_threshold),
            )
            self.embedder = Embedder(self.factory.create("arcface", self._model("arcface"), static_shapes={"None": 1}))
            sw_path = self._model("inswapper")
            self.swapper = Swapper(self.factory.create("inswapper", sw_path), load_emap(sw_path))
            if cfg.face.occlusion:
                self.occluder = OcclusionMasker(self.factory.create("xseg", self._model("xseg"), static_shapes={"unk__1495": 1}))
            if cfg.face.mask == "region" or cfg.body.hair_recolor:
                self.parser = FaceParser(self.factory.create("bisenet", self._model("bisenet"), static_shapes={"batch_size": 1}))
        if cfg.enhance.enabled and cfg.face.enabled:
            name = cfg.enhance.model
            self.enhancer = Enhancer(self.factory.create(name, self._model(name)), name)
        if cfg.body.enabled:
            if cfg.body.matting or cfg.body.background != "none":
                self.matter = Matter(self.factory.create("rvm", self._model("rvm"), providers=cfg.runtime.get("overrides", {}).get("rvm", ["cpu"])),
                                     input_width=int(cfg.body.matting_size))
            if cfg.body.pose:
                self.pose_est = PoseEstimator(self.factory.create("yolov8n_pose", self._model("yolov8n_pose")), size=320)

    def load_identity(self, profile: Path) -> None:
        prof = load_profile(profile)
        self.set_identity(prof["embedding"])
        log.info("identity profile loaded from %s", profile)

    def set_identity(self, embedding: np.ndarray) -> None:
        self.identity = embedding
        if self.swapper is not None:
            self.swapper.set_identity(embedding)

    def identity_from_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Convenience for tests and quick starts: embed the largest face in an image."""
        faces = self.detector(image_bgr)
        face = pick_face(faces)
        if face is None:
            raise ValueError("no face found in reference image")
        emb = self.embedder.embed(image_bgr, face.kps)
        self.set_identity(emb)
        return emb

    def apply_config(self) -> None:
        """Push live-changeable settings into stateful components."""
        cfg = self.cfg
        self.tracker.detect_every = max(1, int(cfg.face.detect_every))
        self.tracker.fade_frames = max(1, int(cfg.face.fade_frames))
        self.tracker.set_smoothing(float(cfg.face.smoothing), float(cfg.camera.fps))
        if self.detector is not None:
            self.detector.threshold = float(cfg.face.detect_threshold)
        if self.reshaper is not None:
            self.reshaper.smoothing = float(cfg.body.warp_smoothing)
        self._bg_key = None

    # -- per frame ----------------------------------------------------------------

    def process(self, frame: np.ndarray, dt: float | None = None, hud: bool = False) -> tuple[np.ndarray, dict]:
        t_start = time.perf_counter()
        sw = Stopwatch()
        cfg = self.cfg
        self.frame_index += 1
        out = frame if frame.flags.writeable else frame.copy()
        stats: dict = {"face": None, "presence": 0.0}

        alpha = None
        if cfg.body.enabled:
            out, alpha = self._body_stage(out, sw)
        if cfg.face.enabled and self.swapper is not None and self.swapper.latent is not None:
            out = self._face_stage(out, dt, sw, stats)
        if cfg.body.enabled and alpha is not None:
            out = self._body_post(out, alpha, sw)
        if cfg.output.disclosure:
            draw_label(out, str(cfg.output.disclosure_text))
        sw.lap("overlay")

        total = (time.perf_counter() - t_start) * 1000
        self._fps_hist.append(time.perf_counter())
        self._fps_hist = [t for t in self._fps_hist if t > time.perf_counter() - 2.0]
        fps = len(self._fps_hist) / 2.0 if len(self._fps_hist) > 1 else 0.0
        stats.update({"total_ms": total, "fps": fps, "stages": {k: v for k, v in sw.times.items() if v >= 0.5}})
        self.last_stats = stats
        return out, stats

    # -- face ---------------------------------------------------------------------

    def _mask(self, size: int) -> np.ndarray:
        cfg = self.cfg.face
        key = (size, round(float(cfg.mask_blur), 3), tuple(float(p) for p in cfg.mask_padding))
        m = self._mask_cache.get(key)
        if m is None:
            m = box_mask(size, float(cfg.mask_blur), key[2])
            self._mask_cache[key] = m
        return m

    def _face_stage(self, out: np.ndarray, dt, sw: Stopwatch, stats: dict) -> np.ndarray:
        cfg = self.cfg
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        ran = self.tracker.should_detect()
        pick = None
        if ran:
            faces = self.detector(out)
            pick = pick_face(faces, self.tracker.face)
        sw.lap("detect")
        face = self.tracker.update(gray, pick, ran, dt)
        presence = self.tracker.presence
        stats["presence"] = presence
        if face is None or presence <= 0.0:
            return out
        stats["face"] = face.score

        # Swap on a 128 arcface crop.
        crop, m = warp_crop(out, face.kps, "arcface_112_v2", 128)
        swapped = self.swapper(crop)
        sw.lap("swap")
        mask = self._mask(128).copy()
        if cfg.face.mask == "region" and self.parser is not None:
            labels = self.parser.labels(crop)
            mask *= FaceParser.region_mask(labels, blur=4)
            if cfg.face.keep_mouth:
                mask *= 1.0 - FaceParser.region_mask(labels, MOUTH_REGION, blur=3)
            sw.lap("parse")
        if self.occluder is not None:
            mask *= self.occluder(crop)
            sw.lap("occlusion")
        if cfg.face.color_transfer:
            swapped = match_color(swapped, crop, mask)
        strength = float(cfg.face.swap_strength) * presence
        paste_back(out, swapped, mask, m, strength)
        sw.lap("paste")

        # Enhance on an FFHQ-aligned crop of the swapped result.
        if self.enhancer is not None and float(cfg.enhance.strength) > 0:
            size = self.enhancer.size
            every = max(1, int(cfg.enhance.every))
            ecrop, em = warp_crop(out, face.kps, "ffhq_512", size)
            if self._enh_cache is None or self.frame_index % every == 0:
                self._enh_cache = self.enhancer(ecrop)
            enhanced = self._enh_cache
            if cfg.face.color_transfer:
                enhanced = match_color(enhanced, ecrop, None, 0.5)
            emask = self._mask(size)
            paste_back(out, enhanced, emask, em, float(cfg.enhance.strength) * presence)
            sw.lap("enhance")

        if cfg.body.enabled and cfg.body.hair_recolor and self.parser is not None:
            if self._parse_cache is None or self.frame_index % max(1, int(cfg.body.pose_every)) == 0:
                hcrop, hm = warp_crop(out, face.kps, "ffhq_512", 512)
                labels = self.parser.labels(hcrop)
                hair = FaceParser.region_mask(labels, (HAIR,), blur=6)
                inv = invert_affine(hm)
                self._hair_mask = cv2.warpAffine(hair, inv, (out.shape[1], out.shape[0]), flags=cv2.INTER_LINEAR)
                self._parse_cache = (self.frame_index, self._hair_mask)
                sw.lap("hair-parse")
        return out

    # -- body ---------------------------------------------------------------------

    def _body_stage(self, out: np.ndarray, sw: Stopwatch):
        cfg = self.cfg.body
        alpha = None
        if self.matter is not None:
            alpha = self.matter(out)
            sw.lap("matte")
        params = ReshapeParams(float(cfg.waist), float(cfg.hips), float(cfg.shoulders), float(cfg.thighs), float(cfg.arms), float(cfg.height))
        if self.pose_est is not None and not params.is_identity():
            if self.last_pose is None or self.frame_index % max(1, int(cfg.pose_every)) == 0:
                face = self.tracker.face
                poses = self.pose_est(out)
                self.last_pose = pick_pose(poses, face.center if face is not None else None)
                sw.lap("pose")
            if self.reshaper is None or (self.reshaper.h, self.reshaper.w) != out.shape[:2]:
                self.reshaper = BodyReshaper(out.shape, smoothing=float(cfg.warp_smoothing))
            face = self.tracker.face
            cp = None
            if self.last_pose is not None:
                cp = build_control_points(self.last_pose, alpha, params, out.shape, face.bbox if face is not None else None)
            self.reshaper.update(cp)
            if self.reshaper.active:
                out = self.reshaper.apply(out)
                if alpha is not None:
                    alpha = self.reshaper.apply(alpha)
            sw.lap("reshape")
        elif self.reshaper is not None:
            self.reshaper.reset()
        return out, alpha

    def _body_post(self, out: np.ndarray, alpha: np.ndarray, sw: Stopwatch) -> np.ndarray:
        cfg = self.cfg.body
        if cfg.hair_recolor and self._hair_mask is not None:
            out = recolor_hair(out, self._hair_mask, cfg.hair_color, float(cfg.hair_strength))
            sw.lap("hair")
        if float(cfg.skin_smooth) > 0:
            out = smooth_skin(out, alpha, float(cfg.skin_smooth))
            sw.lap("skin")
        key = (cfg.background, tuple(cfg.background_color), cfg.background_image, int(cfg.background_blur))
        if key != self._bg_key:
            self.background = Background(cfg.background, cfg.background_color, cfg.background_image, int(cfg.background_blur))
            self._bg_key = key
        bg = self.background.render(out) if self.background is not None else None
        if bg is not None:
            out = composite(out, bg, alpha)
            sw.lap("background")
        return out
