"""Build an identity profile (.npz) from one or more reference images of the avatar.

    avatarcam-prepare --out lulu.npz ref1.png ref2.png ...

Every usable face is embedded with ArcFace; embeddings are averaged with weights from
detector confidence and face size, which gives a more stable identity than a single image.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..face.align import warp_crop
from ..face.detect import Detector, pick_face
from ..face.embed import Embedder, average_embedding, normalize, save_profile
from ..models.registry import ensure_model
from ..runtime import SessionFactory

log = logging.getLogger("avatarcam.prepare")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="avatarcam-prepare")
    p.add_argument("images", nargs="+", help="reference images of the avatar")
    p.add_argument("--out", required=True, help="output profile path (.npz)")
    p.add_argument("--config", default=None)
    p.add_argument("--min-size", type=int, default=80, help="ignore faces smaller than this many pixels")
    p.add_argument("--providers", default="cpu", help="execution providers for this one-off job")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = Config.load(args.config)
    models_dir = cfg.models_dir()
    factory = SessionFactory([a.strip() for a in args.providers.split(",")])
    detector = Detector(factory.create("scrfd", ensure_model("scrfd", models_dir), static_shapes={"?": 320}), size=320, threshold=0.4)
    embedder = Embedder(factory.create("arcface", ensure_model("arcface", models_dir), static_shapes={"None": 1}))

    embeddings, weights, thumbs, used = [], [], [], []
    for path in args.images:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("%s: cannot read, skipped", path)
            continue
        face = pick_face(detector(img))
        if face is None:
            log.warning("%s: no face found, skipped", path)
            continue
        if face.size < args.min_size:
            log.warning("%s: face too small (%.0f px), skipped", path, face.size)
            continue
        emb = embedder.embed(img, face.kps)
        weight = face.score * min(1.0, face.size / 200.0)
        embeddings.append(emb)
        weights.append(weight)
        crop, _ = warp_crop(img, face.kps, "arcface_112_v2", 112)
        thumbs.append(crop)
        used.append(path)
        log.info("%s: face %.0f px, confidence %.2f, weight %.2f", path, face.size, face.score, weight)
    if not embeddings:
        raise SystemExit("no usable faces; supply clearer, larger reference images")

    mean = average_embedding(embeddings, weights)
    sims = [float(normalize(e) @ mean) for e in embeddings]
    for path, s in zip(used, sims):
        flag = "" if s >= 0.5 else "  <- looks like a different person or a poor angle; consider removing"
        log.info("%s: similarity to profile %.2f%s", path, s, flag)
    out = Path(args.out)
    if out.suffix != ".npz":
        out = out.with_suffix(".npz")
    save_profile(out, mean, {"sources": np.array(used), "similarity": np.array(sims, np.float32)}, np.stack(thumbs))
    log.info("profile written to %s from %d image(s)", out, len(used))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
