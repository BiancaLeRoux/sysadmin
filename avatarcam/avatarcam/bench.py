"""Benchmark every model on every configured execution provider.

Usage: avatarcam-bench [--config ...] [--providers qnn-htp,cpu] [--iters 30] [--image face.jpg]

Prints a table of per-model latency for each provider and, when --image is given, an
end-to-end pipeline timing on that image. Writes bench.json next to the models.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np

from .config import Config
from .models.registry import ensure_model
from .runtime import SessionFactory, available_aliases, describe

log = logging.getLogger("avatarcam.bench")

# key -> (registry key, feeds builder, static_shapes)
def _feeds(key: str):
    if key == "scrfd":
        return {"input": np.random.rand(1, 3, 320, 320).astype(np.float32)}, {"?": 320}
    if key == "arcface":
        return {"input": np.random.rand(1, 3, 112, 112).astype(np.float32)}, {"None": 1}
    if key == "inswapper":
        return {"target": np.random.rand(1, 3, 128, 128).astype(np.float32), "source": np.random.rand(1, 512).astype(np.float32)}, {}
    if key == "gpen_bfr_256":
        return {"input": np.random.rand(1, 3, 256, 256).astype(np.float32)}, {}
    if key == "gfpgan_1.4":
        return {"input": np.random.rand(1, 3, 512, 512).astype(np.float32)}, {}
    if key == "bisenet":
        return {"input": np.random.rand(1, 3, 512, 512).astype(np.float32)}, {"batch_size": 1}
    if key == "xseg":
        return {"input": np.random.rand(1, 256, 256, 3).astype(np.float32)}, {"unk__1495": 1}
    if key == "rvm":
        z = np.zeros((1, 1, 1, 1), np.float32)
        return {"src": np.random.rand(1, 3, 216, 384).astype(np.float32), "r1i": z, "r2i": z, "r3i": z, "r4i": z,
                "downsample_ratio": np.array([1.0], np.float32)}, {}
    if key == "yolov8n_pose":
        return {"images": np.random.rand(1, 3, 320, 320).astype(np.float32)}, {}
    raise KeyError(key)


BENCH_MODELS = ["scrfd", "arcface", "inswapper", "gpen_bfr_256", "gfpgan_1.4", "xseg", "bisenet", "rvm", "yolov8n_pose"]


def bench_model(factory: SessionFactory, key: str, path: Path, alias: str, iters: int) -> dict:
    feeds, static = _feeds(key)
    t0 = time.perf_counter()
    info = factory.create(key, path, static_shapes=static, providers=[alias])
    load = (time.perf_counter() - t0) * 1000
    if info.alias != alias:
        return {"skipped": f"fell back to {info.alias}"}
    # RVM keeps recurrent state; feed zeros each time which is fine for timing.
    info.run(feeds)  # warm-up
    info.run(feeds)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        info.run(feeds)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"load_ms": round(load), "median_ms": round(times[len(times) // 2], 2), "p90_ms": round(times[int(len(times) * 0.9)], 2)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="avatarcam-bench")
    p.add_argument("--config", default=None)
    p.add_argument("--providers", default=None, help="comma list; default: all available from config order")
    p.add_argument("--models", default=None, help="comma list of model keys to benchmark")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--image", default=None, help="run the full pipeline on this image for end-to-end timing")
    p.add_argument("--json", default=None, help="where to write results (default: <models_dir>/bench.json)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = Config.load(args.config)
    models_dir = cfg.models_dir()
    factory = SessionFactory(list(cfg.runtime.providers), cache_dir=models_dir / "qnn-cache" if cfg.runtime.context_cache else None)
    aliases = [a.strip() for a in args.providers.split(",")] if args.providers else [a for a in cfg.runtime.providers if a in available_aliases()]
    keys = [k.strip() for k in args.models.split(",")] if args.models else BENCH_MODELS
    print(describe())
    print(f"{platform.platform()}  {platform.processor() or platform.machine()}")
    results: dict = {"platform": platform.platform(), "providers": aliases, "models": {}}
    header = f"{'model':16s}" + "".join(f"{a:>22s}" for a in aliases)
    print(header)
    print("-" * len(header))
    for key in keys:
        path = ensure_model(key, models_dir)
        row = f"{key:16s}"
        results["models"][key] = {}
        for alias in aliases:
            try:
                r = bench_model(factory, key, path, alias, args.iters)
            except Exception as exc:  # noqa: BLE001
                r = {"error": str(exc)[:80]}
            results["models"][key][alias] = r
            if "median_ms" in r:
                row += f"{r['median_ms']:>12.1f} ms p90 {r['p90_ms']:>5.0f}"
            elif "skipped" in r:
                row += f"{'(fallback)':>22s}"
            else:
                row += f"{'error':>22s}"
        print(row)

    if args.image:
        import cv2

        from .pipeline import Pipeline

        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
        pipe = Pipeline(cfg, factory)
        pipe.identity_from_image(img)
        pipe.process(img.copy())
        t = time.perf_counter()
        n = 20
        for _ in range(n):
            _, stats = pipe.process(img.copy(), 1 / 30)
        per = (time.perf_counter() - t) / n * 1000
        print(f"\nend-to-end on {img.shape[1]}x{img.shape[0]}: {per:.1f} ms/frame ({1000 / per:.1f} fps)  stages: {{{', '.join(f'{k}: {v:.0f}' for k, v in stats['stages'].items())}}}")
        results["end_to_end_ms"] = per
        results["stages"] = stats["stages"]

    out = Path(args.json) if args.json else models_dir / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
