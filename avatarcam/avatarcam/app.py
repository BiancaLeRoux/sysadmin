"""AvatarCam entry point: camera -> pipeline -> preview / virtual camera / video file."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import Config
from .overlay import draw_hud
from .pipeline import Pipeline
from .runtime import describe

log = logging.getLogger("avatarcam")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="avatarcam", description="Real-time face and body persona camera")
    p.add_argument("--config", default=None, help="path to config.toml (default: bundled)")
    p.add_argument("--profile", default=None, help="identity profile .npz from avatarcam-prepare")
    p.add_argument("--source-image", default=None, help="quick start: take the identity from this image instead of a profile")
    p.add_argument("--camera", type=int, default=None, help="camera index override")
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--input", default=None, help="process a video/image file instead of the camera (headless)")
    p.add_argument("--output", default=None, help="write processed video/image here (with --input)")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--no-panel", action="store_true", help="do not open the control panel window")
    p.add_argument("--body", action="store_true", help="enable the body stage regardless of config")
    p.add_argument("--providers", default=None, help="comma list overriding runtime.providers, e.g. cpu")
    p.add_argument("--licences", action="store_true", help="print model licence summary and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"avatarcam {__version__}")
    return p.parse_args(argv)


def build(args) -> tuple[Config, Pipeline]:
    cfg = Config.load(args.config)
    if args.profile:
        cfg.avatar.profile = args.profile
    if args.camera is not None:
        cfg.camera.index = int(args.camera)
    if args.body:
        cfg.body.enabled = True
    if args.providers:
        cfg.runtime.providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    log.info("%s", describe())
    pipe = Pipeline(cfg)
    if args.source_image:
        img = cv2.imread(args.source_image, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cannot read {args.source_image}")
        pipe.identity_from_image(img)
    if pipe.identity is None and cfg.face.enabled:
        log.warning("no identity loaded: run avatarcam-prepare or pass --profile / --source-image; face swap is inactive")
    return cfg, pipe


def run_file(args, cfg: Config, pipe: Pipeline) -> int:
    src = Path(args.input)
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is not None:
        # Run the tracker to steady state so fade-in and smoothing do not hide the result.
        for _ in range(int(cfg.face.fade_frames) + 2):
            out, stats = pipe.process(img.copy(), 1.0 / 30)
        if args.output:
            cv2.imwrite(args.output, out)
        log.info("image processed in %.0f ms; stages %s", stats["total_ms"], stats["stages"])
        return 0
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    n = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, stats = pipe.process(frame, 1.0 / fps)
        if args.output:
            if writer is None:
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
            writer.write(out)
        n += 1
        if n % 30 == 0:
            log.info("frame %d: %.0f ms %s", n, stats["total_ms"], {k: round(v) for k, v in stats["stages"].items()})
        if args.frames and n >= args.frames:
            break
    if writer is not None:
        writer.release()
    el = time.perf_counter() - t0
    log.info("%d frames in %.1fs (%.1f fps)", n, el, n / el if el else 0)
    return 0


class LiveRunner:
    """Camera capture thread -> processing thread -> preview (main thread) + virtual camera."""

    def __init__(self, args, cfg: Config, pipe: Pipeline):
        from .capture import Camera
        from .output import VirtualCamera

        self.args, self.cfg, self.pipe = args, cfg, pipe
        self.camera = Camera(int(cfg.camera.index), int(cfg.camera.width), int(cfg.camera.height), int(cfg.camera.fps), bool(cfg.camera.mirror)).open()
        self.vcam = VirtualCamera(self.camera.width, self.camera.height, int(cfg.output.virtual_camera_fps)) if cfg.output.virtual_camera else None
        self._lock = threading.Lock()
        self._out: np.ndarray | None = None
        self._stats: dict = {}
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="pipeline", daemon=True)

    def _loop(self) -> None:
        last_seq = -1
        last_t = time.perf_counter()
        while not self._stop.is_set():
            frame, stamp, seq = self.camera.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq
            now = time.perf_counter()
            dt, last_t = now - last_t, now
            try:
                out, stats = self.pipe.process(frame.copy(), dt)
            except Exception:  # noqa: BLE001
                log.exception("pipeline error; passing frame through")
                out, stats = frame, {}
            stats["latency_ms"] = (time.perf_counter() - stamp) * 1000
            if self.vcam is not None:
                self.vcam.send(out)
            with self._lock:
                self._out, self._stats = out, stats

    def run(self) -> int:
        from .output import PreviewWindow, is_headless

        self._worker.start()
        if self.args.no_preview or is_headless():
            log.info("running without preview; press Ctrl+C to stop")
            try:
                while True:
                    time.sleep(0.5)
                    if self._stats:
                        log.info("%.1f fps, %.0f ms latency", self._stats.get("fps", 0), self._stats.get("latency_ms", 0))
            except KeyboardInterrupt:
                pass
            finally:
                self.stop()
            return 0

        win = PreviewWindow(scale=float(self.cfg.output.preview_scale), capture_mode=bool(self.cfg.output.capture_mode), on_key=self._on_key)
        self.win = win
        panel = None
        if not self.args.no_panel:
            from .ui import ControlPanel

            panel = ControlPanel(win.root, self.cfg, self._on_change, self._save)
        self.panel = panel
        interval = 1.0 / max(1, int(self.cfg.camera.fps))
        try:
            while win.pump():
                t = time.perf_counter()
                with self._lock:
                    out, stats = self._out, self._stats
                if out is not None:
                    if self.cfg.output.hud:
                        out = draw_hud(out.copy(), stats)
                    win.show(out)
                spare = interval - (time.perf_counter() - t)
                if spare > 0:
                    time.sleep(spare)
        finally:
            self.stop()
        return 0

    def _on_change(self, sec: str, key: str) -> None:
        if (sec, key) == ("output", "capture_mode") and hasattr(self, "win"):
            self.win.set_capture_mode(bool(self.cfg.output.capture_mode))
        if sec in ("face", "enhance", "body") and key in ("enabled", "mask", "hair_recolor") or (sec, key) == ("enhance", "model"):
            try:
                self.pipe.load_models()
            except Exception:  # noqa: BLE001
                log.exception("could not load models for the new setting")
        self.pipe.apply_config()

    def _save(self) -> None:
        p = self.cfg.save()
        log.info("preset saved to %s", p)

    def _on_key(self, key: str) -> None:
        cfg = self.cfg
        if key in ("q", "escape"):
            self.win.close()
        elif key == "c":
            cfg.output.capture_mode = not cfg.output.capture_mode
            self.win.set_capture_mode(bool(cfg.output.capture_mode))
        elif key == "h":
            cfg.output.hud = not cfg.output.hud
        elif key == "d":
            cfg.output.disclosure = not cfg.output.disclosure
        elif key == "space":
            cfg.face.swap_strength = 0.0 if cfg.face.swap_strength > 0 else 1.0
        elif key == "s":
            self._save()
        if getattr(self, "panel", None) is not None:
            self.panel.refresh()

    def stop(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)
        self.camera.close()
        if self.vcam is not None:
            self.vcam.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.licences:
        from .models.registry import licence_report

        print(licence_report())
        return 0
    if args.list_cameras:
        from .capture import list_cameras

        for idx, desc in list_cameras():
            print(f"camera {idx}: {desc}")
        return 0
    cfg, pipe = build(args)
    if args.input:
        return run_file(args, cfg, pipe)
    return LiveRunner(args, cfg, pipe).run()


if __name__ == "__main__":
    sys.exit(main())
