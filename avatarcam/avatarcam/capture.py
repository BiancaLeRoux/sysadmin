"""Threaded camera capture that always hands out the newest frame."""

from __future__ import annotations

import logging
import sys
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _backend() -> int:
    if sys.platform == "win32":
        return cv2.CAP_MSMF
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def list_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _backend())
        ok = cap.isOpened()
        if ok:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, f"{w}x{h}"))
        cap.release()
    return found


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30, mirror: bool = False):
        self.index, self.width, self.height, self.fps, self.mirror = index, width, height, fps, mirror
        self.cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> "Camera":
        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.index} could not be opened")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        log.info("camera %d opened at %dx%d", self.index, self.width, self.height)
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self.cap is not None
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._stamp = time.perf_counter()
                self._seq += 1

    def latest(self, wait: float = 0.5) -> tuple[np.ndarray | None, float, int]:
        deadline = time.perf_counter() + wait
        while True:
            with self._lock:
                if self._frame is not None:
                    return self._frame, self._stamp, self._seq
            if time.perf_counter() > deadline:
                return None, 0.0, self._seq
            time.sleep(0.002)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
