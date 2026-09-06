"""Preview window (tkinter) and optional virtual camera output."""

from __future__ import annotations

import logging
import sys

import numpy as np

log = logging.getLogger(__name__)


class VirtualCamera:
    """Thin wrapper over pyvirtualcam; silently disabled when it is not installed."""

    def __init__(self, width: int, height: int, fps: int):
        self.cam = None
        try:
            import pyvirtualcam  # type: ignore

            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
            log.info("virtual camera: %s", self.cam.device)
        except Exception as exc:  # noqa: BLE001
            log.warning("virtual camera unavailable: %s", exc)

    @property
    def active(self) -> bool:
        return self.cam is not None

    def send(self, frame_bgr: np.ndarray) -> None:
        if self.cam is None:
            return
        if frame_bgr.shape[1] != self.cam.width or frame_bgr.shape[0] != self.cam.height:
            import cv2

            frame_bgr = cv2.resize(frame_bgr, (self.cam.width, self.cam.height))
        self.cam.send(frame_bgr)

    def close(self) -> None:
        if self.cam is not None:
            self.cam.close()
            self.cam = None


class PreviewWindow:
    """Tk window that displays BGR frames. Create and use it from the main thread only."""

    def __init__(self, title: str = "AvatarCam", scale: float = 1.0, capture_mode: bool = False, on_key=None):
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="black")
        self.label = tk.Label(self.root, bg="black", bd=0, highlightthickness=0)
        self.label.pack()
        self.scale = float(scale)
        self.capture_mode = False
        self._photo = None
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        if on_key:
            self.root.bind("<Key>", lambda e: on_key(e.keysym.lower()))
        self.set_capture_mode(capture_mode)

    def set_capture_mode(self, enabled: bool) -> None:
        """Borderless, fixed-size window for OBS Window Capture."""
        self.capture_mode = bool(enabled)
        try:
            self.root.overrideredirect(self.capture_mode)
            self.root.attributes("-topmost", self.capture_mode)
        except Exception:  # noqa: BLE001 - some window managers reject these
            pass

    def show(self, frame_bgr: np.ndarray) -> None:
        if self._closed:
            return
        from PIL import Image, ImageTk

        rgb = frame_bgr[:, :, ::-1]
        img = Image.fromarray(np.ascontiguousarray(rgb))
        if self.scale != 1.0:
            img = img.resize((int(img.width * self.scale), int(img.height * self.scale)))
        self._photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo)

    def pump(self) -> bool:
        if self._closed:
            return False
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self._closed = True
        return not self._closed

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.root.destroy()
            except Exception:  # noqa: BLE001
                pass


def is_headless() -> bool:
    if sys.platform.startswith("linux"):
        import os

        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False
