"""Text overlays: disclosure label (burned into output) and HUD (preview only)."""

from __future__ import annotations

import cv2
import numpy as np


def draw_label(frame: np.ndarray, text: str, corner: str = "bottom-left", scale: float = 0.6) -> np.ndarray:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    pad = 6
    if corner == "bottom-left":
        x, y = 10, h - 10
    elif corner == "top-left":
        x, y = 10, 10 + th
    elif corner == "top-right":
        x, y = w - tw - 10, 10 + th
    else:
        x, y = w - tw - 10, h - 10
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, stats: dict) -> np.ndarray:
    lines = [f"{stats.get('fps', 0):.1f} fps  latency {stats.get('latency_ms', 0):.0f} ms"]
    stages = stats.get("stages", {})
    if stages:
        lines.append("  ".join(f"{k} {v:.0f}" for k, v in stages.items()))
    if stats.get("face") is not None:
        lines.append(f"face {stats['face']:.2f}  presence {stats.get('presence', 0):.2f}")
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)
        y += 20
    return frame
