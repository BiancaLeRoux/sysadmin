"""Configuration loading, validation, and live updates.

The TOML file is loaded into a plain nested dict and wrapped in :class:`Config`, which
exposes attribute access (``cfg.face.swap_strength``) while keeping the dict semantics
that the control panel and the preset writer need.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Iterator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


class Section:
    """Attribute view over a nested dict. Missing keys raise AttributeError."""

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            return Section(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = value

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)


class Config(Section):
    """Top-level configuration with load/save helpers."""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        super().__init__(data)
        object.__setattr__(self, "path", path)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(p, "rb") as f:
            data = tomllib.load(f)
        defaults = _load_defaults()
        merged = _deep_merge(defaults, data)
        validate(merged)
        return cls(merged, p)

    def save(self, path: str | os.PathLike | None = None) -> Path:
        p = Path(path) if path else self.path or DEFAULT_CONFIG_PATH
        p.write_text(dump_toml(self._data), encoding="utf-8")
        return p

    def models_dir(self) -> Path:
        configured = self._data.get("runtime", {}).get("models_dir") or ""
        if configured:
            return Path(configured).expanduser()
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            return base / "AvatarCam" / "models"
        return Path.home() / ".cache" / "avatarcam" / "models"


def _load_defaults() -> dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_RANGES: dict[str, tuple[float, float]] = {
    "face.swap_strength": (0.0, 1.0),
    "face.mask_blur": (0.0, 1.0),
    "face.smoothing": (0.0, 1.0),
    "face.detect_threshold": (0.05, 0.99),
    "enhance.strength": (0.0, 1.0),
    "body.waist": (-1.0, 1.0),
    "body.hips": (-1.0, 1.0),
    "body.shoulders": (-1.0, 1.0),
    "body.thighs": (-1.0, 1.0),
    "body.arms": (-1.0, 1.0),
    "body.height": (-0.3, 0.3),
    "body.warp_smoothing": (0.0, 1.0),
    "body.hair_strength": (0.0, 1.0),
    "body.skin_smooth": (0.0, 1.0),
}

_CHOICES: dict[str, tuple[str, ...]] = {
    "face.mask": ("box", "region"),
    "enhance.model": ("gpen_bfr_256", "gfpgan_1.4"),
    "body.background": ("none", "blur", "color", "image"),
}

KNOWN_PROVIDERS = ("qnn-htp", "qnn-gpu", "dml", "cuda", "coreml", "openvino", "cpu")


def validate(data: dict[str, Any]) -> None:
    """Raise ValueError on out-of-range or unknown values."""
    for key, (lo, hi) in _RANGES.items():
        sec, name = key.split(".")
        v = data[sec][name]
        if not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}, got {v!r}")
    for key, choices in _CHOICES.items():
        sec, name = key.split(".")
        v = data[sec][name]
        if v not in choices:
            raise ValueError(f"{key} must be one of {choices}, got {v!r}")
    for p in data["runtime"]["providers"]:
        if p not in KNOWN_PROVIDERS:
            raise ValueError(f"runtime.providers: unknown provider {p!r}; known: {KNOWN_PROVIDERS}")
    for ints in ("face.detect_every", "enhance.every", "body.pose_every", "face.fade_frames"):
        sec, name = ints.split(".")
        if int(data[sec][name]) < 1:
            raise ValueError(f"{ints} must be >= 1")
    if data["face"]["keep_mouth"] and data["face"]["mask"] != "region":
        data["face"]["mask"] = "region"


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    raise TypeError(f"cannot serialise {type(value).__name__}")


def dump_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer for the flat-sections layout used by config.toml."""
    lines: list[str] = []

    def emit(prefix: str, section: dict[str, Any]) -> None:
        scalars = {k: v for k, v in section.items() if not isinstance(v, dict)}
        subs = {k: v for k, v in section.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {_fmt(v)}")
        if prefix:
            lines.append("")
        for k, v in subs.items():
            emit(f"{prefix}.{k}" if prefix else k, v)

    emit("", data)
    return "\n".join(lines).rstrip() + "\n"
