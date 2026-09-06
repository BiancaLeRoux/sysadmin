"""Control panel: sliders and toggles bound to the live config."""

from __future__ import annotations

from typing import Callable

SLIDERS = [
    ("face", "swap_strength", "Face swap strength", 0.0, 1.0),
    ("face", "mask_blur", "Face mask feather", 0.0, 1.0),
    ("face", "smoothing", "Landmark smoothing", 0.0, 1.0),
    ("enhance", "strength", "Enhance strength", 0.0, 1.0),
    ("body", "waist", "Waist", -1.0, 1.0),
    ("body", "hips", "Hips", -1.0, 1.0),
    ("body", "shoulders", "Shoulders", -1.0, 1.0),
    ("body", "thighs", "Thighs", -1.0, 1.0),
    ("body", "arms", "Upper arms", -1.0, 1.0),
    ("body", "height", "Height", -0.3, 0.3),
    ("body", "warp_smoothing", "Warp smoothing", 0.0, 1.0),
    ("body", "skin_smooth", "Skin smoothing", 0.0, 1.0),
    ("body", "hair_strength", "Hair recolour strength", 0.0, 1.0),
]

TOGGLES = [
    ("face", "enabled", "Face swap"),
    ("face", "color_transfer", "Colour transfer"),
    ("face", "keep_mouth", "Keep my mouth (region mask)"),
    ("enhance", "enabled", "Face enhancer"),
    ("body", "enabled", "Body stage"),
    ("body", "hair_recolor", "Hair recolour"),
    ("output", "disclosure", "Disclosure label"),
    ("output", "hud", "HUD in preview"),
    ("output", "capture_mode", "Capture mode (borderless)"),
]

CHOICES = [
    ("body", "background", "Background", ("none", "blur", "color", "image")),
    ("face", "mask", "Face mask", ("box", "region")),
]


class ControlPanel:
    def __init__(self, root, cfg, on_change: Callable[[str, str], None], on_save: Callable[[], None]):
        import tkinter as tk
        from tkinter import ttk

        self.cfg = cfg
        self.on_change = on_change
        self.win = tk.Toplevel(root)
        self.win.title("AvatarCam controls")
        self.win.resizable(False, False)
        frame = ttk.Frame(self.win, padding=8)
        frame.grid()
        row = 0
        self.vars = {}
        for sec, key, label, lo, hi in SLIDERS:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            var = tk.DoubleVar(value=float(cfg[sec][key]))
            scale = ttk.Scale(frame, from_=lo, to=hi, variable=var, length=220,
                              command=lambda v, s=sec, k=key, vv=var: self._set(s, k, round(float(vv.get()), 3)))
            scale.grid(row=row, column=1, sticky="we", padx=4)
            self.vars[(sec, key)] = var
            row += 1
        for sec, key, label in TOGGLES:
            var = tk.BooleanVar(value=bool(cfg[sec][key]))
            ttk.Checkbutton(frame, text=label, variable=var,
                            command=lambda s=sec, k=key, vv=var: self._set(s, k, bool(vv.get()))).grid(row=row, column=0, columnspan=2, sticky="w")
            self.vars[(sec, key)] = var
            row += 1
        for sec, key, label, choices in CHOICES:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=str(cfg[sec][key]))
            box = ttk.Combobox(frame, textvariable=var, values=list(choices), state="readonly", width=12)
            box.grid(row=row, column=1, sticky="w", padx=4)
            box.bind("<<ComboboxSelected>>", lambda e, s=sec, k=key, vv=var: self._set(s, k, vv.get()))
            self.vars[(sec, key)] = var
            row += 1
        ttk.Button(frame, text="Save preset to config.toml", command=on_save).grid(row=row, column=0, columnspan=2, pady=(8, 0))

    def _set(self, sec: str, key: str, value) -> None:
        if self.cfg[sec][key] != value:
            self.cfg[sec][key] = value
            self.on_change(sec, key)

    def refresh(self) -> None:
        for (sec, key), var in self.vars.items():
            try:
                var.set(self.cfg[sec][key])
            except Exception:  # noqa: BLE001
                pass
