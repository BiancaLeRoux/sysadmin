# Baseline: Deep-Live-Cam on this PC

Before investing in the native pipeline, it is worth 20 minutes to see how the best
existing face-swap tool behaves on a Snapdragon X under Windows' x64 emulation. It gives
a reference for "what does DirectML-under-emulation deliver" and a sanity check for
identity quality with the same `inswapper_128` model AvatarCam uses.

Deep-Live-Cam is AGPL-3.0 and has no body features; this is a measurement, not a
component of AvatarCam.

## Steps

1. Install **x64** Python 3.11 or 3.12 from python.org (the default installer, not ARM64).
   It runs under the Prism emulator.
2. Clone <https://github.com/hacksider/Deep-Live-Cam> and follow its manual install:
   create a venv with the x64 Python, `pip install -r requirements.txt`, download
   `inswapper_128_fp16.onnx` and the GFPGAN model into `models/`.
3. Replace `onnxruntime`/`onnxruntime-gpu` in the venv with DirectML:

   ```powershell
   pip uninstall -y onnxruntime onnxruntime-gpu
   pip install onnxruntime-directml
   ```

4. Launch with `python run.py --execution-provider dml`, pick a source face, click
   *Live*, and watch the fps counter with the enhancer off, then on. Also try
   `--execution-provider cpu` for comparison.

## Record

| Setting | fps | Notes |
|---|---|---|
| dml, enhancer off | | |
| dml, enhancer on | | |
| cpu, enhancer off | | |

## Interpret

- ≥ 15 fps with `dml`: DirectML under emulation is a viable fallback for AvatarCam too
  (`requirements-x64.txt`).
- Native AvatarCam on the NPU (`run_bench.bat`) should beat the emulated `cpu` row
  comfortably; if it does not, something is wrong with the QNN setup (check that
  `run_bench` prints `qnn-htp` among available providers and that the inswapper row is not
  marked `(fallback)`).
