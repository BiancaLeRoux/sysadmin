# AvatarCam

Real-time face and body persona for your webcam. AvatarCam takes your live camera feed,
replaces your face with the identity from reference images you supply, reshapes and
restyles your body, and hands the result to OBS Studio, which streams it or exposes it as
a virtual camera for Zoom, Teams, Meet, or Discord.

It is built for a Windows-on-ARM PC (Snapdragon X, Qualcomm NPU) but runs anywhere ONNX
Runtime does: CPU, DirectML, CUDA, CoreML.

## What it does, honestly

| Stage | What you get | Ceiling |
|---|---|---|
| Face swap | Your reference identity on your live face, colour-matched, feathered, with hands/hair/glasses kept in front (XSeg occlusion) | 128 px swap resolution plus an enhancer. Looks like you with a different face; not a pixel-perfect film. |
| Face enhance | GPEN-256 (fast) or GFPGAN 1.4 (sharper) restores detail on the swapped face | Adds latency; the enhancer cadence is adjustable. |
| Body | Person matting (RVM), pose keypoints, then sliders for waist, hips, shoulders, thighs, upper arms, and height; hair recolour; skin smoothing; background blur/colour/image | This reshapes *your* filmed body. It cannot replace it with a generated one in real time; nothing consumer-grade can. |
| Output | Preview window (borderless "capture mode" for OBS), optional virtual camera, burned-in "AI-altered video" label toggle | Virtual camera needs pyvirtualcam and the OBS driver; on Windows ARM use OBS Window Capture. |

Video platforms (YouTube, TikTok, Twitch) require realistic synthetic or altered video to be
labelled. The disclosure toggle exists for that. What you do with it is your call.

## Install (Windows)

1. Install **Python 3.12**. On a Snapdragon PC choose the *ARM64* installer at
   <https://www.python.org/downloads/windows/> (not the default x64 one).
2. Install **OBS Studio 31.1 or newer**, the *Windows ARM64* build on a Snapdragon PC.
3. In PowerShell inside this folder:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\install.ps1
   ```

   This creates `.venv`, installs the native ARM64 stack (`onnxruntime` + `onnxruntime-qnn`
   for the NPU, the community `opencv-python-headless` ARM64 wheel), downloads about 1 GB
   of models with checksum verification, and writes `run.bat`, `run_bench.bat`, and
   `prepare_avatar.bat`.

4. Benchmark first, so you know what this PC can do:

   ```powershell
   .\run_bench.bat --image path\to\a\photo\of\you.jpg
   ```

   The table shows per-model latency on each accelerator (`qnn-htp` is the NPU, `qnn-gpu`
   the Adreno GPU, `cpu` the fallback). Edit `runtime.providers` in `config.toml` to match
   what worked best. Results also land in `%LOCALAPPDATA%\AvatarCam\models\bench.json`.

5. Build your identity profile from reference images of the avatar (several angles help):

   ```powershell
   .\prepare_avatar.bat --out me.npz ref1.png ref2.png ref3.png
   ```

   It reports each image's face size and its similarity to the averaged profile; drop any
   image it flags.

6. Run:

   ```powershell
   .\run.bat --profile me.npz
   ```

   A preview window and a control panel open. Keys in the preview: `c` capture mode,
   `h` HUD, `d` disclosure label, `space` swap on/off, `s` save preset, `q` quit.

### If the native ARM64 stack fails

`requirements-x64.txt` is the fallback: install x64 Python 3.12, run `install.ps1` from it,
and it will use DirectML under Windows' x64 emulation. Slower, but every wheel is official.

## OBS, calls, and streaming

See [docs/obs-setup.md](docs/obs-setup.md). Short version: run AvatarCam in capture mode,
add a *Window Capture* of the "AvatarCam" window in OBS, use OBS for streaming and turn on
OBS Virtual Camera for call apps. On Windows ARM you must enable the virtual camera for
native ARM64 apps once, in OBS settings.

## Configuration

Everything lives in `config.toml`; the control panel edits it live and "Save preset" writes
it back. Key knobs:

- `runtime.providers` accelerator order; `runtime.overrides.<model>` per-model override
  (the matting network defaults to CPU because its graph has dynamic shapes).
- `face.detect_every` detector cadence (landmarks are tracked with optical flow in between).
- `face.mask` `box` (fast) or `region` (BiSeNet parsing, needed for "keep my mouth").
- `enhance.every` run the enhancer every N frames if it is the bottleneck.
- `body.*` sliders, background, hair recolour, skin smoothing.
- `output.capture_mode`, `output.virtual_camera`, `output.disclosure`.

## Offline use and testing

```powershell
.\run.bat --source-image ref.png --input clip.mp4 --output clip_out.mp4
```

processes a file without a camera (also how the test suite exercises the pipeline).

```powershell
.venv\Scripts\python -m pytest
```

runs the unit tests. Set `AVATARCAM_MODELS_DIR` to the models folder and
`AVATARCAM_SAMPLES` to a folder with `zidane.jpg`/`bus.jpg` style samples to include the
model integration tests.

## Models and licences

Run `run.bat --licences`. The detector, embedder, and swapper are InsightFace models
published for non-commercial research; the pose model is Ultralytics YOLOv8 (AGPL-3.0,
bundled in `assets/models` with its notice); matting is RVM (GPL-3.0); XSeg is from
DeepFaceLab (GPL-3.0). AvatarCam's own code is MIT. If you monetise streams, check those
terms first.

## Layout

```
avatarcam/           package: pipeline, runtime, capture, output, ui, bench
avatarcam/face/      SCRFD detect, ArcFace embed, inswapper, enhancers, parsing/XSeg, tracker
avatarcam/body/      RVM matting, YOLOv8-pose, MLS reshape, background/hair/skin
avatarcam/tools/     prepare_avatar (identity profile builder)
assets/models/       bundled pose model + MANIFEST
docs/                OBS setup, Deep-Live-Cam baseline test
tests/               unit tests (no models needed) + gated integration tests
```
