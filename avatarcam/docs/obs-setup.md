# OBS Studio setup: streaming and video calls

AvatarCam produces frames; OBS Studio distributes them. This keeps one integration for
every platform: OBS streams to Twitch, YouTube, Kick, TikTok Live (RTMP), and its virtual
camera feeds Zoom, Teams, Google Meet, and Discord.

## 1. Install the right OBS

- Snapdragon / Windows on ARM: OBS Studio **31.1 or newer**, the **Windows ARM64** build
  (the file name ends in `Windows-arm64`). The default x64 installer runs under emulation
  and its virtual camera does not reach native ARM64 apps.
- x64 PC: the normal installer.

ARM64 limitations as of OBS 31.1: no hardware encoders (x264 software encoding only),
x64 plugins do not load, and the virtual camera must be enabled manually once.

## 2. Capture AvatarCam

1. Start AvatarCam (`run.bat --profile me.npz`).
2. Press `c` in the preview (or tick *Capture mode* in the control panel). The window
   becomes borderless and sized exactly to the camera frame.
3. In OBS: *Sources* → `+` → **Window Capture** → pick the `AvatarCam` window. Set
   *Capture Method* to *Windows 10 (1903 and up)* if the picture is black.
4. Right-click the source → *Transform* → *Fit to screen*.

Keep the AvatarCam preview window unobstructed; Window Capture copies what is drawn.
The HUD is drawn only in the preview, so turn it off (`h`) before capturing.

## 3. Video calls (virtual camera)

1. OBS → *Tools* → *Virtual Camera* (or the *Start Virtual Camera* button).
2. On Windows ARM, first go to *Settings* → *Video* (or *Advanced*, depending on the
   build) and enable the virtual camera for **native ARM64 applications**. This disables
   it for emulated x64 apps, so use the ARM64 builds of Zoom, Teams, Discord, and Chrome/Edge
   (for Meet). All four ship native ARM64 versions.
3. In the call app choose **OBS Virtual Camera** as the camera.

Alternative without OBS in the loop: set `output.virtual_camera = true` in `config.toml`.
This needs the `pyvirtualcam` package, which has no prebuilt Windows ARM64 wheel; it can be
built from source with Visual Studio's ARM64 tools, but the OBS route works without that.

## 4. Streaming

- *Settings* → *Stream*: pick the service and paste your stream key.
- *Settings* → *Output*: encoder **x264**, preset `veryfast`, 720p at 30 fps, 3500 to
  4500 kbps. The Snapdragon X's 12 cores handle 720p30 x264 comfortably while AvatarCam
  runs on the NPU; 1080p is possible but leaves less headroom for the pipeline.
- Add your microphone as an audio source; AvatarCam does not touch audio.
- Every major platform asks that realistic synthetic or altered video be labelled.
  AvatarCam's `d` key burns a small "AI-altered video" label into the frame; the platforms
  also have their own disclosure checkboxes when you go live.

## 5. Latency

Camera → AvatarCam → Window Capture → OBS encode adds roughly 100 to 200 ms before the
network. For calls, that is on par with a normal webcam through OBS. If it drifts higher,
lower `camera.width/height` to 960x540, raise `enhance.every` to 2 or 3, or switch the
enhancer to `gpen_bfr_256`.
