import numpy as np

from avatarcam.face.detect import Face, nms, pick_face
from avatarcam.face.track import FaceTracker, OneEuro


def test_one_euro_smooths_noise_but_follows_motion():
    f = OneEuro(freq=30, min_cutoff=1.0, beta=0.01)
    rng = np.random.default_rng(0)
    xs = []
    for i in range(60):
        x = np.array([i * 2.0, 0.0]) + rng.normal(0, 3.0, 2)
        xs.append(f(x))
    xs = np.array(xs)
    # follows the ramp within a few frames of lag and reduces jitter
    assert abs(xs[-1, 0] - 118) < 12
    assert np.std(np.diff(xs[20:, 1])) < 2.0


def test_nms_removes_overlaps():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 300, 300]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms(boxes, scores, 0.4)
    assert keep == [0, 2]


def test_pick_face_prefers_previous_neighbour():
    big = Face(np.array([0, 0, 200, 200], np.float32), np.zeros((5, 2), np.float32), 0.9)
    small = Face(np.array([400, 400, 460, 460], np.float32), np.zeros((5, 2), np.float32), 0.8)
    assert pick_face([big, small]) is big
    prev = Face(np.array([395, 395, 465, 465], np.float32), np.zeros((5, 2), np.float32), 0.8)
    assert pick_face([big, small], prev) is small


def test_tracker_fades_in_and_out():
    tr = FaceTracker(detect_every=2, smoothing=0.5, fps=30, fade_frames=4)
    gray = np.zeros((100, 100), np.uint8)
    face = Face(np.array([20, 20, 60, 60], np.float32), np.array([[30, 30], [50, 30], [40, 40], [32, 50], [48, 50]], np.float32), 0.9)
    presences = []
    for i in range(6):
        ran = tr.should_detect()
        tr.update(gray, face if ran else None, ran, 1 / 30)
        presences.append(tr.presence)
    assert presences[0] == 0.25 and presences[3] == 1.0
    for i in range(6):
        ran = tr.should_detect()
        tr.update(gray, None, ran, 1 / 30)
    assert tr.face is None and tr.presence == 0.0
