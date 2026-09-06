import numpy as np

from avatarcam.body.pose import Pose
from avatarcam.body.reshape import BodyReshaper, ControlPoints, ReshapeParams, build_control_points, mls_similarity_backward


def test_mls_pure_translation_is_exact():
    p = np.array([[100, 100], [500, 100], [100, 300], [500, 300]], np.float32)
    q = p + np.array([10, -5], np.float32)
    g = np.array([[300, 200], [120, 110]], np.float32)
    out = mls_similarity_backward(p, q, g)
    assert np.allclose(out, g + [10, -5], atol=1e-2)


def test_mls_control_points_are_interpolated():
    p = np.array([[0, 0], [100, 0], [0, 100], [100, 100], [50, 50]], np.float32)
    q = p.copy()
    q[4] = [60, 50]
    out = mls_similarity_backward(p, q, p)
    assert np.allclose(out, q, atol=1e-3)


def test_reshaper_identity_when_no_points():
    r = BodyReshaper((90, 160), smoothing=0.0)
    r.update(None)
    assert not r.active
    img = np.random.default_rng(1).integers(0, 255, (90, 160, 3), np.uint8)
    assert (r.apply(img) == img).all()


def test_reshaper_moves_content_and_caps_displacement():
    r = BodyReshaper((180, 320), smoothing=0.0, max_shift=0.05)
    cp = ControlPoints()
    for x in np.linspace(0, 319, 5):
        cp.anchor((x, 0))
        cp.anchor((x, 179))
    cp.add((160, 90), (300, 90))  # asks for 140 px, cap is 16 px
    r.update(cp)
    disp = np.abs(r.field - r.identity).max()
    assert r.active and disp <= 16.01


def _pose(shift=0.0):
    kps = np.zeros((17, 3), np.float32)
    pts = {0: (160, 40), 5: (130, 80), 6: (190, 80), 7: (115, 130), 8: (205, 130), 9: (110, 175), 10: (210, 175),
           11: (140, 170), 12: (180, 170), 13: (140, 250), 14: (180, 250), 15: (140, 330), 16: (180, 330)}
    for i, (x, y) in pts.items():
        kps[i] = (x + shift, y, 0.9)
    return Pose(np.array([100, 20, 220, 340], np.float32), kps, 0.9)


def test_control_points_slim_waist_moves_edges_inward():
    alpha = np.zeros((360, 320), np.float32)
    alpha[60:340, 110:210] = 1.0
    cp = build_control_points(_pose(), alpha, ReshapeParams(waist=-1.0), (360, 320))
    moved = np.abs(cp.dst - cp.src).sum(1) > 0.5
    assert moved.sum() >= 2
    # moved points end closer to the torso axis (x=160) than they started
    for s, d in zip(cp.src[moved], cp.dst[moved]):
        assert abs(d[0] - 160) < abs(s[0] - 160)


def test_control_points_identity_when_no_change():
    cp = build_control_points(_pose(), None, ReshapeParams(), (360, 320))
    assert np.allclose(cp.src, cp.dst)
