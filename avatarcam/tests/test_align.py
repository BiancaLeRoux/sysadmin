import cv2
import numpy as np

from avatarcam.face.align import box_mask, crop_bbox_in_frame, estimate_affine, paste_back, template_points, warp_crop


def _synthetic_kps(cx, cy, size):
    return template_points("arcface_112_v2", size) - size / 2 + np.array([cx, cy], np.float32)


def test_affine_maps_landmarks_onto_template():
    kps = _synthetic_kps(300, 200, 150)
    m = estimate_affine(kps, "arcface_112_v2", 128)
    proj = cv2.transform(kps[None], m)[0]
    assert np.allclose(proj, template_points("arcface_112_v2", 128), atol=0.5)


def test_warp_crop_and_paste_back_roundtrip():
    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(frame, (300, 200), 60, (0, 255, 0), -1)
    kps = _synthetic_kps(300, 200, 150)
    crop, m = warp_crop(frame, kps, "arcface_112_v2", 128)
    assert crop.shape == (128, 128, 3)
    red = np.zeros_like(crop)
    red[:, :, 2] = 255
    out = paste_back(frame.copy(), red, np.ones((128, 128), np.float32), m, 1.0)
    x0, y0, x1, y1 = crop_bbox_in_frame(m, 128, frame.shape)
    assert x0 < 300 < x1 and y0 < 200 < y1
    assert out[200, 300, 2] == 255 and out[200, 300, 1] == 0
    # far away pixels untouched
    assert (out[10, 10] == frame[10, 10]).all()


def test_paste_back_strength_blends():
    frame = np.zeros((200, 200, 3), np.uint8)
    kps = _synthetic_kps(100, 100, 80)
    crop, m = warp_crop(frame, kps, "arcface_112_v2", 128)
    white = np.full_like(crop, 255)
    out = paste_back(frame.copy(), white, np.ones((128, 128), np.float32), m, 0.5)
    assert 120 <= out[100, 100, 0] <= 135


def test_box_mask_shape_and_feather():
    m = box_mask(128, blur=0.3, padding=(0.1, 0.0, 0.0, 0.0))
    assert m.shape == (128, 128) and m.dtype == np.float32
    assert m.max() <= 1.0 and m.min() >= 0.0
    assert m[64, 64] > 0.95
    assert m[2, 64] < 0.2  # padded top edge
    assert m[64, 2] < m[64, 20]  # feathered edge
