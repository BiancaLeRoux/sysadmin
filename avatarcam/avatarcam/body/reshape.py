"""Body reshaping with Moving Least Squares (similarity) deformation.

Control points come from pose keypoints and the silhouette (matte). The warp field is
solved on a coarse grid, upsampled to frame size, smoothed over time, and applied with
``cv2.remap`` to the frame and the alpha matte together.

The deformation is expressed as a backward map: for every output pixel we ask where in
the input it comes from, so control points are given as (target -> source) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .pose import (L_ANK, L_EAR, L_ELB, L_EYE, L_HIP, L_KNE, L_SHO, L_WRI, NOSE, R_ANK, R_EAR, R_ELB, R_EYE, R_HIP, R_KNE, R_SHO, R_WRI, Pose)


@dataclass
class ReshapeParams:
    waist: float = 0.0
    hips: float = 0.0
    shoulders: float = 0.0
    thighs: float = 0.0
    arms: float = 0.0
    height: float = 0.0

    def is_identity(self) -> bool:
        return all(abs(v) < 1e-3 for v in (self.waist, self.hips, self.shoulders, self.thighs, self.arms, self.height))


@dataclass
class ControlPoints:
    src: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    dst: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))

    def add(self, s, d) -> None:
        self.src = np.vstack([self.src, np.asarray(s, np.float32).reshape(1, 2)])
        self.dst = np.vstack([self.dst, np.asarray(d, np.float32).reshape(1, 2)])

    def anchor(self, p) -> None:
        self.add(p, p)


def _edges_along(alpha: np.ndarray, center, direction, max_len: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Walk from ``center`` along +/-``direction`` until the matte ends; return both edge points."""
    h, w = alpha.shape
    d = np.asarray(direction, np.float32)
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    d /= n
    pts = []
    for sign in (-1.0, 1.0):
        last = np.asarray(center, np.float32)
        for t in np.arange(0, max_len, 2.0):
            p = np.asarray(center, np.float32) + sign * t * d
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < w and 0 <= y < h) or alpha[y, x] < 0.5:
                break
            last = p
        pts.append(last)
    return pts[0], pts[1]


def build_control_points(pose: Pose, alpha: np.ndarray | None, params: ReshapeParams, frame_shape, face_bbox=None) -> ControlPoints:
    """Derive (source -> target) control pairs from pose + silhouette."""
    h, w = frame_shape[:2]
    cp = ControlPoints()

    # Frame border anchors keep the background still.
    for x in np.linspace(0, w - 1, 7):
        cp.anchor((x, 0))
        cp.anchor((x, h - 1))
    for y in np.linspace(0, h - 1, 5)[1:-1]:
        cp.anchor((0, y))
        cp.anchor((w - 1, y))

    ls, rs, lh, rh = pose.pt(L_SHO), pose.pt(R_SHO), pose.pt(L_HIP), pose.pt(R_HIP)
    if ls is None or rs is None:
        return cp
    ls, rs = np.array(ls), np.array(rs)
    sho_c = (ls + rs) / 2
    sho_w = float(np.linalg.norm(ls - rs))
    if lh is not None and rh is not None:
        lh, rh = np.array(lh), np.array(rh)
        hip_c = (lh + rh) / 2
    else:
        hip_c = sho_c + np.array([0, sho_w * 1.3])
        lh, rh = hip_c + np.array([sho_w * 0.4, 0]), hip_c - np.array([sho_w * 0.4, 0])
    axis = hip_c - sho_c
    axis_len = float(np.linalg.norm(axis)) or 1.0
    axis_u = axis / axis_len
    perp = np.array([-axis_u[1], axis_u[0]])

    # Height: everything above the floor line moves up by ``height`` * distance to floor.
    la, ra = pose.pt(L_ANK), pose.pt(R_ANK)
    floor_y = max(la[1] if la else 0.0, ra[1] if ra else 0.0) or float(hip_c[1] + axis_len)
    stretch = params.height if abs(params.height) > 1e-3 else 0.0

    def lift(p):
        p = np.asarray(p, np.float32)
        return p if stretch == 0.0 else np.array([p[0], p[1] - (floor_y - p[1]) * stretch], np.float32)

    # Keep the face rigid (it still rides the height stretch).
    if face_bbox is not None:
        x0, y0, x1, y1 = face_bbox
        for p in ((x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)):
            cp.add(p, lift(p))
    # Head keypoints ride along rigidly too, whether or not the face stage supplied a box.
    for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR):
        p = pose.pt(i)
        if p is not None:
            cp.add(p, lift(p))

    def slim_pair(center, amount: float, fallback_half: float, reach: float):
        """Move the silhouette edges at ``center`` toward the axis by ``amount``."""
        edges = _edges_along(alpha, center, perp, reach) if alpha is not None else None
        if edges is None or np.linalg.norm(edges[0] - edges[1]) < 4:
            edges = (center - perp * fallback_half, center + perp * fallback_half)
        for e in edges:
            off = e - center
            cp.add(e, lift(center + off * (1.0 + amount)))

    # Shoulders, waist, hips, thighs.
    slim_pair(sho_c, params.shoulders * 0.25, sho_w / 2, sho_w * 1.5)
    waist_c = sho_c + axis_u * axis_len * 0.62
    slim_pair(waist_c, params.waist * 0.35, sho_w * 0.4, sho_w * 1.5)
    slim_pair(hip_c, params.hips * 0.3, sho_w * 0.45, sho_w * 1.5)
    lk, rk = pose.pt(L_KNE), pose.pt(R_KNE)
    for hip, knee in ((lh, lk), (rh, rk)):
        if knee is None:
            continue
        knee = np.array(knee)
        mid = (np.array(hip) + knee) / 2
        leg_u = knee - np.array(hip)
        leg_n = np.linalg.norm(leg_u) or 1.0
        leg_perp = np.array([-leg_u[1], leg_u[0]]) / leg_n
        edges = _edges_along(alpha, mid, leg_perp, sho_w) if alpha is not None else None
        if edges is None:
            edges = (mid - leg_perp * sho_w * 0.2, mid + leg_perp * sho_w * 0.2)
        for e in edges:
            cp.add(e, lift(mid + (e - mid) * (1.0 + params.thighs * 0.35)))
    # Upper arms.
    for sho, elb in ((ls, pose.pt(L_ELB)), (rs, pose.pt(R_ELB))):
        if elb is None:
            continue
        elb = np.array(elb)
        mid = (sho + elb) / 2
        arm_u = elb - sho
        arm_n = np.linalg.norm(arm_u) or 1.0
        arm_perp = np.array([-arm_u[1], arm_u[0]]) / arm_n
        edges = _edges_along(alpha, mid, arm_perp, sho_w * 0.6) if alpha is not None else None
        if edges is None:
            edges = (mid - arm_perp * sho_w * 0.12, mid + arm_perp * sho_w * 0.12)
        for e in edges:
            cp.add(e, lift(mid + (e - mid) * (1.0 + params.arms * 0.35)))
    # Ankles stay on the floor; wrists ride the stretch but do not otherwise drift.
    for i in (L_ANK, R_ANK):
        p = pose.pt(i)
        if p is not None:
            cp.anchor(p)
    for i in (L_WRI, R_WRI):
        p = pose.pt(i)
        if p is not None:
            cp.add(p, lift(p))
    if stretch:
        for p in (sho_c, hip_c):
            cp.add(p, lift(p))
    return cp


def mls_similarity_backward(dst_pts: np.ndarray, src_pts: np.ndarray, grid_xy: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """MLS similarity deformation (Schaefer 2006) evaluated at ``grid_xy`` (Nx2).

    ``dst_pts`` are where the points end up in the output, ``src_pts`` where they come
    from. Returns, for every grid point in output space, the sampling position in input
    space (Nx2). This is the backward map ``cv2.remap`` wants.
    """
    p = dst_pts.astype(np.float32)   # control positions in output space
    q = src_pts.astype(np.float32)   # corresponding positions in input space
    v = grid_xy.astype(np.float32)
    if p.shape[0] == 0:
        return v.copy()
    d2 = ((v[:, None, :] - p[None, :, :]) ** 2).sum(-1)  # N x M
    wgt = 1.0 / np.maximum(d2, 1e-6) ** alpha
    wsum = wgt.sum(1, keepdims=True)
    p_star = (wgt[:, :, None] * p[None]).sum(1) / wsum
    q_star = (wgt[:, :, None] * q[None]).sum(1) / wsum
    p_hat = p[None] - p_star[:, None, :]   # N x M x 2
    q_hat = q[None] - q_star[:, None, :]
    mu = (wgt * (p_hat ** 2).sum(-1)).sum(1)  # N
    vp = v - p_star  # N x 2
    # A_i = w_i * [p_hat_i; -p_hat_i^perp] [v-p*; -(v-p*)^perp]^T  -> accumulate q_hat_i^T A_i
    ph = p_hat
    ph_perp = np.stack([-ph[..., 1], ph[..., 0]], -1)
    vp_perp = np.stack([-vp[:, 1], vp[:, 0]], -1)
    # A_i = w_i [p_hat; -p_hat_perp] [vp; -vp_perp]^T  (Schaefer et al. 2006, eq. 7)
    a11 = (ph * vp[:, None, :]).sum(-1)            # p_hat . vp
    a12 = -(ph * vp_perp[:, None, :]).sum(-1)      # -p_hat . vp_perp
    a21 = -(ph_perp * vp[:, None, :]).sum(-1)      # -p_hat_perp . vp
    a22 = (ph_perp * vp_perp[:, None, :]).sum(-1)  # p_hat_perp . vp_perp
    fx = (wgt * (q_hat[..., 0] * a11 + q_hat[..., 1] * a21)).sum(1)
    fy = (wgt * (q_hat[..., 0] * a12 + q_hat[..., 1] * a22)).sum(1)
    out = np.stack([fx, fy], -1) / np.maximum(mu, 1e-6)[:, None] + q_star
    exact = d2.min(1) < 1e-6
    if exact.any():
        out[exact] = q[d2[exact].argmin(1)]
    return out.astype(np.float32)


class BodyReshaper:
    """Maintains a temporally smoothed remap field."""

    def __init__(self, frame_shape, grid: tuple[int, int] = (48, 27), smoothing: float = 0.7, max_shift: float = 0.12):
        self.h, self.w = frame_shape[:2]
        self.max_shift = float(max_shift)  # cap on displacement as a fraction of frame width
        self.gw, self.gh = grid
        self.smoothing = float(smoothing)
        xs = np.linspace(0, self.w - 1, self.gw, dtype=np.float32)
        ys = np.linspace(0, self.h - 1, self.gh, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        self.grid_xy = np.stack([gx.ravel(), gy.ravel()], -1)
        self.identity = self.grid_xy.reshape(self.gh, self.gw, 2).copy()
        self.field = self.identity.copy()
        self._maps: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        self.field = self.identity.copy()
        self._maps = None

    def update(self, cp: ControlPoints | None) -> None:
        if cp is None or cp.src.shape[0] == 0:
            target = self.identity
        else:
            target = mls_similarity_backward(cp.dst, cp.src, self.grid_xy).reshape(self.gh, self.gw, 2)
            limit = self.max_shift * self.w
            disp = np.clip(target - self.identity, -limit, limit)
            if not np.isfinite(disp).all():
                disp = np.nan_to_num(disp)
            target = self.identity + disp
        s = self.smoothing
        self.field = self.field * s + target * (1 - s)
        fx = cv2.resize(self.field[..., 0], (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        fy = cv2.resize(self.field[..., 1], (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        self._maps = (fx, fy)

    @property
    def active(self) -> bool:
        return self._maps is not None and float(np.abs(self.field - self.identity).max()) > 0.25

    def apply(self, img: np.ndarray) -> np.ndarray:
        if self._maps is None or not self.active:
            return img
        fx, fy = self._maps
        return cv2.remap(img, fx, fy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
