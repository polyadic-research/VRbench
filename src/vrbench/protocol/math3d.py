import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def pose(position, quaternion_xyzw):
    p = np.asarray(position, dtype=float)
    q = np.asarray(quaternion_xyzw, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(np.r_[p, q])):
        raise ValueError("Pose must contain finite position[3] and quaternion_xyzw[4]")
    if abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError("Quaternion must be unit length; predictions are not silently normalized")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q).as_matrix()
    t[:3, 3] = p
    return t


def pose_dict(t):
    return {"position": t[:3, 3].tolist(), "quaternion_xyzw": Rotation.from_matrix(t[:3, :3]).as_quat().tolist()}


def inverse(t):
    r = np.eye(4)
    r[:3, :3] = t[:3, :3].T
    r[:3, 3] = -r[:3, :3] @ t[:3, 3]
    return r


def angle(a, b):
    return float(np.linalg.norm(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec()))


def lerp_pose(a, b, u):
    t = np.eye(4)
    t[:3, 3] = a[:3, 3] * (1-u) + b[:3, 3] * u
    t[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))([u]).as_matrix()[0]
    return t


def joint_motion(kind, axis, anchor, delta):
    """Joint axis/anchor expressed in the baseline parent frame."""
    axis, anchor = np.asarray(axis), np.asarray(anchor)
    t = np.eye(4)
    if kind == "slide":
        t[:3, 3] = axis * delta
    elif kind == "hinge":
        t[:3, :3] = Rotation.from_rotvec(axis * delta).as_matrix()
        t[:3, 3] = anchor - t[:3, :3] @ anchor
    else:
        raise ValueError(f"Unsupported moving joint {kind}")
    return t
