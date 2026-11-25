from dataclasses import dataclass
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from ..protocol.math3d import joint_motion, angle


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    position_error: float
    orientation_error: float
    iterations: int


class Robot:
    """Frozen base, exported body tree; obstacles are absent from this class."""
    def __init__(self, spec):
        spec = copy.deepcopy(spec)
        self.spec = spec
        self.nodes = spec["nodes"]
        self.arm = spec["arm_joints"]
        self.q0 = np.array([spec["joints"][j]["q0"] for j in self.arm], float)
        self.lower = np.array([spec["joints"][j]["range"][0] for j in self.arm])
        self.upper = np.array([spec["joints"][j]["range"][1] for j in self.arm])
        self.tcp_local = np.asarray(spec["tcp_local"], float)

    def fk(self, q, gripper="open"):
        values = dict(zip(self.arm, q))
        values.update(self.spec["gripper_values"][gripper])
        out = {}
        for n in self.nodes:
            parent = np.eye(4) if n["parent"] is None else out[n["parent"]]
            local = np.asarray(n["baseline_local"], float)
            for j in n.get("joints", []):
                v = values.get(j, self.spec["joints"][j]["q0"])
                js = self.spec["joints"][j]
                local = joint_motion(js["type"], js["axis_parent"], js["anchor_parent"], v-js["q0"]) @ local
            out[n["name"]] = parent @ local
        return out, out[self.spec["tcp_body"]] @ self.tcp_local

    def jacobian(self, q):
        frames, tcp = self.fk(q)
        jac = np.zeros((6, len(q)))
        for k, name in enumerate(self.arm):
            j = self.spec["joints"][name]
            parent = np.eye(4) if j["parent"] is None else frames[j["parent"]]
            axis = parent[:3, :3] @ np.asarray(j["axis_parent"])
            anchor = parent[:3, :3] @ np.asarray(j["anchor_parent"]) + parent[:3, 3]
            if j["type"] == "hinge":
                jac[:3, k] = np.cross(axis, tcp[:3, 3]-anchor)
                jac[3:, k] = axis
            else:
                jac[:3, k] = axis
        return jac

    def solve(self, target, seed, config):
        q = np.asarray(seed, float).copy()
        max_it = config.get("ik_max_iterations", 100)
        damping = config.get("ik_damping", 0.02)
        orient_weight = config.get("ik_orientation_weight", 0.25)
        wp = config.get("ik_position_tolerance_m", 0.002)
        wr = config.get("ik_orientation_tolerance_rad", 0.03)
        for it in range(max_it+1):
            _, now = self.fk(q)
            ep = target[:3, 3] - now[:3, 3]
            er = Rotation.from_matrix(target[:3, :3] @ now[:3, :3].T).as_rotvec()
            pe, re = np.linalg.norm(ep), np.linalg.norm(er)
            if pe <= wp and re <= wr:
                return IKResult(True, q, float(pe), float(re), it)
            if it == max_it:
                break
            jac = self.jacobian(q)
            jac[3:] *= orient_weight
            err = np.r_[ep, er * orient_weight]
            reg = config.get("ik_posture_weight", 1e-5)
            dq = np.linalg.solve(jac.T @ jac + (damping*damping+reg)*np.eye(len(q)), jac.T @ err + reg*(self.q0-q))
            step = config.get("ik_iteration_joint_step", 0.1)
            dq *= min(1, step/max(np.max(abs(dq)), 1e-12))
            q = np.clip(q+dq, self.lower, self.upper)
        return IKResult(False, q, float(pe), float(re), max_it)

    def interpolate_gripper(self, a, b, u):
        key = "_interpolated"
        self.spec["gripper_values"][key] = {
            j: float((1-u)*v + u*self.spec["gripper_values"][b][j])
            for j, v in self.spec["gripper_values"][a].items()
        }
        return key
