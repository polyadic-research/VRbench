"""Coordinates and bookkeeping. No expert future state is used."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from ..protocol.math3d import inverse, angle, lerp_pose
from ..protocol.validation import validate_prediction


def dense_reference(robot, prediction, cfg):
    ws = validate_prediction(prediction, 96)
    refs = [robot.fk(robot.q0)[1]]
    segments = []
    for wi, (target, _, _) in enumerate(ws):
        start = refs[-1]
        intervals = max(1, math.ceil(np.linalg.norm(target[:3, 3]-start[:3, 3])/cfg['reference_step_m']),
                        math.ceil(angle(start, target)/cfg['reference_step_rad']))
        first = len(refs)-1
        refs.extend(lerp_pose(start, target, float(u)) for u in np.linspace(0, 1, intervals+1)[1:])
        segments.append(dict(waypoint_index=wi, start_node=first, end_node=len(refs)-1,
                             gripper_event=ws[wi][1], contact_event=ws[wi][2]))
        if len(refs) > cfg['max_reference_nodes']:
            raise ValueError('Reference-node budget exceeded; do not truncate the input')
    return refs, segments


def interpolation_samples(robot, qa, qb, ra, rb, cfg, depth=0):
    """Subdivide the GIVEN joint interpolation; never solve/repair IK."""
    qm = (qa+qb)/2
    aa, ab, am = (robot.fk(q)[1] for q in (qa, qb, qm))
    expected = lerp_pose(aa, ab, .5)
    need = (np.max(abs(qb-qa)) > cfg['max_joint_step_rad'] or
            np.linalg.norm(am[:3, 3]-expected[:3, 3]) > cfg['midpoint_position_tolerance_m'] or
            angle(am, expected) > cfg['midpoint_rotation_tolerance_rad'])
    if need:
        if depth >= cfg['max_depth']:
            raise ValueError('Joint replay subdivision exhausted; remaining geometry unknown')
        rm = lerp_pose(ra, rb, .5)
        return (interpolation_samples(robot, qa, qm, ra, rm, cfg, depth+1) +
                interpolation_samples(robot, qm, qb, rm, rb, cfg, depth+1))
    return [(qm, lerp_pose(ra, rb, .5)), (qb, rb)]
