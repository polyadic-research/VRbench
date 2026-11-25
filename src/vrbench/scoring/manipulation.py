import numpy as np
from scipy.spatial.transform import Rotation
from .region import distance as geometry_distance
from .bindings import materialize_receiver
PROTOCOL="coupled_manipulation"
DISTANCE_SCALE=.05
ANGLE_SCALE=np.deg2rad(5)
ANGLE_TOLERANCE=1e-9

def trajectory(prediction, initial):
    waypoints = prediction["waypoints"]
    positions = np.asarray([initial["position"], *[w["tcp_position"] for w in waypoints]], float)
    quaternions = np.asarray([initial["quaternion"], *[w["tcp_quaternion_xyzw"] for w in waypoints]], float)
    events = ["none", *["none" if w["gripper_event"] is None else w["gripper_event"] for w in waypoints]]
    if (positions.shape != (len(events), 3) or quaternions.shape != (len(events), 4)
            or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
            or np.any(np.abs(np.linalg.norm(quaternions, axis=1)-1) > 1e-3)
            or any(event not in ("none", "close", "open", "hold") for event in events)):
        raise ValueError("Invalid TCP trajectory or gripper events")
    poses = np.tile(np.eye(4), (len(events), 1, 1))
    poses[:, :3, 3] = positions
    poses[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    return positions, poses, events

def grip_cycles(events):
    cycles, close = [], None
    for index, event in enumerate(events):
        if event == "close" and close is None:
            close = index
        elif event == "open" and close is not None:
            cycles.append((close, index))
            close = None
    if close is not None:
        cycles.append((close, None))
    return cycles

def distance(points, regions):
    return np.min([geometry_distance(np.atleast_2d(points), region) for region in regions], axis=0)

def stage(name, value, index, blocked=(), hard=None, scale=DISTANCE_SCALE):
    eligible = value is not None and not blocked
    hit = bool(eligible and (value <= 1e-9 if hard is None else hard))
    return dict(id=name, ordered_score=float(100*np.exp(-value/scale)) if eligible else 0.,
                hard_region_hit=hit, hit_index=index if hit else None,
                best_path_index=index, blocked_by=list(blocked),
                constraint_violation=value, constraint_scale=float(scale))

def dependencies(rows):
    return [row["id"] for row in rows if not row["hard_region_hit"]]

def finish(rows, **details):
    return dict(score_0_to_100=float(np.mean([row["ordered_score"] for row in rows])),
                milestones=rows, protocol=PROTOCOL, **details)

def object_pose(poses, close, index, initial_pose):
    return poses[index] @ np.linalg.inv(poses[close]) @ np.asarray(initial_pose)

def stack(positions, poses, events, definition,trace=None):
    cycles = grip_cycles(events)
    best = None
    for order in ((1, 2), (2, 1)):
        rows, placed, audit = [], {}, []
        for number, bowl in enumerate(order):
            obj = definition["objects"][str(bowl)]
            close, release = cycles[number] if number < len(cycles) else (None, None)
            from .interaction import cycle_value
            value,close = cycle_value(trace,obj["source_regions"],close,release,positions) if close is not None else (None,None)
            row = stage(f"bowl{bowl}_source", value, close, dependencies(rows))
            rows.append(row)
            target_pose = None
            if row["hard_region_hit"] and release is not None:
                body=np.array(obj["initial_body_pose"],copy=True)
                if trace:body[:3,3]+=trace.centering[id(obj["source_regions"]),close]
                target_pose = object_pose(poses, close, release, body)
            value = None
            if target_pose is not None:
                value = float(distance(target_pose[:3, 3], definition["sink_regions"])[0])
                if number == 1:
                    base = definition["objects"][str(order[0])]
                    bbox = placed[order[0]] @ np.linalg.inv(np.asarray(base["initial_body_pose"])) @ np.asarray(base["initial_bbox_pose"])
                    receiver = materialize_receiver(base["receiver_template"], bbox[:3, 3], bbox[:3, :3])
                    value = max(value, float(distance(target_pose[:3, 3], [receiver])[0]))
                audit.append(dict(object=f"receptacle{bowl}", close_index=close, release_index=release,
                                  placed_body_pose=target_pose.tolist()))
            row = stage(f"bowl{bowl}_in_sink" if number == 0 else f"bowl{bowl}_in_moved_bowl",
                        value, release, dependencies(rows))
            rows.append(row)
            if row["hard_region_hit"]:
                placed[bowl] = target_pose
        value = None
        if len(placed) == 2:
            separation = min(np.linalg.norm(positions[-1]-pose[:3, 3]) for pose in placed.values())
            value = float(max(0, definition["retreat_distance_m"]-separation))
        rows.append(stage("released_and_retreated", value, len(positions)-1, dependencies(rows),
                          hard=value is not None and separation > definition["retreat_distance_m"]))
        result = finish(rows, bowl_order=list(order), placement_anchors=audit,
                        ignored_additional_grip_cycles=max(0, len(cycles)-2))
        if best is None or result["score_0_to_100"] > best["score_0_to_100"]:
            best = result
    return best

def tilt(positions, poses, events, definition,trace=None):
    cycles = grip_cycles(events)
    close, release = cycles[0] if cycles else (None, None)
    from .interaction import cycle_value
    value,close=cycle_value(trace,definition["source_regions"],close,release,positions) if close is not None else (None,None)
    rows = [stage("pan_grasp", value, close)]
    end = release if release is not None else len(positions)-1
    indices = np.arange(close+1, end+1) if close is not None else np.array([], int)
    angles, modeled = np.array([]), None
    if len(indices):
        body=np.array(definition["initial_body_pose"],copy=True)
        if trace:body[:3,3]+=trace.centering[id(definition["source_regions"]),close]
        modeled = np.asarray([object_pose(poses, close, int(index), body) for index in indices])
        initial_rotation = np.asarray(definition["initial_body_pose"])[:3, :3]
        angles = Rotation.from_matrix(modeled[:, :3, :3] @ initial_rotation.T).magnitude()
    hit = np.flatnonzero(angles > definition["tilt_threshold_rad"] + ANGLE_TOLERANCE)
    choice = int(hit[0]) if len(hit) else int(np.argmax(angles)) if len(angles) else None
    value = float(max(0, definition["tilt_threshold_rad"]-angles[choice])) if choice is not None else None
    index = int(indices[choice]) if choice is not None else None
    rows.append(stage("held_pan_tilt", value, index, dependencies(rows), hard=bool(len(hit)), scale=ANGLE_SCALE))
    value, returned = None, None
    if rows[-1]["hard_region_hit"]:
        later = np.flatnonzero(indices > index)
        if len(later):
            distances = distance(modeled[later, :3, 3], [definition["return_region"]])
            hits = np.flatnonzero(distances <= 1e-9)
            best = int(hits[0]) if len(hits) else int(np.argmin(distances))
            value, returned = float(distances[best]), int(indices[later[best]])
    rows.append(stage("pan_back_on_original_burner", value, returned, dependencies(rows)))
    return finish(rows, close_index=close, release_index=release,
                  maximum_held_orientation_change_rad=float(angles.max()) if len(angles) else None,
                  tilt_duration_assessed=False, food_containment_assessed=False,
                  physical_stove_contact_assessed=False,
                  upright_return_required=False)
