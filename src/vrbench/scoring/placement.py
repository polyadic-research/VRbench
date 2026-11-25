import numpy as np
from . import manipulation as common
from .sequences import densify
PROTOCOL="coupled_placement"

def points_and_events(prediction, initial):
    if prediction.get("coordinate_frame") != "world" or prediction.get("position_unit") not in ("m", "meter"):
        raise ValueError("Expected world-frame meters")
    if not prediction.get("waypoints"):
        raise ValueError("Missing trajectory")
    if any(w.get("gripper_event") not in (None, "close", "open") for w in prediction["waypoints"]):
        raise ValueError("Invalid gripper event")
    return common.trajectory(prediction, initial)

def region_distance(point, regions):
    if not regions:
        raise ValueError("Empty region union")
    return float(common.distance(point, regions)[0])

def pick_cycle(cycles, positions, regions, after=-1, excluded=(), trace=None):
    eligible = [(close, release) for close, release in cycles if close > after and close not in excluded]
    distances = {close: region_distance(positions[close], regions) for close, _ in eligible}
    if trace:
        probes=[(trace.grip(regions,c,e),e) for c,e in eligible]
        eligible=[(k,e) for (_,k),e in probes]
        distances={k:d for (d,k),_ in probes}
    selected = next((cycle for cycle in eligible if distances[cycle[0]] <= 1e-9), None)
    if selected is None and eligible:
        selected = min(eligible, key=lambda cycle: distances[cycle[0]])
    return selected, distances.get(selected[0]) if selected is not None else None

def numeric_stage(name, value, index=None, blocked=(), scale=.05):
    return common.stage(name, value, index, blocked=blocked, scale=scale)

def finish(rows, **audit):
    return dict(protocol=PROTOCOL, score_0_to_100=float(np.mean([r["ordered_score"] for r in rows])),
                milestones=rows, index_policy="index0 is initial TCP; waypoint n has index n", **audit)

def container_transfer(positions, cycles, definition,trace=None):
    cycle, value = pick_cycle(cycles, positions, definition["container_source_regions"],trace=trace)
    close, release = cycle if cycle is not None else (None, None)
    rows = [numeric_stage("container_source_close", value, close, scale=definition["grasp_distance_scale_m"])]
    value = region_distance(positions[release], definition["counter_regions"]) if release is not None else None
    rows.append(numeric_stage("container_counter_open", value, release, common.dependencies(rows)))
    placed = rows[-1]["hard_region_hit"]
    meat_cycle, value = pick_cycle(cycles, positions, definition["meat_source_regions"],
                                  after=release,trace=trace) if placed else (None, None)
    meat_close, meat_release = meat_cycle if meat_cycle is not None else (None, None)
    rows.append(numeric_stage("meat_source_close", value, meat_close, common.dependencies(rows),
                              scale=definition["grasp_distance_scale_m"]))
    receiver_distance = None
    if placed and meat_release is not None:
        separation = float(np.linalg.norm(positions[meat_release] - positions[release]))
        receiver_distance = max(0., separation - definition["receiver_tcp_uncertainty_radius_m"])
        value = max(receiver_distance, region_distance(positions[meat_release], definition["counter_regions"]))
    else:
        value = None
    rows.append(numeric_stage("meat_receiver_open", value, meat_release, common.dependencies(rows)))
    return finish(rows, container_close_index=close, container_release_index=release,
                  meat_close_index=meat_close, meat_release_index=meat_release,
                  receiver_anchor_tcp=positions[release].tolist() if placed else None,
                  receiver_radius_m=definition["receiver_tcp_uncertainty_radius_m"],
                  receiver_ball_violation_m=receiver_distance,
                  containment_identity_assessed=False)

def rack_placements(positions, poses, cycles, definition,trace=None):
    rows, audit, used, releases = [], {}, set(), []
    for label, obj in definition["objects"].items():
        cycle, value = pick_cycle(cycles, positions, obj["source_regions"], excluded=used,trace=trace)
        close, release = cycle if cycle is not None else (None, None)
        if close is not None:
            used.add(next(c for c,e in cycles if e==release and c<=close))
        grasp = numeric_stage(label + "_source_close", value, close, scale=definition["grasp_distance_scale_m"])
        rows.append(grasp)
        body=np.array(obj["initial_body_pose"],copy=True)
        if trace and close is not None:body[:3,3]+=trace.centering[id(obj["source_regions"]),close]
        placed = common.object_pose(poses, close, release, body) if release is not None else None
        value = region_distance(placed[:3, 3], definition["destination_regions"]) if placed is not None else None
        destination = numeric_stage(label + "_rack_open", value, release,
                                    () if grasp["hard_region_hit"] else [grasp["id"]])
        rows.append(destination)
        if destination["hard_region_hit"]:
            releases.append(release)
        audit[label] = dict(close_index=close, release_index=release,
                            placed_body_pose=placed.tolist() if placed is not None else None)
    if definition.get("door_evaluated_by_stage_graph"):
        return finish(rows, placements=audit)
    blocked = common.dependencies(rows)
    value, door_index = None, None
    if not blocked:
        start = max(releases) + 1
        if start < len(positions):
            samples = densify(positions[start - 1:])
            ds = common.distance(samples, definition["door_regions"])
            value = float(np.min(ds))
            door_index = int(np.argmin(ds))
    rows.append(numeric_stage("door_after_both_placements", value, None, blocked))
    return finish(rows, placements=audit, door_best_densified_index=door_index,
                  door_densified_origin_index=max(releases) if len(releases) == 2 else None,
                  door_joint_closed_assessed=False)

def drawer_pick_place(positions, cycles, definition,trace=None):
    cycle, distance = pick_cycle(cycles, positions, definition["drawer_handle_regions"])
    close, release = cycle if cycle is not None else (None, None)
    if trace:
        distance,close,release,_=trace.drawer_pull(definition["drawer_handle_regions"])
    displacement = (float(np.dot(positions[release] - positions[close], definition["drawer_outward_axis_world"]))
                    if release is not None else None)
    handle_hit = distance is not None and distance <= 1e-9
    opened = bool(handle_hit and displacement is not None and displacement > definition["drawer_pull_minimum_m"] + 1e-9)
    first = dict(id="drawer_handle_pull", ordered_score=100. if opened else 0.,
                 hard_region_hit=opened, hit_index=release if opened else None,
                 blocked_by=[] if opened else ["legitimate_handle_close_open_outward_pull"])
    rows = [first]
    cup_cycle, value = pick_cycle(cycles, positions, definition["possible_cup_source_regions"],
                                 after=release) if opened else (None, None)
    if opened and trace and 'hidden_cup_initial_region' in definition:
        from copy import deepcopy
        region = deepcopy(definition['hidden_cup_initial_region'])
        axis = np.asarray(definition['drawer_outward_axis_world'])
        shift_world = axis*min(displacement, definition['hidden_cup_outward_limit_m'])
        region['vertices_world'] = (np.asarray(region['vertices_world'])+shift_world).tolist()
        planes = np.asarray(region['halfspaces_world'])
        planes[:,3] -= planes[:,:3] @ shift_world
        region['halfspaces_world'] = planes.tolist()
        solids = trace.solids([region])
        candidates = []
        for c, e in cycles:
            if c <= release:
                continue
            rotation = trace.poses[c,:3,:3]
            point = positions[c]
            gap = min(trace.gap(shape, rotation, point + rotation @ shift * aperture, solids)
                      for shape, shift, _ in trace.parts for aperture in (0., 1.))
            gap = max(gap, region_distance(point, [region]))
            candidates.append((gap, c, e))
        if candidates:
            value, c, e = min(candidates, key=lambda item: (item[0], item[1]))
            cup_cycle = (c,e)
    cup_close, cup_release = cup_cycle if cup_cycle is not None else (None, None)
    rows.append(numeric_stage("possible_cup_source_close", value, cup_close, common.dependencies(rows),
                              scale=definition["grasp_distance_scale_m"]))
    value = region_distance(positions[cup_release], definition["counter_regions"]) if cup_release is not None else None
    separation = float(np.linalg.norm(positions[-1] - positions[cup_release])) if cup_release is not None else None
    blocked = common.dependencies(rows)
    if separation is None or separation <= definition["retreat_distance_m"]:
        blocked.append("release_then_terminal_retreat")
    if cup_release is not None and any(c > cup_release for c, _ in cycles):
        blocked.append("gripper_reclosed_after_cup_release")
    rows.append(numeric_stage("counter_release_and_retreat", value, cup_release, blocked))
    return finish(rows, drawer_close_index=close, drawer_release_index=release,
                  drawer_outward_displacement_m=displacement, cup_close_index=cup_close,
                  cup_release_index=cup_release, terminal_release_tcp_separation_m=separation,
                  cup_identity_assessed=False, smaller_cup_selection_verified=False)
