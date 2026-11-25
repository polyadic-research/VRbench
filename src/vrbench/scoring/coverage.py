import numpy as np
from scipy.spatial import ConvexHull
from . import manipulation as common
from .bindings import clip_polygon
PROTOCOL="held_object_coverage"

def contact_candidate(vertices, definition):
    vertices = np.asarray(vertices, float)
    gap = float(vertices[:, 2].min() - definition["board_top_world_z"])
    polygon = ConvexHull(vertices[:, :2])
    overlap = clip_polygon(
        polygon.points[polygon.vertices], np.asarray(definition["board_polygon_world_xy"]))
    area = (float(abs(np.sum(overlap[:, 0] * np.roll(overlap[:, 1], -1)
                             - overlap[:, 1] * np.roll(overlap[:, 0], -1))) / 2)
            if len(overlap) >= 3 else 0.)
    hit = bool(abs(gap) <= definition["contact_vertical_tolerance_m"] + 1e-9 and area > 1e-10)
    return hit, gap, area

def coverage(centers, definition):
    retained = []
    for point in centers:
        point = np.asarray(point, float)
        if all(np.linalg.norm(point - other) >= definition["distinct_contact_separation_m"] - 1e-9
               for other in retained):
            retained.append(point)
    diagonal = float(np.linalg.norm(np.ptp(retained, axis=0))) if retained else 0.
    value = 100 * min(1., len(retained) / definition["required_distinct_contacts"],
                      diagonal / definition["required_coverage_bbox_diagonal_m"])
    hit = (len(retained) >= definition["required_distinct_contacts"]
           and diagonal >= definition["required_coverage_bbox_diagonal_m"] - 1e-9)
    return float(value), hit, retained, diagonal

def score(prediction, initial, definition,trace=None):
    if prediction.get("coordinate_frame") != "world" or prediction.get("position_unit") not in ("meter", "m"):
        raise ValueError("Expected world-frame meter trajectory")
    if not prediction.get("waypoints"):
        raise ValueError("Missing coverage waypoints")
    for waypoint in prediction["waypoints"]:
        if waypoint.get("gripper_event") not in (None, "close", "open"):
            raise ValueError("Invalid gripper event")
    positions, poses, events = common.trajectory(prediction, initial)
    closes = [i for i, event in enumerate(events) if event == "close"]
    distances = {i: float(common.distance(positions[i], definition["source_regions"])[0]) for i in closes}
    if trace:
        distances={k:d for c,e in trace.cycles for d,k in [trace.grip(definition["source_regions"],c,e)]}
        closes=list(distances)
    close = next((i for i in closes if distances[i] <= 1e-9), None)
    best = close if close is not None else min(closes, key=distances.get, default=None)
    source_hit = close is not None
    rows = [dict(id="sponge_grasp", ordered_score=(
        float(100 * np.exp(-distances[best] / definition["source_distance_scale_m"])) if best is not None else 0.),
        hard_region_hit=source_hit, hit_index=close, best_path_index=best, blocked_by=[])]
    release = next((i for i in range(close + 1, len(events)) if events[i] == "open"), None) if source_hit else None
    candidates, gaps, areas, centers = [], [], [], []
    complete_index = None
    body = np.asarray(definition["initial_sponge_body_pose"])
    original_vertices = np.asarray(definition["initial_sponge_vertices_world"])
    if source_hit and trace:
        shift=trace.centering[id(definition["source_regions"]),close]
        original_vertices=original_vertices + shift
        body=body.copy();body[:3,3]+=shift
    if source_hit:
        for i in range(close, release if release is not None else len(events)):
            transform = poses[i] @ np.linalg.inv(poses[close])
            vertices = original_vertices @ transform[:3, :3].T + transform[:3, 3]
            contact, gap, area = contact_candidate(vertices, definition)
            if contact:
                center = (transform @ body)[:2, 3]
                candidates.append(i)
                gaps.append(gap)
                areas.append(area)
                centers.append(center)
                if complete_index is None and coverage(centers, definition)[1]:
                    complete_index = i
    value, covered, retained, diagonal = coverage(centers, definition)
    rows.append(dict(id="sponge_board_coverage", ordered_score=value if source_hit else 0.,
                     hard_region_hit=source_hit and covered, hit_index=complete_index,
                     blocked_by=[] if source_hit else ["sponge_grasp"]))
    final_separation = None
    release_after_coverage = bool(covered and release is not None and release > complete_index)
    reopened = bool(release is not None and "close" in events[release + 1:])
    blocked = [row["id"] for row in rows if not row["hard_region_hit"]]
    if not release_after_coverage:
        blocked.append("release_after_coverage")
    if reopened:
        blocked.append("gripper_reclosed_after_release")
    released_body = None
    if release is not None:
        released_body = common.object_pose(poses, close, release, body)
        final_separation = float(np.linalg.norm(positions[-1] - released_body[:3, 3]))
    retreat_hit = bool(not blocked and final_separation > definition["retreat_distance_m"])
    retreat_value = 100. if retreat_hit else 0.
    rows.append(dict(id="released_and_retreated", ordered_score=retreat_value,
                     hard_region_hit=retreat_hit, hit_index=len(positions) - 1 if retreat_hit else None,
                     blocked_by=blocked))
    return dict(protocol=PROTOCOL, score_0_to_100=float(np.mean([r["ordered_score"] for r in rows])),
                milestones=rows, close_index=close, release_index=release,
                index_policy="initial pose at index0; authored waypoint n at index n",
                contact_candidate_indices=candidates, contact_vertical_gaps_m=gaps,
                contact_overlap_areas_m2=areas, contact_candidate_count=len(candidates),
                retained_distinct_contact_centers_xy=[p.tolist() for p in retained],
                distinct_contact_count=len(retained), coverage_bbox_diagonal_m=diagonal,
                coverage_completed_index=complete_index, final_tcp_sponge_separation_m=final_separation,
                released_sponge_body_pose=released_body.tolist() if released_body is not None else None,
                waypoint_count=len(prediction["waypoints"]), interpolated_contact_evidence_count=0)
