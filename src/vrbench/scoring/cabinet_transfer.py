"""Score opening a cupboard and transferring food to a previously placed container."""
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from . import placement
from .manipulation import dependencies, stage


def first_opening(trace, doors, threshold):
    clock = trace.clock(.01)
    positions = np.column_stack([np.interp(clock, np.arange(len(trace.ps)), a) for a in trace.ps.T])
    rotations = trace.slerp(clock).as_matrix()
    arc = np.interp(clock, np.arange(len(trace.ps)), trace.travel)
    best = (float('inf'), None, None, None)
    hits = []
    for door in doors:
        objects = trace.solids(door['regions'])
        vertices = np.vstack([r['vertices_world'] for r in door['regions']])
        near = np.all((positions >= vertices.min(0)-.16) & (positions <= vertices.max(0)+.16), axis=1)
        for shape, shift, name in trace.parts:
            for i in np.flatnonzero(near):
                opened = not trace.closed[int(clock[i])]
                offset = np.asarray(trace.config['part_centers'][name]) + shift*opened
                point = positions[i] + rotations[i] @ offset
                tangent = np.cross(door['axis'], point-door['pivot'])*door['sign']
                tangent /= np.linalg.norm(tangent)
                gap = trace.gap(shape, rotations[i], positions[i]+rotations[i]@(opened*shift), objects)
                if gap > .10:
                    continue
                j_end = np.searchsorted(arc, arc[i]+trace.config['jaw_travel_m'], side='right')
                for j in range(i+1, j_end):
                    finger = positions[j] + rotations[j] @ offset
                    progress = (finger-point) @ tangent
                    if progress <= 0:
                        continue
                    error = gap - .01*np.log(min(1., progress/threshold))
                    event = (error, int(np.ceil(clock[j])), float(clock[i]), door['side'])
                    if error < best[0]:
                        best = event
                    if error <= 1e-9:
                        hits.append(event)
                        break
    if hits:
        return min(hits, key=lambda event: (event[1],event[2]))
    return best if best[1] is not None else (None,None,None,None)


def receiver(trace, close, release, counter, params):
    template = params['opening_template']
    rim = np.asarray(template['rim_xy'])
    inward = np.array([*params['cabinet_inward_xy'],0.])
    across = np.array([inward[1],-inward[0],0.])
    rotation = trace.poses[release,:3,:3] @ trace.poses[close,:3,:3].T
    x = (rotation @ across)[:2]
    if np.linalg.norm(x) < 1e-6:
        y = (rotation @ inward)[:2]
        x = np.array([y[1],-y[0]])
    x /= np.linalg.norm(x)
    basis = np.column_stack((x, [-x[1],x[0]]))
    center = trace.ps[release,:2] + (rotation @ (-rim[:,1].min()*inward))[:2]
    planes = ConvexHull(rim).equations.copy()
    planes[:,2] -= template['lateral_margin_m']
    polygon = HalfspaceIntersection(planes, rim.mean(0)).intersections
    xy = polygon @ basis.T + center
    floor = min(v[2] for v in counter['vertices_world'])
    z0 = floor + template['inner_floor_above_base_m']
    z1 = floor + template['rim_above_base_m'] + template['above_rim_m']
    verts = np.array([[*p,z] for p in xy for z in (z0,z1)])
    hull = ConvexHull(verts)
    return dict(name='placed_box_opening',type='convex_polytope',
                vertices_world=verts[hull.vertices].tolist(),halfspaces_world=hull.equations.tolist())


def evaluate(trace, cycles, params):
    action = trace.config['actions'].get('cabinet_open')
    details = {}
    if action and action['kind'] == 'hinge_path':
        door, = params['doors']
        side = door['side']
        ds, details = trace.hinge_path(door['regions'], trace.clock(), 0, action, .01)
        best = int(np.argmin(ds))
        hits = np.flatnonzero(ds <= 1e-9)
        start = int(hits[0]) if len(hits) else best
        value = float(ds[start])
        contact = next(iter(details.get('handle_contact_times', [])), None)
    else:
        value, start, contact, side = first_opening(trace, params['doors'], params['opening_travel_m'])
    rows = [dict(stage('cabinet_open', value, start, scale=.01), **details)]
    def grasp(label, regions, after):
        cycle, value = (placement.pick_cycle(cycles, trace.ps, regions, after=after, trace=trace)
                        if not dependencies(rows) else (None,None))
        close, release = cycle or (None,None)
        rows.append(stage(label, value, close, dependencies(rows), scale=params['grasp_distance_scale_m']))
        return close, release
    close, release = grasp('container_source_close', params['container_source_regions'], start)
    support = None
    value = None
    if release is not None and not dependencies(rows):
        distances = [placement.region_distance(trace.ps[release], [r]) for r in params['counter_regions']]
        support = params['counter_regions'][int(np.argmin(distances))]
        value = min(distances)
    rows.append(stage('container_counter_open', value, release, dependencies(rows)))
    zone = receiver(trace, close, release, support, params) if not dependencies(rows) else None
    regions = params['meat_source_regions']
    m, e = grasp('meat_source_close', regions, release)
    center = value = None
    if e is not None and not dependencies(rows):
        initial = np.asarray(params['meat_center_world']) + trace.centering[id(regions),m]
        transform = trace.poses[e] @ np.linalg.inv(trace.poses[m])
        center = (transform[:3,:3]@initial + transform[:3,3]).tolist()
        value = placement.region_distance(center, [zone])
    rows.append(stage('meat_receiver_open', value, e, dependencies(rows)))
    return placement.finish(rows, door_side=side, door_contact_time=contact,
        container_close_index=close, container_release_index=release, meat_close_index=m,
        meat_release_index=e, receiver_region=zone, released_meat_center=center)
