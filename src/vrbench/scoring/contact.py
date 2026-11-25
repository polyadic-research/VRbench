import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from . import manipulation as common
from .manipulation import distance as raw_distances
PROTOCOL="contact_evidence"

def distances(points, regions):
    tolerance=regions[0].get("distance_tolerance_m",0.)
    return np.maximum(raw_distances(points,regions)-tolerance,0.)

def pad_distances(positions, poses, regions, offsets):
    values = distances(positions, regions)
    for offset in np.asarray(offsets):
        pts = positions + np.einsum('nij,j->ni', poses[:, :3, :3], offset)
        values = np.minimum(values, distances(pts, regions))
    return values

def row(stage, ds, eligible, scale, blocked=()):
    indices = np.asarray(eligible, int) if not blocked else np.array([], int)
    best = int(indices[np.argmin(ds[indices])]) if len(indices) else None
    hits = indices[ds[indices] <= 1e-9] if len(indices) else []
    hit = int(hits[0]) if len(hits) else None
    return dict(id=stage['id'], ordered_score=float(100*np.exp(-ds[best]/scale)) if best is not None else 0.,
                hard_region_hit=hit is not None, hit_index=hit, best_path_index=best,
                minimum_eligible_distance_m=float(ds[best]) if best is not None else None,
                blocked_by=list(blocked), eligible_samples=len(indices), distance_scale_m=scale)

def score(task, prediction, initial, definition,trace=None):
    positions, poses, events = common.trajectory(prediction, initial)
    n = len(positions)
    stages = definition['milestones']
    pad = definition.get('pad_offsets_tcp_local', [])
    if task in ('grasp_release', 'held_tool_contact'):
        ds = pad_distances(positions, poses, stages[0]['regions'], pad)
        cycles = common.grip_cycles(events)
        if trace:
            ds=np.full(n,1e6)
            effective=[]
            for c,e in cycles:
                shift=trace.drawer_shift(c) if trace.config['actions'].get(stages[0]['id'],{}).get('in_drawer') else None
                gap,k=trace.grip(stages[0]['regions'],c,e,shift)
                ds[k]=gap
                effective.append((k,e))
            cycles=effective
        source = row(stages[0], ds, [a for a, b in cycles], .05)
        anchor = next(((a, b) for a, b in cycles if ds[a] <= 1e-9), None)
        if anchor:
            source.update(hard_region_hit=True, hit_index=anchor[0])
        blocked = [] if anchor else [stages[0]['id']]
        target = np.full(n, np.inf)
        if task == 'grasp_release':
            if anchor and anchor[1] is not None:
                target[anchor[1]] = distances(positions[anchor[1]], stages[1]['regions'])[0]
            destination = row(stages[1], target,
                              [anchor[1]] if anchor and anchor[1] is not None else [], .05, blocked)
            destination['event_requirement'] = 'first explicit open after the valid source close'
            rows = [source, destination]
        elif 'press_depth_m' in definition:
            if trace is None:
                raise ValueError('Press scoring requires gripper and held-part geometry')
            eligible = range(anchor[0]+1, anchor[1] if anchor[1] is not None else n) if anchor else []
            witnesses = {}
            if anchor:
                target, witnesses = trace.held_press(anchor, definition, .05)
            dest = row(stages[1], target, eligible, .05, blocked)
            dest.update(witnesses.get(dest['best_path_index'], {}))
            dest['event_requirement'] = 'downward stroke of the held part during continuous target contact'
            return dict(protocol=PROTOCOL, score_0_to_100=(source['ordered_score']+dest['ordered_score'])/2,
                        milestones=[source, dest], close_index=anchor[0] if anchor else None,
                        release_index=anchor[1] if anchor else None, author_points_only=False,
                        geometry_proxy_not_physical_success=True)
        else:
            eligible = range(anchor[0]+1, anchor[1] if anchor[1] is not None else n) if anchor else []
            if anchor:
                local = np.asarray(definition['tool_head_surface_samples_world'])
                inv = np.linalg.inv(poses[anchor[0]])
                selected = np.asarray(list(eligible), int)
                if len(selected):
                    transforms = poses[selected] @ inv
                    moved = np.einsum('nij,kj->nki', transforms[:, :3, :3], local) + transforms[:, None, :3, 3]
                    target[selected] = distances(moved.reshape(-1, 3), stages[1]['regions']).reshape(len(selected), -1).min(1)
            if trace and anchor and 'moving_parts' in definition:
                target=trace.held_contact(poses,anchor,definition)
            rows = [source, row(stages[1], target, list(eligible), .05, blocked)]
            rows[1]['event_requirement'] = 'strictly after genuine grasp and before first release; authored poses only'
        return dict(protocol=PROTOCOL, score_0_to_100=float(np.mean([r['ordered_score'] for r in rows])),
                    milestones=rows, close_index=anchor[0] if anchor else None,
                    release_index=anchor[1] if anchor else None,
                    author_points_only=True, interpolated_contact_evidence_count=0,
                    geometry_proxy_not_physical_success=True)
    if task == 'held_handle' and trace:
        gap,k,e,j=trace.drawer_pull(stages[0]['regions'])
        result=common.stage(stages[0]['id'],gap,j,scale=.01)
        return dict(protocol=PROTOCOL,score_0_to_100=result['ordered_score'],milestones=[result])
    if task == 'held_handle':
        ds = pad_distances(positions, poses, stages[0]['regions'], pad)
        eligible = sorted({i for a, b in common.grip_cycles(events) for i in range(a, b if b is not None else n)})
        rows = [row(stages[0], ds, eligible, .01)]
        return dict(protocol=PROTOCOL, score_0_to_100=rows[0]['ordered_score'], milestones=rows,
                    author_points_only=True, geometry_proxy_not_physical_success=True)
    if task != 'pad_sequence':
        raise ValueError(task)
    ps, qs = [positions[0]], [Rotation.from_matrix(poses[0, :3, :3]).as_quat()]
    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    for i in range(n-1):
        count = max(1, int(np.ceil(np.linalg.norm(positions[i+1]-positions[i])/.01)))
        u = np.linspace(0, 1, count+1)[1:]
        ps.extend(positions[i]+u[:, None]*(positions[i+1]-positions[i]))
        qs.extend(Slerp([0, 1], Rotation.from_quat(quats[i:i+2]))(u).as_quat())
    ps = np.asarray(ps)
    pp = np.tile(np.eye(4), (len(ps), 1, 1))
    pp[:, :3, :3] = Rotation.from_quat(qs).as_matrix()
    hits, rows = {}, []
    for st in stages:
        deps = st.get('after', [])
        blocked = [d for d in deps if hits.get(d) is None]
        start = max([hits[d]+1 for d in deps if hits.get(d) is not None], default=0)
        ds = (pad_distances(ps, pp, st['regions'], pad) if st['id'] == 'faucet'
              else distances(ps, st['regions']))
        if trace and st['id'] in trace.config['actions']:
            ds=trace.series(st['regions'],trace.clock(.01),trace.config['actions'][st['id']])
        rr = row(st, ds, range(start, len(ps)), .05, blocked)
        hits[st['id']] = rr['hit_index']
        rows.append(rr)
    return dict(protocol=PROTOCOL, score_0_to_100=float(np.mean([r['ordered_score'] for r in rows])),
                milestones=rows, dense_samples=len(ps), geometry_proxy_not_physical_success=True)
