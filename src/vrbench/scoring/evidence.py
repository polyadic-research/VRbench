"""Fixed coupled checks. Dataset JSON supplies geometry, never executable code."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from . import manipulation, sequences, coverage, placement, contact
from .region import distance
from .bindings import materialize_receiver

KINDS = ('rigid_release', 'event_sequence', 'release_anchored_sequence', 'two_receivers', 'held_object_coverage', 'container_transfer',
         'rack_placements', 'drawer_pick_place', 'stack_objects', 'tilt_return',
         'grasp_release', 'held_tool_contact', 'held_handle', 'pad_sequence', 'sampled_contact')


def evaluate(group, prediction, initial, trace=None):
    kind, d = group['kind'], copy.deepcopy(group['parameters'])
    if kind not in KINDS: raise ValueError('Unsupported coupled evidence: ' + kind)
    if not isinstance(initial, dict) or 'quaternion' not in initial:
        raise ValueError('Coupled geometry needs explicit initial TCP position and quaternion')
    ps, poses, events = manipulation.trajectory(prediction, initial)
    cycles = manipulation.grip_cycles(events)
    if kind == 'rigid_release':
        from .rigid_release import evaluate as rigid_release
        return rigid_release(ps, poses, cycles, d,trace)
    if kind == 'held_object_coverage': return coverage.score(prediction, initial, d,trace)
    if kind == 'container_transfer' and 'opening_template' in d:
        from .cabinet_transfer import evaluate as cabinet_transfer
        return cabinet_transfer(trace, cycles, d)
    if kind == 'container_transfer': return placement.container_transfer(ps, cycles, d,trace)
    if kind == 'rack_placements': return placement.rack_placements(ps, poses, cycles, d,trace)
    if kind == 'drawer_pick_place': return placement.drawer_pick_place(ps, cycles, d,trace)
    if kind == 'stack_objects':
        result=manipulation.stack(ps, poses, events, d,trace)
        for row,name in zip(result['milestones'],('base_source','base_in_sink','top_source','top_in_base','released_and_retreated')):
            row['object_stage_id']=row['id'];row['id']=name
        return result
    if kind == 'tilt_return': return manipulation.tilt(ps, poses, events, d,trace)
    if kind in ('grasp_release', 'held_tool_contact', 'held_handle', 'pad_sequence'):
        return contact.score(kind, prediction, initial, d,trace)
    if kind == 'two_receivers': return sequences.two_receivers(ps, prediction, d['matching'], d['geometry'],trace)
    if kind in ('event_sequence', 'release_anchored_sequence'):
        opened = sequences.releases(prediction)
        for node in d['milestones']:
            if node.get('release_receiver'):
                k = node['release_receiver']['ordinal']
                node['regions'] = [materialize_receiver(node['release_receiver']['template'], ps[opened[k]])] if k < len(opened) else [{'type':'unresolved_dynamic'}]
        if kind == 'release_anchored_sequence':
            return sequences.hard_score(sequences.densify(ps), d, d.get('distance_scale_m', .05),trace, trace.clock(.01) if trace else None)
        return sequences.event_sequence(ps, prediction['waypoints'], d)
    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    if d['sampling']['type'] == 'linear':
        points, rotations = [ps[0]], [quats[0]]
        for i in range(len(ps)-1):
            n = max(1, int(np.ceil(np.linalg.norm(ps[i+1]-ps[i])/d['sampling']['step_m'])))
            u = np.linspace(0,1,n+1)[1:]
            points.extend(ps[i]+u[:,None]*(ps[i+1]-ps[i]))
            rotations.extend(Slerp([0,1],Rotation.from_quat(quats[i:i+2]))(u).as_quat())
        ps=np.asarray(points);rotations=Rotation.from_quat(rotations)
    else: rotations=Rotation.from_quat(quats)
    hits, rows = {}, []
    for node in d['milestones']:
        refs=node.get('after',[]);blocked=[x for x in refs if hits.get(x) is None]
        start=max([hits[x]+1 for x in refs if hits.get(x) is not None],default=0)
        ds=np.min([distance(ps,r) for r in node['regions']],axis=0)
        for offset in node.get('pad_offsets_tcp_local',[]):
            ds=np.minimum(ds,np.min([distance(ps+rotations.apply(offset),r) for r in node['regions']],axis=0))
        eligible=not blocked and start<len(ds)
        matches=np.flatnonzero(ds[start:]<=node.get('gate_tolerance_m',0.)+1e-9)+start if eligible else []
        hit=int(matches[0]) if len(matches) else None
        value=float(100*np.exp(-ds[start:].min()/d['evaluation_distance_scale_m'])) if eligible else 0.
        hits[node['id']]=hit
        rows.append(dict(id=node['id'],ordered_score=value,hard_region_hit=hit is not None,hit_index=hit,blocked_by=blocked,best_path_index=start+int(np.argmin(ds[start:])) if eligible else None))
    return dict(score_0_to_100=float(np.mean([r['ordered_score'] for r in rows])),milestones=rows)
