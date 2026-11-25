import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .region import distance

def authored_stages(prediction,initial,definition):
    """Event/contact/orientation rules operate on author points, never synthetic coverage."""
    from ..protocol.validation import validate_prediction
    validate_prediction(prediction,36)
    waypoints=prediction['waypoints'];positions=np.asarray([initial,*[w['tcp_position'] for w in waypoints]],float)
    quats=[None,*[w['tcp_quaternion_xyzw'] for w in waypoints]]
    grip=[False];contact=[False];releases=[]
    for i,w in enumerate(waypoints,1):
        held=grip[-1];engaged=contact[-1]
        if w['gripper_event']=='close':held=True
        if w['gripper_event']=='open':
            if held:releases.append(i)
            held=False
        if w['contact_event']=='engage':engaged=True
        if w['contact_event']=='disengage':engaged=False
        grip.append(held);contact.append(engaged)
    hits={};results=[]
    for stage in definition['milestones']:
        name=stage['id'];deps=stage.get('after',[])
        if name in hits or set(deps)-set(hits):raise ValueError('Invalid authored-stage dependencies')
        blocked=[d for d in deps if hits[d] is None];start=max(stage.get('_eligible_start', 0), max([hits[d]+1 for d in deps if hits[d] is not None],default=0))
        mode=stage['mode'];indices=list(range(max(1,start),len(positions)))
        if stage.get('last_only'):indices=[len(positions)-1] if len(positions)-1>=start else []
        if stage.get('event'):
            key,value=stage['event'];indices=[i for i in indices if waypoints[i-1].get(key)==value]
        if stage.get('held') is not None:indices=[i for i in indices if grip[i]==stage['held']]
        if stage.get('engaged') is not None:indices=[i for i in indices if contact[i]==stage['engaged']]
        regions=copy.deepcopy(stage.get('regions',[]))
        if 'release_anchor' in stage:
            anchor=stage['release_anchor'];ordinal=anchor['index']
            if ordinal<0 or ordinal>=len(releases):indices=[]
            else:
                event=releases[ordinal];indices=[i for i in indices if i>event]
                for r in regions:
                    pose=np.asarray(r['world_pose'],float);pose[:3,3]=positions[event]+np.asarray(anchor.get('offset',[0,0,0]));r['world_pose']=pose.tolist()
        if blocked or not indices:value=0.;hit=None;detail={'blocked_by':blocked}
        elif mode in ('distance','event_distance'):
            ds=np.min([distance(positions[indices],r) for r in regions],axis=0);scale=stage.get('distance_scale_m',definition.get('distance_scale_m',.05))
            if scale<=0:raise ValueError('Invalid scale')
            value=float(100*np.exp(-ds.min()/scale));candidates=np.flatnonzero(ds<=1e-9);hit=indices[int(candidates[0])] if len(candidates) else None;detail={'minimum_distance_m':float(ds.min()),'best_path_index':indices[int(np.argmin(ds))]}
        elif mode=='coverage':
            points=positions[indices].copy()
            if 'tool_offset' in stage:points+=Rotation.from_quat([quats[i] for i in indices]).apply(np.asarray(stage['tool_offset'],float))
            ds=np.min([distance(points,r) for r in regions],axis=0);valid=[j for j,x in enumerate(ds) if x<=stage['tolerance_m']];kept=[]
            for j in valid:
                if all(np.linalg.norm(points[j]-points[k])>=stage['separation_m'] for k in kept):kept.append(j)
            diagonal=float(np.linalg.norm(np.ptp(points[kept,:2],axis=0))) if kept else 0.
            quality=min(1.,len(kept)/stage['minimum_points'],diagonal/stage['minimum_xy_diagonal_m'])
            value=100*quality;hit=indices[kept[-1]] if quality>=1 and kept else None;detail={'author_waypoint_indices':[indices[k] for k in kept],'xy_diagonal_m':diagonal,'interpolation_used_for_coverage':False}
        elif mode=='orientation':
            angles=(Rotation.from_quat(stage['target_quaternion_xyzw']).inv()*Rotation.from_quat([quats[i] for i in indices])).magnitude();scale=stage['angle_scale_rad']
            if scale<=0:raise ValueError('Invalid angular scale')
            value=float(100*np.exp(-angles.min()/scale));matches=np.flatnonzero(angles<=stage['angle_tolerance_rad']);hit=indices[int(matches[0])] if len(matches) else None;detail={'minimum_angle_rad':float(angles.min()),'best_path_index':indices[int(np.argmin(angles))]}
        else:raise ValueError('Unsupported authored-stage mode '+mode)
        hits[name]=hit;results.append({'id':name,'ordered_score':value,'hard_region_hit':hit is not None,'hit_index':hit,**detail})
    if not results:raise ValueError('Empty authored stages')
    return {'score_0_to_100':float(np.mean([r['ordered_score'] for r in results])),'milestones':results}

def check(prediction, initial, parameters, start):
    node = copy.deepcopy(parameters)
    node['_eligible_start'] = start
    pos = initial['position'] if isinstance(initial, dict) else initial
    return authored_stages(prediction, pos, {'milestones': [node]})['milestones'][0]
