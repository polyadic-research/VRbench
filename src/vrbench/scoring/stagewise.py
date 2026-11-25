"""Stage-local collision and tracking aggregation."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .stages import order
from .reliability import components

PARAMETERS = {'tracking_cutoff': .40, 'collision_cutoff': .30, 'region_floor_weight': .10}



def contribution(weight, region, collision, tracking):
    if not all(math.isfinite(x) for x in (weight,region,collision,tracking)):
        raise ValueError('Non-finite stage component')
    if not 0<weight<=1 or not 0<=region<=100 or not 0<=collision<=1 or not 0<=tracking<=1:
        raise ValueError('Stage component out of bounds')
    correction = (collision*tracking if collision>=PARAMETERS['collision_cutoff']
                  and tracking>=PARAMETERS['tracking_cutoff'] else 0.)
    floor = PARAMETERS['region_floor_weight']
    return weight*region*(floor+(1-floor)*correction)

def require(condition, message):
    if not condition:
        raise ValueError(message)


def dense_clock(positions,step=.01,origin=0):
    ts=[float(origin)]
    for i,(a,b) in enumerate(zip(positions[:-1],positions[1:]),origin):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/step)))
        ts.extend(i+np.linspace(0,1,n+1)[1:])
    return np.array(ts)
def clock_for(group,ps):
    kind=group['kind'];d=group['parameters']
    if kind in ('release_anchored_sequence','pad_sequence'):return dense_clock(ps)
    if kind=='sampled_contact' and d['sampling']['type']=='linear':return dense_clock(ps,d['sampling']['step_m'])
    return np.arange(37,dtype=float)
def time_at(clock,index,label):
    if index is None:return None
    k=int(index);require(k==index and 0<=k<len(clock),f'bad evidence index {label}: {index}/{len(clock)}')
    return float(clock[k])
def flatten(definition,detail,ps,prefix='',factor=1.):
    rule=definition.get('rule')
    if rule in ('mean','best_assignment'):
        chosen=[detail['selected_alternative']] if rule=='best_assignment' else range(len(definition['rules']))
        out=[]
        for i in chosen:out.extend(flatten(definition['rules'][i],detail['children'][i],ps,prefix+f'{i}/',factor/(len(definition['rules']) if rule=='mean' else 1)))
        return out
    nodes=order(definition['stages']);rows=detail.get('stages',detail.get('milestones'));require(len(nodes)==len(rows),'stage shape mismatch')
    total=sum(n.get('weight',1.) for n in nodes);require(total>0,'zero weights')
    groups={g['id']:g for g in definition.get('evidence',[])}
    sampling=definition.get('sampling',{'type':'authored'})
    clock=dense_clock(ps,sampling['step_m']) if isinstance(sampling,dict) and sampling.get('type')=='linear' else np.arange(37,dtype=float)
    out=[]
    for node,row in zip(nodes,rows):
        require(node['id']==row['id'],'stage ID mismatch')
        support=[];clock0=clock;kind='authored';groupdetail={};idx=row.get('best_path_index',row.get('hit_index'))
        c=node.get('check',{})
        if isinstance(c,dict) and c['type']=='evidence_stage':
            g=groups[c['evidence']];kind=g['kind'];groupdetail=detail['evidence'][g['id']];clock0=clock_for(g,ps)
            if kind=='two_receivers' and row['id'].startswith('food'):
                anchors=groupdetail['release_anchor_indices'];offset=max(anchors.values())+1 if anchors else 37
                clock0=np.arange(offset,37,dtype=float)
            if kind=='rack_placements' and row['id']=='door_after_both_placements':
                origin=groupdetail['door_densified_origin_index'];idx=groupdetail['door_best_densified_index']
                if origin is not None:clock0=dense_clock(ps[origin:],origin=origin)
            if kind=='held_object_coverage' and row['id']=='sponge_board_coverage':
                support=list(map(float,groupdetail['contact_candidate_indices']))
                idx=max(support) if support else None
                if groupdetail.get('close_index') is not None:support.append(float(groupdetail['close_index']))
        if row.get('author_waypoint_indices'):
            support.extend(map(float,row['author_waypoint_indices']));idx=max(support)
        witness=time_at(clock0,idx,prefix+row['id']) if idx is not None else None
        if 'retreat' in row['id'] and float(row['ordered_score'])>0:witness=36.
        support=sorted(set(support+([witness] if witness is not None else [])))
        value=float(row['ordered_score']);require(value==0 or witness is not None,'positive stage without trace '+row['id'])
        dep=node.get('dependencies',{});refs=dep.get('all_of',dep.get('any_of',node.get('after',[])))
        out.append(dict(id=prefix+row['id'],region_stage_score=value,weight=factor*node.get('weight',1.)/total,
                        witness_author_time=witness,support_author_times=support,evidence_kind=kind,
                        dependency_ids=[prefix+r for r in refs],region_row=row))
    return out


def windows(definition, detail, positions):
    rows = flatten(definition, detail, positions)
    for row in rows:
        row['support_author_times'].extend(row['region_row'].get('handle_contact_times', []))
    active = [x for x in rows if x['region_stage_score']>0]
    anchors = sorted({x['witness_author_time'] for x in active})
    by_id = {x['id']: x for x in rows}
    for x in active:
        t = x['witness_author_time']
        start = max((a for a in anchors if a<t-1e-10),default=0.)
        start = min([start,*x['support_author_times'],*[by_id[k]['witness_author_time'] for k in x['dependency_ids'] if by_id[k]['witness_author_time'] is not None]])
        x['interval'] = [start, 36. if t==anchors[-1] else t]
    require(math.isclose(sum(x['weight']*x['region_stage_score'] for x in rows),detail['score_0_to_100'],abs_tol=1e-8),'stage total mismatch')
    return rows


def tracking_samples(robot, joints, prediction, nodes):
    pos, rot = [], []
    for node, w in zip(nodes, prediction['waypoints']):
        pose = robot.fk(joints[node])[1]
        pos.append(1000*np.linalg.norm(pose[:3,3]-w['tcp_position']))
        rot.append(np.degrees(Rotation.from_matrix(pose[:3,:3].T@Rotation.from_quat(w['tcp_quaternion_xyzw']).as_matrix()).magnitude()))
    step = np.degrees(np.max(abs(np.diff(joints[[0,*nodes]],axis=0)),axis=1))
    return np.array(pos), np.array(rot), step


def aggregate(rows, clock, penetration, boundary, tracking):
    author = np.arange(1,37)
    covered = np.zeros(clock.shape,bool)
    pos,rot,step = tracking
    for x in rows:
        x['contribution'] = 0.
        if 'interval' not in x:
            continue
        start,end = x['interval']
        mask = (clock>start+1e-10)&(clock<=end+1e-10)
        if start<=1e-10:
            mask |= np.isclose(clock,0.,atol=1e-10,rtol=0)
        if not mask.any():
            mask = np.isclose(clock,end,atol=1e-10,rtol=0)
        covered |= mask
        p = penetration[mask] if mask.any() else np.array([boundary[end]])
        c = (math.exp(-p.mean()/10)+math.exp(-p.max()/10)+1-np.mean(p>0))/3
        idx = np.flatnonzero((author>start+1e-10)&(author<=end+1e-10))
        if not len(idx):
            idx = [max(1,min(36,math.ceil(end)))-1]
        w = components(float(np.sqrt(np.mean(pos[idx]**2))),float(np.sqrt(np.mean(rot[idx]**2))),float(np.percentile(step[idx],95)),float(step[idx].max()))['reliability_weight']
        x.update(collision_coefficient=float(c),tracking_coefficient=w,
                 contribution=contribution(x['weight'],x['region_stage_score'],c,w))
    require(not any('interval' in x for x in rows) or covered.all(),'stage windows omit collision samples')
    return math.fsum(x['contribution'] for x in rows)
