import copy,math
import numpy as np
from .region import distance
from .bindings import materialize_receiver

def densify(p):
    out=[p[0]]
    for a,b in zip(p[:-1],p[1:]):
        n=max(1,int(math.ceil(float(np.linalg.norm(b-a))/.01)))
        out.extend(a+(b-a)*(i/n) for i in range(1,n+1))
    return np.asarray(out)

def hard_score(points,definition,scale=None,trace=None,clock=None):
    scale=.05 if scale is None else scale
    rows=[];hits={}
    for node in definition['milestones']:
        deps=node.get('after',[]);blocked=[d for d in deps if hits.get(d) is None]
        start=max([hits[d]+1 for d in deps if hits.get(d) is not None],default=0)
        action=trace.config['actions'].get(node['id']) if trace else None
        ds=trace.series(node['regions'],clock,action) if action else np.min([distance(points,r) for r in node['regions']],axis=0)
        eligible=not blocked and start<len(points)
        candidates=np.flatnonzero(ds[start:]<=1e-9)+start if eligible else np.array([],int)
        hit=int(candidates[0]) if len(candidates) else None
        k=start+int(np.argmin(ds[start:])) if eligible else None
        value=float(100*np.exp(-ds[k]/scale)) if eligible else 0.
        hits[node['id']]=hit
        rows.append(dict(id=node['id'],label=node.get('label',node['id']),ordered_score=value,
            raw_distance_score=float(100*np.exp(-np.min(ds)/scale)) if len(ds) else 0.,
            hard_region_hit=hit is not None,hit_index=hit,best_path_index=k if k is not None else 0,
            minimum_distance_m=float(np.min(ds)) if len(ds) else None,
            blocked_by=blocked,eligible_start_index=start))
    return dict(score_0_to_100=float(np.mean([r['ordered_score'] for r in rows])),milestones=rows)

def gate_rows(rows,definition=None):
    """Gate event-scored rows with their actual event/held-phase validity."""
    byid={n['id']:n for n in definition['milestones']} if definition else {}
    done={};out=[]
    for i,item in enumerate(rows):
        r=copy.deepcopy(item);deps=byid[r['id']].get('after',[]) if definition else ([rows[i-1]['id']] if i else [])
        blocked=[d for d in deps if not done.get(d,False)]
        r['pre_gate_score']=r['ordered_score'];r['blocked_by']=blocked
        if blocked:r['ordered_score']=0.;r['hard_region_hit']=False
        done[r['id']]=bool(r['hard_region_hit'])
        out.append(r)
    return out

def releases(pred):
    pending=False;out=[]
    for i,w in enumerate(pred['waypoints'],1):
        if w.get('gripper_event')=='close':pending=True
        if w.get('gripper_event')=='open' and pending:out.append(i);pending=False
    return out

def two_receivers(points,pred,definition,new,trace=None):
    """Evaluate possible stage orders and assignments using centers from release events."""
    opens=releases(pred);byid={n['id']:n for n in definition['milestones']};best=None
    templates={b:next(n for n in new['milestones'] if n['id']=='food1_bowl')['regions'][b-1] for b in (1,2)}
    for bowls in ((1,2),(2,1)):
        nodes=[];prev=()
        for b in bowls:
            src=copy.deepcopy(byid[f'bowl{b}_source']);src['after']=list(prev)
            dst=copy.deepcopy(byid[f'bowl{b}_counter']);dst['after']=[src['id']];nodes.extend([src,dst]);prev=(dst['id'],)
        bowlscore=hard_score(points,dict(milestones=nodes),trace=trace,clock=np.arange(len(points)))
        anchor={b:opens[i] for i,b in enumerate(bowls) if i<len(opens)}
        for foods in ((1,2),(2,1)):
            for assignment in ((1,2),(2,1)):
                targets=[];prev=()
                start=max(anchor.values())+1 if anchor else len(points)
                for f,b in zip(foods,assignment):
                    src=copy.deepcopy(byid[f'food{f}_source']);src['after']=list(prev)
                    rr=materialize_receiver(templates[b],points[anchor[b]]) if b in anchor else dict(type='unresolved_dynamic')
                    dst=dict(id=f'food{f}_bowl',label=f'food {f} in moved bowl {b}',after=[src['id']],regions=[rr]);targets.extend([src,dst]);prev=(dst['id'],)
                foodscore=hard_score(points[start:],dict(milestones=targets),trace=trace,clock=np.arange(start,len(points))) if start<len(points) else dict(milestones=[dict(id=n['id'],ordered_score=0,hard_region_hit=False,minimum_distance_m=None) for n in targets])
                rows=gate_rows(bowlscore['milestones']+foodscore['milestones']);score=float(np.mean([r['ordered_score'] for r in rows]))
                row=dict(score_0_to_100=score,milestones=rows,selected_bowl_order=list(bowls),selected_food_order=list(foods),food_to_bowl=dict(zip(map(str,foods),assignment)),
                    release_anchor_indices=anchor,anchor_approximation='first two held-phase release TCPs as moved bowl bbox centers',
                    unresolved_anchors=sorted(set((1,2))-set(anchor)))
                if best is None or score>best['score_0_to_100']:best=row
    return best

def event_sequence(points, waypoints, definition):
    rows=[];cursor=0
    for node in definition['milestones']:
        candidates=[i+1 for i,w in enumerate(waypoints) if w.get('gripper_event')==node['event'] and i+1>cursor]
        k=candidates[0] if candidates else min(cursor,len(points)-1)
        action=trace.config['actions'].get(node['id']) if trace else None
        ds=trace.series(node['regions'],clock,action) if action else np.min([distance(points,r) for r in node['regions']],axis=0)
        value=float(100*np.exp(-ds[k]/definition.get('distance_scale_m',.01))) if candidates else 0.
        hit=bool(candidates and ds[k]<=1e-9)
        rows.append(dict(id=node['id'],ordered_score=value,minimum_distance_m=float(ds[k]),best_path_index=k,hit_index=k if hit else None,hard_region_hit=hit))
        cursor=k
    rows=gate_rows(rows,definition)
    return dict(score_0_to_100=float(np.mean([r['ordered_score'] for r in rows])),milestones=rows)
