"""Construct explicit graphs from spatial or authored stage specifications."""
import copy


def spatial(milestones, scale=.05, sampling=None):
    return {'rule': 'stage_graph', 'sampling': sampling or {'type':'linear','step_m':.01},
            'stages': [dict(id=s['id'], label=s.get('label',s['id']),
                **({'dependencies': {'all_of':s['after'], 'relation':'strictly_after'}} if s.get('after') else {}),
                check={'type':'distance', 'regions':copy.deepcopy(s['regions']),
                       'distance_scale_m':s.get('distance_scale_m',scale),
                       'hit_tolerance_m':s.get('gate_tolerance_m',0.)}) for s in milestones]}


def authored(milestones, scale=.05):
    stages=[]
    for s in milestones:
        p=copy.deepcopy(s);deps=p.pop('after',[]);name=p.pop('id');p.setdefault('distance_scale_m',scale)
        stages.append(dict(id=name,check={'type':'authored','parameters':p},
            **({'dependencies':{'all_of':deps,'relation':'strictly_after'}} if deps else {})))
    return {'rule':'stage_graph','sampling':{'type':'authored'},'stages':stages}


def coupled(kind, parameters, stages):
    """Coupled checks own shared anchors; graph edges expose satisfaction gates."""
    return {'rule':'stage_graph','sampling':{'type':'authored'},
            'evidence':[{'id':'operation','kind':kind,'parameters':copy.deepcopy(parameters)}],
            'stages':[dict(id=s['id'],check={'type':'evidence_stage','evidence':'operation','stage':s['id']},
                **({'dependencies':{'all_of':s['after'],'relation':'gate'}} if s.get('after') else {})) for s in stages]}
