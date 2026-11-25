"""Declarative stage DAGs: order-free branches, joins and explicit evidence."""
import copy
import math
import numpy as np
from . import region


def order(stages):
    by_id = {s['id']: s for s in stages}
    if len(by_id) != len(stages) or not by_id:
        raise ValueError('Empty or duplicate stage IDs')
    result, visiting, done = [], set(), set()
    def visit(name):
        if name not in by_id: raise ValueError('Unknown prerequisite: ' + name)
        if name in visiting: raise ValueError('Cyclic stage dependencies')
        if name in done: return
        visiting.add(name)
        dep = by_id[name].get('dependencies', {})
        if set(dep) - {'all_of', 'any_of', 'relation'}: raise ValueError('Unknown dependency field')
        if dep.get('all_of') and dep.get('any_of'): raise ValueError('Use one dependency operator')
        if dep.get('relation', 'strictly_after') not in ('strictly_after', 'not_before', 'gate'):
            raise ValueError('Unknown dependency relation')
        for ref in dep.get('all_of', []) + dep.get('any_of', []): visit(ref)
        visiting.remove(name); done.add(name); result.append(by_id[name])
    for name in by_id: visit(name)
    return result


def validate_graph(definition):
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value): raise ValueError('Non-finite rule parameter')
        if isinstance(value, dict):
            for v in value.values(): finite(v)
        if isinstance(value, list):
            for v in value: finite(v)
    finite(definition)
    from ..schema import validate
    validate('regions', definition)
    if definition['rule'] == 'stage_graph':
        stages = order(definition['stages'])
        pools = {p['id'] for p in definition.get('evidence', [])}
        if len(pools) != len(definition.get('evidence', [])): raise ValueError('Duplicate evidence IDs')
        from .evidence import KINDS
        for p in definition.get('evidence', []):
            if p['kind'] not in KINDS: raise ValueError('Unsupported coupled evidence kind')
        actions = definition.get('metadata', {}).get('interaction_model', {}).get('actions', {})
        if set(actions)-{stage['id'] for stage in stages}:
            raise ValueError('Interaction action references an unknown stage')
        for action in actions.values():
            if action.get('kind') not in {'grasp','operate','hinge_path','release','hidden_grasp'}:
                raise ValueError('Unsupported interaction action')
        for s in stages:
            if not math.isfinite(s.get('weight', 1.)): raise ValueError('Non-finite weight')
            c = s['check']
            if c['type'] == 'evidence_stage' and c['evidence'] not in pools: raise ValueError('Unknown evidence group')
    else:
        for child in definition['rules']: validate_graph(child)


def score(prediction, initial, definition):
    from ..protocol.validation import validate_prediction
    validate_prediction(prediction, 36)
    validate_graph(definition)
    rule = definition['rule']
    if rule in ('mean', 'best_assignment'):
        children = [score(prediction, initial, child) for child in definition['rules']]
        values = [c['score_0_to_100'] for c in children]
        return {'score_0_to_100': float(np.mean(values) if rule == 'mean' else max(values)),
                'children': children, 'aggregation': rule,
                'selected_alternative': int(np.argmax(values)) if rule == 'best_assignment' else None}
    from .evidence import evaluate
    config=definition.get('metadata',{}).get('interaction_model')
    trace=None
    if config:
        from .interaction import Trace
        trace=Trace(prediction,initial,config)
    evidence = {p['id']: evaluate(p, prediction, initial, trace) for p in definition.get('evidence', [])}
    sampling = definition.get('sampling', {'type': 'authored'})
    if sampling['type'] == 'linear':
        pos = initial['position'] if isinstance(initial, dict) else initial
        points = region.densify([pos, *[w['tcp_position'] for w in prediction['waypoints']]], sampling['step_m'])
    else:
        pos = initial['position'] if isinstance(initial, dict) else initial
        points = np.asarray([pos, *[w['tcp_position'] for w in prediction['waypoints']]])
    rows, completed = [], {}
    credited_releases = set()
    for s in order(definition['stages']):
        dep = s.get('dependencies', {}); relation = dep.get('relation', 'strictly_after')
        refs = dep.get('all_of', dep.get('any_of', [])); is_any = bool(dep.get('any_of'))
        successes = [completed[x] for x in refs if completed[x]['satisfied']]
        eligible = not refs or (bool(successes) if is_any else len(successes) == len(refs))
        blocked = [] if eligible else [x for x in refs if not completed[x]['satisfied']]
        start = 0
        if eligible and successes and relation != 'gate':
            times = [r['hit_index'] for r in successes]
            if any(x is None for x in times): raise ValueError('Temporal dependency needs comparable hit indices')
            start = (min(times) if is_any else max(times)) + (1 if relation == 'strictly_after' else 0)
        c = s['check']; kind = c['type']
        if kind == 'evidence_stage':
            group = evidence[c['evidence']]
            matches = [r for r in group['milestones'] if r['id'] == c['stage']]
            if len(matches) != 1: raise ValueError('Missing/ambiguous coupled evidence stage: ' + c['stage'])
            row = copy.deepcopy(matches[0]); value = row['ordered_score']
            satisfied = bool(row['hard_region_hit']); hit = row.get('hit_index')
            if relation != 'gate' and refs: raise ValueError('Coupled evidence uses gate dependencies; timing is internal')
        elif kind == 'distance':
            action=config['actions'].get(s['id']) if config else None
            contact_detail = {}
            if action and action['kind'] == 'hinge_path':
                ds, contact_detail = trace.hinge_path(c['regions'], trace.clock(sampling.get('step_m')),
                                                     start, action, c['distance_scale_m'])
            elif action and action['kind'] in ('release', 'hidden_grasp'):
                clock = trace.clock(sampling.get('step_m'))
                ds = np.full(len(clock), 1e6)
                if action['kind'] == 'hidden_grasp':
                    for close, _ in trace.cycles:
                        index = np.searchsorted(clock, close)
                        ds[index] = min(region.distance([trace.ps[close]], r)[0] for r in c['regions'])
                elif eligible:
                    source = completed[action['source']]
                    grasp = int(clock[source['hit_index']])
                    release = next((e for c0,e in trace.cycles if c0 <= grasp and (e is None or grasp < e)), None)
                    unique_group = action.get('unique_release_group')
                    if release is not None and (unique_group, release) not in credited_releases:
                        point = trace.ps[release]
                        if 'object_point_world' in action:
                            source_regions = next(n['check']['regions'] for n in definition['stages'] if n['id']==action['source'])
                            offset = trace.centering.get((id(source_regions), grasp), np.zeros(3))
                            transform = trace.poses[release] @ np.linalg.inv(trace.poses[grasp])
                            point = transform[:3,:3] @ (np.asarray(action['object_point_world'])+offset)+transform[:3,3]
                        index = np.searchsorted(clock, release)
                        gap = min(region.distance([point], r)[0] for r in c['regions'])
                        if 'orientation_tolerance_deg' in action:
                            tilt = np.arccos(np.clip(transform[2,2], -1., 1.))
                            gap = max(gap, .025*max(0., tilt-np.deg2rad(action['orientation_tolerance_deg'])))
                        if 'alignment_xy' in action:
                            gap = max(gap, np.linalg.norm(point[:2]-action['alignment_xy'])-action['alignment_radius_m'])
                        ds[index] = gap
                        if unique_group and index >= start and gap <= c.get('hit_tolerance_m', 0.)+1e-9:
                            credited_releases.add((unique_group, release))
                        contact_detail['release_point_world'] = point.tolist()
            elif action:
                ds=trace.series(c['regions'],trace.clock(sampling.get('step_m')),action,c['distance_scale_m'])
            else:
                ds = np.min([region.distance(points, r) for r in c['regions']], axis=0)
            value = float(100*np.exp(-np.min(ds[start:])/c['distance_scale_m'])) if start < len(ds) else 0.
            hits = np.flatnonzero(ds[start:] <= c.get('hit_tolerance_m', 0.) + 1e-9) + start
            hit = int(hits[0]) if len(hits) else None; satisfied = hit is not None
            row = {'minimum_eligible_distance_m': float(ds[start:].min()) if start < len(ds) else None, 'best_path_index': start+int(np.argmin(ds[start:])) if start<len(ds) else None}
            if eligible:
                row.update(contact_detail)
        elif kind == 'authored':
            if sampling['type'] != 'authored': raise ValueError('Event checks require authored sampling')
            node = copy.deepcopy(c['parameters']); node.update(id=s['id'], after=[])
            from .authored import check
            row = check(prediction, initial, node, start)
            value = row['ordered_score']; hit = row.get('hit_index'); satisfied = row['hard_region_hit']
        else: raise ValueError('Unsupported check type: ' + str(kind))
        if not eligible: value = 0.; satisfied = False; hit = None
        if not math.isfinite(value) or not 0 <= value <= 100: raise ValueError('Invalid stage score')
        row.update(id=s['id'], score=value, ordered_score=value, satisfied=satisfied,
                   hard_region_hit=satisfied, hit_index=hit, dependency_blocked_by=blocked,
                   eligible_start_index=start, weight=s.get('weight', 1.))
        completed[s['id']] = row; rows.append(row)
    total = math.fsum(r['weight'] for r in rows)
    value = math.fsum(r['score']*r['weight'] for r in rows)/total
    return {'score_0_to_100': value, 'stages': rows, 'milestones': rows,
            'all_stages_satisfied': all(r['satisfied'] for r in rows),
            'evidence': evidence, 'aggregation': 'weighted_mean', 'physical_success_verified': False}
