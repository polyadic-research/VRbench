import math


def combined(region,collision,weight):
    if not all(math.isfinite(x) for x in (region,collision,weight)) or not 0<=region<=100 or not 0<=collision<=100 or not 0<=weight<=1:
        raise ValueError("Invalid component range")
    return {"region_score":region,"collision_score":collision,"reliability_weight":weight,"adjusted_collision_score":collision*weight,"combined_score":(region+collision*weight)/2}


def summarize(results,tasks):
    index={row["id"]:row for row in results}
    if len(index)!=len(results) or set(index)-set(tasks):raise ValueError("Invalid result identity")
    identities=set()
    for row in results:
        if row.get('status')!='scored':continue
        expected=combined(row['region_score'],row['collision_score'],row['reliability_weight'])
        if row.get('score_basis')=='stagewise_contact_tracking':
            from .stagewise import PARAMETERS, contribution
            if row.get('composite_parameters') != PARAMETERS:
                raise ValueError('Unexpected stagewise scoring parameters')
            stages = row['stage_scores']
            if not math.isclose(math.fsum(s['weight'] for s in stages),1.,abs_tol=1e-10):
                raise ValueError('Stage weights do not sum to one')
            if not math.isclose(math.fsum(s['weight']*s['region_stage_score'] for s in stages),row['region_score'],abs_tol=1e-8):
                raise ValueError('Stage credits do not match region score')
            terms = []
            for s in stages:
                term = contribution(s['weight'],s['region_stage_score'],s['collision_coefficient'],s['tracking_coefficient']) if s['region_stage_score'] else 0.
                if not math.isclose(term,s['contribution'],abs_tol=1e-9):
                    raise ValueError('Stage contribution does not match its components')
                terms.append(term)
            expected['combined_score'] = math.fsum(terms)
            if not 0<=expected['combined_score']<=row['region_score']+1e-8:
                raise ValueError('Invalid stagewise score')
        if any(not math.isclose(row[k],v,rel_tol=1e-10,abs_tol=1e-9) for k,v in expected.items()):
            raise ValueError('Inconsistent score components')
        if row.get('prediction_status')=='no_answer':
            if row.get('score_basis')!='no_answer_zero_penalty' or any(row[k]!=0 for k in expected):
                raise ValueError('No-answer results must use the explicit all-zero penalty')
        identity=row.get('identity')
        if identity:identities.add(tuple(identity.get(k) for k in ('model','reasoning_effort','transport','dataset_sha256')))
    if len(identities)>1:raise ValueError('Cannot aggregate different model or dataset identities')
    groups={"all":set(tasks),"atomic":{k for k,v in tasks.items() if v=="atomic"},"composite":{k for k,v in tasks.items() if v=="composite"}}
    report={}
    for name,expected in groups.items():
        present=[index[k] for k in sorted(expected) if k in index and index[k].get("status")=="scored"]
        complete=len(present)==len(expected) and bool(expected)
        report[name]={"expected":len(expected),"scored":len(present),"no_answer":sum(r.get('prediction_status')=='no_answer' for r in present),"complete":complete,"means":{key:sum(r[key] for r in present)/len(present) for key in ("region_score","collision_score","reliability_weight","adjusted_collision_score","combined_score")} if complete else None}
    return report
