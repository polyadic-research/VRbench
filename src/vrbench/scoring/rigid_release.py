"""Credit a placement at the open event of a unique, geometrically valid grasp."""
from itertools import product

import numpy as np

from . import manipulation as motion


def evaluate(positions, poses, cycles, rule,trace=None):
    objects = rule['objects']
    distances = [[motion.distance(positions[c], obj['source_regions'])[0] for c, _ in cycles]
                 for obj in objects]

    anchors=[[c for c,_ in cycles] for _ in objects]
    if trace:
        for i,obj in enumerate(objects):
            for j,(c,e) in enumerate(cycles):
                distances[i][j],anchors[i][j]=trace.grip(obj['source_regions'],c,e)

    def cost(indices):
        ds = [values[i] for values, i in zip(distances, indices) if i >= 0]
        return (-sum(d <= 1e-9 for d in ds), len(indices) - len(ds), sum(ds), indices)

    assignments = (indices for indices in product(range(-1, len(cycles)), repeat=len(objects))
                   if len({i for i in indices if i >= 0}) == sum(i >= 0 for i in indices))
    rows = []
    for obj, values, valid, idx in zip(objects, distances, anchors, min(assignments, key=cost)):
        close, release = cycles[idx] if idx >= 0 else (None, None)
        if idx>=0:close=valid[idx]
        source = motion.stage(obj['source'], values[idx] if idx >= 0 else None, close)
        rows.append(source)
        distance = None
        if release is not None and source['hard_region_hit']:
            shift=trace.centering.get((id(obj['source_regions']),close),np.zeros(3)) if trace else np.zeros(3)
            offset = np.linalg.solve(poses[close], [*(np.asarray(obj['center_world'])+shift), 1.])
            center = (poses[release] @ offset)[:3]
            distance = motion.distance(center, obj['destination_regions'])[0]
            if trace and 'support_vertices_world' in obj:
                from .interaction import support_distance
                T=poses[release]@np.linalg.inv(poses[close])
                moved=np.einsum('ij,kj->ki',T[:3,:3],np.asarray(obj['support_vertices_world'])+shift)+T[:3,3]
                distance=support_distance(center,moved,obj['destination_regions'])
        rows.append(motion.stage(obj['destination'], distance, release, motion.dependencies([source])))
    return motion.finish(rows)
