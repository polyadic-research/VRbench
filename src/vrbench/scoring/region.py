"""Declarative spatial milestones; event-specific rules are explicit handlers."""
import numpy as np
from scipy.spatial import ConvexHull

def base_distance(points, r):
    points = np.asarray(points, float)
    if r['type'] in ('box', 'cylinder'):
        transform = np.asarray(r['world_pose'], float)
        local = np.einsum('ij,jk->ik', points - transform[:3, 3], transform[:3, :3])
        if r['type'] == 'box':
            return np.linalg.norm(np.maximum(abs(local) - r['size'], 0), axis=1)
        return np.hypot(np.maximum(np.linalg.norm(local[:, :2], axis=1) - r['radius'], 0), np.maximum(abs(local[:, 2]) - r['half_height'], 0))
    if r['type'] == 'unresolved_dynamic':
        return np.full(len(points), 1e+100)
    if r['type'] != 'convex_polytope':
        raise ValueError('Unsupported region geometry: ' + r['type'])
    vertices = np.asarray(r['vertices_world'])
    h = np.asarray(r['halfspaces_world'])
    outside = np.any(np.einsum('ij,kj->ik', points, h[:, :3]) + h[:, 3] > 1e-09, axis=1)
    result = np.zeros(len(points))
    ps = points[outside]
    if not len(ps):
        return result
    best = np.full(len(ps), np.inf)
    for tri in ConvexHull(vertices).simplices:
        a, b, c = vertices[tri]
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        area = np.linalg.norm(n)
        if area < 1e-14:
            continue
        n /= area
        signed = np.einsum('ij,j->i', ps - a, n)
        projected = ps - signed[:, None] * n
        rel = projected - a
        u = np.einsum('ij,j->i', np.cross(rel, ac), n) / area
        w = np.einsum('ij,j->i', np.cross(ab, rel), n) / area
        ds = np.where((u >= -1e-10) & (w >= -1e-10) & (u + w <= 1 + 1e-10), abs(signed), np.inf)
        for x, y in ((a, b), (b, c), (c, a)):
            edge = y - x
            t = np.clip(np.einsum('ij,j->i', ps - x, edge) / (np.dot(edge, edge)), 0, 1)
            ds = np.minimum(ds, np.linalg.norm(ps - x - t[:, None] * edge, axis=1))
        best = np.minimum(best, ds)
    result[outside] = best
    return result

def densify(points, step):
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Invalid region sampling step')
    result = [np.asarray(points[0], float)]
    for a, b in zip(points, points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(np.asarray(b) - a) / step)))
        result.extend(((1 - u) * np.asarray(a) + u * np.asarray(b) for u in np.linspace(0, 1, count + 1)[1:]))
    return np.asarray(result)

def score(prediction, initial, definition):
    from .stages import score as graph_score
    return graph_score(prediction, initial, definition)


def sphere_surface_distance(points, region):
    points=np.asarray(points,float)
    surface=region['surface'];center=np.asarray(region['sphere_center_world'],float)
    radius=region['radius_m'];vertices=np.asarray(surface['vertices_world'],float)
    low,high=vertices[:,2].min(),vertices[:,2].max()
    footprint=ConvexHull(np.unique(vertices[:,:2],axis=0))
    polygon=footprint.points[footprint.vertices]
    expected=np.array([[*xy,z] for xy in polygon for z in (low,high)])
    if not all(np.min(np.linalg.norm(expected-v,axis=1))<1e-8 for v in vertices):
        raise ValueError('Counter intersection requires a horizontal prism')
    edges=np.roll(polygon,-1,axis=0)-polygon
    norms=np.sum(edges**2,axis=1)
    def project(ps):
        ps=np.atleast_2d(ps);xy=ps[:,:2].copy()
        outside=np.max(np.einsum('ij,kj->ik', xy, footprint.equations[:,:2])+footprint.equations[:,2],axis=1)>0
        if np.any(outside):
            chosen=xy[outside]
            t=np.clip(np.sum((chosen[:,None,:]-polygon)*edges,axis=2)/norms,0,1)
            candidates=polygon+t[:,:,None]*edges
            best=np.argmin(np.sum((candidates-chosen[:,None,:])**2,axis=2),axis=1)
            xy[outside]=candidates[np.arange(len(best)),best]
        return np.column_stack((xy,np.clip(ps[:,2],low,high)))
    if np.linalg.norm(project(center)[0]-center)>=radius:
        raise ValueError('Counter/sink intersection is empty or tangent')
    projected=project(points)
    active=np.linalg.norm(projected-center,axis=1)>radius
    if np.any(active):
        ps=points[active];lo=np.zeros(len(ps));hi=np.ones(len(ps))
        for _ in range(50):
            mid=(lo+hi)/2
            q=project((1-mid[:,None])*ps+mid[:,None]*center)
            outside=np.linalg.norm(q-center,axis=1)>radius
            lo=np.where(outside,mid,lo);hi=np.where(outside,hi,mid)
        mid=(lo+hi)/2
        projected[active]=project((1-mid[:,None])*ps+mid[:,None]*center)
    return np.linalg.norm(points-projected,axis=1)

def distance(points, region):
    points = np.asarray(points, float)
    if region["type"] == "counter_ball_intersection":
        return sphere_surface_distance(points, region)
    if region["type"] == "annular_cylinder":
        local = np.einsum('ij,jk->ik', points - np.asarray(region['world_pose'])[:3, 3], np.asarray(region['world_pose'])[:3, :3])
        radial = np.linalg.norm(local[:, :2], axis=1)
        radial_gap = np.maximum(np.maximum(region["inner_radius_m"] - radial,
                                           radial - region["outer_radius_m"]), 0)
        return np.hypot(radial_gap, np.maximum(np.abs(local[:, 2]) - region["half_height_m"], 0))
    return base_distance(points, region)
