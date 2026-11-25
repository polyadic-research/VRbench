import numpy as np
from scipy.spatial import ConvexHull

def materialize_receiver(template,center_world,rotation_world=None):
    """Caller binds an explicit placement pose. Never selects a path point itself."""
    center=np.asarray(center_world,float);r=np.asarray(rotation_world if rotation_world is not None else template['initial_rotation_world'],float)
    if center.shape!=(3,) or not np.isfinite(center).all() or r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-6) or np.linalg.det(r)<0:
        raise ValueError('Invalid placed receiver pose')
    h=np.asarray(template['half_size'])
    p=np.array([[x,y,z] for x in (-h[0],h[0]) for y in (-h[1],h[1]) for z in (-h[2],h[2])])@r.T+center
    vertices=np.concatenate([p,p+[0,0,template['upward_extension_m']]])
    return dict(name=template['name'],type='convex_polytope',vertices_world=vertices.tolist(),
                halfspaces_world=ConvexHull(vertices).equations.tolist(),source_names=template['source_names'],
                role='placed_object_bbox_extruded_upwards',anchor_stage_id=template['anchor_stage_id'])

def clip_polygon(subject,clip):
    subject=subject[ConvexHull(subject).vertices];clip=clip[ConvexHull(clip).vertices]
    for start,end in zip(clip,np.roll(clip,-1,axis=0)):
        if len(subject)==0:break
        result=[];edge=end-start
        for p,q in zip(subject,np.roll(subject,-1,axis=0)):
            dp=np.cross(edge,p-start);dq=np.cross(edge,q-start)
            if dp>=-1e-10:result.append(p)
            if (dp>=0)!=(dq>=0):result.append(p+(q-p)*dp/(dp-dq))
        subject=np.asarray(result)
    return subject
