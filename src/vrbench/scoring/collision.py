"""Geometric collision replay aligned to expert frames."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from ..io import read,relative
from ..protocol.math3d import inverse


def validate_xml(text, assets):
    """Require a self-contained XML; MuJoCo must not resolve host files."""
    root = ET.fromstring(text)
    for node in root.iter():
        if node.tag == 'include':
            raise ValueError('External XML includes are not supported; flatten the scene')
        if 'file' in node.attrib and node.attrib['file'] not in assets:
            raise ValueError('Scene file is absent from declared assets: ' + node.attrib['file'])
        if node.tag == 'compiler' and any(node.get(k) for k in ('meshdir', 'texturedir', 'assetdir')):
            raise ValueError('Flatten compiler asset paths into the declared asset mapping')


def descendants(model,root):
    ids={root}
    for body in range(root+1,model.nbody):
        if int(model.body_parentid[body]) in ids:ids.add(body)
    return ids


def self_exclusions(model,bodies,geoms):
    weld={}
    for body in sorted(bodies):
        parent=int(model.body_parentid[body]);weld[body]=weld[parent] if parent in weld and model.body_jntnum[body]==0 else body
    excluded=set()
    for i,a in enumerate(geoms):
        for b in geoms[i+1:]:
            wa,wb=weld[int(model.geom_bodyid[a])],weld[int(model.geom_bodyid[b])]
            if wa==wb or int(model.body_parentid[wa]) in bodies and weld[int(model.body_parentid[wa])]==wb or int(model.body_parentid[wb]) in bodies and weld[int(model.body_parentid[wb])]==wa:excluded.add(frozenset((model.geom(a).name,model.geom(b).name)))
    return excluded


def shape(model,i):
    import mujoco
    types={mujoco.mjtGeom.mjGEOM_BOX:'box',mujoco.mjtGeom.mjGEOM_SPHERE:'sphere',mujoco.mjtGeom.mjGEOM_CAPSULE:'capsule',mujoco.mjtGeom.mjGEOM_CYLINDER:'cylinder',mujoco.mjtGeom.mjGEOM_ELLIPSOID:'ellipsoid',mujoco.mjtGeom.mjGEOM_PLANE:'plane'}
    kind=int(model.geom_type[i])
    if kind in types:return {'type':types[kind],'size':model.geom_size[i].tolist()}
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
        mid=int(model.geom_dataid[i]);start,count=int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid]);return {'type':'convex','vertices':model.mesh_vert[start:start+count].astype(float).tolist()}
    raise ValueError('Unsupported scene geometry')


def components(raw,policy):
    n=raw['checked_samples']
    if n<=0:raise ValueError('No checked collision samples')
    per={};robot=lambda name:any(name.startswith(x) for x in policy['robot_name_prefixes'])
    for record in raw['records']:
        for hit in record['pairs']:
            a,b=hit['pair']
            if robot(a)==robot(b) or any(token.lower() in (a+' '+b).lower() for token in policy.get('exclude_pair_name_contains',[])):continue
            p=max(0.,float(hit['depth_mm'])-10);per[record['sample']]=max(per.get(record['sample'],0),p)
    mean=sum(per.values())/n;maximum=max(per.values(),default=0);fraction=sum(p>0 for p in per.values())/n
    return {'collision_score':100*(math.exp(-mean/10)+math.exp(-maximum/10)+1-fraction)/3,'mean_excess_penetration_mm':mean,'max_excess_penetration_mm':maximum,'environment_collision_sample_fraction':fraction}


def replay(dataset,task,robot,joints,refs,boundary_nodes=()):
    import mujoco
    from .geometry import Geometry
    scene=read(dataset.path(task,'scene'));policy=dataset.collision_policy(task)
    if policy.get('joint_sample_step_rad')!=.04 or policy.get('scoring_tolerance_mm')!=10 or policy.get('record_cap')!=1000:raise ValueError('Different collision policy requires a new declared protocol')
    xml=relative(dataset.root,scene['model']);assets={key:relative(dataset.root,value).read_bytes() for key,value in scene.get('assets',{}).items()}
    xml_text=xml.read_text(encoding='utf8');validate_xml(xml_text,assets)
    model=mujoco.MjModel.from_xml_string(xml_text,assets=assets);data=mujoco.MjData(model)
    meta=read(dataset.path(task,'reference'));states=np.load(relative(dataset.root,meta['states']),allow_pickle=False)[meta.get('state_key','states')]
    expert=np.load(relative(dataset.root,meta['tcp']),allow_pickle=False);positions=expert[meta.get('position_key','tcp_position_world')];quats=expert[meta.get('quaternion_key','tcp_quaternion_xyzw_world')]
    if not np.isfinite(states).all() or not np.isfinite(positions).all() or len(states)!=len(positions) or states.shape[1]!=1+model.nq+model.nv+model.na:raise ValueError('Invalid expert replay arrays')
    rotations=Rotation.from_quat(quats);frames=[];lower=0
    for ref in refs:
        errors=np.linalg.norm(positions[lower:]-ref[:3,3],axis=1)/.005+(rotations[lower:].inv()*Rotation.from_matrix(ref[:3,:3])).magnitude()/.03
        lower+=int(np.argmin(errors));frames.append(lower)
    root=model.body(scene['robot_root']).id;robot_bodies=descendants(model,root);active=[i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    robot_ids=[i for i in active if int(model.geom_bodyid[i]) in robot_bodies];other=[i for i in active if i not in robot_ids]
    excluded_bodies=set();unresolved=[];resolved=[]
    for name in policy['excluded_object_roots']:
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if body<0:unresolved.append(name)
        else:resolved.append(name);excluded_bodies.update(descendants(model,body))
    if unresolved and not policy.get('allow_unresolved_exclusions',False):raise ValueError('Unresolved collision exclusion '+str(unresolved))
    names=[model.geom(i).name for i in robot_ids if not any(model.geom(i).name.startswith(p) for p in policy.get('skip_robot_geom_prefixes',['gripper0_']))]
    others=[model.geom(i).name for i in other if int(model.geom_bodyid[i]) not in excluded_bodies]
    exclusions=self_exclusions(model,robot_bodies,robot_ids)
    geometry=Geometry([{'name':model.geom(i).name,'body':model.body(int(model.geom_bodyid[i])).name,'group':'dynamic',**shape(model,i)} for i in active])
    def pose(pos,rot):
        t=np.eye(4);t[:3,:3]=np.asarray(rot).reshape(3,3);t[:3,3]=pos;return t
    data.qpos[:]=states[0,1:1+model.nq];data.qvel[:]=states[0,1+model.nq:1+model.nq+model.nv];mujoco.mj_forward(model,data)
    local={model.geom(i).name:inverse(pose(data.xpos[int(model.geom_bodyid[i])],data.xmat[int(model.geom_bodyid[i])]))@pose(data.geom_xpos[i],data.geom_xmat[i]) for i in robot_ids}
    link=model.body(scene['registration_body']).id;records=[];checked=0;previous=[]
    sampled_nodes=[]; penetration=[]; boundaries={}
    by_layer={}
    for node in boundary_nodes:
        layer=math.ceil(node)
        by_layer.setdefault(layer,[]).append(node)
    prefixes=policy['robot_name_prefixes']
    exclude=[x.lower() for x in policy.get('exclude_pair_name_contains',[])]
    for layer,q in enumerate(joints):
        count=1 if layer==0 else max(1,int(np.ceil(np.max(abs(q-joints[layer-1]))/.04)))
        samples=[(0.,q)] if layer==0 else [(u,(1-u)*joints[layer-1]+u*q) for u in np.linspace(0,1,count+1)[1:]]
        state=states[frames[layer]];data.qpos[:]=state[1:1+model.nq];data.qvel[:]=state[1+model.nq:1+model.nq+model.nv];mujoco.mj_forward(model,data)
        samples=[(u,qs,None) for u,qs in samples]
        for node in by_layer.get(layer, []):
            u=node-(layer-1) if layer else 0.
            qs=(1-u)*joints[layer-1]+u*q if layer else q
            samples.append((u,qs,node))
        for u,qs,boundary in samples:
            fk,_=robot.fk(qs);registration=pose(data.xpos[link],data.xmat[link])@inverse(fk[scene['registration_body']])
            geometry.update({model.geom(i).name:registration@fk[model.body(int(model.geom_bodyid[i])).name]@local[model.geom(i).name] if int(model.geom_bodyid[i]) in robot_bodies else pose(data.geom_xpos[i],data.geom_xmat[i]) for i in active})
            pairs=list(geometry.broad_pairs(names,others))+list(geometry.broad_pairs(names,names,self_pairs=True));seen=set();hits=[]
            for a,b in previous+pairs:
                key=frozenset((a,b))
                if key in seen or key in exclusions:continue
                seen.add(key)
                if geometry.collide_any(a,b):
                    hit,depth=geometry.collide(a,b,max_contacts=1)
                    if hit:hits.append((a,b,depth))
            excess=max((max(0.,d*1000-10) for a,b,d in hits
                        if a.startswith(tuple(prefixes)) != b.startswith(tuple(prefixes))
                        and not any(x in (a+' '+b).lower() for x in exclude)),default=0.)
            if boundary is not None:
                boundaries[boundary]=excess
                continue
            sampled_nodes.append(layer-1+float(u) if layer else 0.)
            penetration.append(excess)
            previous=[(a,b) for a,b,_ in hits];bad=[(a,b,d) for a,b,d in hits if d>.001]
            if bad and len(records)<1000:records.append({'sample':checked,'layer':layer,'expert_frame':frames[layer],'segment_fraction':float(u),'pairs':[{'pair':[a,b],'depth_mm':d*1000} for a,b,d in bad]})
            checked+=1
    raw={'checked_samples':checked,'records':records,'resolved_excluded_object_roots':resolved,'unresolved_exclusion_roots':unresolved,'records_may_be_truncated':len(records)>=1000}
    raw.update(sampled_nodes=sampled_nodes,excess_penetration_mm=penetration,boundary_penetration_mm=boundaries)
    scores=components(raw,policy)
    p=np.asarray(penetration)
    scores.update(collision_score=float(100*(math.exp(-p.mean()/10)+math.exp(-p.max()/10)+1-np.mean(p>0))/3),
                  mean_excess_penetration_mm=float(p.mean()),max_excess_penetration_mm=float(p.max()),
                  environment_collision_sample_fraction=float(np.mean(p>0)))
    return raw,scores
