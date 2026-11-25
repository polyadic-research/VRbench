"""Score a frozen prediction and an existing joint path; no model or IK calls."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..io import read, write, sha, relative, unchanged
from ..protocol.validation import validate_prediction
from ..failures import no_answer, zero_score, prediction_path
from ..solver.robot import Robot
from . import region, reliability, combined, stagewise


def score(dataset,task,prediction_file,path_directory,output):
    failure=no_answer(dataset,task,prediction_file)
    if failure:
        out=Path(output);out.mkdir(parents=True,exist_ok=False)
        result=zero_score(dataset,task,failure)
        write(out/'result.json',result);return result
    if path_directory is None:
        raise ValueError('A solved --path is required for a valid prediction')
    prediction_file=prediction_path(prediction_file)
    from .collision import replay
    out=Path(output);out.mkdir(parents=True,exist_ok=False);path_directory=Path(path_directory)
    report=read(path_directory/'result.json');prediction=read(prediction_file);validate_prediction(prediction,36)
    if report['id']!=task or not report['complete_finite'] or report['prediction_sha256']!=sha(prediction_file):raise ValueError('Prediction/path identity mismatch')
    jp=relative(path_directory,report['joint_path']);rp=relative(path_directory,report['reference_path']);hashes={str(prediction_file):sha(prediction_file),str(jp):report['joint_sha256'],str(rp):report['reference_sha256']};unchanged(hashes)
    q=np.load(jp,allow_pickle=False);refs=np.load(rp,allow_pickle=False);nodes=report['waypoint_nodes']
    robot_path=relative(dataset.root,dataset.manifest['robots'][dataset.task(task)['robot']]);robot=Robot(read(robot_path))
    if q.shape!=(len(refs),len(robot.arm)) or refs.shape!=(len(q),4,4) or not np.isfinite(q).all() or not np.isfinite(refs).all() or not np.allclose(q[0],robot.q0,atol=1e-8,rtol=0) or np.any(q<robot.lower-1e-7) or np.any(q>robot.upper+1e-7):raise ValueError('Invalid path arrays or bounds')
    for node,w in zip(nodes,prediction['waypoints']):
        if not np.allclose(refs[node,:3,3],w['tcp_position'],atol=1e-8,rtol=0) or not np.allclose(refs[node,:3,:3],Rotation.from_quat(w['tcp_quaternion_xyzw']).as_matrix(),atol=1e-8,rtol=0):raise ValueError('Reference waypoint mismatch')
    definition=read(dataset.path(task,'grading.regions'));details=region.score(prediction,{'position': robot.fk(robot.q0)[1][:3,3].tolist(), 'quaternion': Rotation.from_matrix(robot.fk(robot.q0)[1][:3,:3]).as_quat().tolist()},definition);write(out/'regions.json',details)
    write(out/'definition.json',definition)
    inventory=read(dataset.root/'checksums.json')
    for name,expected in inventory.items():
        hashes[str(relative(dataset.root,name))]=expected['sha256'] if isinstance(expected,dict) else expected
    hashes[str(dataset.root/'checksums.json')]=sha(dataset.root/'checksums.json')
    initial_tcp=robot.fk(robot.q0)[1][:3,3]
    positions=np.array([initial_tcp,*[w['tcp_position'] for w in prediction['waypoints']]])
    windows=stagewise.windows(definition,details,positions)
    ends={x['interval'][1] for x in windows if 'interval' in x}
    mapped={t:float(np.interp(t,np.arange(37),[0,*nodes])) for t in ends}
    raw,contact=replay(dataset,task,robot,q,refs,mapped.values())
    clock=np.interp(raw['sampled_nodes'],[0,*nodes],np.arange(37))
    boundary={t:raw['boundary_penetration_mm'][node] for t,node in mapped.items()}
    overall=stagewise.aggregate(windows,clock,np.asarray(raw['excess_penetration_mm']),boundary,
                                stagewise.tracking_samples(robot,q,prediction,nodes))
    write(out/'collision.json',raw)
    weight=reliability.measure(robot,q,prediction,nodes);result={'id':task,'type':dataset.task(task)['type'],'status':'scored',**contact,**weight,
       **combined.combined(details['score_0_to_100'],contact['collision_score'],weight['reliability_weight']),
       'identity':report.get('identity'),'strict_default':report['strict_default'],'path_source':report['selected'],'collision_records_may_be_truncated':raw['records_may_be_truncated'],
       'unresolved_exclusions':raw['unresolved_exclusion_roots'],'model_calls':0,'ik_calls':0,'input_sha256':hashes,
       'waypoint_nodes':nodes,'path_report_sha256':sha(path_directory/'result.json'),
       'implementation_sha256':{str(p.relative_to(Path(__file__).parent.parent)):sha(p) for p in Path(__file__).parent.parent.rglob('*.py')},
       'geometric_proxy_not_physical_success':True}
    result.update(combined_score=overall,stage_scores=windows,composite_parameters=stagewise.PARAMETERS,
                  score_basis='stagewise_contact_tracking')
    unchanged(hashes);write(out/'result.json',result);return result
