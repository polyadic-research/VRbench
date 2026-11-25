"""The frozen Guided Anytime default followed by SparseAdaptive fallback."""
from importlib.resources import files
from pathlib import Path
import json
import time
import numpy as np
from .robot import Robot
from .path import dense_reference
from .guided import GuidedAnytimeSolver, assess_tracking
from .sparse import SparseAdaptive
from .timing import virtual_times
from ..io import read,write,sha
from ..protocol.validation import validate_prediction
from ..failures import no_answer, prediction_path


def solve(dataset,task,prediction_file,output,config_file=None):
    failure=no_answer(dataset,task,prediction_file)
    if failure:
        record,marker=failure
        out=Path(output);out.mkdir(parents=True,exist_ok=False)
        result={'id':task,'status':'no_answer','prediction_status':'no_answer',
                'complete_finite':False,'identity':record['identity'],
                'no_answer_sha256':sha(marker),'attempts_used':record['attempts_used'],
                'model_calls':0,'ik_calls':0,'score_policy':'all_components_zero'}
        write(out/'result.json',result);return result
    prediction_file=prediction_path(prediction_file)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    config=read(config_file) if config_file else json.loads(files('vrbench').joinpath('resources/solver.json').read_text(encoding='utf8'))
    prediction=read(prediction_file);points=validate_prediction(prediction,36)
    identity_path=Path(prediction_file).parent/'identity.json'
    identity=read(identity_path) if identity_path.exists() else None
    if identity and identity['prediction_sha256']!=sha(prediction_file):raise ValueError('Prediction identity hash mismatch')
    if len(points)!=36:raise ValueError('Expected frozen 36-point prediction')
    name=dataset.task(task)['robot'];specpath=dataset.root/dataset.manifest['robots'][name]
    from ..io import relative
    specpath=relative(dataset.root,dataset.manifest['robots'][name]);robot=Robot(read(specpath))
    refs,segments=dense_reference(robot,prediction,config)
    times,velocities,timing=virtual_times(robot,len(refs),config['virtual_time'])
    start=time.monotonic();planner=GuidedAnytimeSolver(robot,refs,times,velocities,config['planner'],config['replay'],config['seed'])
    path,report=planner.solve();q=np.empty((0,len(robot.arm))) if path is None else np.asarray(path)
    refs=np.asarray(refs);np.save(out/'joint_path_default.npy',q);np.save(out/'reference_default.npy',refs)
    write(out/'default.json',report);strict=bool(report.get('trackable_path_found'));selected='default';fallback=None
    if not strict:
        sparse_refs=np.asarray([robot.fk(robot.q0)[1],*[p[0] for p in points]])
        planner=SparseAdaptive(robot,sparse_refs,config,seed=1907);candidate,chosen=planner.run()
        candidate=np.empty((0,len(robot.arm))) if candidate is None else np.asarray(candidate)
        tracking=assess_tracking(robot,candidate,sparse_refs,config['planner'],config['replay']) if len(candidate)==len(sparse_refs) else None
        accepted=[] if chosen is None else [bool(x.accepted) for x in chosen]
        fallback={'ik_calls':planner.calls,'phase_calls':planner.phase_calls,'accepted':accepted,'tracking':tracking,
                  'status':'accepted_continuous_path' if accepted and all(accepted) and tracking and tracking['valid'] else 'accepted_sparse_path' if accepted and all(accepted) else 'best_effort_sparse_path' if len(candidate) else 'no_path'}
        np.save(out/'joint_path_sparse.npy',candidate);np.save(out/'reference_sparse.npy',sparse_refs);write(out/'fallback.json',fallback)
        if candidate.shape==(len(sparse_refs),len(robot.arm)) and np.isfinite(candidate).all():selected='sparse';q=candidate;refs=sparse_refs
    nodes=list(range(1,37)) if selected=='sparse' else [int(s['end_node']) for s in segments]
    complete=bool(q.shape==(len(refs),len(robot.arm)) and len(q) and np.isfinite(q).all() and max(nodes)<len(q))
    jp=out/f'joint_path_{selected}.npy';rp=out/f'reference_{selected}.npy'
    result={'id':task,'identity':identity,'selected':selected,'strict_default':strict,'complete_finite':complete,'waypoint_nodes':nodes,
            'prediction_sha256':sha(prediction_file),'joint_sha256':sha(jp),'reference_sha256':sha(rp),
            'joint_path':jp.name,'reference_path':rp.name,'wall_seconds':time.monotonic()-start,'fallback':fallback,
            'model_calls':0,'robot_sha256':sha(specpath),'config':config,'timing':timing}
    write(out/'result.json',result);return result
