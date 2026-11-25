"""Quaternion normalization with per-waypoint diagnostics."""
import copy
from pathlib import Path
import math
from .validation import normalize_count,validate_prediction,QuaternionNormError
from ..io import read,write,sha


def normalize_quaternions(prediction):
    """Normalize quaternion lengths and validate the resulting prediction."""
    try:
        validate_prediction(prediction,39)
    except QuaternionNormError:
        pass
    result=copy.deepcopy(prediction);audit=[]
    for i,w in enumerate(result['waypoints'],1):
        q=w['tcp_quaternion_xyzw'];norm=math.hypot(*q)
        audit.append({'waypoint':i,'original':list(q),'norm':norm,
                      'outside_tolerance':abs(norm-1)>1e-3})
        w['tcp_quaternion_xyzw']=[v/norm for v in q]
    fixed,adjustment=normalize_count(result);validate_prediction(fixed,36)
    return fixed,audit,adjustment


def repair(source,output,reason):
    if not reason.strip():raise ValueError('Repair selection reason required')
    prediction=read(source);before=sha(source)
    fixed,audit,adjustment=normalize_quaternions(prediction)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'original.json').write_bytes(Path(source).read_bytes());write(out/'prediction.json',fixed)
    record={'source_sha256':before,'prediction_sha256':sha(out/'prediction.json'),'selection_reason':reason,'original_protocol_valid':False,'classification':'quaternion_normalized_postprocessed','points':audit,'point_count_adjustment':adjustment,'model_calls':0,'ik_calls':0}
    if sha(source)!=before:raise ValueError('Source changed')
    write(out/'audit.json',record);return record
