import math
import numpy as np
from scipy.spatial.transform import Rotation


def components(position_rms_mm,orientation_rms_deg,joint_step_p95_deg,joint_step_max_deg):
    values=(position_rms_mm,orientation_rms_deg,joint_step_p95_deg,joint_step_max_deg)
    if any(not math.isfinite(x) or x<0 for x in values):raise ValueError("Invalid reliability measurements")
    P=math.exp(-position_rms_mm/10);O=math.exp(-orientation_rms_deg/5)
    J=math.exp(-.5*(max(0,joint_step_p95_deg-30)/30+max(0,joint_step_max_deg-90)/90))
    return {"position_score":P,"orientation_score":O,"joint_score":J,"reliability_weight":J*math.sqrt(P*O)}


def measure(robot,joints,prediction,nodes):
    if len(nodes)!=36 or len(prediction["waypoints"])!=36 or not all(a<b for a,b in zip(nodes,nodes[1:])) or min(nodes)<1 or max(nodes)>=len(joints):
        raise ValueError("Invalid author-waypoint mapping")
    pos=[];rot=[]
    for node,w in zip(nodes,prediction["waypoints"]):
        actual=robot.fk(joints[node])[1]
        pos.append(1000*np.linalg.norm(actual[:3,3]-w["tcp_position"]))
        rot.append(np.degrees(Rotation.from_matrix(actual[:3,:3].T@Rotation.from_quat(w["tcp_quaternion_xyzw"]).as_matrix()).magnitude()))
    step=np.degrees(np.max(abs(np.diff(np.vstack([joints[0],joints[nodes]]),axis=0)),axis=1))
    result={"position_rms_mm":float(np.sqrt(np.mean(np.square(pos)))),"orientation_rms_deg":float(np.sqrt(np.mean(np.square(rot)))),"joint_step_p95_deg":float(np.percentile(step,95)),"joint_step_max_deg":float(step.max())}
    return {**result,**components(**result)}
