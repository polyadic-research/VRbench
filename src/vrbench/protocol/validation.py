import copy
import math
import numpy as np
from .math3d import pose


class QuaternionNormError(ValueError):
    """Otherwise valid pose data contains a nonunit quaternion."""


def validate_prediction(prediction, limit=36):
    if not isinstance(prediction, dict):
        raise ValueError("Expected a prediction object")
    if set(prediction) != {"coordinate_frame", "position_unit", "waypoints"}:
        raise ValueError("Unexpected prediction root fields")
    if prediction["coordinate_frame"] != "world" or prediction["position_unit"] != "meter":
        raise ValueError("Expected world coordinates and meters")
    points = prediction["waypoints"]
    if not isinstance(points, list) or not 1 <= len(points) <= limit:
        raise ValueError("Invalid waypoint count")
    result = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("Expected a waypoint object")
        if set(point) != {"tcp_position", "tcp_quaternion_xyzw", "gripper_event", "contact_event"}:
            raise ValueError("Unexpected waypoint fields")
        for key, length in (("tcp_position", 3), ("tcp_quaternion_xyzw", 4)):
            values = point[key]
            if not isinstance(values, list) or len(values) != length or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
                raise ValueError("Expected finite numeric pose arrays")
        if point["gripper_event"] not in (None, "open", "close") or point["contact_event"] not in (None, "engage", "disengage"):
            raise ValueError("Invalid gripper or contact event")
        norm = math.hypot(*point["tcp_quaternion_xyzw"])
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError("Quaternion must be finite and nonzero")
    if any(abs(math.hypot(*p["tcp_quaternion_xyzw"])-1) > 1e-3 for p in points):
        raise QuaternionNormError("Quaternion must be unit length; predictions are not silently normalized")
    for point in points:
        result.append((pose(point["tcp_position"],point["tcp_quaternion_xyzw"]),point["gripper_event"],point["contact_event"]))
    return result


def normalize_count(prediction):
    validate_prediction(prediction,39)
    count=len(prediction["waypoints"])
    if not 33<=count<=39:
        raise ValueError("Expected 33 to 39 raw waypoints")
    result=copy.deepcopy(prediction)
    if count>36:result["waypoints"]=result["waypoints"][:36]
    while len(result["waypoints"])<36:result["waypoints"].append(copy.deepcopy(result["waypoints"][-1]))
    validate_prediction(result,36)
    return result,{"raw_waypoint_count":count,"action":"truncate_tail" if count>36 else "repeat_last" if count<36 else "unchanged"}
