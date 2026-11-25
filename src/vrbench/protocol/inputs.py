import hashlib
import json
from importlib.resources import files
from pathlib import Path
from ..io import read,relative

PROMPT_SHA256="222aedb83688a7cd4bd50dd4e2899a57134e2a2e253c6e401799c01db4b03018"
FIELDS=("observation_profile","task_instruction","coordinate_frame","position_unit","initial_tcp","initial_gripper","robot_context","cameras")


def build(dataset,task):
    path=dataset.path(task,"input")
    observation=read(path)
    public={key:observation[key] for key in FIELDS}
    prompt=files("vrbench.protocol").joinpath("prompt.txt").read_text(encoding="utf8")
    if hashlib.sha256(prompt.encode()).hexdigest()!=PROMPT_SHA256:
        raise ValueError("Prediction prompt checksum mismatch")
    images=[]
    for camera in observation["cameras"]:
        image=relative(path.parent,camera["image"])
        if not image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Expected initial PNG image")
        images.append((camera["name"],image))
    if len(images)!=3:raise ValueError("Expected three initial images")
    text=prompt+"\n\n"+json.dumps(public,ensure_ascii=False,indent=2)+"\nAttached images, in order: "+", ".join(n for n,_ in images)+".\n"
    return {"prompt":text,"public":public,"images":images,"system_sha256":PROMPT_SHA256}
