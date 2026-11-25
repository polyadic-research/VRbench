"""Declarative local/Hugging Face dataset loader; no remote Python execution."""
import json
import re
from pathlib import Path
from .io import read, relative, sha
from .schema import validate


class Dataset:
    def __init__(self, root, verify=True):
        self.root = Path(root).resolve()
        self.manifest = read(self.root / "dataset.json")
        validate('dataset',self.manifest)
        if self.manifest.get("coordinate_frame") != "world" or self.manifest.get("position_unit") != "meter":
            raise ValueError("Expected world coordinates and meters")
        self.records = {}
        for line in (self.root / "tasks.jsonl").read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get("id", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in self.records:
                raise ValueError("Invalid or duplicate task ID")
            if row.get("type") not in ("atomic", "composite"):
                raise ValueError("Task type must be atomic or composite")
            path = relative(self.root, row["path"])
            if sha(path) != row["sha256"]:
                raise ValueError("Task manifest hash mismatch: " + name)
            task = read(path)
            validate('task',task)
            if task["id"] != name or task["type"] != row["type"]:
                raise ValueError("Task index identity mismatch")
            self.records[name] = (row, task, path.parent)
        if len(self.records) != self.manifest.get("task_count") or not self.records:
            raise ValueError("Dataset count mismatch")
        if verify:
            checks = read(self.root / "checksums.json")
            if not isinstance(checks, dict) or not checks:
                raise ValueError("Missing checksum inventory")
            for name, expected in checks.items():
                path = relative(self.root, name)
                if sha(path) != (expected["sha256"] if isinstance(expected, dict) else expected):
                    raise ValueError("Dataset checksum mismatch: " + name)
            required = {"dataset.json", "tasks.jsonl"}
            required.update(self.manifest['robots'].values())
            for row, task, folder in self.records.values():
                for name in (row["path"],):
                    required.add(name)
                for name in (task["input"], task["scene"], *task["grading"].values(), task["reference"]):
                    required.add(relative(folder, name).relative_to(self.root).as_posix())
                observation = read(relative(folder, task["input"]))
                validate('observation',observation)
                for camera in observation["cameras"]:
                    path = relative(relative(folder, task["input"]).parent, camera["image"])
                    required.add(path.relative_to(self.root).as_posix())
                scene=read(relative(folder,task['scene']));validate('scene',scene)
                for name in [scene['model'],*scene['assets'].values()]:
                    required.add(relative(self.root,name).relative_to(self.root).as_posix())
                for name in scene['assets']:
                    relative(self.root,name)
                meta=read(relative(folder,task['reference']))
                for key in ('states','tcp'):
                    required.add(relative(self.root,meta[key]).relative_to(self.root).as_posix())
                validate('regions', read(relative(folder, task['grading']['regions'])))
                self.collision_policy(task['id'])
                from .scoring.stages import validate_graph
                validate_graph(read(relative(folder,task['grading']['regions'])))
            if required - set(checks):
                raise ValueError("Checksum inventory omits required files: " + str(sorted(required-set(checks))))

    def task(self, name):
        return self.records[name][1]

    def path(self, name, key):
        _, task, folder = self.records[name]
        value = task
        for part in key.split("."):
            value = value[part]
        return relative(folder, value)

    def collision_policy(self, task):
        policy = read(self.path(task, 'grading.collision'))
        validate('collision', policy)
        return policy

    def select(self, tasks=None):
        selected = list(self.records) if tasks is None else list(tasks)
        if len(selected) != len(set(selected)) or not selected or set(selected)-set(self.records):
            raise ValueError("Invalid task selection")
        return selected


def download(repo, revision, destination=None):
    if not revision:
        raise ValueError("Explicit dataset revision is required")
    from huggingface_hub import HfApi, snapshot_download
    commit = HfApi().dataset_info(repo, revision=revision).sha
    if not destination:
        destination=Path.home()/'.cache/vrbench'/repo.replace('/','--')/commit
    path = snapshot_download(repo_id=repo, repo_type="dataset", revision=commit, local_dir=destination)
    Dataset(path)
    return {"path": str(path), "repo": repo, "commit": commit}
