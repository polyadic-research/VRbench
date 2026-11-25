"""Explicit exhausted-prediction records and their zero-score penalty."""
from pathlib import Path
from .io import read, sha


def prediction_path(value):
    path = Path(value)
    return path / 'prediction.json' if path.is_dir() else path


def no_answer(dataset, task, value):
    """Accept a task directory, marker, or its missing prediction.json path."""
    path = Path(value)
    prediction = prediction_path(path)
    marker = path / 'no_answer.json' if path.is_dir() else path.parent / 'no_answer.json'
    if path.name == 'no_answer.json':
        marker = path
        prediction = path.parent / 'prediction.json'
    elif not path.is_dir() and path.name != 'prediction.json':
        return None
    if not marker.exists():
        return None
    if prediction.exists():
        raise ValueError('Ambiguous task: both prediction.json and no_answer.json exist')
    record = read(marker)
    attempts = record.get('attempts', [])
    limit = record.get('max_attempts')
    identity = record.get('identity', {})
    if (record.get('id') != task or record.get('status') != 'no_answer'
            or record.get('reason') != 'attempt_limit_exhausted'
            or type(limit) is not int or limit < 1
            or record.get('attempts_used') != limit or len(attempts) != limit
            or [a.get('attempt') for a in attempts] != list(range(1, limit + 1))
            or any(a.get('status') != 'error' or a.get('quaternion_repair_eligible')
                   for a in attempts)
            or identity.get('dataset_sha256') != sha(dataset.root / 'dataset.json')
            or not identity.get('model')):
        raise ValueError('Invalid no-answer identity or attempt history')
    dataset.task(task)
    return record, marker


def zero_score(dataset, task, failure):
    record, marker = failure
    return {
        'id': task, 'type': dataset.task(task)['type'], 'status': 'scored',
        'prediction_status': 'no_answer', 'score_basis': 'no_answer_zero_penalty',
        'region_score': 0.0, 'collision_score': 0.0, 'reliability_weight': 0.0,
        'adjusted_collision_score': 0.0, 'combined_score': 0.0,
        'physical_measurements_available': False,
        'identity': record['identity'], 'attempts_used': record['attempts_used'],
        'no_answer_sha256': sha(marker), 'model_calls': 0, 'ik_calls': 0,
        'geometric_proxy_not_physical_success': True,
    }
