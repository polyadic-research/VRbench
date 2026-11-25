"""Score the predictions selected by a batch solve and summarize the requested tasks."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

from ..failures import no_answer, prediction_path
from ..io import read, sha, write
from .combined import summarize
from .pipeline import score


def _inputs(dataset, selected, paths):
    manifest = read(paths / 'summary.json')
    cases = manifest['cases']
    index = {case['id']: case for case in cases}
    if len(index) != len(cases):
        raise ValueError('Duplicate task IDs in solve summary')
    base = paths / manifest['prediction_base'] if 'prediction_base' in manifest else Path.cwd()
    dataset_hash = sha(dataset.root / 'dataset.json')
    inputs = {}
    identities = set()
    for task in selected:
        case = index.get(task)
        if case is None or case['status'] not in ('complete', 'no_answer'):
            raise ValueError(f'{task}: missing or unfinished joint solve')
        prediction = base / case['prediction']
        report = read(paths / task / 'result.json')
        identity = report.get('identity') or {}
        if report['id'] != task or identity.get('dataset_sha256') != dataset_hash:
            raise ValueError(f'{task}: joint path identity does not match the dataset')
        identities.add(tuple(identity.get(k) for k in ('model', 'reasoning_effort', 'transport')))
        failure = no_answer(dataset, task, prediction)
        if case['status'] == 'no_answer':
            if (not failure or report.get('status') != 'no_answer'
                    or report.get('no_answer_sha256') != sha(failure[1])):
                raise ValueError(f'{task}: no-answer record differs from the solve')
        elif (failure or not report.get('complete_finite')
              or sha(prediction_path(prediction)) != report.get('prediction_sha256')):
            raise ValueError(f'{task}: prediction differs from the joint solve')
        inputs[task] = prediction
    if len(identities) != 1:
        raise ValueError('Cannot score a batch with mixed model identities')
    return inputs


def _score_task(dataset, task, prediction, paths, output):
    with TemporaryDirectory(dir=output, prefix=f'.{task}-') as staging:
        work = Path(staging) / 'score'
        result = score(dataset, task, prediction, paths / task, work)
        work.rename(output / task)
    result.pop('input_sha256', None)
    result.pop('implementation_sha256', None)
    return result


def score_batch(dataset, tasks, paths, output, concurrency):
    selected = dataset.select(tasks)
    inputs = _inputs(dataset, selected, paths)
    types = {task: dataset.task(task)['type'] for task in selected}
    output.mkdir(parents=True, exist_ok=False)
    cases = {task: {'id': task, 'status': 'pending'} for task in selected}
    summary = {'cases': list(cases.values()), 'batch_completed': False}
    destination = output / 'summary.json'
    write(destination, summary)
    results = []
    print(f'Scoring {len(selected)} tasks in {output}', flush=True)
    with ProcessPoolExecutor(max_workers=min(concurrency, len(selected))) as pool:
        futures = {pool.submit(_score_task, dataset, task, source, paths, output): task
                   for task, source in inputs.items()}
        for count, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                cases[task].update(status='scored', combined_score=result['combined_score'])
            except Exception as error:
                cases[task].update(status='error', error=f'{type(error).__name__}: {error}')
            write(destination, summary)
            print(f"[{count}/{len(selected)}] {task}: {cases[task]['status']}", flush=True)
    summary['groups'] = summarize(results, types)
    summary['batch_completed'] = len(results) == len(selected)
    write(destination, summary)
    if not summary['batch_completed']:
        raise RuntimeError(f'Scoring errors recorded in {destination}')
    print(f'Summary: {destination}', flush=True)
