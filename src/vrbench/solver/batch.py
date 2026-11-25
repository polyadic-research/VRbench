"""Resolve prediction runs and reconstruct one joint path per selected task."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import relpath
from pathlib import Path
from tempfile import TemporaryDirectory

from ..failures import no_answer
from ..io import read, sha, write
from ..protocol.validation import validate_prediction
from .pipeline import solve


def select_inputs(dataset, tasks, runs):
    selected = dataset.select(tasks)
    valid, exhausted = {}, {}
    dataset_hash = sha(dataset.root / 'dataset.json')
    batch_identity = None
    fields = ('model', 'transport', 'reasoning_effort', 'dataset_sha256')
    for run in runs:
        summary = read(run / 'summary.json')
        if summary.get('dataset_sha256') != dataset_hash or not summary.get('execute'):
            raise ValueError(f'{run}: expected an executed prediction run for this dataset')
        identity = {key: summary[key] for key in fields}
        if batch_identity is not None and identity != batch_identity:
            raise ValueError(f'{run}: model, transport or reasoning effort differs between runs')
        batch_identity = identity
        for case in summary['cases']:
            task = case['id']
            if task not in selected:
                continue
            if case['status'] == 'prediction_valid':
                valid[task] = (run / task, case['prediction_sha256'])
            elif case['status'] == 'no_answer':
                exhausted[task] = run / task

    missing = [task for task in selected if task not in valid and task not in exhausted]
    if missing:
        raise ValueError('Missing completed prediction records for: ' + ', '.join(missing))
    inputs = {}
    for task in selected:
        if task in valid:
            folder, expected = valid[task]
            path = folder / 'prediction.json'
            identity = read(folder / 'identity.json')
            if (any(identity.get(key) != value for key, value in batch_identity.items())
                    or identity.get('prediction_sha256') != expected or sha(path) != expected):
                raise ValueError(f'{task}: prediction identity or digest mismatch in {folder}')
            if (folder / 'no_answer.json').exists() or len(validate_prediction(read(path))) != 36:
                raise ValueError(f'{task}: expected one valid 36-point prediction in {folder}')
        else:
            folder = exhausted[task]
            failure = no_answer(dataset, task, folder)
            if not failure or any(failure[0]['identity'].get(k) != v for k, v in batch_identity.items()):
                raise ValueError(f'{task}: invalid no-answer record in {folder}')
        inputs[task] = folder
    return inputs


def _solve_task(dataset, task, source, output, config):
    with TemporaryDirectory(dir=output, prefix=f'.{task}-') as staging:
        work = Path(staging) / 'path'
        result = solve(dataset, task, source, work, config)
        work.rename(output / task)
    if result.get('status') == 'no_answer':
        return 'no_answer'
    return 'complete' if result['complete_finite'] else 'incomplete'


def solve_batch(dataset, tasks, runs, output, config, concurrency):
    inputs = select_inputs(dataset, tasks, runs)
    if config is not None:
        read(config)
    output.mkdir(parents=True, exist_ok=False)
    cases = {task: {'id': task, 'prediction': source.as_posix(), 'status': 'pending'}
             for task, source in inputs.items()}
    cwd = Path.cwd()
    base = Path(relpath(cwd, output)) if cwd.drive == output.resolve().drive else cwd
    summary = {'cases': list(cases.values()), 'batch_completed': False,
               'prediction_base': base.as_posix()}
    write(output / 'summary.json', summary)
    print(f'Solving {len(inputs)} tasks in {output}', flush=True)
    with ProcessPoolExecutor(max_workers=min(concurrency, len(inputs))) as pool:
        futures = {pool.submit(_solve_task, dataset, task, source, output, config): task
                   for task, source in inputs.items()}
        for count, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            record = cases[task]
            try:
                record['status'] = future.result()
            except Exception as error:
                record.update(status='error', error=f'{type(error).__name__}: {error}')
            write(output / 'summary.json', summary)
            print(f"[{count}/{len(inputs)}] {task}: {record['status']}", flush=True)
    summary['batch_completed'] = True
    write(output / 'summary.json', summary)
    if any(case['status'] == 'error' for case in cases.values()):
        raise RuntimeError(f"Solver errors recorded in {output / 'summary.json'}")
