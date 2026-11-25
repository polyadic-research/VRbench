"""Public-input prediction with bounded attempts and task-level concurrency."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock, Event
import json
import shutil
from .io import read, write, sha
from .failures import no_answer, zero_score
from .protocol.inputs import build
from .protocol.validation import normalize_count, QuaternionNormError
from .protocol.repairs import normalize_quaternions, repair
from .providers.responses import call, load_config, ProviderBlocked


def predict(dataset, model_file, output, tasks=None, execute=False,
            max_attempts=3, max_concurrency=1):
    for name, value in [('max_attempts', max_attempts), ('max_concurrency', max_concurrency)]:
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    config = load_config(model_file)
    selected = dataset.select(tasks)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = {'model': config['model'], 'reasoning_effort': config['reasoning_effort'],
                'transport': config['transport'], 'dataset_sha256': sha(dataset.root / 'dataset.json')}
    summary = {**identity, 'execute': execute, 'model_config_sha256': sha(model_file),
               'max_attempts': max_attempts, 'max_concurrency': max_concurrency,
               'cases': [{'id': task, 'status': 'pending'} for task in selected], 'model_calls': 0}
    lock = Lock()
    paused = Event()
    positions = {task: i for i, task in enumerate(selected)}

    def publish(record, calling=False):
        with lock:
            if calling:
                summary['model_calls'] += 1
            summary['cases'][positions[record['id']]] = deepcopy(record)
            write(out / record['id'] / 'record.json', record)
            write(out / 'summary.json', summary)

    def accept(task, folder, record, prediction, raw_path, response_path, adjustment,
               selected_attempt, repaired=False):
        shutil.copy2(raw_path, folder / 'prediction_raw.json')
        shutil.copy2(response_path, folder / 'response.txt')
        write(folder / 'prediction.json', prediction)
        write(folder / 'point_count.json', adjustment)
        record.update(status='prediction_valid', selected_attempt=selected_attempt,
                      prediction_sha256=sha(folder / 'prediction.json'),
                      prediction_selection='quaternion_normalized_postprocessed' if repaired else 'first_valid',
                      original_protocol_valid=not repaired and adjustment['action'] == 'unchanged')
        write(folder / 'identity.json', {**identity, 'prediction_sha256': record['prediction_sha256'],
              'original_protocol_valid': record['original_protocol_valid'],
              'prediction_selection': record['prediction_selection'], 'selected_attempt': selected_attempt,
              'attempts_used': record['attempts_used']})
        publish(record)
        print(task + ': ' + record['prediction_selection'], flush=True)

    def run(task):
        folder = out / task
        folder.mkdir()
        record = {'id': task, 'status': 'preparing', 'attempts': [], 'attempts_used': 0}
        try:
            inputs = build(dataset, task)
            write(folder / 'public.json', inputs['public'])
            (folder / 'prompt.txt').write_text(inputs['prompt'], encoding='utf8')
            record.update(status='input_generated', system_sha256=inputs['system_sha256'],
                          image_sha256=[sha(p) for _, p in inputs['images']])
            publish(record)
            if not execute:
                return
            for number in range(1, max_attempts + 1):
                if paused.is_set():
                    record.update(status='provider_paused')
                    publish(record)
                    return
                attempt_folder = folder / 'attempts' / f'{number:03d}'
                attempt_folder.mkdir(parents=True)
                attempt = {'attempt': number, 'status': 'calling_model',
                           'quaternion_repair_eligible': False}
                record['attempts'].append(attempt)
                record.update(status='calling_model', attempts_used=number)
                publish(record, calling=True)
                print(f'{task}: attempt {number}/{max_attempts}', flush=True)
                raw = None
                raw_path = attempt_folder / 'prediction_raw.json'
                response_path = attempt_folder / 'response.txt'
                try:
                    text = call(config, inputs, attempt_folder / 'transport')
                    response_path.write_text(text, encoding='utf8')
                    raw = json.loads(text)
                    prediction, adjustment = normalize_count(raw)
                    write(raw_path, raw)
                    attempt.update(status='prediction_valid', prediction_sha256=sha(raw_path))
                except ProviderBlocked as error:
                    paused.set()
                    attempt.update(status='provider_blocked',error=str(error))
                    write(attempt_folder/'result.json',attempt)
                    record.update(status='provider_blocked',error=str(error))
                    publish(record)
                    return
                except Exception as error:
                    attempt.update(status='error', error_type=type(error).__name__, error=str(error))
                    if isinstance(error, QuaternionNormError):
                        try:
                            normalize_quaternions(raw)
                            write(raw_path, raw)
                            attempt.update(quaternion_repair_eligible=True,
                                           prediction_sha256=sha(raw_path))
                        except (ValueError, TypeError, KeyError):
                            pass
                    write(attempt_folder / 'result.json', attempt)
                    publish(record)
                    if attempt['quaternion_repair_eligible']:
                        audit = repair(raw_path, folder/'repair',
                                       f'Quaternion-only failure on attempt {number}')
                        record['repair_audit_sha256'] = sha(folder/'repair'/'audit.json')
                        accept(task,folder,record,read(folder/'repair'/'prediction.json'),
                               raw_path,response_path,audit['point_count_adjustment'],number,repaired=True)
                        return
                    continue
                write(attempt_folder / 'result.json', attempt)
                accept(task, folder, record, prediction, raw_path, response_path,
                       adjustment, number)
                return
            record.update(status='no_answer', max_attempts=max_attempts,
                          reason='attempt_limit_exhausted', identity=identity)
            write(folder / 'no_answer.json', record)
            write(folder / 'failure_score.json', zero_score(dataset, task, no_answer(dataset, task, folder)))
            publish(record)
            print(f'{task}: no valid answer after {max_attempts} attempts; all scores are zero', flush=True)
        except Exception as error:
            record.update(status='error', error_type=type(error).__name__, error=str(error))
            publish(record)

    write(out / 'summary.json', summary)
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [pool.submit(run, task) for task in selected]
        for future in as_completed(futures):
            future.result()
    summary.update(batch_completed=not paused.is_set(), provider_paused=paused.is_set(),
                   prediction_valid=sum(r['status'] == 'prediction_valid' for r in summary['cases']),
                   no_answer=sum(r['status'] == 'no_answer' for r in summary['cases']))
    write(out / 'summary.json', summary)
    return summary
