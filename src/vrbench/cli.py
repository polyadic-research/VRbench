import argparse
from pathlib import Path
from .dataset import Dataset,download
from .io import read,write


def positive_integer(value):
    number=int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return number


def main():
    p=argparse.ArgumentParser(prog='vrbench');sub=p.add_subparsers(dest='command',required=True)
    hub = sub.add_parser('download', help='Download and verify a Hugging Face dataset')
    hub.add_argument('--repo', required=True, help='Hugging Face dataset ID, e.g. cyx712/VirtualRoboBench')
    hub.add_argument('--revision', required=True, help='Dataset revision; use a commit hash for reproducible runs')
    hub.add_argument('--output', type=Path, help='Local dataset directory (default: revision-specific cache)')
    check=sub.add_parser('validate');check.add_argument('--dataset',required=True,type=Path)
    pred=sub.add_parser('predict');pred.add_argument('--dataset',required=True,type=Path);pred.add_argument('--model',required=True,type=Path);pred.add_argument('--output',required=True,type=Path);pred.add_argument('--tasks');pred.add_argument('--execute',action='store_true')
    pred.add_argument('--max-attempts',type=positive_integer,default=3,help='Maximum attempts per task before a zero-score prediction failure (default: 3)')
    pred.add_argument('--max-concurrency',type=positive_integer,default=1,help='Maximum tasks running concurrently (default: 1)')
    solver = sub.add_parser('solve')
    solver.add_argument('--dataset', required=True, type=Path)
    selection = solver.add_mutually_exclusive_group()
    selection.add_argument('--task', help='Single task with its own prediction file or task directory')
    selection.add_argument('--tasks', help='Comma-separated batch task IDs (default: all dataset tasks)')
    solver.add_argument('--prediction', required=True, type=Path, action='append',
                        help='Prediction run directory (repeatable), or a single prediction with --task')
    solver.add_argument('--output', required=True, type=Path)
    solver.add_argument('--config', type=Path)
    solver.add_argument('--max-concurrency', type=positive_integer, default=1,
                        help='Maximum parallel solver processes in a batch (default: 1)')
    scoring = sub.add_parser('score')
    scoring.add_argument('--dataset', required=True, type=Path)
    selection = scoring.add_mutually_exclusive_group()
    selection.add_argument('--task', help='Score one prediction and its matching joint path')
    selection.add_argument('--tasks', help='Comma-separated batch task IDs (default: all dataset tasks)')
    scoring.add_argument('--prediction', type=Path, help='Prediction file or directory for --task')
    scoring.add_argument('--path', '--paths', type=Path,
                         help='Solved task directory with --task, or batch solve output directory')
    scoring.add_argument('--output', required=True, type=Path)
    scoring.add_argument('--max-concurrency', type=positive_integer, default=1,
                         help='Maximum parallel scoring processes in a batch (default: 1)')
    summary=sub.add_parser('summarize');summary.add_argument('--dataset',required=True,type=Path);summary.add_argument('--results',required=True,type=Path);summary.add_argument('--output',required=True,type=Path)
    repair=sub.add_parser('repair');repair.add_argument('--prediction',required=True,type=Path);repair.add_argument('--output',required=True,type=Path);repair.add_argument('--reason',required=True)
    a=p.parse_args()
    if a.command=='download':print(download(a.repo,a.revision,a.output));return
    if a.command=='repair':
        from .protocol.repairs import repair
        print(repair(a.prediction,a.output,a.reason));return
    dataset=Dataset(a.dataset)
    if a.command=='validate':print('Validated',len(dataset.records),'tasks');return
    if a.command=='predict':
        from .prediction import predict
        r=predict(dataset,a.model,a.output,a.tasks.split(',') if a.tasks else None,a.execute,a.max_attempts,a.max_concurrency);print('Processed',len(r['cases']),'tasks;',r['prediction_valid'],'valid;',r['no_answer'],'without an answer');return
    if a.command=='solve':
        from .solver.pipeline import solve
        if a.task is not None:
            if len(a.prediction) != 1:
                p.error('--task takes exactly one prediction file or task directory')
            print(solve(dataset, a.task, a.prediction[0], a.output, a.config)['complete_finite'])
        else:
            from .solver.batch import solve_batch
            tasks = a.tasks.split(',') if a.tasks is not None else None
            try:
                solve_batch(dataset, tasks, a.prediction, a.output, a.config, a.max_concurrency)
            except (ValueError, OSError, RuntimeError) as error:
                p.exit(1, f'vrbench solve: {error}\n')
        return
    if a.command=='score':
        if a.task is not None:
            if a.prediction is None:
                p.error('single-task scoring requires --prediction')
            from .scoring.pipeline import score
            print(score(dataset, a.task, a.prediction, a.path, a.output)['combined_score'])
        else:
            if a.path is None or a.prediction is not None:
                p.error('batch scoring requires --path pointing to the batch solve output; predictions come from its summary.json')
            from .scoring.batch import score_batch
            tasks = a.tasks.split(',') if a.tasks is not None else None
            try:
                score_batch(dataset, tasks, a.path, a.output, a.max_concurrency)
            except (ValueError, OSError, RuntimeError) as error:
                p.exit(1, f'vrbench score: {error}\n')
        return
    if a.command=='summarize':
        from .scoring.combined import summarize
        rows=read(a.results);types={k:v[0]['type'] for k,v in dataset.records.items()};write(a.output,summarize(rows,types));print(a.output)

if __name__=='__main__':main()
