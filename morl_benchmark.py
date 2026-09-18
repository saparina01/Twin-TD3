"""Matched-budget sampled-preference vs fixed-preference seed experiment."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
from morl_experiment import ROOT, train_morl, evaluate_morl, write_json
from morl_plot import compare_evaluations


def run_benchmark(output_dir, seeds=(0, 1, 2), episodes=300, eval_seeds=(1000, 1001, 1002),
                  preferences=None, device='cpu', threads=1):
    if len(set(seeds)) < 3:
        raise ValueError('A comparison requires at least three distinct training seeds')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    evaluations = []
    manifest = dict(episodes=episodes, seeds=list(seeds), eval_seeds=list(eval_seeds),
                    status='running', runs=[])
    write_json(output / 'benchmark_manifest.json', manifest)
    for seed in seeds:
        for fixed in (None, 0.0, 0.5, 1.0):
            label = 'sampled' if fixed is None else f'fixed_{fixed:g}'
            run = train_morl(episodes=episodes, seeds=[seed], preference=fixed,
                             output_dir=output / f'{label}_seed_{seed}', device=device, threads=threads)
            evaluation = evaluate_morl(run, preferences=preferences if fixed is None else [fixed],
                                       eval_seeds=eval_seeds, device=device, threads=threads)
            evaluations.append(evaluation)
            manifest['runs'].append(dict(seed=seed, preference=fixed, model=str(run), evaluation=str(evaluation)))
            write_json(output / 'benchmark_manifest.json', manifest)
    compare_evaluations(evaluations, output / 'comparison')
    manifest['status'] = 'complete'
    write_json(output / 'benchmark_manifest.json', manifest)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(ROOT / 'data' / 'storage' / 'morl' / 'benchmark'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--ep-num', type=int, default=300)
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[1000, 1001, 1002])
    parser.add_argument('--preferences', nargs='+', type=float, default=None,
                        help='Sampled-model evaluation grid; default 0:0.05:1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    print(run_benchmark(args.output_dir, seeds=args.seeds, episodes=args.ep_num,
                        eval_seeds=args.eval_seeds, preferences=args.preferences,
                        device=args.device, threads=args.threads))
