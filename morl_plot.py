"""Plots and seed-level comparisons from named MORL CSV fields only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def plot_training(run_path):
    path = Path(run_path)
    data = pd.read_csv(path / 'training_summary.csv')
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for ax, field, label in zip(axes, ('mean_secrecy_rate', 'total_energy_j', 'rate_weight'),
                                ('Mean sum secrecy rate (bits/s/Hz)', 'Propulsion energy (J)', 'Rate preference')):
        ax.plot(data['episode'] + 1, data[field], marker='.' if len(data) < 20 else None)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel('Training episode')
    fig.suptitle('MORL training (exploration enabled)')
    fig.tight_layout()
    fig.savefig(path / 'training.png', dpi=160)
    plt.close(fig)


def plot_evaluation(eval_path):
    path = Path(eval_path)
    summary = pd.read_csv(path / 'evaluation_summary.csv')
    steps = pd.read_csv(path / 'evaluation_steps.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    dots = ax.scatter(summary['total_energy_j'] / 1000, summary['mean_secrecy_rate'],
                      c=summary['rate_weight'], vmin=0, vmax=1, cmap='viridis', s=55)
    front = summary[summary['nondominated'].astype(str).str.lower() == 'true'].sort_values('total_energy_j')
    ax.scatter(front['total_energy_j'] / 1000, front['mean_secrecy_rate'],
               facecolors='none', edgecolors='black', s=115, linewidths=1,
               label='Non-dominated evaluated policies')
    ax.set(xlabel='Total propulsion energy (kJ)', ylabel='Mean sum secrecy rate (bits/s/Hz)',
           title='Preference-conditioned policy evaluation')
    fig.colorbar(dots, ax=ax, label='Rate preference')
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis='both', style='plain', useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'pareto.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    available = np.array(sorted(steps['rate_weight'].unique()))
    selected = sorted({float(available[np.argmin(abs(available - w))]) for w in (0, 0.5, 1)})
    first_seed = steps['eval_seed'].iloc[0]
    for weight in selected:
        group = steps[(steps['rate_weight'] == weight) & (steps['eval_seed'] == first_seed)].sort_values('step')
        xs = [group['x_start'].iloc[0], *group['x']]
        ys = [group['y_start'].iloc[0], *group['y']]
        ax.plot(xs, ys, label=f'lambda={weight:.2f}')
        ax.scatter(xs[0], ys[0], c='black', marker='o', s=20)
        ax.scatter(xs[-1], ys[-1], marker='x', s=40)
    ax.set(xlabel='x (m)', ylabel='y (m)', xlim=(-25, 25), ylim=(0, 50),
           title=f'UAV trajectories (evaluation seed {first_seed})')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'trajectories.png', dpi=160)
    plt.close(fig)


def compare_evaluations(eval_paths, output_dir):
    """Average eval repetitions first, then report variation across trained models."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    common_signature = None
    seen_models = set()
    for folder in eval_paths:
        folder = Path(folder).resolve()
        config = json.loads((folder / 'evaluation_config.json').read_text(encoding='utf-8'))
        meta = config['model_metadata']
        if config['model_path'] in seen_models:
            raise ValueError('Do not count multiple evaluations of the same model as independent training seeds')
        seen_models.add(config['model_path'])
        signature = dict(environment=meta['environment'], reward_scales=meta['reward_scales'],
                         observation_noise_std=meta['observation_noise_std'], eval_seeds=config['eval_seeds'],
                         training_episodes=meta['completed_episodes'],
                         hidden_sizes=[a['hidden_sizes'] for a in meta['agents']])
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError('Comparison requires matching environment, scales, evaluation seeds, architecture and training budget')
        method = 'sampled' if meta['preference'] is None else f'fixed_{meta["preference"]:g}'
        data = pd.read_csv(folder / 'evaluation_episodes.csv')
        for weight, group in data.groupby('rate_weight'):
            if meta['preference'] is not None and not np.isclose(weight, meta['preference']):
                continue  # A fixed-preference baseline is only valid at its training weight.
            row = dict(method=method, training_seed=meta['seeds'][0],
                        seed_pair=str(meta['seeds']), rate_weight=float(weight), model_path=config['model_path'])
            for field in ('mean_secrecy_rate', 'total_energy_j', 'scalar_return', 'boundary_projections', 'power_projections'):
                row[field] = float(group[field].mean())
            records.append(row)
    if not records:
        raise ValueError('No comparable evaluation rows')
    raw = pd.DataFrame(records)
    if raw.duplicated(['method', 'seed_pair', 'rate_weight']).any():
        raise ValueError('Duplicate training seed within a method')
    raw.to_csv(output / 'comparison_per_model.csv', index=False)
    rows = []
    for (method, weight), group in raw.groupby(['method', 'rate_weight']):
        row = dict(method=method, rate_weight=weight, training_seeds=len(group))
        for field in ('mean_secrecy_rate', 'total_energy_j', 'scalar_return', 'boundary_projections', 'power_projections'):
            row[field] = float(group[field].mean())
            row[field + '_std'] = float(group[field].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'comparison_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in summary.groupby('method'):
        ax.errorbar(group['total_energy_j']/1000, group['mean_secrecy_rate'],
                     xerr=group['total_energy_j_std']/1000, yerr=group['mean_secrecy_rate_std'],
                     fmt='o', capsize=3, label=method, alpha=0.85)
    ax.set(xlabel='Total propulsion energy (kJ)', ylabel='Mean sum secrecy rate (bits/s/Hz)',
           title='Mean and standard deviation across training seeds')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)

    matched = []
    for method, fixed in raw[raw['method'] != 'sampled'].groupby('method'):
        shared = raw[raw['method'] == 'sampled']
        pairs = fixed.merge(shared, on=['seed_pair', 'rate_weight'], suffixes=('_fixed', '_sampled'))
        for _, pair in pairs.iterrows():
            matched.append(dict(baseline=method, seed_pair=pair['seed_pair'], rate_weight=pair['rate_weight'],
                                fixed_utility=pair['scalar_return_fixed'], sampled_utility=pair['scalar_return_sampled'],
                                sampled_minus_fixed=pair['scalar_return_sampled'] - pair['scalar_return_fixed']))
    if matched:
        pd.DataFrame(matched).to_csv(output / 'matched_utility.csv', index=False)
    report = ['# MORL comparison', '',
              f'Training budget per model: {common_signature["training_episodes"]} episodes.',
              'Evaluation repetitions are averaged within each model before computing variation across training seeds.', '',
              'These measurements are empirical diagnostics, not evidence of convergence or complete Pareto coverage.', '']
    if common_signature['training_episodes'] < 300:
        report += ['This is a short-run smoke comparison; do not interpret it as a final algorithm ranking.', '']
    report += ['| Method | Rate preference | Seeds | Rate (bits/s/Hz) | Energy (J) |',
               '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        report.append(f'| {row["method"]} | {row["rate_weight"]:.2f} | {row["training_seeds"]} | '
                      f'{row["mean_secrecy_rate"]:.4f} +/- {row["mean_secrecy_rate_std"]:.4f} | '
                      f'{row["total_energy_j"]:.2f} +/- {row["total_energy_j_std"]:.2f} |')
    sampled = raw[raw['method'] == 'sampled']
    if not sampled.empty:
        report += ['', '## Preference sensitivity', '']
        for seed, group in sampled.groupby('seed_pair'):
            rate_span = group['mean_secrecy_rate'].max() - group['mean_secrecy_rate'].min()
            energy_span = group['total_energy_j'].max() - group['total_energy_j'].min()
            report.append(f'- Seed {seed}: rate span {rate_span:.4f} bits/s/Hz; energy span {energy_span:.2f} J.')
        report += ['', 'Small spans call for checking the objective conflict, normalization and preference training coverage.']
    (output / 'comparison_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare MORL evaluations across training seeds and baselines')
    parser.add_argument('--paths', nargs='+', required=True, help='Evaluation directories containing evaluation_config.json')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(compare_evaluations(args.paths, args.output_dir))
