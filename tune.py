"""
Hyperparameter tuning
=====================

Eén schoon Optuna-script dat alle trainings-scripts in dit project tuned:

    deeplob       -> train.train_deeplob_pretrain   (metric: val accuracy)
    ppo_only      -> train.train_ppo_only           (metric: eval composite)
    sac_only      -> train.train_sac_only           (metric: eval composite)
    ppo_deeplob   -> train.train_ppo_with_deeplob   (metric: eval composite)
    sac_deeplob   -> train.train_sac_with_deeplob   (metric: eval composite)

Elke trial draait het bestaande training-script als subprocess met een
gereduceerd budget (minder steps + minder data) en een uniek experiment.
De score wordt uit de output-CSV van dat script gelezen — geen wijziging aan
de training-scripts nodig.

Gebruik
-------
    # PPO met DeepLOB backbone, 25 trials
    python tune.py --algo ppo_deeplob --n_trials 25

    # SAC (MLP), korter budget per trial
    python tune.py --algo sac_only --n_trials 30 --trial_steps 500000

    # DeepLOB pretrain
    python tune.py --algo deeplob --n_trials 20 --trial_epochs 5

Resultaten (Optuna study + beste params) worden opgeslagen in tuning_results/.
De studies zijn persistent (SQLite): hervat met hetzelfde --algo/--study_name.
"""

import os
import sys
import csv
import json
import argparse
import datetime
import subprocess

import optuna

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, 'tuning_results')
TRIAL_LOG_DIR = os.path.join(RESULTS_DIR, 'trial_logs')


# =============================================================================
# SEARCH SPACES  (één functie per algoritme; geeft een {arg: waarde} dict terug)
# =============================================================================

def _space_deeplob(trial):
    return {
        'learning_rate':   trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
        'batch_size':      trial.suggest_categorical('batch_size', [128, 256, 512]),
        'dropout':         trial.suggest_float('dropout', 0.0, 0.4),
        'hidden_dim':      trial.suggest_categorical('hidden_dim', [32, 64, 128]),
        'lstm_hidden':     trial.suggest_categorical('lstm_hidden', [32, 64, 128]),
        'weight_decay':    trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
    }


def _space_ppo(trial, with_deeplob):
    space = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        'gamma':         trial.suggest_float('gamma', 0.95, 0.999),
        'gae_lambda':    trial.suggest_float('gae_lambda', 0.90, 0.99),
        'clip_epsilon':  trial.suggest_float('clip_epsilon', 0.1, 0.3),
        'entropy_coef':  trial.suggest_float('entropy_coef', 1e-4, 3e-2, log=True),
        'value_coef':    trial.suggest_float('value_coef', 0.25, 1.0),
        'n_epochs':      trial.suggest_categorical('n_epochs', [5, 10, 15]),
        'n_steps':       trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
        'batch_size':    trial.suggest_categorical('batch_size', [256, 512, 1024]),
    }
    if not with_deeplob:  # alleen de MLP-variant heeft een --hidden_dims arg
        space['hidden_dims'] = trial.suggest_categorical(
            'hidden_dims', ['256 256', '512 256', '256 256 128'])
    return space


def _space_sac(trial, with_deeplob):
    space = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 3e-4, log=True),
        'gamma':         trial.suggest_float('gamma', 0.95, 0.999),
        'tau':           trial.suggest_float('tau', 1e-3, 2e-2, log=True),
        'batch_size':    trial.suggest_categorical('batch_size', [256, 512, 1024]),
    }
    if not with_deeplob:
        space['hidden_dims'] = trial.suggest_categorical(
            'hidden_dims', ['128 128', '256 256', '256 256 128'])
    return space


# =============================================================================
# METRIC PARSING  (lees de score uit de output van het training-script)
# =============================================================================

def _metric_rl(run_dir):
    """Beste eval composite score uit logs/<run>/training_monitor.csv."""
    csv_path = os.path.join(run_dir, 'training_monitor.csv')
    if not os.path.exists(csv_path):
        return None
    best = None
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                score = float(row['eval_composite_score'])
            except (KeyError, ValueError):
                continue
            if best is None or score > best:
                best = score
    return best


def _metric_deeplob(model_name):
    """Beste validatie-accuracy uit models/<name>_training_log.csv."""
    csv_path = os.path.join(ROOT, 'models', f'{model_name}_training_log.csv')
    if not os.path.exists(csv_path):
        return None
    best = None
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                acc = float(row['val_acc'])
            except (KeyError, ValueError):
                continue
            if best is None or acc > best:
                best = acc
    return best


# =============================================================================
# ALGORITME-REGISTER
# =============================================================================

ALGOS = {
    'deeplob':     {'module': 'train.train_deeplob_pretrain', 'kind': 'deeplob'},
    'ppo_only':    {'module': 'train.train_ppo_only',         'kind': 'ppo'},
    'sac_only':    {'module': 'train.train_sac_only',         'kind': 'sac'},
    'ppo_deeplob': {'module': 'train.train_ppo_with_deeplob', 'kind': 'ppo', 'deeplob': True},
    'sac_deeplob': {'module': 'train.train_sac_with_deeplob', 'kind': 'sac', 'deeplob': True},
}


def build_params(algo, trial):
    """Sample de hyperparameters voor één trial."""
    spec = ALGOS[algo]
    kind = spec['kind']
    with_deeplob = spec.get('deeplob', False)
    if kind == 'deeplob':
        return _space_deeplob(trial)
    if kind == 'ppo':
        return _space_ppo(trial, with_deeplob)
    return _space_sac(trial, with_deeplob)


def to_cli(params):
    """Zet een {arg: waarde} dict om naar een CLI-lijst (nargs-veilig)."""
    cli = []
    for key, value in params.items():
        cli.append(f'--{key}')
        cli.extend(str(value).split() if isinstance(value, str) else [str(value)])
    return cli


# =============================================================================
# TRIAL RUNNER
# =============================================================================

def run_trial(algo, args, trial):
    spec = ALGOS[algo]
    params = build_params(algo, trial)
    tag = f'tune_{algo}_t{trial.number}'

    cmd = [sys.executable, '-m', spec['module'],
           '--data_dir', args.data_dir,
           '--seed', str(args.seed)]
    cmd += to_cli(params)

    if spec['kind'] == 'deeplob':
        cmd += ['--epochs', str(args.trial_epochs),
                '--max_rows', str(args.deeplob_max_rows),
                '--model_name', tag]
        metric_target = tag
    else:
        cmd += ['--total_steps', str(args.trial_steps),
                '--max_rows', str(args.max_rows),
                '--num_envs', str(args.num_envs),
                '--experiment_name', tag,
                '--log_dir', args.log_dir]
        if spec.get('deeplob'):
            cmd += ['--deeplob_model', args.deeplob_model]
        else:
            cmd += ['--no_tensorboard']  # alleen de MLP-only scripts kennen deze flag
        metric_target = os.path.join(args.log_dir, tag)

    # Schrijf de volledige output van de trial naar een logbestand.
    os.makedirs(TRIAL_LOG_DIR, exist_ok=True)
    log_path = os.path.join(TRIAL_LOG_DIR, f'{tag}.log')
    print(f"  [trial {trial.number}] {params}")
    print(f"  [trial {trial.number}] log -> {os.path.relpath(log_path, ROOT)}")

    with open(log_path, 'w') as log_file:
        try:
            subprocess.run(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                           timeout=args.trial_timeout or None, check=True)
        except subprocess.TimeoutExpired:
            print(f"  [trial {trial.number}] timeout — partiële score wordt gebruikt")
        except subprocess.CalledProcessError as e:
            print(f"  [trial {trial.number}] FAILED (exit {e.returncode}) — zie log")
            raise optuna.TrialPruned()

    score = (_metric_deeplob(metric_target) if spec['kind'] == 'deeplob'
             else _metric_rl(metric_target))
    if score is None:
        print(f"  [trial {trial.number}] geen score gevonden — pruned")
        raise optuna.TrialPruned()
    print(f"  [trial {trial.number}] score = {score:.6f}")
    return score


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Hyperparameter tuning (Optuna) voor DataDeepRL')
    p.add_argument('--algo', required=True, choices=list(ALGOS))
    p.add_argument('--n_trials', type=int, default=25)
    p.add_argument('--study_name', type=str, default=None)
    p.add_argument('--data_dir', type=str, default='./coreData')
    p.add_argument('--log_dir', type=str, default='./logs')
    p.add_argument('--seed', type=int, default=42)

    # Per-trial budget (klein houden: tuning zoekt richting, geen eindmodel).
    p.add_argument('--trial_steps', type=int, default=1_000_000,
                   help='total_steps per RL-trial')
    p.add_argument('--max_rows', type=int, default=3_000_000,
                   help='data-subset per RL-trial')
    p.add_argument('--num_envs', type=int, default=64)
    p.add_argument('--trial_epochs', type=int, default=5,
                   help='epochs per DeepLOB-trial')
    p.add_argument('--deeplob_max_rows', type=int, default=5_000_000,
                   help='data-subset per DeepLOB-trial')
    p.add_argument('--trial_timeout', type=int, default=0,
                   help='max seconden per trial (0 = geen limiet)')

    # Alleen voor ppo_deeplob / sac_deeplob.
    p.add_argument('--deeplob_model', type=str, default='./models/deeplob_pretrained.pt')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    study_name = args.study_name or f'{args.algo}_{datetime.datetime.now():%Y%m%d}'
    storage = f'sqlite:///{os.path.join(RESULTS_DIR, study_name + ".db")}'

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    print(f"\n{'='*60}")
    print(f"Tuning: {args.algo}")
    print(f"Study:  {study_name}  ({storage})")
    print(f"Trials: {args.n_trials}")
    print(f"{'='*60}\n")

    study.optimize(lambda t: run_trial(args.algo, args, t), n_trials=args.n_trials)

    best_path = os.path.join(RESULTS_DIR, f'best_params_{args.algo}.json')
    payload = {
        'algo': args.algo,
        'study_name': study_name,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'updated': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(best_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Beste score: {study.best_value:.6f}")
    print(f"Beste params:")
    for k, v in study.best_params.items():
        print(f"  {k:16s} {v}")
    print(f"\nOpgeslagen: {os.path.relpath(best_path, ROOT)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
