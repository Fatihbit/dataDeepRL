"""
Hyperparameter Tuning met Optuna
================================

Dit script voert automatische hyperparameter optimalisatie uit voor
PPO en SAC agents met Optuna's Bayesian optimization.

Wat is Optuna?
--------------
Optuna is een hyperparameter optimization framework dat:
- Pruning gebruikt (stop slechte trials vroeg)
- Bayesian optimization (slimmer dan grid search)
- Parallelisatie ondersteunt
- Resultaten in database opslaat voor hervatting

Gebruik:
    # Quick search (10 trials)
    python tune_hyperparams.py --algorithm ppo --trials 10
    
    # Full search met meer trials
    python tune_hyperparams.py --algorithm sac --trials 100 --timeout 3600
    
    # Hervat eerdere search
    python tune_hyperparams.py --algorithm ppo --resume --study_name my_study

Output:
    - SQLite database met alle trial resultaten
    - JSON file met beste hyperparameters
    - Optioneel: Optuna visualization dashboard

Auteur: DataDeepRL Team
"""

import os
import sys
import json
import signal
import argparse
import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch

# GPU optimalisaties
torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')  # Gebruik tensor cores

# Graceful shutdown: Ctrl+C laat huidige trial afmaken
_stop_after_trial = False

def _signal_handler(sig, frame):
    global _stop_after_trial
    if _stop_after_trial:
        print("\n\n⚠ Tweede Ctrl+C — forceer stop!", flush=True)
        sys.exit(1)
    _stop_after_trial = True
    print("\n\n⏸ Ctrl+C ontvangen — huidige trial wordt afgemaakt, daarna stoppen...", flush=True)
    print("  (Druk nogmaals Ctrl+C om direct te stoppen)\n", flush=True)

signal.signal(signal.SIGINT, _signal_handler)

# Voeg project root toe aan path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import optuna
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Run: pip install optuna")

from src.data.dataloader import BTCDataLoader
from src.envs.trading_env import CryptoTradingEnv
from src.models.ppo import PPODeepLOBAgent
from src.models.sac import SACDeepLOBAgent

# DeepLOB agents (lazy import om circulaire imports te vermijden)
try:
    from train.train_sac_with_deeplob import SACWithPretrainedDeepLOB
    from train.train_ppo_with_deeplob import PPOWithPretrainedDeepLOB
    from train.common.setup import load_coredata
    DEEPLOB_AVAILABLE = True
except ImportError as e:
    DEEPLOB_AVAILABLE = False
    print(f"Warning: DeepLOB agents niet beschikbaar: {e}")


def create_ppo_agent(trial: 'optuna.Trial', env_info: Dict[str, Any]) -> PPODeepLOBAgent:
    """
    Maak PPO agent met hyperparameters gesuggereerd door Optuna.
    
    Args:
        trial: Optuna trial object
        env_info: Dictionary met environment informatie
        
    Returns:
        Geconfigureerde PPODeepLOBAgent
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    value_coef = trial.suggest_float('value_coef', 0.25, 0.75)
    entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.1, log=True)
    n_epochs = trial.suggest_int('n_epochs', 3, 15)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    # DeepLOB hyperparameters
    deeplob_hidden = trial.suggest_categorical('deeplob_hidden', [32, 64, 128])
    deeplob_lstm = trial.suggest_categorical('deeplob_lstm', [32, 64, 128])
    
    # MLP hidden dims
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    
    agent = PPODeepLOBAgent(
        window_size=env_info['window_size'],
        num_features=env_info['num_features'],
        portfolio_dim=env_info['portfolio_dim'],
        action_dim=env_info['action_dim'],
        deeplob_hidden=deeplob_hidden,
        deeplob_lstm=deeplob_lstm,
        hidden_dims=(hidden_dim, hidden_dim),
        lr=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        num_epochs=n_epochs,
        batch_size=batch_size,
        buffer_size=2048,
        device=env_info['device']
    )
    
    return agent


def create_sac_agent(trial: 'optuna.Trial', env_info: Dict[str, Any]) -> SACDeepLOBAgent:
    """
    Maak SAC agent met hyperparameters gesuggereerd door Optuna.
    
    Args:
        trial: Optuna trial object
        env_info: Dictionary met environment informatie
        
    Returns:
        Geconfigureerde SACDeepLOBAgent
    """
    # Suggest hyperparameters (bewust wijder gezet, vergelijkbaar met PPO search space)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    tau = trial.suggest_float('tau', 0.001, 0.01)
    alpha = trial.suggest_float('alpha', 0.05, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024])
    update_every = trial.suggest_categorical('update_every', [1, 2, 4, 8, 16])

    # DeepLOB hyperparameters
    deeplob_hidden = trial.suggest_categorical('deeplob_hidden', [32, 64, 128])
    deeplob_lstm = trial.suggest_categorical('deeplob_lstm', [32, 64, 128])

    # MLP hidden dims
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    
    agent = SACDeepLOBAgent(
        window_size=env_info['window_size'],
        num_features=env_info['num_features'],
        portfolio_dim=env_info['portfolio_dim'],
        action_dim=1,  # SAC typically uses continuous actions
        deeplob_hidden=deeplob_hidden,
        deeplob_lstm=deeplob_lstm,
        hidden_dims=(hidden_dim, hidden_dim),
        lr=learning_rate,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        auto_alpha=True,
        batch_size=batch_size,
        start_steps=500,  # Korte warmup voor tuning
        device=env_info['device']
    )
    
    return agent


def create_sac_deeplob_agent(trial: 'optuna.Trial', deeplob_model_path: str, device: str) -> 'SACWithPretrainedDeepLOB':
    """Maak SAC+DeepLOB agent met hyperparameters gesuggereerd door Optuna."""
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    tau = trial.suggest_float('tau', 0.001, 0.01)
    alpha = trial.suggest_float('alpha', 0.05, 0.5)

    return SACWithPretrainedDeepLOB(
        deeplob_model_path=deeplob_model_path,
        portfolio_dim=4,
        action_dim=3,
        freeze_deeplob=True,
        lr=lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        auto_alpha=True,
        device=device
    )


def create_ppo_deeplob_agent(trial: 'optuna.Trial', deeplob_model_path: str, device: str) -> 'PPOWithPretrainedDeepLOB':
    """Maak PPO+DeepLOB agent met hyperparameters gesuggereerd door Optuna."""
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    clip_epsilon = trial.suggest_float('clip_epsilon', 0.1, 0.3)
    value_coef = trial.suggest_float('value_coef', 0.25, 0.75)
    entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.05, log=True)
    n_epochs = trial.suggest_int('n_epochs', 3, 15)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    return PPOWithPretrainedDeepLOB(
        deeplob_model_path=deeplob_model_path,
        portfolio_dim=4,
        action_dim=3,
        freeze_deeplob=True,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device
    )


def evaluate_agent(agent, eval_env, n_episodes: int = 2, max_steps: int = 5000) -> float:
    """
    Evalueer agent op validation environment.

    Returns:
        Composite score: 0.5 * sharpe + 0.5 * net_return (beide genormaliseerd)
    """
    episode_scores = []

    for _ in range(n_episodes):
        _, _ = eval_env.reset()
        obs = eval_env.get_flat_observation()
        done = False
        portfolio_values = [10000.0]
        ep_steps = 0

        while not done and ep_steps < max_steps:
            if hasattr(agent, 'select_action'):
                # Gebruik stochastische acties — deterministic mean blijft ~0 vroeg in training
                result = agent.select_action(obs, deterministic=False)
                action = result[0] if isinstance(result, tuple) else result
            else:
                action = 0

            _, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            obs = eval_env.get_flat_observation()
            pv = info.get('portfolio_value', portfolio_values[-1])
            portfolio_values.append(pv)
            ep_steps += 1

        # Net return
        net_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Sharpe ratio (dagelijkse stap-returns)
        pv_arr = np.array(portfolio_values)
        step_returns = np.diff(pv_arr) / pv_arr[:-1]
        if len(step_returns) > 1 and step_returns.std() > 1e-8:
            sharpe = step_returns.mean() / step_returns.std() * np.sqrt(len(step_returns))
        else:
            sharpe = 0.0

        # Max drawdown penalty
        peak = np.maximum.accumulate(pv_arr)
        drawdown = np.max((peak - pv_arr) / (peak + 1e-8))

        # Composite score
        score = 0.5 * np.clip(sharpe, -5, 5) / 5 + 0.5 * np.clip(net_return, -1, 1) - 0.2 * drawdown
        episode_scores.append(score)

    return float(np.mean(episode_scores))


def objective(trial: 'optuna.Trial', 
              algorithm: str,
              train_env: CryptoTradingEnv,
              eval_env: CryptoTradingEnv,
              env_info: Dict[str, Any],
              training_steps: int,
              eval_interval: int,
              max_trial_minutes: Optional[float] = None) -> float:
    """
    Objective functie voor Optuna.
    
    Args:
        trial: Optuna trial
        algorithm: 'ppo' of 'sac'
        train_env: Training environment
        eval_env: Evaluation environment
        env_info: Environment informatie
        training_steps: Aantal training steps
        eval_interval: Evaluatie interval
        
    Returns:
        Validation reward (te maximaliseren)
    """
    # Maak agent
    if algorithm == 'ppo':
        agent = create_ppo_agent(trial, env_info)
    else:
        agent = create_sac_agent(trial, env_info)

    import time as _time

    print(f"\n[Trial {trial.number}] Params: { {k: round(v, 6) if isinstance(v, float) else v for k, v in trial.params.items()} }", flush=True)

    max_steps = training_steps

    _, _ = train_env.reset()
    obs = train_env.get_flat_observation()
    episode_reward = 0
    best_eval_reward = float('-inf')
    t_start = _time.time()

    # Debug counters
    action_counts = {'+buy': 0, '-sell': 0, '=hold': 0}
    total_trades = 0
    episodes = 0
    bankruptcies = 0

    # SAC: update niet elke step maar elke N steps (betere GPU benutting)
    update_every = trial.params.get('update_every', 1) if algorithm == 'sac' else 1

    step = 0
    stopped_reason = "max_steps"

    while step < max_steps:
        if algorithm == 'ppo':
            action, log_prob, value = agent.select_action(obs)
        else:
            action = agent.select_action(obs)

        _, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        next_obs = train_env.get_flat_observation()

        # Track acties voor logging (+buy / -sell / =hold)
        if algorithm == 'sac':
            action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            if action_val > 0.1:
                action_counts['+buy'] += 1
            elif action_val < -0.1:
                action_counts['-sell'] += 1
            else:
                action_counts['=hold'] += 1
        else:
            # PPO gebruikt discrete acties: 0=hold, 1=buy, 2=sell
            if isinstance(action, (np.ndarray, list, tuple)):
                action_val = int(action[0])
            else:
                action_val = int(action)

            if action_val == 1:
                action_counts['+buy'] += 1
            elif action_val == 2:
                action_counts['-sell'] += 1
            else:
                action_counts['=hold'] += 1
        trade_info = info.get('trade_info', {})
        if trade_info.get('executed', False):
            total_trades += 1

        if algorithm == 'ppo':
            agent.collect_rollout(obs, action, reward, value, log_prob, done)
            if agent.should_update():
                agent.update(next_obs)
        else:
            agent.buffer.add(obs, action, reward, next_obs, done)
            agent.total_steps += 1
            if agent.total_steps >= agent.start_steps and step % update_every == 0:
                agent.update()

        episode_reward += reward

        if done:
            episodes += 1
            if info.get('portfolio_value', 10000) < 1000:
                bankruptcies += 1
            _, _ = train_env.reset()
            next_obs = train_env.get_flat_observation()
            episode_reward = 0

        obs = next_obs

        # Harde tijdlimiet per trial (optioneel via CLI)
        elapsed_min = (_time.time() - t_start) / 60
        if max_trial_minutes is not None and max_trial_minutes > 0 and elapsed_min >= max_trial_minutes:
            stopped_reason = f"tijdslimiet ({max_trial_minutes:.1f} min)"
            print(f"  [Trial {trial.number}] EARLY STOP @ step {step:,} — {stopped_reason}", flush=True)
            break

        # Vroege voortgang na 100 steps
        if step == 100:
            elapsed = _time.time() - t_start
            steps_per_sec = 100 / elapsed
            eta_min = (training_steps - 100) / steps_per_sec / 60
            print(f"  [Trial {trial.number}] Speed: {steps_per_sec:.1f} steps/s | ETA ~{eta_min:.0f} min", flush=True)

        # Voortgang tonen elke 2000 steps
        if step > 0 and step % 2000 == 0:
            elapsed = _time.time() - t_start
            steps_per_sec = step / elapsed
            remaining_min = (max_steps - step) / steps_per_sec / 60

            if step % eval_interval == 0:
                eval_reward = evaluate_agent(agent, eval_env)
                total_actions = sum(action_counts.values()) or 1
                act_str = " | ".join(f"{k}:{v/total_actions*100:.0f}%" for k, v in action_counts.items())

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    status = "NEW BEST"
                else:
                    status = "no improvement"

                print(f"  [Trial {trial.number}] Step {step:,} | Score: {eval_reward:.4f} | Best: {best_eval_reward:.4f} | {status} | {act_str} | Trades: {total_trades} | Bankrupt: {bankruptcies}", flush=True)

                trial.report(eval_reward, step)
                if trial.should_prune():
                    print(f"  [Trial {trial.number}] GEPRUNED op step {step:,}", flush=True)
                    raise optuna.TrialPruned()
            else:
                print(f"  [Trial {trial.number}] Step {step:,} | ETA: {remaining_min:.0f}m", flush=True)

        step += 1

    final_reward = evaluate_agent(agent, eval_env, n_episodes=3, max_steps=10000)
    total_actions = sum(action_counts.values()) or 1
    act_str = " | ".join(f"{k}:{v/total_actions*100:.0f}%" for k, v in action_counts.items())
    elapsed_min = (_time.time() - t_start) / 60
    print(f"  [Trial {trial.number}] KLAAR @ {step:,} steps ({elapsed_min:.1f}min) | Score: {final_reward:.4f} | Reden: {stopped_reason} | {act_str} | Trades: {total_trades} | Bankrupt: {bankruptcies}", flush=True)

    return final_reward


def objective_deeplob(trial: 'optuna.Trial',
                      algorithm: str,
                      train_env: 'CryptoTradingEnv',
                      eval_env: 'CryptoTradingEnv',
                      deeplob_model_path: str,
                      device: str,
                      training_steps: int,
                      eval_interval: int) -> float:
    """Objective functie voor SAC+DeepLOB en PPO+DeepLOB tuning."""
    if algorithm == 'sac_deeplob':
        agent = create_sac_deeplob_agent(trial, deeplob_model_path, device)
    else:
        agent = create_ppo_deeplob_agent(trial, deeplob_model_path, device)

    import time as _time

    obs, _ = train_env.reset()
    best_score = float('-inf')
    t_start = _time.time()

    for step in range(training_steps):
        if algorithm == 'ppo_deeplob':
            action, log_prob, value = agent.select_action(obs)
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        if algorithm == 'ppo_deeplob':
            agent.collect_rollout(obs, action, reward, value, log_prob, done)
            if agent.should_update():
                agent.update(next_obs)
        else:
            agent.store_transition(obs, action, reward, next_obs, done)
            if agent.total_steps >= agent.batch_size:
                agent.update()

        if done:
            obs, _ = train_env.reset()
        else:
            obs = next_obs

        # Vroege voortgang na 100 steps
        if step == 100:
            elapsed = _time.time() - t_start
            steps_per_sec = 100 / elapsed
            eta_min = (training_steps - 100) / steps_per_sec / 60
            print(f"  [Trial {trial.number}] Speed: {steps_per_sec:.1f} steps/s | ETA deze trial: ~{eta_min:.0f} min", flush=True)

        # Voortgang tonen elke 2000 steps
        if step > 0 and step % 2000 == 0:
            elapsed = _time.time() - t_start
            steps_per_sec = step / elapsed
            eta_min = (training_steps - step) / steps_per_sec / 60
            progress = step / training_steps * 100
            if step % eval_interval == 0:
                score = evaluate_agent(agent, eval_env)
                print(f"  [Trial {trial.number}] Step {step:,}/{training_steps:,} ({progress:.0f}%) | Score: {score:.4f} | Best: {best_score:.4f} | ETA: {eta_min:.0f}m", flush=True)
                trial.report(score, step)
                if trial.should_prune():
                    print(f"  [Trial {trial.number}] GEPRUNED op step {step:,}", flush=True)
                    raise optuna.TrialPruned()
                best_score = max(best_score, score)
            else:
                print(f"  [Trial {trial.number}] Step {step:,}/{training_steps:,} ({progress:.0f}%)", flush=True)

    final_score = evaluate_agent(agent, eval_env, n_episodes=3, max_steps=10000)
    print(f"  [Trial {trial.number}] KLAAR | Finale score: {final_score:.4f}")
    return final_score


def run_deeplob_hyperparameter_search(args):
    """Voer hyperparameter search uit voor SAC+DeepLOB of PPO+DeepLOB."""
    if not DEEPLOB_AVAILABLE:
        print("Error: DeepLOB agents niet beschikbaar.")
        return

    print(f"\n{'='*60}")
    print(f"Hyperparameter Search: {args.algorithm.upper()}")
    print(f"{'='*60}")
    print(f"Trials: {args.trials}, Steps/trial: {args.steps_per_trial:,}")
    print(f"DeepLOB model: {args.deeplob_model}")
    print(f"{'='*60}\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading coreData...")
    train_seq, train_prices, val_seq, val_prices, _, _ = load_coredata(
        data_dir=args.coredata_dir,
        sequence_length=args.sequence_length,
        max_rows=args.max_rows_tune
    )
    print(f"  Train: {len(train_seq):,}, Val: {len(val_seq):,}")

    use_continuous = args.algorithm in ('sac_deeplob', 'sac')
    train_env = CryptoTradingEnv(
        sequences=train_seq, prices=train_prices,
        initial_balance=100000.0, transaction_fee=0.0, flat_fee=1.0, random_start=True,
        discrete_actions=not use_continuous
    )
    eval_env = CryptoTradingEnv(
        sequences=val_seq, prices=val_prices,
        initial_balance=100000.0, transaction_fee=0.0, flat_fee=1.0, random_start=False,
        discrete_actions=not use_continuous
    )

    study_name = args.study_name or f"{args.algorithm}_tuning_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    storage = f"sqlite:///{args.output_dir}/optuna_{study_name}.db"
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.resume:
        db_path = os.path.join(args.output_dir, f"optuna_{study_name}.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Vorige study verwijderd, start fresh.")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=args.steps_per_trial // 10),
        load_if_exists=args.resume
    )

    trial_counter = {'current': 0, 'total': args.trials}

    def objective_wrapper(trial):
        global _stop_after_trial
        trial_counter['current'] += 1
        print(f"\n{'─'*60}")
        print(f"  Trial {trial_counter['current']}/{trial_counter['total']} (Optuna #{trial.number})")
        print(f"{'─'*60}")
        result = objective_deeplob(
            trial=trial,
            algorithm=args.algorithm,
            train_env=train_env,
            eval_env=eval_env,
            deeplob_model_path=args.deeplob_model,
            device=device,
            training_steps=args.steps_per_trial,
            eval_interval=args.eval_interval
        )
        if _stop_after_trial:
            print(f"\n✅ Trial afgerond en opgeslagen. Stoppen...", flush=True)
            study.stop()
        return result

    study.optimize(objective_wrapper, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"Best score: {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    best_params_path = os.path.join(args.output_dir, f"best_params_{args.algorithm}.json")
    with open(best_params_path, 'w') as f:
        json.dump({
            'algorithm': args.algorithm,
            'best_value': study.best_trial.value,
            'best_params': study.best_trial.params,
            'n_trials': len(study.trials),
            'timestamp': datetime.datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nBeste parameters opgeslagen: {best_params_path}")


def run_hyperparameter_search(args):
    """
    Voer hyperparameter search uit.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print(f"Hyperparameter Search: {args.algorithm.upper()}")
    print(f"{'='*60}")
    print(f"Trials: {args.trials}")
    print(f"Training steps per trial: {args.steps_per_trial:,}")
    if args.max_trial_minutes is not None and args.max_trial_minutes > 0:
        print(f"Max trial duur: {args.max_trial_minutes:.1f} min")
    print(f"{'='*60}\n")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data vanuit coreData (preprocessed)
    print("\nLoading coreData...")
    train_sequences, train_prices, val_sequences, val_prices, _, _ = load_coredata(
        data_dir=args.coredata_dir,
        sequence_length=args.sequence_length,
        max_rows=args.max_rows_tune
    )
    print(f"Train: {len(train_sequences):,}, Val: {len(val_sequences):,}")
    
    # Create environments
    # SAC gebruikt continuous actions, PPO discrete
    use_continuous = (args.algorithm == 'sac')

    train_env = CryptoTradingEnv(
        sequences=train_sequences,
        prices=train_prices,
        initial_balance=100000.0,
        transaction_fee=0.0,
        flat_fee=1.0,
        random_start=True,
        discrete_actions=not use_continuous
    )

    eval_env = CryptoTradingEnv(
        sequences=val_sequences,
        prices=val_prices,
        initial_balance=100000.0,
        transaction_fee=0.0,
        flat_fee=1.0,
        random_start=False,
        discrete_actions=not use_continuous
    )
    
    # Get environment info
    _, _ = train_env.reset()
    flat_obs = train_env.get_flat_observation()
    
    env_info = {
        'window_size': args.sequence_length,
        'num_features': train_sequences.shape[2] if len(train_sequences.shape) == 3 else train_sequences.shape[1],
        'portfolio_dim': 4,
        'action_dim': train_env.action_space.n if hasattr(train_env.action_space, 'n') else train_env.action_space.shape[0],
        'obs_dim': flat_obs.shape[0],
        'device': device
    }
    
    # Create/load Optuna study
    study_name = args.study_name or f"{args.algorithm}_tuning_{datetime.datetime.now().strftime('%Y%m%d')}"
    db_path = os.path.join(args.output_dir, f"optuna_{study_name}.db")
    storage = f"sqlite:///{db_path}"

    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        print(f"Resuming study: {study_name}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
    else:
        # Verwijder bestaande database zodat we fresh starten
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Vorige study verwijderd, start fresh.", flush=True)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=args.steps_per_trial // 10
            )
        )
    
    # Run optimization
    print(f"\nStarting optimization...")
    
    trial_counter = {'current': 0, 'total': args.trials}

    def objective_wrapper(trial):
        global _stop_after_trial
        trial_counter['current'] += 1
        print(f"\n{'─'*60}")
        print(f"  Trial {trial_counter['current']}/{trial_counter['total']} (Optuna #{trial.number})")
        print(f"{'─'*60}")
        result = objective(
            trial=trial,
            algorithm=args.algorithm,
            train_env=train_env,
            eval_env=eval_env,
            env_info=env_info,
            training_steps=args.steps_per_trial,
            eval_interval=args.eval_interval,
            max_trial_minutes=args.max_trial_minutes
        )
        if _stop_after_trial:
            print(f"\n✅ Trial afgerond en opgeslagen. Stoppen...", flush=True)
            study.stop()
        return result

    study.optimize(
        objective_wrapper,
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=False
    )
    
    # Print results
    completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])

    print(f"\n{'='*60}")
    if _stop_after_trial:
        print("OPTIMIZATION PAUSED (Ctrl+C)")
    else:
        print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")

    print(f"\nTrial statistics:")
    print(f"  Completed: {completed}")
    print(f"  Pruned: {pruned}")
    print(f"  Total: {len(study.trials)}")

    if completed > 0:
        print(f"\nBest trial:")
        print(f"  Value (reward): {study.best_trial.value:.4f}")
        print(f"  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        best_params_path = os.path.join(args.output_dir, f"best_params_{args.algorithm}.json")
        with open(best_params_path, 'w') as f:
            json.dump({
                'algorithm': args.algorithm,
                'best_value': study.best_trial.value,
                'best_params': study.best_trial.params,
                'n_trials': completed,
                'timestamp': datetime.datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nBest hyperparameters saved to: {best_params_path}")

        # Overzicht tekst bestand
        summary_path = os.path.join(args.output_dir, f"overzicht_{args.algorithm}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"  HYPERPARAMETER TUNING OVERZICHT — {args.algorithm.upper()}\n")
            f.write(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"Trials voltooid:  {completed}\n")
            f.write(f"Trials gepruned:  {pruned}\n")
            f.write(f"Steps per trial:  {args.steps_per_trial:,} (adaptive)\n\n")

            f.write(f"{'-'*60}\n")
            f.write(f"  BESTE TRIAL (#{study.best_trial.number})\n")
            f.write(f"{'-'*60}\n")
            f.write(f"  Score: {study.best_trial.value:.4f}\n\n")
            f.write(f"  Hyperparameters:\n")
            for key, value in study.best_trial.params.items():
                if isinstance(value, float):
                    f.write(f"    {key:20s} = {value:.6f}\n")
                else:
                    f.write(f"    {key:20s} = {value}\n")

            f.write(f"\n{'-'*60}\n")
            f.write(f"  ALLE TRIALS (gesorteerd op score)\n")
            f.write(f"{'-'*60}\n")
            sorted_trials = sorted(
                [t for t in study.trials if t.state == TrialState.COMPLETE],
                key=lambda t: t.value, reverse=True
            )
            for i, t in enumerate(sorted_trials):
                marker = " <-- BEST" if t.number == study.best_trial.number else ""
                f.write(f"  #{t.number:2d}  Score: {t.value:+.4f}  "
                        f"lr={t.params.get('learning_rate', 0):.6f}  "
                        f"gamma={t.params.get('gamma', 0):.4f}  "
                        f"batch={t.params.get('batch_size', 0)}{marker}\n")

            f.write(f"\n{'-'*60}\n")
            f.write(f"  GEBRUIK DEZE PARAMS\n")
            f.write(f"{'-'*60}\n")
            bp = study.best_trial.params
            cmd_parts = [f"python train/train_sac_only.py --adaptive --total_steps 500000"]
            if 'learning_rate' in bp:
                cmd_parts.append(f"  --learning_rate {bp['learning_rate']}")
            if 'gamma' in bp:
                cmd_parts.append(f"  --gamma {bp['gamma']}")
            if 'tau' in bp:
                cmd_parts.append(f"  --tau {bp['tau']}")
            if 'alpha' in bp:
                cmd_parts.append(f"  --alpha {bp['alpha']}")
            if 'batch_size' in bp:
                cmd_parts.append(f"  --batch_size {bp['batch_size']}")
            if 'hidden_dim' in bp:
                cmd_parts.append(f"  --hidden_dims {bp['hidden_dim']} {bp['hidden_dim']}")
            f.write("  " + " \\\n  ".join(cmd_parts) + "\n")

        print(f"Overzicht opgeslagen: {summary_path}")
    else:
        print("\nGeen voltooide trials om op te slaan.")

    if _stop_after_trial:
        remaining = args.trials - completed
        print(f"\nHervat later met: python tune_hyperparams.py --algorithm {args.algorithm} --trials {remaining} --steps_per_trial {args.steps_per_trial} --resume --study_name {study_name}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning met Optuna'
    )
    
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='ppo',
                        choices=['ppo', 'sac', 'ppo_deeplob', 'sac_deeplob'],
                        help='RL algoritme om te tunen')
    
    # Optuna settings
    parser.add_argument('--trials', type=int, default=20,
                        help='Aantal Optuna trials')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum tijd in seconden (optioneel)')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Naam voor Optuna study')
    parser.add_argument('--resume', action='store_true',
                        help='Hervat vorige study')
    
    # Training settings
    parser.add_argument('--steps_per_trial', type=int, default=50000,
                        help='Training steps per trial')
    parser.add_argument('--eval_interval', type=int, default=3000,
                        help='Evaluatie interval')
    parser.add_argument('--max_trial_minutes', type=float, default=20.0,
                        help='Harde tijdlimiet per trial in minuten (<=0 schakelt uit)')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='./btc_l2_data',
                        help='Data directory')
    parser.add_argument('--max_files', type=int, default=50,
                        help='Maximum data bestanden')
    parser.add_argument('--sequence_length', type=int, default=100,
                        help='Sequence lengte')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./tuning_results',
                        help='Output directory voor resultaten')

    # DeepLOB specifiek
    parser.add_argument('--deeplob_model', type=str, default='./models/deeplob_pretrained.pt',
                        help='Pad naar pre-trained DeepLOB model (voor *_deeplob algoritmes)')
    parser.add_argument('--coredata_dir', type=str, default='./coreData',
                        help='CoreData directory (voor *_deeplob algoritmes)')
    parser.add_argument('--max_rows_tune', type=int, default=500_000,
                        help='Max rijen voor tuning data (klein houden = sneller)')

    return parser.parse_args()


def main():
    """Main entry point."""
    if not OPTUNA_AVAILABLE:
        print("Error: optuna is required. Install with: pip install optuna")
        return
    
    args = parse_args()
    if args.algorithm in ('sac_deeplob', 'ppo_deeplob'):
        run_deeplob_hyperparameter_search(args)
    else:
        run_hyperparameter_search(args)


if __name__ == '__main__':
    main()
