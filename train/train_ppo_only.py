"""
Training Script: PPO Only (MLP)
===============================

Dit script traint een Proximal Policy Optimization (PPO) agent met simpele MLP
feature extraction voor cryptocurrency trading.

Gebruik:
    python train_ppo_only.py --data_dir ../btc_l2_data --steps 1000000

Auteur: DataDeepRL Team
"""

import os
import sys
import argparse
import datetime
import time
import signal
import warnings
from pathlib import Path

import numpy as np
import torch

# Voeg src toe aan path anders ModuleNotFoundError: No module named 'src' voor de onderstaande bestanden
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch.nn.functional as F

from src.data.dataloader import BTCDataLoader
from src.envs.trading_env import FlatCryptoTradingEnv
from src.envs.vec_env import VectorizedTradingEnv
from src.models.ppo import PPOAgent
from src.utils.logger import TrainingLogger, setup_logging
from src.utils.callbacks import PauseResumeCallback, load_checkpoint
from src.utils.trade_logger import TradeLogger
from train.common.setup import load_coredata, load_coredata_streaming

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PPO agent (MLP only) for crypto trading'
    )
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./coreData')
    parser.add_argument('--max_files', type=int, default=100)
    parser.add_argument('--max_rows', type=int, default=90_000_000)
    parser.add_argument('--sequence_length', type=int, default=100)
    
    # Training
    parser.add_argument('--total_steps', type=int, default=10_000_000)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.25)
    parser.add_argument('--entropy_coef', type=float, default=0.005)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256])

    # Environment
    parser.add_argument('--initial_balance', type=float, default=100000.0)
    parser.add_argument('--transaction_fee', type=float, default=0.0)
    parser.add_argument('--flat_fee', type=float, default=0.0, help='Flat fee per trade in USDT')
    parser.add_argument('--max_position', type=float, default=1.0)
    parser.add_argument('--max_episode_steps', type=int, default=3600,
                        help='Max steps per training episode (0=unlimited)')
    parser.add_argument('--num_envs', type=int, default=256,
                        help='Number of parallel training environments')

    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--n_eval_episodes', type=int, default=5)
    parser.add_argument('--max_eval_steps', type=int, default=2000, help='Max steps per eval episode (0=unlimited)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=10)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--no_tensorboard', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help='Pad naar checkpoint om training te hervatten')
    
    return parser.parse_args()


def main():
    """Main training functie."""
    args = parse_args()

    # Setup
    if args.experiment_name is None:
        args.experiment_name = f"ppo_mlp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    setup_logging(log_dir=args.log_dir, experiment_name=args.experiment_name)
    
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    
    # GPU optimizations
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    
    print(f"\n{'='*60}")
    print(f"PPO (MLP Only) Training")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"cudnn.benchmark: True")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Steps per rollout: {args.n_steps}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"{'='*60}\n")
    
    # Random seed torch is om nn te bouwen 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    print("Loading data...")
    is_coredata = os.path.exists(os.path.join(args.data_dir, 'train.parquet'))
    
    if is_coredata:
        train_features, train_prices, val_features, val_prices, _, _ = load_coredata_streaming(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            max_rows=args.max_rows
        )
    else:
        data_loader = BTCDataLoader(
            window_size=args.sequence_length,
            data_dir=args.data_dir
        )
        df = data_loader.load_data(max_files=args.max_files)
        if df is None or len(df) == 0:
            print("Error: Could not load data!")
            return
        df = data_loader.create_features()
        sequences, targets, prices = data_loader.prepare_sequences(df)
        train_data, val_data, _ = data_loader.split_data(sequences, targets, prices)
        train_sequences, _, train_prices = train_data
        val_sequences, _, val_prices = val_data
        train_features, val_features = None, None
    
    print(f"Train: {len(train_prices):,}, Val: {len(val_prices):,}")
    
    # Create environments
    num_envs = args.num_envs
    
    if train_features is not None:
        train_env_kwargs = dict(
            raw_features=train_features,
            prices=train_prices,
            window_size=args.sequence_length,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            flat_fee=args.flat_fee,
            max_position=args.max_position,
            max_episode_steps=args.max_episode_steps,
            random_start=True,
            random_start_range=1.0
        )
        eval_env = FlatCryptoTradingEnv(
            raw_features=val_features,
            prices=val_prices,
            window_size=args.sequence_length,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            flat_fee=args.flat_fee,
            max_position=args.max_position
        )
    else:
        train_env_kwargs = dict(
            sequences=train_sequences,
            prices=train_prices,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            flat_fee=args.flat_fee,
            max_position=args.max_position,
            max_episode_steps=args.max_episode_steps,
            random_start=True,
            random_start_range=1.0
        )
        eval_env = FlatCryptoTradingEnv(
            sequences=val_sequences,
            prices=val_prices,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            flat_fee=args.flat_fee,
            max_position=args.max_position
        )
    
    print(f"Creating {num_envs} parallel training environments...")
    train_vec_env = VectorizedTradingEnv(num_envs=num_envs, env_kwargs=train_env_kwargs, env_class=FlatCryptoTradingEnv)
    
    obs_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.n
    
    # Create agent
    print("\nCreating PPO agent...")
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device
    )
    
    # Logger
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        use_tensorboard=not args.no_tensorboard,
        log_interval=args.log_interval
    )
    logger.save_config(vars(args))
    
    # =====================================
    # RESUME FROM CHECKPOINT
    # =====================================
    total_steps = 0
    update_count = 0
    episode_count = 0
    episode_rewards = []
    portfolio_values = []
    all_policy_losses = []
    all_value_losses = []
    best_eval_reward = float('-inf')
    # Financial tracking
    win_rates = []
    trade_counts = []
    episode_returns = []
    total_fees_list = []
    total_money_lost = 0.0
    total_money_gained = 0.0
    total_fees_paid = 0.0
    max_drawdowns = []
    last_eval_composite = 0.0

    ckpt_dir = os.path.join(args.log_dir, args.experiment_name, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_ckpt_path = os.path.join(ckpt_dir, 'resume_checkpoint.pt')
    
    save_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Episode finance CSV (written at episode end)
    episode_csv_path = os.path.join(save_dir, 'episode_finance.csv')
    if not os.path.exists(episode_csv_path):
        with open(episode_csv_path, 'w') as f:
            f.write('episode,reward,portfolio_value,pnl,return_pct,win_rate,total_trades,fees,drawdown,money_gained,money_lost,composite_score\n')
    
    # Training monitor CSV (written every eval period - comprehensive diagnostics)
    monitor_csv_path = os.path.join(save_dir, 'training_monitor.csv')
    if not os.path.exists(monitor_csv_path):
        with open(monitor_csv_path, 'w') as f:
            f.write('step,episode,timestamp,reward_avg100,portfolio_value,'
                    'total_trades,buys,sells,winning_sells,losing_sells,win_pct,'
                    'total_profit,total_loss,net_pnl,total_fees,'
                    'avg_buy_size,avg_sell_size,avg_profit_per_win,avg_loss_per_loss,'
                    'trade_freq,policy_loss,value_loss,entropy,eval_composite_score\n')
    
    # Per-trade logger
    trade_logger = TradeLogger(save_dir)
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, weights_only=False, map_location=device)
            if 'policy_state_dict' in ckpt:
                agent.policy.load_state_dict(ckpt['policy_state_dict'])
            if 'value_state_dict' in ckpt:
                agent.value_net.load_state_dict(ckpt['value_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            total_steps = ckpt.get('step', 0)
            update_count = ckpt.get('update_count', 0)
            episode_count = ckpt.get('episode_count', 0)
            episode_rewards = ckpt.get('episode_rewards', [])
            portfolio_values = ckpt.get('portfolio_values', [])
            all_policy_losses = ckpt.get('all_policy_losses', [])
            all_value_losses = ckpt.get('all_value_losses', [])
            best_eval_reward = ckpt.get('best_eval_reward', float('-inf'))
            win_rates = ckpt.get('win_rates', [])
            trade_counts = ckpt.get('trade_counts', [])
            episode_returns = ckpt.get('episode_returns', [])
            total_fees_list = ckpt.get('total_fees_list', [])
            total_money_lost = ckpt.get('total_money_lost', 0.0)
            total_money_gained = ckpt.get('total_money_gained', 0.0)
            total_fees_paid = ckpt.get('total_fees_paid', 0.0)
            max_drawdowns = ckpt.get('max_drawdowns', [])
            # Restore per-trade history
            saved_trades = ckpt.get('trade_history', [])
            if saved_trades:
                trade_logger.restore_from_list(saved_trades)
            print(f"  Continuing from step {total_steps:,}, updates: {update_count}, episodes: {episode_count}")
            print(f"  Total gained: ${total_money_gained:,.2f}, Total lost: ${total_money_lost:,.2f}, Fees: ${total_fees_paid:,.2f}")
            print(f"  Restored {len(saved_trades)} trades from checkpoint")
        else:
            print(f"Warning: Checkpoint not found: {args.resume}")
    
    # =====================================
    # PAUSE SIGNAL HANDLER
    # =====================================
    _pause_requested = False
    _original_sigint = signal.getsignal(signal.SIGINT)
    
    def _signal_handler(signum, frame):
        nonlocal _pause_requested
        if _pause_requested:
            print("\nForce quit!")
            sys.exit(1)
        _pause_requested = True
        print("\n[PAUSE] Pause requested! Saving checkpoint after current update...")
    
    signal.signal(signal.SIGINT, _signal_handler)

    actual_max_steps = args.total_steps

    # Training
    start_time = time.time()
    print(f"\nStarting training... (total_steps target: {args.total_steps:,}, n_steps: {args.n_steps})")
    print(f"  Train data: {len(train_vec_env.prices):,}, Val data: {len(eval_env.prices):,}")
    print(f"  Parallel envs: {num_envs}, transitions per rollout: {args.n_steps * num_envs:,}")
    print(f"  Eval freq: every {args.eval_freq} updates, {args.n_eval_episodes} episodes, max {args.max_eval_steps} steps each")
    
    # Episode state persists across rollouts — per-environment tracking
    obs_list = train_vec_env.reset()
    ep_rewards = [0.0] * num_envs
    ep_lengths = [0] * num_envs
    ep_buy_counts = [0] * num_envs
    ep_sell_counts = [0] * num_envs
    ep_hold_counts = [0] * num_envs
    ep_fees_list_env = [0.0] * num_envs
    prev_balances = [args.initial_balance] * num_envs
    prev_btc_helds = [0.0] * num_envs
    
    try:
        while total_steps < actual_max_steps:
            # ===== COLLECT ROLLOUT =====
            # Storage for vectorized rollout (n_steps, num_envs, ...)
            buf_obs = np.zeros((args.n_steps, num_envs, obs_dim), dtype=np.float32)
            buf_actions = np.zeros((args.n_steps, num_envs), dtype=np.int64)
            buf_rewards = np.zeros((args.n_steps, num_envs), dtype=np.float32)
            buf_dones = np.zeros((args.n_steps, num_envs), dtype=np.float32)
            buf_values = np.zeros((args.n_steps, num_envs), dtype=np.float32)
            buf_log_probs = np.zeros((args.n_steps, num_envs), dtype=np.float32)
            
            for step in range(args.n_steps):
                # Batched action selection — single GPU forward pass for all envs
                obs_batch = np.stack(obs_list)  # (num_envs, obs_dim)
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_batch).to(agent.device)
                    values_t = agent.value_net(obs_t).squeeze(-1)
                    if agent.discrete:
                        actions_t, log_probs_t = agent.policy.get_action(obs_t, deterministic=False)
                    else:
                        actions_t, log_probs_t = agent.policy.sample(obs_t, deterministic=False)
                    actions = actions_t.cpu().numpy().flatten().astype(int)
                    log_probs = log_probs_t.cpu().numpy().flatten()
                    values = values_t.cpu().numpy().flatten()
                
                # Store in buffer
                buf_obs[step] = obs_batch
                buf_actions[step] = actions
                buf_values[step] = values
                buf_log_probs[step] = log_probs
                
                # Step all environments
                next_obs_list, rewards, dones, infos = train_vec_env.step(actions.tolist())
                buf_rewards[step] = rewards
                buf_dones[step] = dones.astype(np.float32)
                total_steps += num_envs
                
                # Per-environment episode tracking
                for i in range(num_envs):
                    ep_rewards[i] += rewards[i]
                    ep_lengths[i] += 1
                    
                    if actions[i] == 1:
                        ep_buy_counts[i] += 1
                    elif actions[i] == 2:
                        ep_sell_counts[i] += 1
                    else:
                        ep_hold_counts[i] += 1
                    
                    info = infos[i]
                    trade_info = info.get('trade_info', {})
                    if trade_info.get('executed', False):
                        ep_fees_list_env[i] += trade_info.get('fee', 0.0)
                        trade_logger.log_trade(
                            episode=episode_count,
                            step=total_steps,
                            trade_info=trade_info,
                            balance_before=prev_balances[i],
                            balance_after=info.get('balance', prev_balances[i]),
                            btc_held_before=prev_btc_helds[i],
                            btc_held_after=info.get('btc_held', prev_btc_helds[i]),
                            portfolio_value=info.get('portfolio_value', args.initial_balance),
                        )
                    
                    prev_balances[i] = info.get('balance', prev_balances[i])
                    prev_btc_helds[i] = info.get('btc_held', prev_btc_helds[i])
                    
                    if dones[i]:
                        terminal_info = info.get('terminal_info', info)
                        pv = terminal_info.get('portfolio_value', args.initial_balance)
                        ep_return = terminal_info.get('total_return', 0.0)
                        ep_pnl = pv - args.initial_balance
                        ep_win_rate = terminal_info.get('win_rate', 0.0)
                        ep_trades = terminal_info.get('total_trades', 0)
                        ep_drawdown = terminal_info.get('max_drawdown', 0.0)
                        ep_sharpe = terminal_info.get('sharpe_ratio', 0.0)
                        ep_composite = 0.5 * np.clip(ep_sharpe, -5, 5) / 5 + 0.5 * np.clip(ep_return, -1, 1) - 0.2 * ep_drawdown

                        if ep_pnl >= 0:
                            total_money_gained += ep_pnl
                        else:
                            total_money_lost += abs(ep_pnl)
                        total_fees_paid += ep_fees_list_env[i]
                        
                        logger.log_episode(episode_count, ep_rewards[i], ep_lengths[i])
                        episode_rewards.append(ep_rewards[i])
                        portfolio_values.append(pv)
                        win_rates.append(ep_win_rate)
                        trade_counts.append({'buy': ep_buy_counts[i], 'sell': ep_sell_counts[i], 'hold': ep_hold_counts[i]})
                        episode_returns.append(ep_return)
                        total_fees_list.append(ep_fees_list_env[i])
                        max_drawdowns.append(ep_drawdown)
                        episode_count += 1
                        
                        with open(episode_csv_path, 'a') as f:
                            f.write(f'{episode_count},{ep_rewards[i]:.4f},{pv:.2f},{ep_pnl:.2f},'
                                    f'{ep_return*100:.4f},{ep_win_rate:.4f},{ep_trades},{ep_fees_list_env[i]:.4f},'
                                    f'{ep_drawdown:.4f},{total_money_gained:.2f},{total_money_lost:.2f},{ep_composite:.6f}\n')
                        
                        if episode_count % 10 == 0:
                            net_pnl = total_money_gained - total_money_lost - total_fees_paid
                            print(
                                f"  [FINANCE] Ep {episode_count} | "
                                f"PnL: ${ep_pnl:+,.2f} | "
                                f"Portfolio: ${pv:,.2f} | "
                                f"Return: {ep_return*100:+.2f}% | "
                                f"Win Rate: {ep_win_rate*100:.0f}% | "
                                f"Trades: {ep_trades} (B:{ep_buy_counts[i]} S:{ep_sell_counts[i]} H:{ep_hold_counts[i]}) | "
                                f"Fees: ${ep_fees_list_env[i]:.2f} | "
                                f"Drawdown: {ep_drawdown*100:.1f}%"
                            )
                            print(
                                f"  [CUMULATIVE] Gained: ${total_money_gained:,.2f} | "
                                f"Lost: ${total_money_lost:,.2f} | "
                                f"Fees: ${total_fees_paid:,.2f} | "
                                f"Net: ${net_pnl:+,.2f}"
                            )
                        
                        # Reset per-env trackers
                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0
                        ep_buy_counts[i] = 0
                        ep_sell_counts[i] = 0
                        ep_hold_counts[i] = 0
                        ep_fees_list_env[i] = 0.0
                        prev_balances[i] = args.initial_balance
                        prev_btc_helds[i] = 0.0
                
                obs_list = next_obs_list
            
            # ===== COMPUTE GAE PER ENV =====
            with torch.no_grad():
                last_obs_batch = np.stack(obs_list)
                last_obs_t = torch.FloatTensor(last_obs_batch).to(agent.device)
                last_values = agent.value_net(last_obs_t).squeeze(-1).cpu().numpy()
            
            n_steps = args.n_steps
            adv = np.zeros((n_steps, num_envs), dtype=np.float32)
            gae = np.zeros(num_envs, dtype=np.float32)
            
            for t in reversed(range(n_steps)):
                next_value = last_values if t == n_steps - 1 else buf_values[t + 1]
                delta = buf_rewards[t] + args.gamma * next_value * (1 - buf_dones[t]) - buf_values[t]
                gae = delta + args.gamma * args.gae_lambda * (1 - buf_dones[t]) * gae
                adv[t] = gae
            
            returns = adv + buf_values
            
            # Flatten for mini-batch PPO update
            total_samples = n_steps * num_envs
            flat_obs = torch.FloatTensor(buf_obs.reshape(total_samples, obs_dim)).to(agent.device)
            flat_actions = torch.LongTensor(buf_actions.reshape(total_samples)).to(agent.device)
            flat_log_probs = torch.FloatTensor(buf_log_probs.reshape(total_samples)).to(agent.device)
            flat_adv = torch.FloatTensor(adv.reshape(total_samples)).to(agent.device)
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
            flat_returns = torch.FloatTensor(returns.reshape(total_samples)).to(agent.device)
            
            # ===== PPO UPDATE =====
            all_pl = []
            all_vl = []
            for epoch in range(args.n_epochs):
                indices = np.random.permutation(total_samples)
                for start in range(0, total_samples, args.batch_size):
                    end = min(start + args.batch_size, total_samples)
                    idx = indices[start:end]
                    
                    obs_b = flat_obs[idx]
                    act_b = flat_actions[idx]
                    old_lp_b = flat_log_probs[idx]
                    adv_b = flat_adv[idx]
                    ret_b = flat_returns[idx]
                    
                    log_probs_new, entropy = agent.policy.evaluate_actions(obs_b, act_b)
                    ratio = torch.exp(log_probs_new - old_lp_b)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    vals = agent.value_net(obs_b).squeeze()
                    value_loss = F.mse_loss(vals, ret_b)
                    
                    entropy_loss = -entropy.mean()
                    
                    loss = policy_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss
                    
                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(agent.policy.parameters()) + list(agent.value_net.parameters()),
                        args.max_grad_norm
                    )
                    agent.optimizer.step()
                    
                    all_pl.append(policy_loss.item())
                    all_vl.append(value_loss.item())
            
            losses = {'policy_loss': np.mean(all_pl), 'value_loss': np.mean(all_vl)}
            update_count += 1
            all_policy_losses.append(losses['policy_loss'])
            all_value_losses.append(losses['value_loss'])
            
            # Log
            logger.log_step(step=total_steps, losses=losses)
            
            # Progress
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / max(elapsed, 1)
            progress = total_steps / actual_max_steps * 100
            remaining = (actual_max_steps - total_steps) / max(steps_per_sec, 0.01)
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_pv = np.mean(portfolio_values[-100:]) if portfolio_values else args.initial_balance
            avg_return = np.mean(episode_returns[-100:]) * 100 if episode_returns else 0
            
            print(
                f"Update {update_count:4d} | "
                f"Steps: {total_steps:,}/{actual_max_steps:,} ({progress:.1f}%) | "
                f"{steps_per_sec:.0f} steps/s | "
                f"Ep: {episode_count} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg PV: ${avg_pv:,.0f} | "
                f"Avg Return: {avg_return:+.2f}% | "
                f"Policy: {losses.get('policy_loss', 0):.4f} | "
                f"Value: {losses.get('value_loss', 0):.4f} | "
                f"ETA: {remaining/60:.1f}min"
            )
            
            # Evaluation
            if update_count % args.eval_freq == 0:
                eval_rewards = []
                eval_infos = []
                print(f"  Starting eval ({args.n_eval_episodes} episodes, max {args.max_eval_steps} steps)...")
                for ep_i in range(args.n_eval_episodes):
                    eval_obs, _ = eval_env.reset()
                    eval_done = False
                    eval_reward = 0
                    eval_steps = 0
                    eval_step_info = {}
                    eval_cum_profit = 0.0
                    eval_cum_loss = 0.0
                    eval_buys = 0
                    eval_sells = 0
                    while not eval_done:
                        eval_action, _, _ = agent.select_action(eval_obs, deterministic=True)
                        eval_obs, r, term, trunc, eval_step_info = eval_env.step(eval_action)
                        eval_done = term or trunc
                        eval_reward += r
                        eval_steps += 1
                        trade_info = eval_step_info.get('trade_info', {})
                        if trade_info and trade_info.get('executed', False):
                            if trade_info.get('type') == 'buy':
                                eval_buys += 1
                            elif trade_info.get('type') == 'sell':
                                eval_sells += 1
                                p = trade_info.get('profit', 0.0)
                                if p > 0: eval_cum_profit += p
                                else: eval_cum_loss += abs(p)
                        if args.max_eval_steps > 0 and eval_steps >= args.max_eval_steps:
                            break
                    eval_rewards.append(eval_reward)
                    eval_infos.append(eval_step_info)
                    eval_pv = eval_step_info.get('portfolio_value', args.initial_balance)
                    eval_pnl = eval_pv - args.initial_balance
                    eval_net = eval_cum_profit - eval_cum_loss
                    print(f"    Eval ep {ep_i+1}/{args.n_eval_episodes}: {eval_steps} steps | "
                          f"PV=${eval_pv:,.2f} | PnL=${eval_pnl:+,.2f} | "
                          f"Realized=${eval_net:+,.2f} (W:${eval_cum_profit:,.2f} L:${eval_cum_loss:,.2f}) | "
                          f"Trades={eval_buys+eval_sells} (B:{eval_buys}/S:{eval_sells})")
                _c_scores = [0.5 * np.clip(i.get('sharpe_ratio', 0.0), -5, 5) / 5
                             + 0.5 * np.clip(i.get('total_return', 0.0), -1, 1)
                             - 0.2 * i.get('max_drawdown', 0.0) for i in eval_infos]
                last_eval_composite = float(np.mean(_c_scores)) if _c_scores else 0.0
                
                mean_eval = np.mean(eval_rewards)
                logger.log_evaluation(total_steps, eval_rewards)
                
                print(f"  Eval: {mean_eval:.2f} ± {np.std(eval_rewards):.2f}")
                
                # Adaptive: check verbetering VOOR best_eval_reward update
                is_improving = mean_eval > best_eval_reward

                if is_improving:
                    best_eval_reward = mean_eval
                    save_dict = {
                        'step': total_steps,
                        'policy_state_dict': agent.policy.state_dict(),
                        'value_state_dict': agent.value_net.state_dict(),
                        'mean_reward': mean_eval
                    }
                    best_path = os.path.join(args.log_dir, args.experiment_name, 'best_model.pt')
                    torch.save(save_dict, best_path)

                    # Backup met prestatie-info
                    backup_dir = os.path.join(args.log_dir, args.experiment_name, 'backups')
                    os.makedirs(backup_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_name = f"best_model_step{total_steps}_r{mean_eval:.1f}_{ts}.pt"
                    torch.save(save_dict, os.path.join(backup_dir, backup_name))
                    print(f"  [*] New best model!")
                    print(f"  [BACKUP] {backup_name}")
                # Save snapshot
                try:
                    snapshot_dir = os.path.join(args.log_dir, args.experiment_name, 'plots', f'step_{total_steps:07d}')
                    os.makedirs(snapshot_dir, exist_ok=True)
                    avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    with open(os.path.join(snapshot_dir, 'params.txt'), 'w') as pf:
                        pf.write(f"Snapshot at step {total_steps:,}\n")
                        pf.write(f"{'='*50}\n")
                        pf.write(f"\n--- Hyperparameters ---\n")
                        pf.write(f"Learning rate:     {args.learning_rate}\n")
                        pf.write(f"Batch size:        {args.batch_size}\n")
                        pf.write(f"N steps:           {args.n_steps}\n")
                        pf.write(f"N epochs:          {args.n_epochs}\n")
                        pf.write(f"Gamma:             {args.gamma}\n")
                        pf.write(f"GAE lambda:        {args.gae_lambda}\n")
                        pf.write(f"Clip epsilon:      {args.clip_epsilon}\n")
                        pf.write(f"Value coef:        {args.value_coef}\n")
                        pf.write(f"Entropy coef:      {args.entropy_coef}\n")
                        pf.write(f"Max grad norm:     {args.max_grad_norm}\n")
                        pf.write(f"Hidden dims:       {args.hidden_dims}\n")
                        pf.write(f"Sequence length:   {args.sequence_length}\n")
                        pf.write(f"Total steps:       {args.total_steps:,}\n")
                        pf.write(f"Initial balance:   {args.initial_balance}\n")
                        pf.write(f"Transaction fee:   {args.transaction_fee}\n")
                        pf.write(f"Max position:      {args.max_position}\n")
                        pf.write(f"Seed:              {args.seed}\n")
                        pf.write(f"\n--- Training Progress ---\n")
                        pf.write(f"Current step:      {total_steps:,}\n")
                        pf.write(f"Updates:           {update_count}\n")
                        pf.write(f"Episodes:          {episode_count}\n")
                        pf.write(f"Avg reward (100):  {avg_r:.2f}\n")
                        pf.write(f"Eval reward:       {mean_eval:.2f}\n")
                        pf.write(f"Best eval reward:  {best_eval_reward:.2f}\n")
                        pf.write(f"\n--- Financial Summary ---\n")
                        net_pnl = total_money_gained - total_money_lost - total_fees_paid
                        pf.write(f"Total gained:      ${total_money_gained:,.2f}\n")
                        pf.write(f"Total lost:        ${total_money_lost:,.2f}\n")
                        pf.write(f"Total fees:        ${total_fees_paid:,.2f}\n")
                        pf.write(f"Net PnL:           ${net_pnl:+,.2f}\n")
                        pf.write(f"Avg win rate:      {np.mean(win_rates[-100:])*100:.1f}%\n" if win_rates else '')
                    print(f"  [SNAPSHOT] Snapshot saved: {snapshot_dir}")
                except Exception as e:
                    print(f"  [WARN] Could not save snapshot: {e}")
            
            # Write training monitor CSV (comprehensive diagnostics)
            try:
                ts = trade_logger.get_summary()
                trade_freq = ts['total_trades'] / max(total_steps, 1)
                last_pl = all_policy_losses[-1] if all_policy_losses else 0
                last_vl = all_value_losses[-1] if all_value_losses else 0
                avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                pv = portfolio_values[-1] if portfolio_values else args.initial_balance
                import datetime as _dt
                with open(monitor_csv_path, 'a') as f:
                    f.write(f"{total_steps},{episode_count},{_dt.datetime.now().isoformat()},"
                            f"{avg_r:.4f},{pv:.2f},"
                            f"{ts['total_trades']},{ts['total_buys']},{ts['total_sells']},"
                            f"{ts['winning_sells']},{ts['losing_sells']},{ts['win_rate']*100:.2f},"
                            f"{ts['total_profit']:.2f},{ts['total_loss']:.2f},{ts['net_pnl']:.2f},{ts['total_fees']:.2f},"
                            f"{ts['avg_buy_size_usd']:.2f},{ts['avg_sell_size_usd']:.2f},"
                            f"{ts['avg_profit_per_win']:.2f},{ts['avg_loss_per_loss']:.2f},"
                            f"{trade_freq:.6f},{last_pl:.6f},{last_vl:.6f},0.000000,{last_eval_composite:.6f}\n")
            except Exception as e:
                print(f"  [WARN] Could not write monitor CSV: {e}")
            
            # Save resume checkpoint
            if update_count % args.save_freq == 0:
                _save_ppo_checkpoint(agent, total_steps, update_count, episode_count,
                                     episode_rewards, portfolio_values,
                                     all_policy_losses, all_value_losses,
                                     best_eval_reward, resume_ckpt_path,
                                     win_rates=win_rates, trade_counts=trade_counts,
                                     episode_returns=episode_returns, total_fees_list=total_fees_list,
                                     total_money_lost=total_money_lost, total_money_gained=total_money_gained,
                                     total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
                                     max_drawdowns=max_drawdowns)
                print(f"  [SAVE] Checkpoint saved at step {total_steps:,}")
            
            # Check pause
            if _pause_requested:
                _save_ppo_checkpoint(agent, total_steps, update_count, episode_count,
                                     episode_rewards, portfolio_values,
                                     all_policy_losses, all_value_losses,
                                     best_eval_reward, resume_ckpt_path,
                                     win_rates=win_rates, trade_counts=trade_counts,
                                     episode_returns=episode_returns, total_fees_list=total_fees_list,
                                     total_money_lost=total_money_lost, total_money_gained=total_money_gained,
                                     total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
                                     max_drawdowns=max_drawdowns)
                print(f"\n[PAUSE] Training paused at step {total_steps:,}")
                print(f"  Resume: python train_ppo_only.py --resume {resume_ckpt_path} [other args]")
                break
    
    finally:
        # Save final model
        _save_ppo_checkpoint(agent, total_steps, update_count, episode_count,
                             episode_rewards, portfolio_values,
                             all_policy_losses, all_value_losses,
                             best_eval_reward,
                             os.path.join(args.log_dir, args.experiment_name, 'final_model.pt'),
                             win_rates=win_rates, trade_counts=trade_counts,
                             episode_returns=episode_returns, total_fees_list=total_fees_list,
                             total_money_lost=total_money_lost, total_money_gained=total_money_gained,
                             total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
                             max_drawdowns=max_drawdowns)
        signal.signal(signal.SIGINT, _original_sigint)
        logger.close()
        
        # Generate final plots
        try:
            final_dir = os.path.join(args.log_dir, args.experiment_name, 'plots', 'final')
            os.makedirs(final_dir, exist_ok=True)
            trade_logger.print_summary()
            avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            with open(os.path.join(final_dir, 'params.txt'), 'w') as pf:
                pf.write(f"FINAL RESULTS\n")
                pf.write(f"{'='*50}\n")
                pf.write(f"\n--- Hyperparameters ---\n")
                pf.write(f"Learning rate:     {args.learning_rate}\n")
                pf.write(f"Batch size:        {args.batch_size}\n")
                pf.write(f"N steps:           {args.n_steps}\n")
                pf.write(f"N epochs:          {args.n_epochs}\n")
                pf.write(f"Gamma:             {args.gamma}\n")
                pf.write(f"GAE lambda:        {args.gae_lambda}\n")
                pf.write(f"Clip epsilon:      {args.clip_epsilon}\n")
                pf.write(f"Value coef:        {args.value_coef}\n")
                pf.write(f"Entropy coef:      {args.entropy_coef}\n")
                pf.write(f"Max grad norm:     {args.max_grad_norm}\n")
                pf.write(f"Hidden dims:       {args.hidden_dims}\n")
                pf.write(f"Sequence length:   {args.sequence_length}\n")
                pf.write(f"Total steps:       {args.total_steps:,}\n")
                pf.write(f"Initial balance:   {args.initial_balance}\n")
                pf.write(f"Transaction fee:   {args.transaction_fee}\n")
                pf.write(f"Max position:      {args.max_position}\n")
                pf.write(f"Seed:              {args.seed}\n")
                pf.write(f"\n--- Final Results ---\n")
                pf.write(f"Total steps:       {total_steps:,}\n")
                pf.write(f"Updates:           {update_count}\n")
                pf.write(f"Episodes:          {episode_count}\n")
                pf.write(f"Avg reward (100):  {avg_r:.2f}\n")
                pf.write(f"Best eval reward:  {best_eval_reward:.2f}\n")
                pf.write(f"\n--- Financial Summary ---\n")
                gross_pnl = total_money_gained - total_money_lost
                net_pnl = gross_pnl - total_fees_paid
                fee_impact = (total_fees_paid / gross_pnl * 100) if gross_pnl > 0 else 0
                pf.write(f"Total gained:      ${total_money_gained:,.2f}\n")
                pf.write(f"Total lost:        ${total_money_lost:,.2f}\n")
                pf.write(f"Gross PnL (no fee):${gross_pnl:+,.2f}\n")
                pf.write(f"Total fees:        ${total_fees_paid:,.2f} ({fee_impact:.1f}% of gross)\n")
                pf.write(f"Net PnL (w/ fee):  ${net_pnl:+,.2f}\n")
                pf.write(f"Avg win rate:      {np.mean(win_rates[-100:])*100:.1f}%\n" if win_rates else '')
                # Trade summary
                ts = trade_logger.get_summary()
                pf.write(f"\n--- Trade Summary ---\n")
                pf.write(f"Total trades:      {ts['total_trades']:,}\n")
                pf.write(f"Buys:              {ts['total_buys']:,}\n")
                pf.write(f"Sells:             {ts['total_sells']:,}\n")
                pf.write(f"Winning sells:     {ts['winning_sells']:,} ({ts['win_rate']*100:.1f}%)\n")
                pf.write(f"Losing sells:      {ts['losing_sells']:,}\n")
                pf.write(f"Total profit:      ${ts['total_profit']:,.2f}\n")
                pf.write(f"Total loss:        ${ts['total_loss']:,.2f}\n")
                pf.write(f"Net PnL:           ${ts['net_pnl']:+,.2f}\n")
                pf.write(f"Total fees:        ${ts['total_fees']:,.2f}\n")
                pf.write(f"Avg buy size:      ${ts['avg_buy_size_usd']:,.2f}\n")
                pf.write(f"Avg sell size:     ${ts['avg_sell_size_usd']:,.2f}\n")
                pf.write(f"Avg profit/win:    ${ts['avg_profit_per_win']:,.2f}\n")
                pf.write(f"Avg loss/loss:     ${ts['avg_loss_per_loss']:,.2f}\n")
            print(f"  [SNAPSHOT] Final snapshot saved: {final_dir}")
        except Exception as e:
            print(f"[WARN] Could not generate plots: {e}")
        
        elapsed = time.time() - start_time
        gross_pnl_final = total_money_gained - total_money_lost
        net_pnl_final = gross_pnl_final - total_fees_paid
        print(f"\n{'='*60}")
        print(f"Training completed! Total time: {elapsed/60:.1f}min")
        print(f"  Steps: {total_steps:,} | Updates: {update_count} | Episodes: {episode_count}")
        print(f"  Gross PnL (no fee): ${gross_pnl_final:+,.2f}")
        print(f"  Total fees:         ${total_fees_paid:,.2f}")
        print(f"  Net PnL (w/ fee):   ${net_pnl_final:+,.2f}")
        print(f"  Best eval reward:   {best_eval_reward:.2f}")
        if win_rates:
            print(f"  Avg win rate:       {np.mean(win_rates[-100:])*100:.1f}%")
        print(f"{'='*60}")
        
        # Auto-evaluate on val and test
        best_model_path = os.path.join(save_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            try:
                import subprocess
                python_exe = sys.executable
                eval_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluate.py')
                for split in ['val', 'test']:
                    print(f"\n{'='*60}")
                    print(f"Auto-evaluating on {split} data...")
                    print(f"{'='*60}")
                    subprocess.run([
                        python_exe, eval_script,
                        '--model_path', best_model_path,
                        '--algo', 'ppo_only',
                        '--split', split,
                        '--data_dir', args.data_dir,
                        '--n_episodes', '10',
                        '--max_steps', '2000',
                        '--initial_balance', str(args.initial_balance),
                        '--transaction_fee', str(args.transaction_fee),
                        '--flat_fee', str(args.flat_fee),
                    ], check=False)
            except Exception as e:
                print(f"[WARN] Auto-evaluation failed: {e}")


def _save_ppo_checkpoint(agent, step, update_count, episode_count,
                         episode_rewards, portfolio_values,
                         all_policy_losses, all_value_losses,
                         best_eval_reward, path,
                         win_rates=None, trade_counts=None,
                         episode_returns=None, total_fees_list=None,
                         total_money_lost=0.0, total_money_gained=0.0,
                         total_fees_paid=0.0, trade_history=None,
                         max_drawdowns=None):
    """Save a checkpoint with all state needed for resuming."""
    ckpt = {
        'step': step,
        'update_count': update_count,
        'episode_count': episode_count,
        'episode_rewards': episode_rewards,
        'portfolio_values': portfolio_values,
        'all_policy_losses': all_policy_losses,
        'all_value_losses': all_value_losses,
        'best_eval_reward': best_eval_reward,
        'policy_state_dict': agent.policy.state_dict(),
        'value_state_dict': agent.value_net.state_dict(),
        'win_rates': win_rates or [],
        'trade_counts': trade_counts or [],
        'episode_returns': episode_returns or [],
        'total_fees_list': total_fees_list or [],
        'total_money_lost': total_money_lost,
        'total_money_gained': total_money_gained,
        'total_fees_paid': total_fees_paid,
        'trade_history': trade_history or [],
        'max_drawdowns': max_drawdowns or [],
    }
    if hasattr(agent, 'optimizer'):
        ckpt['optimizer_state_dict'] = agent.optimizer.state_dict()
    torch.save(ckpt, path)


if __name__ == '__main__':
    main()
