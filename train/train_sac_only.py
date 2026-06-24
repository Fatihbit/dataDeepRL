"""
Training Script: SAC Only (MLP)
===============================

Dit script traint een Soft Actor-Critic (SAC) agent met simpele MLP
feature extraction voor cryptocurrency trading.

Gebruik:
    python train_sac_only.py --data_dir ../btc_l2_data --steps 1000000

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

# Voeg src toe aan path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.trading_env import FlatCryptoTradingEnv
from src.envs.vec_env import VectorizedTradingEnv
from src.models.sac import SACAgent
from src.utils.logger import TrainingLogger, setup_logging
from src.utils.trade_logger import TradeLogger
from train.common.setup import load_coredata_streaming
from src.utils.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback,
    ProgressCallback, LearningRateScheduler,
    PauseResumeCallback, load_checkpoint
)

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SAC agent (MLP only) for crypto trading'
    )
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./coreData')
    parser.add_argument('--max_files', type=int, default=100)
    parser.add_argument('--max_rows', type=int, default=90_000_000)
    parser.add_argument('--sequence_length', type=int, default=100)

    # Training — best hyperparams from Optuna tuning (trial 0, score 0.155)
    parser.add_argument('--total_steps', type=int, default=25_000_000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=3.5e-5)
    parser.add_argument('--gamma', type=float, default=0.993)
    parser.add_argument('--tau', type=float, default=0.0047)
    parser.add_argument('--alpha', type=float, default=0.30)
    parser.add_argument('--auto_alpha', action='store_true', default=True)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--num_envs', type=int, default=64, help='Number of parallel environments')

    # Environment
    parser.add_argument('--initial_balance', type=float, default=100000.0)
    parser.add_argument('--transaction_fee', type=float, default=0.0)
    parser.add_argument('--flat_fee', type=float, default=0.0, help='Flat fee per trade in USDT')
    parser.add_argument('--max_position', type=float, default=1.0)
    parser.add_argument('--max_episode_steps', type=int, default=3600,
                        help='Max steps per training episode (0=unlimited)')

    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=1, help='Eval na elke N updates')
    parser.add_argument('--n_eval_episodes', type=int, default=5)
    parser.add_argument('--max_eval_steps', type=int, default=2000, help='Max steps per eval episode (0=unlimited)')

    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--save_freq', type=int, default=25000)
    
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
        args.experiment_name = f"sac_mlp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    setup_logging(log_dir=args.log_dir, experiment_name=args.experiment_name)
    
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    
    # GPU optimizations
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    
    print(f"\n{'='*60}")
    print(f"SAC (MLP Only) Training")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"cudnn.benchmark: True")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"{'='*60}\n")
    
    # Random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    print("Loading data...")
    train_features, train_prices, val_features, val_prices, _, _ = load_coredata_streaming(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        max_rows=args.max_rows,
    )
    print(f"Train: {len(train_prices):,}, Val: {len(val_prices):,}")

    # Create environments
    num_envs = args.num_envs

    train_env_kwargs = dict(
        raw_features=train_features,
        prices=train_prices,
        window_size=args.sequence_length,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        flat_fee=args.flat_fee,
        max_position=args.max_position,
        discrete_actions=False,
        max_episode_steps=args.max_episode_steps,
        random_start=True,
        random_start_range=1.0,
    )
    eval_env = FlatCryptoTradingEnv(
        raw_features=val_features,
        prices=val_prices,
        window_size=args.sequence_length,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        flat_fee=args.flat_fee,
        max_position=args.max_position,
        discrete_actions=False,
    )

    print(f"Creating {num_envs} parallel training environments...")
    train_vec_env = VectorizedTradingEnv(num_envs=num_envs, env_kwargs=train_env_kwargs, env_class=FlatCryptoTradingEnv)
    
    obs_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    
    # Create agent
    print("\nCreating SAC agent...")
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        lr=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
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
    start_step = 0
    episode_count = 0
    episode_rewards = []
    portfolio_values = []
    all_critic_losses = []
    all_actor_losses = []
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
    
    ckpt_dir = os.path.join(args.log_dir, args.experiment_name, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_ckpt_path = os.path.join(ckpt_dir, 'resume_checkpoint.pt')
    
    # Step-level CSV + Trade logger
    save_dir = os.path.join(args.log_dir, args.experiment_name)
    episode_csv_path = os.path.join(save_dir, 'episode_finance.csv')
    
    # Training monitor CSV (written every eval — not every step)
    monitor_csv_path = os.path.join(save_dir, 'training_monitor.csv')
    if not os.path.exists(monitor_csv_path):
        with open(monitor_csv_path, 'w') as f:
            f.write('step,episode,timestamp,reward_avg100,portfolio_value,'
                    'total_trades,buys,sells,winning_sells,losing_sells,win_pct,'
                    'total_profit,total_loss,net_pnl,total_fees,'
                    'avg_buy_size,avg_sell_size,avg_profit_per_win,avg_loss_per_loss,'
                    'trade_freq,critic_loss,actor_loss,eval_composite_score,'
                    'eval_sharpe,eval_return,eval_drawdown\n')

    # Eval results CSV — overzicht of agent leert
    eval_csv_path = os.path.join(save_dir, 'eval_results.csv')
    if not os.path.exists(eval_csv_path):
        with open(eval_csv_path, 'w') as f:
            f.write('step,timestamp,mean_reward,std_reward,best_reward,'
                    'eval_buys,eval_sells,eval_holds,avg_action,'
                    'is_improving,status,composite_score\n')
    
    trade_logger = TradeLogger(save_dir)
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, weights_only=False, map_location=device)
            if 'actor_state_dict' in ckpt:
                agent.actor.load_state_dict(ckpt['actor_state_dict'])
            if 'critic_state_dict' in ckpt:
                agent.critic.load_state_dict(ckpt['critic_state_dict'])
            if 'critic_target_state_dict' in ckpt:
                agent.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
            if 'actor_optimizer' in ckpt and hasattr(agent, 'actor_optimizer'):
                agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
            if 'critic_optimizer' in ckpt and hasattr(agent, 'critic_optimizer'):
                agent.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
            start_step = ckpt.get('step', 0)
            episode_count = ckpt.get('episode_count', 0)
            episode_rewards = ckpt.get('episode_rewards', [])
            portfolio_values = ckpt.get('portfolio_values', [])
            all_critic_losses = ckpt.get('all_critic_losses', [])
            all_actor_losses = ckpt.get('all_actor_losses', [])
            best_eval_reward = ckpt.get('best_eval_reward', float('-inf'))
            win_rates = ckpt.get('win_rates', [])
            trade_counts = ckpt.get('trade_counts', [])
            episode_returns = ckpt.get('episode_returns', [])
            total_fees_list = ckpt.get('total_fees_list', [])
            total_money_lost = ckpt.get('total_money_lost', 0.0)
            total_money_gained = ckpt.get('total_money_gained', 0.0)
            total_fees_paid = ckpt.get('total_fees_paid', 0.0)
            max_drawdowns = ckpt.get('max_drawdowns', [])
            if 'trade_history' in ckpt:
                trade_logger.restore_from_list(ckpt['trade_history'])
            print(f"  Continuing from step {start_step:,}, episodes: {episode_count}")
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
        print("\n[PAUSE] Pause requested! Saving checkpoint...")
    
    signal.signal(signal.SIGINT, _signal_handler)
    
    # Training
    print(f"\nStarting training... (total_steps: {args.total_steps:,})")
    print(f"  Train data: {len(train_vec_env.prices):,}, Val data: {len(eval_env.prices):,}")
    print(f"  Parallel envs: {num_envs}, transitions per step: {num_envs}")
    print(f"  Eval every {args.eval_freq:,} updates, {args.n_eval_episodes} eval episodes, max {args.max_eval_steps} steps each")
    print(f"  Log interval: {args.log_interval:,} steps")
    
    obs_list = train_vec_env.reset()
    # Per-env episode trackers
    ep_rewards = [0.0] * num_envs
    ep_lengths = [0] * num_envs
    ep_buy_counts = [0] * num_envs
    ep_sell_counts = [0] * num_envs
    ep_hold_counts = [0] * num_envs
    ep_fees_env = [0.0] * num_envs
    prev_balances = [args.initial_balance] * num_envs
    prev_btc_helds = [0.0] * num_envs
    training_start = time.time()

    actual_max_steps = args.total_steps

    update_count = 0
    step = start_step
    try:
        for step in range(start_step, actual_max_steps, num_envs):
            # Batched action selection
            obs_batch = np.stack(obs_list)  # (num_envs, obs_dim)
            if agent.total_steps < agent.start_steps:
                actions_np = np.random.uniform(-1, 1, size=(num_envs, action_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_batch).to(agent.device)
                    actions_t, _ = agent.actor.sample(obs_t, deterministic=False)
                    actions_np = actions_t.cpu().numpy()
            
            # Step all environments
            next_obs_list, rewards, dones, infos = train_vec_env.step(actions_np.tolist())
            
            # Store N transitions in replay buffer and track episodes
            for i in range(num_envs):
                info = infos[i]
                trade_info = info.get('trade_info', {})
                if trade_info and trade_info.get('executed', False):
                    trade_type = trade_info.get('type', 'unknown')
                    fee = trade_info.get('fee', 0)
                    profit = trade_info.get('profit', 0)
                    ep_fees_env[i] += fee
                    
                    if trade_type == 'buy':
                        ep_buy_counts[i] += 1
                    elif trade_type == 'sell':
                        ep_sell_counts[i] += 1
                        if profit > 0:
                            total_money_gained += profit
                        else:
                            total_money_lost += abs(profit)
                    total_fees_paid += fee
                    
                    trade_logger.log_trade(
                        episode=episode_count,
                        step=step + i,
                        trade_info=trade_info,
                        balance_before=prev_balances[i],
                        balance_after=info.get('balance', prev_balances[i]),
                        btc_held_before=prev_btc_helds[i],
                        btc_held_after=info.get('btc_held', prev_btc_helds[i]),
                        portfolio_value=info.get('portfolio_value', 0),
                    )
                else:
                    ep_hold_counts[i] += 1
                
                prev_balances[i] = info.get('balance', prev_balances[i])
                prev_btc_helds[i] = info.get('btc_held', prev_btc_helds[i])
                
                # Store transition in replay buffer
                agent.buffer.add(obs_batch[i], actions_np[i], rewards[i],
                                 np.array(next_obs_list[i], dtype=np.float32), float(dones[i]))
                agent.total_steps += 1
                ep_rewards[i] += rewards[i]
                ep_lengths[i] += 1
                
                # Episode end for env i
                if dones[i]:
                    terminal_info = info.get('terminal_info', info)
                    pv = terminal_info.get('portfolio_value', args.initial_balance)
                    wr = terminal_info.get('win_rate', 0)
                    tc = terminal_info.get('total_trades', 0)
                    ret = terminal_info.get('total_return', 0)
                    ep_drawdown = terminal_info.get('max_drawdown', 0.0)
                    ep_sharpe = terminal_info.get('sharpe_ratio', 0.0)
                    ep_composite = 0.5 * np.clip(ep_sharpe, -5, 5) / 5 + 0.5 * np.clip(ret, -1, 1) - 0.2 * ep_drawdown

                    logger.log_episode(
                        episode=episode_count,
                        total_reward=ep_rewards[i],
                        episode_length=ep_lengths[i],
                        info={'portfolio_value': pv}
                    )
                    episode_rewards.append(ep_rewards[i])
                    portfolio_values.append(pv)
                    win_rates.append(wr)
                    trade_counts.append(tc)
                    episode_returns.append(ret)
                    total_fees_list.append(ep_fees_env[i])
                    max_drawdowns.append(ep_drawdown)
                    
                    if not os.path.exists(episode_csv_path):
                        with open(episode_csv_path, 'w') as f:
                            f.write('episode,step,reward,portfolio_value,win_rate,total_trades,total_return,fees,buys,sells,holds,drawdown,composite_score\n')
                    with open(episode_csv_path, 'a') as f:
                        f.write(f"{episode_count},{step},{ep_rewards[i]:.4f},{pv:.2f},{wr:.4f},{tc},{ret:.6f},{ep_fees_env[i]:.4f},{ep_buy_counts[i]},{ep_sell_counts[i]},{ep_hold_counts[i]},{ep_drawdown:.4f},{ep_composite:.6f}\n")
                    
                    if episode_count % 10 == 0:
                        net_pnl = total_money_gained - total_money_lost - total_fees_paid
                        print(f"  [FINANCE] Ep {episode_count}: PV=${pv:,.2f} | WR={wr*100:.1f}% | "
                              f"Trades={tc} (B:{ep_buy_counts[i]}/S:{ep_sell_counts[i]}/H:{ep_hold_counts[i]}) | Fees=${ep_fees_env[i]:.2f}")
                        print(f"  [CUMULATIVE] Gained=${total_money_gained:,.2f} | Lost=${total_money_lost:,.2f} | "
                              f"Fees=${total_fees_paid:,.2f} | Net=${net_pnl:+,.2f}")
                    
                    episode_count += 1
                    # Reset per-env trackers
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0
                    ep_buy_counts[i] = 0
                    ep_sell_counts[i] = 0
                    ep_hold_counts[i] = 0
                    ep_fees_env[i] = 0.0
                    prev_balances[i] = args.initial_balance
                    prev_btc_helds[i] = 0.0
            
            obs_list = next_obs_list
            
            # SAC update (already batched from replay buffer)
            if agent.total_steps >= agent.start_steps:
                losses = agent.update()
                if losses:
                    all_critic_losses.append(losses.get('critic_loss', 0))
                    all_actor_losses.append(losses.get('actor_loss', 0))
                    update_count += 1
            
            if step % args.log_interval == 0:
                elapsed = time.time() - training_start
                steps_done = step - start_step + num_envs
                steps_per_sec = steps_done / max(elapsed, 1)
                progress = (step + num_envs) / actual_max_steps * 100
                remaining = (actual_max_steps - step - num_envs) / max(steps_per_sec, 0.01)
                
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                avg_pv = np.mean(portfolio_values[-100:]) if portfolio_values else args.initial_balance
                print(
                    f"Step {step+num_envs:>8,}/{actual_max_steps:,} ({progress:5.1f}%) | "
                    f"{steps_per_sec:.0f} steps/s | "
                    f"Ep: {episode_count} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg PV: ${avg_pv:,.0f} | "
                    f"ETA: {remaining/60:.1f}min"
                )
            
            # Evaluation
            if update_count > 0 and update_count % args.eval_freq == 0:
                eval_rewards = []
                eval_infos = []
                print(f"  Starting eval ({args.n_eval_episodes} episodes, max {args.max_eval_steps} steps)...")
                total_eval_buys = 0
                total_eval_sells = 0
                total_eval_holds = 0
                total_eval_actions = 0.0
                for ep_i in range(args.n_eval_episodes):
                    eval_obs, _ = eval_env.reset()
                    eval_done = False
                    eval_reward = 0
                    eval_steps = 0
                    eval_step_info = {}
                    eval_cum_profit = 0.0
                    eval_cum_loss = 0.0
                    ep_eval_buys = 0
                    ep_eval_sells = 0
                    while not eval_done:
                        eval_action = agent.select_action(eval_obs, deterministic=True)
                        a_val = float(eval_action[0]) if isinstance(eval_action, np.ndarray) else float(eval_action)
                        total_eval_actions += a_val
                        if a_val > 0:
                            total_eval_buys += 1
                            ep_eval_buys += 1
                        elif a_val < 0:
                            total_eval_sells += 1
                            ep_eval_sells += 1
                        else:
                            total_eval_holds += 1
                        eval_obs, r, term, trunc, eval_step_info = eval_env.step(eval_action)
                        eval_done = term or trunc
                        eval_reward += r
                        eval_steps += 1
                        trade_info = eval_step_info.get('trade_info', {})
                        if trade_info and trade_info.get('executed', False) and trade_info.get('type') == 'sell':
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
                          f"Trades={ep_eval_buys+ep_eval_sells} (B:{ep_eval_buys}/S:{ep_eval_sells})")
                _c_scores = [0.5 * np.clip(i.get('sharpe_ratio', 0.0), -5, 5) / 5
                             + 0.5 * np.clip(i.get('total_return', 0.0), -1, 1)
                             - 0.2 * i.get('max_drawdown', 0.0) for i in eval_infos]
                eval_composite = float(np.mean(_c_scores)) if _c_scores else 0.0
                eval_sharpe = float(np.mean([i.get('sharpe_ratio', 0.0) for i in eval_infos])) if eval_infos else 0.0
                eval_return = float(np.mean([i.get('total_return', 0.0) for i in eval_infos])) if eval_infos else 0.0
                eval_drawdown = float(np.mean([i.get('max_drawdown', 0.0) for i in eval_infos])) if eval_infos else 0.0

                mean_eval = np.mean(eval_rewards)
                std_eval = np.std(eval_rewards)
                total_eval_steps = total_eval_buys + total_eval_sells + total_eval_holds
                avg_action = total_eval_actions / max(total_eval_steps, 1)
                logger.log_evaluation(step, eval_rewards)

                is_improving = mean_eval > best_eval_reward

                # Status bepalen
                if total_eval_buys == 0 and total_eval_sells == 0:
                    status = "DEAD (alleen hold)"
                elif is_improving:
                    status = "LEARNING"
                else:
                    status = "OK"

                # Compact eval output
                print(f"  [EVAL] step {step:,} | reward={mean_eval:+.4f} (best={best_eval_reward:+.4f}) | "
                      f"B:{total_eval_buys} S:{total_eval_sells} H:{total_eval_holds} | "
                      f"avg_action={avg_action:.4f} | {status}")

                # Schrijf eval CSV
                import datetime as _dt
                with open(eval_csv_path, 'a') as f:
                    f.write(f"{step},{_dt.datetime.now().isoformat()},{mean_eval:.6f},{std_eval:.6f},{best_eval_reward:.6f},"
                            f"{total_eval_buys},{total_eval_sells},{total_eval_holds},{avg_action:.6f},"
                            f"{int(is_improving)},{status},{eval_composite:.6f}\n")

                # Schrijf monitor CSV (alleen bij eval, niet elke step)
                try:
                    ts = trade_logger.get_summary()
                    trade_freq = ts['total_trades'] / max(step, 1)
                    last_cl = all_critic_losses[-1] if all_critic_losses else 0
                    last_al = all_actor_losses[-1] if all_actor_losses else 0
                    avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    pv = portfolio_values[-1] if portfolio_values else args.initial_balance
                    with open(monitor_csv_path, 'a') as f:
                        f.write(f"{step},{episode_count},{_dt.datetime.now().isoformat()},"
                                f"{avg_r:.4f},{pv:.2f},"
                                f"{ts['total_trades']},{ts['total_buys']},{ts['total_sells']},"
                                f"{ts['winning_sells']},{ts['losing_sells']},{ts['win_rate']*100:.2f},"
                                f"{ts['total_profit']:.2f},{ts['total_loss']:.2f},{ts['net_pnl']:.2f},{ts['total_fees']:.2f},"
                                f"{ts['avg_buy_size_usd']:.2f},{ts['avg_sell_size_usd']:.2f},"
                                f"{ts['avg_profit_per_win']:.2f},{ts['avg_loss_per_loss']:.2f},"
                                f"{trade_freq:.6f},{last_cl:.6f},{last_al:.6f},{eval_composite:.6f},"
                                f"{eval_sharpe:.6f},{eval_return:.6f},{eval_drawdown:.6f}\n")
                except Exception as e:
                    pass

                if mean_eval > best_eval_reward:
                    best_eval_reward = mean_eval
                    save_dict = {
                        'step': step,
                        'actor_state_dict': agent.actor.state_dict(),
                        'critic_state_dict': agent.critic.state_dict(),
                        'mean_reward': mean_eval
                    }
                    best_path = os.path.join(args.log_dir, args.experiment_name, 'best_model.pt')
                    torch.save(save_dict, best_path)
                    
                    # Backup met prestatie-info
                    backup_dir = os.path.join(args.log_dir, args.experiment_name, 'backups')
                    os.makedirs(backup_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_name = f"best_model_step{step}_r{mean_eval:.1f}_{ts}.pt"
                    torch.save(save_dict, os.path.join(backup_dir, backup_name))
                    print(f"  [*] New best model! Reward: {mean_eval:.2f}")
                    print(f"  [BACKUP] {backup_name}")
            
                # Save snapshot
                try:
                    snapshot_dir = os.path.join(args.log_dir, args.experiment_name, 'plots', f'step_{step:07d}')
                    os.makedirs(snapshot_dir, exist_ok=True)
                    avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    with open(os.path.join(snapshot_dir, 'params.txt'), 'w') as pf:
                        pf.write(f"Snapshot at step {step:,}\n")
                        pf.write(f"{'='*50}\n")
                        pf.write(f"\n--- Hyperparameters ---\n")
                        pf.write(f"Learning rate:     {args.learning_rate}\n")
                        pf.write(f"Batch size:        {args.batch_size}\n")
                        pf.write(f"Gamma:             {args.gamma}\n")
                        pf.write(f"Tau:               {args.tau}\n")
                        pf.write(f"Alpha:             {args.alpha}\n")
                        pf.write(f"Auto alpha:        {args.auto_alpha}\n")
                        pf.write(f"Hidden dims:       {args.hidden_dims}\n")
                        pf.write(f"Sequence length:   {args.sequence_length}\n")
                        pf.write(f"Total steps:       {args.total_steps:,}\n")
                        pf.write(f"Initial balance:   {args.initial_balance}\n")
                        pf.write(f"Transaction fee:   {args.transaction_fee}\n")
                        pf.write(f"Max position:      {args.max_position}\n")
                        pf.write(f"Seed:              {args.seed}\n")
                        pf.write(f"\n--- Training Progress ---\n")
                        pf.write(f"Current step:      {step:,}\n")
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

            # Save resume checkpoint
            if step > 0 and step % args.save_freq == 0:
                _save_resume_checkpoint(agent, step, episode_count, episode_rewards,
                                        portfolio_values, all_critic_losses, all_actor_losses,
                                        best_eval_reward, resume_ckpt_path,
                                        win_rates=win_rates, trade_counts=trade_counts,
                                        episode_returns=episode_returns, total_fees_list=total_fees_list,
                                        total_money_lost=total_money_lost, total_money_gained=total_money_gained,
                                        total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
                                        max_drawdowns=max_drawdowns)
                print(f"  [SAVE] Checkpoint saved at step {step:,}")
            
            # Check pause
            if _pause_requested:
                _save_resume_checkpoint(agent, step, episode_count, episode_rewards,
                                        portfolio_values, all_critic_losses, all_actor_losses,
                                        best_eval_reward, resume_ckpt_path,
                                        win_rates=win_rates, trade_counts=trade_counts,
                                        episode_returns=episode_returns, total_fees_list=total_fees_list,
                                        total_money_lost=total_money_lost, total_money_gained=total_money_gained,
                                        total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
                                        max_drawdowns=max_drawdowns)
                print(f"\n[PAUSE] Training paused at step {step:,}")
                print(f"  Resume: python train_sac_only.py --resume {resume_ckpt_path} [other args]")
                break
    
    finally:
        # Save final model
        _save_resume_checkpoint(agent, step, episode_count, episode_rewards,
                                portfolio_values, all_critic_losses, all_actor_losses,
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
                pf.write(f"Gamma:             {args.gamma}\n")
                pf.write(f"Tau:               {args.tau}\n")
                pf.write(f"Alpha:             {args.alpha}\n")
                pf.write(f"Auto alpha:        {args.auto_alpha}\n")
                pf.write(f"Hidden dims:       {args.hidden_dims}\n")
                pf.write(f"Sequence length:   {args.sequence_length}\n")
                pf.write(f"Total steps:       {args.total_steps:,}\n")
                pf.write(f"Initial balance:   {args.initial_balance}\n")
                pf.write(f"Transaction fee:   {args.transaction_fee}\n")
                pf.write(f"Max position:      {args.max_position}\n")
                pf.write(f"Seed:              {args.seed}\n")
                pf.write(f"\n--- Final Results ---\n")
                pf.write(f"Total steps:       {step:,}\n")
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
        
        elapsed = time.time() - training_start
        gross_pnl_final = total_money_gained - total_money_lost
        net_pnl_final = gross_pnl_final - total_fees_paid
        print(f"\n{'='*60}")
        print(f"Training completed! Total time: {elapsed/60:.1f}min")
        print(f"  Steps: {step:,} | Episodes: {episode_count}")
        print(f"  Gross PnL (no fee): ${gross_pnl_final:+,.2f}")
        print(f"  Total fees:         ${total_fees_paid:,.2f}")
        print(f"  Net PnL (w/ fee):   ${net_pnl_final:+,.2f}")
        print(f"  Best eval reward:   {best_eval_reward:.2f}")
        if win_rates:
            print(f"  Avg win rate:       {np.mean(win_rates[-100:])*100:.1f}%")
        print(f"{'='*60}")
        
        # Auto-evaluate on val and test
        best_model_path = os.path.join(args.log_dir, args.experiment_name, 'best_model.pt')
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
                        '--algo', 'sac_only',
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


def _save_resume_checkpoint(agent, step, episode_count, episode_rewards,
                            portfolio_values, all_critic_losses, all_actor_losses,
                            best_eval_reward, path,
                            win_rates=None, trade_counts=None,
                            episode_returns=None, total_fees_list=None,
                            total_money_lost=0.0, total_money_gained=0.0,
                            total_fees_paid=0.0, trade_history=None,
                            max_drawdowns=None):
    """Save a checkpoint with all state needed for resuming."""
    ckpt = {
        'step': step,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'episode_count': episode_count,
        'episode_rewards': episode_rewards,
        'portfolio_values': portfolio_values,
        'all_critic_losses': all_critic_losses,
        'all_actor_losses': all_actor_losses,
        'best_eval_reward': best_eval_reward,
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
    if hasattr(agent, 'critic_target'):
        ckpt['critic_target_state_dict'] = agent.critic_target.state_dict()
    if hasattr(agent, 'actor_optimizer'):
        ckpt['actor_optimizer'] = agent.actor_optimizer.state_dict()
    if hasattr(agent, 'critic_optimizer'):
        ckpt['critic_optimizer'] = agent.critic_optimizer.state_dict()
    torch.save(ckpt, path)


if __name__ == '__main__':
    main()
