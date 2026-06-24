"""
Standalone Evaluation Script
=============================

Evalueer getrainde modellen op validation/test data ZONDER dat het model leert.
Kan op elk moment handmatig worden gestart of wordt automatisch aangeroepen
aan het einde van training.

Ondersteunde model types:
  - ppo_deeplob:  PPOWithPretrainedDeepLOB  (dict obs, CryptoTradingEnv)
  - sac_deeplob:  SACWithPretrainedDeepLOB  (dict obs, CryptoTradingEnv)
  - ppo_only:     PPOAgent                  (flat obs, FlatCryptoTradingEnv)
  - sac_only:     SACAgent                  (flat obs, FlatCryptoTradingEnv)

Gebruik:
  # Evalueer op validation data (default)
  python evaluate.py --model_path logs/ppo_deeplob_frozen_.../best_model.pt --algo ppo_deeplob

  # Evalueer op test data
  python evaluate.py --model_path logs/.../best_model.pt --algo ppo_deeplob --split test

  # Meerdere episodes, custom max steps
  python evaluate.py --model_path logs/.../best_model.pt --algo sac_deeplob --n_episodes 20 --max_steps 5000

  # Alle data (geen max steps limiet)
  python evaluate.py --model_path logs/.../best_model.pt --algo ppo_only --max_steps 0

Resultaten worden opgeslagen in: <model_dir>/eval_<split>_<timestamp>/
  - eval_summary.json      Alle metrics in machine-readable formaat
  - eval_episodes.csv       Per-episode metrics
  - eval_trades.csv         Alle individuele trades
  - eval_report.txt         Leesbaar rapport
  - plots/                  Performance grafieken
"""

import os
import sys
import argparse
import json
import datetime
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs.trading_env import CryptoTradingEnv, FlatCryptoTradingEnv
from src.utils.trade_logger import TradeLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model (no learning)')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint (best_model.pt, final_model.pt, resume_checkpoint.pt)')
    parser.add_argument('--algo', type=str, required=True,
                        choices=['ppo_deeplob', 'sac_deeplob', 'ppo_only', 'sac_only'],
                        help='Algorithm/model type')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Which data split to evaluate on (default: val)')

    # DeepLOB (only needed for ppo_deeplob / sac_deeplob)
    parser.add_argument('--deeplob_model', type=str, default='./models/deeplob_pretrained.pt',
                        help='Path to pre-trained DeepLOB (for deeplob algos)')

    # Data
    parser.add_argument('--data_dir', type=str, default='./coreData')
    parser.add_argument('--max_rows', type=int, default=100_000_000,
                        help='Max rows to load (test/val get max_rows//4)')

    # Eval settings
    parser.add_argument('--n_episodes', type=int, default=1,
                        help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=100_000,
                        help='Max steps per episode (0=unlimited, run until data ends)')

    # Environment
    parser.add_argument('--initial_balance', type=float, default=100000.0)
    parser.add_argument('--transaction_fee', type=float, default=0.0)
    parser.add_argument('--flat_fee', type=float, default=0.0, help='Flat fee per trade in USDT')

    # Other
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory (default: next to model)')
    parser.add_argument('--no_plots', action='store_true', help='Skip plot generation')

    return parser.parse_args()


def load_agent(algo, model_path, deeplob_model, device):
    """Load agent in eval-only mode (no optimizer needed)."""
    print(f"Loading {algo} model from: {model_path}")

    if algo == 'ppo_deeplob':
        from train.train_ppo_with_deeplob import PPOWithPretrainedDeepLOB
        agent = PPOWithPretrainedDeepLOB(
            deeplob_model_path=deeplob_model,
            freeze_deeplob=True,
            device=device
        )
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        agent.deeplob.load_state_dict(ckpt['deeplob_state_dict'])
        agent.portfolio_encoder.load_state_dict(ckpt['portfolio_encoder_state_dict'])
        agent.network.load_state_dict(ckpt['network_state_dict'])
        agent.total_steps = ckpt.get('total_steps', 0)

    elif algo == 'sac_deeplob':
        from train.train_sac_with_deeplob import SACWithPretrainedDeepLOB
        agent = SACWithPretrainedDeepLOB(
            deeplob_model_path=deeplob_model,
            freeze_deeplob=True,
            device=device
        )
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        agent.deeplob.load_state_dict(ckpt['deeplob_state_dict'])
        agent.portfolio_encoder.load_state_dict(ckpt['portfolio_encoder_state_dict'])
        agent.actor.load_state_dict(ckpt['actor_state_dict'])
        agent.critic.load_state_dict(ckpt['critic_state_dict'])
        if 'critic_target_state_dict' in ckpt:
            agent.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
        agent.total_steps = ckpt.get('total_steps', 0)

    elif algo == 'ppo_only':
        from src.models.ppo import PPOAgent
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        # Infer obs_dim from policy network's first layer
        policy_sd = ckpt.get('policy_state_dict', {})
        if not policy_sd:
            # Try resume checkpoint format where policy is stored differently
            policy_sd = {k: v for k, v in ckpt.items() if k.startswith('policy_')}
        first_weight_key = next((k for k in policy_sd if 'weight' in k), None)
        obs_dim = policy_sd[first_weight_key].shape[1] if first_weight_key else 100
        agent = PPOAgent(obs_dim=obs_dim, action_dim=3, device=device)
        if 'policy_state_dict' in ckpt:
            agent.policy.load_state_dict(ckpt['policy_state_dict'])
        if 'value_net_state_dict' in ckpt:
            agent.value_net.load_state_dict(ckpt['value_net_state_dict'])
        agent.total_steps = ckpt.get('total_steps', ckpt.get('step', 0))

    elif algo == 'sac_only':
        from src.models.sac import SACAgent
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        actor_sd = ckpt.get('actor_state_dict', {})
        first_weight_key = next((k for k in actor_sd if 'weight' in k), None)
        obs_dim = actor_sd[first_weight_key].shape[1] if first_weight_key else 100
        # Infer action_dim from mean_head (not stacked)
        if 'mean_head.weight' in actor_sd:
            action_dim = actor_sd['mean_head.weight'].shape[0]
        else:
            weight_keys = [k for k in actor_sd if 'weight' in k]
            last_weight_key = weight_keys[-1] if weight_keys else None
            action_dim = actor_sd[last_weight_key].shape[0] // 2 if last_weight_key else 1
        # Infer hidden_dims from shared layers (only 2D Linear weights, skip LayerNorm)
        hidden_dims = []
        for k in sorted(actor_sd.keys()):
            if k.startswith('shared.') and k.endswith('.weight') and actor_sd[k].dim() == 2:
                hidden_dims.append(actor_sd[k].shape[0])
        print(f"  Inferred hidden_dims: {hidden_dims}, action_dim: {action_dim}")
        agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims, device=device)
        agent.actor.load_state_dict(ckpt['actor_state_dict'])
        if 'critic_state_dict' in ckpt:
            agent.critic.load_state_dict(ckpt['critic_state_dict'])
        agent.total_steps = ckpt.get('total_steps', ckpt.get('step', 0))
        agent.start_steps = 0  # No random exploration during eval

    print(f"  Model loaded (total_steps: {agent.total_steps:,})")
    return agent


def create_eval_env(algo, features, prices, sequence_length, args):
    """Create evaluation environment based on algorithm type."""
    # sac_only gebruikt continue acties, alle andere (PPO en sac_deeplob) discrete
    use_continuous = algo in ('sac_only',)
    env_kwargs = dict(
        raw_features=features,
        prices=prices,
        window_size=sequence_length,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        flat_fee=args.flat_fee,
        max_episode_steps=args.max_steps if args.max_steps > 0 else 0,
        random_start=False,
        discrete_actions=not use_continuous
    )
    if algo in ('ppo_deeplob', 'sac_deeplob'):
        return CryptoTradingEnv(**env_kwargs)
    else:
        return FlatCryptoTradingEnv(**env_kwargs)


def select_action_deterministic(agent, algo, obs):
    """Select action deterministically (no learning, no exploration)."""
    if algo in ('ppo_deeplob', 'ppo_only'):
        action, _, _ = agent.select_action(obs, deterministic=True)
        return action
    else:
        return agent.select_action(obs, deterministic=True)


def run_evaluation(agent, algo, env, n_episodes, trade_logger, args):
    """Run evaluation episodes. Model is in eval mode with no gradients."""
    episodes = []

    for ep_i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_buys = 0
        ep_sells = 0
        ep_holds = 0
        ep_fees = 0.0
        prev_balance = args.initial_balance
        prev_btc_held = 0.0
        portfolio_trajectory = [args.initial_balance]

        log_every = 10_000
        eval_start = time.time()
        cum_profit = 0.0
        cum_loss = 0.0
        progress_log = []

        while not done:
            action = select_action_deterministic(agent, algo, obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
            portfolio_trajectory.append(info.get('portfolio_value', args.initial_balance))

            # Track trades
            trade_info = info.get('trade_info', {})
            if trade_info and trade_info.get('executed', False):
                trade_type = trade_info.get('type', 'unknown')
                if trade_type == 'buy':
                    ep_buys += 1
                elif trade_type == 'sell':
                    ep_sells += 1
                    profit = trade_info.get('profit', 0.0)
                    if profit > 0:
                        cum_profit += profit
                    else:
                        cum_loss += abs(profit)
                ep_fees += trade_info.get('fee', 0.0)

                # Log trade - use the interface the trade_logger expects
                trade_logger.log_trade(
                    episode=ep_i,
                    step=ep_steps,
                    trade_info=trade_info,
                    balance_before=prev_balance,
                    balance_after=info.get('balance', prev_balance),
                    btc_held_before=prev_btc_held,
                    btc_held_after=info.get('btc_held', prev_btc_held),
                    portfolio_value=info.get('portfolio_value', args.initial_balance),
                )
            else:
                ep_holds += 1

            if ep_steps % log_every == 0:
                pv_now = info.get('portfolio_value', args.initial_balance)
                elapsed = time.time() - eval_start
                sps = ep_steps / max(elapsed, 1)
                pnl_now = pv_now - args.initial_balance
                net_realized = cum_profit - cum_loss - ep_fees
                print(f"    Step {ep_steps:>12,} | PV=${pv_now:,.2f} | PnL=${pnl_now:+,.2f} | "
                      f"Realized=${net_realized:+,.2f} (W:${cum_profit:,.2f} L:${cum_loss:,.2f}) | "
                      f"Trades={ep_buys + ep_sells} (B:{ep_buys}/S:{ep_sells}) | "
                      f"{sps:,.0f} steps/s | {elapsed:.0f}s")
                progress_log.append({
                    'step': ep_steps, 'portfolio_value': round(pv_now, 2),
                    'pnl': round(pnl_now, 2), 'realized_profit': round(cum_profit, 2),
                    'realized_loss': round(cum_loss, 2), 'net_realized': round(net_realized, 2),
                    'fees': round(ep_fees, 2), 'buys': ep_buys, 'sells': ep_sells,
                    'steps_per_sec': round(sps, 0), 'elapsed_sec': round(elapsed, 1)
                })

            prev_balance = info.get('balance', prev_balance)
            prev_btc_held = info.get('btc_held', prev_btc_held)
            obs = next_obs

        # End-of-episode metrics from env
        pv = info.get('portfolio_value', args.initial_balance)
        total_return = info.get('total_return', (pv - args.initial_balance) / args.initial_balance)
        win_rate = info.get('win_rate', 0.0)
        total_trades = info.get('total_trades', ep_buys + ep_sells)
        max_drawdown = info.get('max_drawdown', 0.0)
        sharpe = info.get('sharpe_ratio', 0.0)
        pnl = pv - args.initial_balance
        gross_pnl = pnl + ep_fees  # PnL before fees

        # Compute drawdown from trajectory as backup
        pv_arr = np.array(portfolio_trajectory)
        peak = np.maximum.accumulate(pv_arr)
        drawdowns = (peak - pv_arr) / np.maximum(peak, 1e-8)
        max_dd_computed = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
        ep_drawdown = max(max_drawdown, max_dd_computed)

        # Composite score — zelfde formule als tijdens training (vergelijkbaar)
        composite_score = (0.5 * np.clip(sharpe, -5, 5) / 5
                           + 0.5 * np.clip(total_return, -1, 1)
                           - 0.2 * ep_drawdown)

        ep_result = {
            'episode': ep_i,
            'steps': ep_steps,
            'reward': round(ep_reward, 4),
            'portfolio_value': round(pv, 2),
            'gross_pnl': round(gross_pnl, 2),
            'net_pnl': round(pnl, 2),
            'pnl': round(pnl, 2),
            'return_pct': round(total_return * 100, 4),
            'win_rate': round(win_rate, 4),
            'total_trades': total_trades,
            'buys': ep_buys,
            'sells': ep_sells,
            'holds': ep_holds,
            'fees': round(ep_fees, 4),
            'max_drawdown': round(ep_drawdown, 4),
            'sharpe_ratio': round(sharpe, 4),
            'composite_score': round(float(composite_score), 6),
            'avg_reward_per_step': round(ep_reward / max(ep_steps, 1), 6),
            'final_balance': round(info.get('balance', 0), 2),
            'final_btc_held': round(info.get('btc_held', 0), 8),
        }
        episodes.append(ep_result)

        status = "PROFIT" if pnl >= 0 else "LOSS"
        fee_str = f" | Fees=${ep_fees:,.2f}" if ep_fees > 0 else ""
        print(f"  Episode {ep_i+1}/{n_episodes}: {ep_steps:,} steps | "
              f"PV=${pv:,.2f} | Gross=${gross_pnl:+,.2f} | Net=${pnl:+,.2f} ({total_return*100:+.2f}%){fee_str} | "
              f"WR={win_rate*100:.0f}% | Trades={total_trades} | "
              f"DD={max(max_drawdown, max_dd_computed)*100:.1f}% | [{status}]")

    return episodes, progress_log


def compute_buy_and_hold(prices, sequence_length, initial_balance, max_steps):
    """Bereken buy-and-hold return over dezelfde periode als de agent evalueerde."""
    start_idx = sequence_length
    end_idx = len(prices) - 1
    if max_steps > 0:
        end_idx = min(start_idx + max_steps, end_idx)

    start_price = prices[start_idx]
    end_price = prices[end_idx]
    btc_bought = initial_balance / start_price
    final_value = btc_bought * end_price
    pnl = final_value - initial_balance
    return_pct = (end_price - start_price) / start_price * 100

    return {
        'start_price': round(float(start_price), 2),
        'end_price': round(float(end_price), 2),
        'btc_bought': round(float(btc_bought), 8),
        'final_value': round(float(final_value), 2),
        'pnl': round(float(pnl), 2),
        'return_pct': round(float(return_pct), 4),
        'n_steps': end_idx - start_idx,
    }


def compute_summary(episodes, args):
    """Compute aggregate statistics across all episodes."""
    n = len(episodes)
    if n == 0:
        return {}

    pvs = [e['portfolio_value'] for e in episodes]
    pnls = [e['pnl'] for e in episodes]
    gross_pnls = [e['gross_pnl'] for e in episodes]
    returns = [e['return_pct'] for e in episodes]
    rewards = [e['reward'] for e in episodes]
    win_rates = [e['win_rate'] for e in episodes]
    trades = [e['total_trades'] for e in episodes]
    drawdowns = [e['max_drawdown'] for e in episodes]
    sharpes = [e['sharpe_ratio'] for e in episodes]
    composites = [e['composite_score'] for e in episodes]
    fees = [e['fees'] for e in episodes]

    profitable_episodes = sum(1 for p in pnls if p > 0)
    total_fees = float(np.sum(fees))
    total_gross = float(np.sum(gross_pnls))

    # Totaal winst en totaal verlies gesplitst
    total_profit = float(np.sum([p for p in pnls if p > 0]))
    total_loss = float(np.sum([abs(p) for p in pnls if p < 0]))

    return {
        'n_episodes': n,
        'initial_balance': args.initial_balance,
        # Portfolio
        'avg_portfolio_value': round(float(np.mean(pvs)), 2),
        'median_portfolio_value': round(float(np.median(pvs)), 2),
        'min_portfolio_value': round(float(np.min(pvs)), 2),
        'max_portfolio_value': round(float(np.max(pvs)), 2),
        'std_portfolio_value': round(float(np.std(pvs)), 2),
        # PnL (gross = before fees, net = after fees)
        'avg_gross_pnl': round(float(np.mean(gross_pnls)), 2),
        'total_gross_pnl': round(total_gross, 2),
        'avg_pnl': round(float(np.mean(pnls)), 2),
        'median_pnl': round(float(np.median(pnls)), 2),
        'total_pnl': round(float(np.sum(pnls)), 2),
        'min_pnl': round(float(np.min(pnls)), 2),
        'max_pnl': round(float(np.max(pnls)), 2),
        'fee_impact_pct': round(total_fees / total_gross * 100, 2) if total_gross > 0 else 0.0,
        'total_profit': round(total_profit, 2),
        'total_loss': round(total_loss, 2),
        # Returns
        'avg_return_pct': round(float(np.mean(returns)), 4),
        'median_return_pct': round(float(np.median(returns)), 4),
        'min_return_pct': round(float(np.min(returns)), 4),
        'max_return_pct': round(float(np.max(returns)), 4),
        'std_return_pct': round(float(np.std(returns)), 4),
        # Win/Loss
        'profitable_episodes': profitable_episodes,
        'losing_episodes': n - profitable_episodes,
        'episode_win_rate': round(profitable_episodes / n, 4),
        # Rewards
        'avg_reward': round(float(np.mean(rewards)), 4),
        'std_reward': round(float(np.std(rewards)), 4),
        # Trading
        'avg_trades': round(float(np.mean(trades)), 1),
        'avg_win_rate': round(float(np.mean(win_rates)), 4),
        'total_fees': round(total_fees, 4),
        'avg_fees_per_episode': round(total_fees / n, 4) if n > 0 else 0.0,
        # Risk
        'avg_max_drawdown': round(float(np.mean(drawdowns)), 4),
        'worst_drawdown': round(float(np.max(drawdowns)), 4),
        'avg_sharpe': round(float(np.mean(sharpes)), 4),
        'avg_composite_score': round(float(np.mean(composites)), 6),
    }


def save_results(output_dir, episodes, summary, trade_logger, args, algo, elapsed, progress_log=None, bah=None):
    """Save all evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Summary JSON
    full_summary = {
        'evaluation_info': {
            'timestamp': datetime.datetime.now().isoformat(),
            'model_path': args.model_path,
            'algorithm': algo,
            'data_split': args.split,
            'data_dir': args.data_dir,
            'n_episodes': args.n_episodes,
            'max_steps': args.max_steps,
            'initial_balance': args.initial_balance,
            'transaction_fee': args.transaction_fee,
            'elapsed_seconds': round(elapsed, 1),
            'device': args.device,
        },
        'results': summary,
        'buy_and_hold': bah,
        'per_episode': episodes,
    }
    summary_path = os.path.join(output_dir, 'eval_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(full_summary, f, indent=2, default=str)

    # 2. Episode CSV
    csv_path = os.path.join(output_dir, 'eval_episodes.csv')
    cols = list(episodes[0].keys()) if episodes else []
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for ep in episodes:
            vals = [str(ep[c]) for c in cols]
            f.write(','.join(vals) + '\n')

    # 3. Trade CSV
    trades = trade_logger.get_trades()
    if trades:
        trade_csv_path = os.path.join(output_dir, 'eval_trades.csv')
        trade_cols = list(trades[0].keys()) if trades else []
        with open(trade_csv_path, 'w') as f:
            f.write(','.join(trade_cols) + '\n')
            for t in trades:
                vals = [str(t.get(c, '')) for c in trade_cols]
                f.write(','.join(vals) + '\n')

    # 3b. Progress CSV (step-level PnL log)
    if progress_log:
        progress_csv_path = os.path.join(output_dir, 'eval_progress.csv')
        pcols = list(progress_log[0].keys())
        with open(progress_csv_path, 'w') as f:
            f.write(','.join(pcols) + '\n')
            for row in progress_log:
                f.write(','.join(str(row[c]) for c in pcols) + '\n')

    # 4. Plots
    if not args.no_plots:
        try:
            from src.utils.plotting import generate_rl_plots, generate_trade_plots
            plot_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)

            ep_rewards = [e['reward'] for e in episodes]
            portfolio_values = [e['portfolio_value'] for e in episodes]
            win_rates_list = [e['win_rate'] for e in episodes]
            ep_returns = [e['return_pct'] / 100 for e in episodes]
            fees_list = [e['fees'] for e in episodes]
            drawdowns_list = [e['max_drawdown'] for e in episodes]
            trade_counts_list = [{'buy': e['buys'], 'sell': e['sells'], 'hold': e['holds']} for e in episodes]

            generate_rl_plots(
                ep_rewards, portfolio_values, {},
                plot_dir, initial_balance=args.initial_balance,
                win_rates=win_rates_list, trade_counts=trade_counts_list,
                episode_returns=ep_returns, total_fees_list=fees_list,
                max_drawdowns=drawdowns_list
            )
            generate_trade_plots(trade_logger, plot_dir)
            print(f"  Plots saved to: {plot_dir}")
        except Exception as e:
            print(f"  [WARN] Could not generate plots: {e}")

    # 5. Human-readable report
    txt_path = os.path.join(output_dir, 'eval_report.txt')
    with open(txt_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"EVALUATION REPORT\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model:           {args.model_path}\n")
        f.write(f"Algorithm:       {algo}\n")
        f.write(f"Data split:      {args.split}\n")
        f.write(f"Episodes:        {args.n_episodes}\n")
        f.write(f"Max steps/ep:    {args.max_steps if args.max_steps > 0 else 'unlimited'}\n")
        f.write(f"Initial balance: ${args.initial_balance:,.2f}\n")
        f.write(f"Transaction fee: {args.transaction_fee*100:.2f}%\n")
        f.write(f"Date:            {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration:        {elapsed:.1f}s\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"PERFORMANCE SUMMARY\n")
        f.write(f"{'='*60}\n\n")
        s = summary
        f.write(f"--- Portfolio ---\n")
        f.write(f"Avg Portfolio Value:  ${s['avg_portfolio_value']:,.2f}\n")
        f.write(f"Median PV:            ${s['median_portfolio_value']:,.2f}\n")
        f.write(f"Best PV:              ${s['max_portfolio_value']:,.2f}\n")
        f.write(f"Worst PV:             ${s['min_portfolio_value']:,.2f}\n")
        f.write(f"Std PV:               ${s['std_portfolio_value']:,.2f}\n")
        f.write(f"\n--- Profit & Loss ---\n")
        f.write(f"Avg Gross PnL (no fee): ${s['avg_gross_pnl']:+,.2f}\n")
        f.write(f"Total Gross PnL:        ${s['total_gross_pnl']:+,.2f}\n")
        f.write(f"Avg Net PnL (w/ fee):   ${s['avg_pnl']:+,.2f}\n")
        f.write(f"Median Net PnL:         ${s['median_pnl']:+,.2f}\n")
        f.write(f"Best Net PnL:           ${s['max_pnl']:+,.2f}\n")
        f.write(f"Worst Net PnL:          ${s['min_pnl']:+,.2f}\n")
        f.write(f"Total Net PnL:          ${s['total_pnl']:+,.2f}\n")
        f.write(f"Fee Impact:             {s['fee_impact_pct']:.2f}% of gross\n")
        f.write(f"\n--- Returns ---\n")
        f.write(f"Avg Return:           {s['avg_return_pct']:+.2f}%\n")
        f.write(f"Median Return:        {s['median_return_pct']:+.2f}%\n")
        f.write(f"Best Return:          {s['max_return_pct']:+.2f}%\n")
        f.write(f"Worst Return:         {s['min_return_pct']:+.2f}%\n")
        f.write(f"Std Return:           {s['std_return_pct']:.2f}%\n")
        f.write(f"\n--- Win/Loss ---\n")
        f.write(f"Profitable Episodes:  {s['profitable_episodes']}/{s['n_episodes']} ({s['episode_win_rate']*100:.0f}%)\n")
        f.write(f"Avg Win Rate (trades):{s['avg_win_rate']*100:.1f}%\n")
        f.write(f"\n--- Trading ---\n")
        f.write(f"Avg Trades/Episode:   {s['avg_trades']:.1f}\n")
        f.write(f"Total Fees:           ${s['total_fees']:,.2f}\n")
        f.write(f"Avg Fees/Episode:     ${s['avg_fees_per_episode']:,.2f}\n")
        f.write(f"\n--- Risk ---\n")
        f.write(f"Avg Max Drawdown:     {s['avg_max_drawdown']*100:.2f}%\n")
        f.write(f"Worst Drawdown:       {s['worst_drawdown']*100:.2f}%\n")
        f.write(f"Avg Sharpe Ratio:     {s['avg_sharpe']:.3f}\n")
        f.write(f"Avg Composite Score:  {s['avg_composite_score']:.4f}\n")
        f.write(f"\n--- Totaal Winst / Verlies ---\n")
        f.write(f"Totaal Winst:         ${s['total_profit']:+,.2f}\n")
        f.write(f"Totaal Verlies:       ${s['total_loss']:,.2f}\n")
        f.write(f"Netto PnL:            ${s['total_pnl']:+,.2f}\n")
        if bah:
            f.write(f"\n--- Buy-and-Hold Benchmark ---\n")
            f.write(f"BTC prijs begin:      ${bah['start_price']:,.2f}\n")
            f.write(f"BTC prijs eind:       ${bah['end_price']:,.2f}\n")
            f.write(f"BTC return:           {bah['return_pct']:+.2f}%\n")
            f.write(f"BTC PnL:              ${bah['pnl']:+,.2f}\n")
            f.write(f"Agent vs B&H:         {s['avg_return_pct'] - bah['return_pct']:+.2f}% "
                    f"({'beter' if s['avg_return_pct'] > bah['return_pct'] else 'slechter'})\n")
        f.write(f"\n--- Rewards ---\n")
        f.write(f"Avg Episode Reward:   {s['avg_reward']:.2f}\n")
        f.write(f"Std Episode Reward:   {s['std_reward']:.2f}\n")

        # Trade summary from trade_logger
        ts = trade_logger.get_summary()
        f.write(f"\n{'='*60}\n")
        f.write(f"TRADE SUMMARY\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total Trades:         {ts['total_trades']:,}\n")
        f.write(f"Buys:                 {ts['total_buys']:,}\n")
        f.write(f"Sells:                {ts['total_sells']:,}\n")
        f.write(f"Winning Sells:        {ts['winning_sells']:,} ({ts['win_rate']*100:.1f}%)\n")
        f.write(f"Losing Sells:         {ts['losing_sells']:,}\n")
        f.write(f"Total Profit:         ${ts['total_profit']:,.2f}\n")
        f.write(f"Total Loss:           ${ts['total_loss']:,.2f}\n")
        f.write(f"Net PnL:              ${ts['net_pnl']:+,.2f}\n")
        f.write(f"Total Fees:           ${ts['total_fees']:,.2f}\n")
        f.write(f"Avg Buy Size:         ${ts['avg_buy_size_usd']:,.2f}\n")
        f.write(f"Avg Sell Size:        ${ts['avg_sell_size_usd']:,.2f}\n")
        f.write(f"Avg Profit/Win:       ${ts['avg_profit_per_win']:,.2f}\n")
        f.write(f"Avg Loss/Loss:        ${ts['avg_loss_per_loss']:,.2f}\n")

    return summary_path, csv_path, txt_path


def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    algo = args.algo
    uses_deeplob = algo in ('ppo_deeplob', 'sac_deeplob')

    print(f"\n{'='*60}")
    print(f"EVALUATION - {algo.upper()}")
    print(f"{'='*60}")
    print(f"Model:    {args.model_path}")
    print(f"Split:    {args.split}")
    print(f"Device:   {device}")
    print(f"Episodes: {args.n_episodes}, Max steps: {args.max_steps if args.max_steps > 0 else 'unlimited'}")
    print(f"{'='*60}\n")

    # ==========================================
    # DETERMINE SEQUENCE LENGTH
    # ==========================================
    if uses_deeplob:
        deeplob_ckpt = torch.load(args.deeplob_model, map_location='cpu', weights_only=False)
        deeplob_config = deeplob_ckpt['config']
        sequence_length = deeplob_config['sequence_length']
        del deeplob_ckpt
    else:
        sequence_length = 100

    # ==========================================
    # LOAD DATA (alleen de benodigde split)
    # ==========================================
    print("Loading data...")
    import pyarrow.parquet as pq
    import json
    from train.common.setup import STATIONARY_FEATURES

    feature_cols = STATIONARY_FEATURES
    price_col = 'close'
    parquet_path = os.path.join(args.data_dir, f'{args.split}.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data niet gevonden: {parquet_path}")

    # Normalization stats voor price denormalization
    norm_stats_path = os.path.join(args.data_dir, 'normalization_stats.json')
    price_mean, price_std = None, None
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        if 'stats' in norm_stats and price_col in norm_stats['stats']:
            price_mean = norm_stats['stats'][price_col]['mean']
            price_std = norm_stats['stats'][price_col]['std']
            print(f"  Price denormalization: {price_col} mean={price_mean:.2f}, std={price_std:.2f}")

    # Bepaal hoeveel rows we nodig hebben
    load_rows = args.max_rows
    if args.max_steps > 0:
        load_rows = min(load_rows, args.max_steps + sequence_length + 1000)

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    load_cols = list(dict.fromkeys(feature_cols + [price_col]))

    if total_rows <= load_rows:
        df = pf.read(columns=load_cols).to_pandas()
    else:
        # Laad laatste N rows (meest recente data)
        import pyarrow as pa
        num_rg = pf.metadata.num_row_groups
        rg_rows = [pf.metadata.row_group(i).num_rows for i in range(num_rg)]
        cumsum = 0
        start_rg = num_rg
        for i in range(num_rg - 1, -1, -1):
            cumsum += rg_rows[i]
            start_rg = i
            if cumsum >= load_rows:
                break
        tables = [pf.read_row_group(i, columns=load_cols) for i in range(start_rg, num_rg)]
        df = pa.concat_tables(tables).to_pandas()
        del tables
        df = df.tail(load_rows).reset_index(drop=True)

    print(f"  Loaded {len(df):,} / {total_rows:,} rows from {args.split}.parquet")

    eval_features = df[feature_cols].values.astype(np.float32)
    eval_prices = df[price_col].values.astype(np.float64)
    del df

    if price_mean is not None and price_std is not None:
        eval_prices = eval_prices * price_std + price_mean

    print(f"Eval data ({args.split}): {len(eval_features):,} rows, "
          f"{len(eval_features) - sequence_length:,} valid steps")
    print(f"  RAM: {eval_features.nbytes / 1e9:.2f} GB\n")

    # ==========================================
    # LOAD AGENT (eval mode, no gradients)
    # ==========================================
    agent = load_agent(algo, args.model_path, args.deeplob_model, device)

    # Set all networks to eval mode
    for attr_name in ['deeplob', 'portfolio_encoder', 'network', 'actor', 'critic',
                      'policy', 'value_net', 'critic_target']:
        net = getattr(agent, attr_name, None)
        if net is not None and hasattr(net, 'eval'):
            net.eval()

    # ==========================================
    # CREATE ENVIRONMENT
    # ==========================================
    env = create_eval_env(algo, eval_features, eval_prices, sequence_length, args)

    # ==========================================
    # OUTPUT DIRECTORY
    # ==========================================
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_dir = os.path.dirname(args.model_path)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(model_dir, f'eval_{args.split}_{ts}')

    trade_logger = TradeLogger(output_dir)

    # ==========================================
    # RUN EVALUATION
    # ==========================================
    print(f"Running {args.n_episodes} evaluation episodes (deterministic)...\n")
    start_time = time.time()

    with torch.no_grad():
        episodes, progress_log = run_evaluation(agent, algo, env, args.n_episodes, trade_logger, args)

    trade_logger.flush_to_csv()
    elapsed = time.time() - start_time

    # ==========================================
    # COMPUTE & SAVE RESULTS
    # ==========================================
    summary = compute_summary(episodes, args)
    bah = compute_buy_and_hold(eval_prices, sequence_length, args.initial_balance, args.max_steps)

    print(f"\n{'='*60}")
    print(f"RESULTS ({args.n_episodes} episodes on {args.split} data)")
    print(f"{'='*60}")
    print(f"  Avg Portfolio Value:  ${summary['avg_portfolio_value']:,.2f}")
    print(f"  Avg Gross PnL (no fee): ${summary['avg_gross_pnl']:+,.2f}")
    print(f"  Avg Net PnL (w/ fee):   ${summary['avg_pnl']:+,.2f}")
    print(f"  Totaal Winst:         ${summary['total_profit']:+,.2f}")
    print(f"  Totaal Verlies:       ${summary['total_loss']:,.2f}")
    print(f"  Total Fees:          ${summary['total_fees']:,.2f} ({summary['fee_impact_pct']:.1f}% of gross)")
    print(f"  Avg Return:          {summary['avg_return_pct']:+.2f}%")
    print(f"  Episode Win Rate:    {summary['episode_win_rate']*100:.0f}% "
          f"({summary['profitable_episodes']}/{summary['n_episodes']})")
    print(f"  Avg Trade Win Rate:  {summary['avg_win_rate']*100:.1f}%")
    print(f"  Avg Max Drawdown:    {summary['avg_max_drawdown']*100:.2f}%")
    print(f"  Avg Sharpe:          {summary['avg_sharpe']:.3f}")
    print(f"  Avg Composite Score: {summary['avg_composite_score']:.4f}")
    print(f"  Avg Trades/Episode:  {summary['avg_trades']:.1f}")
    print(f"  Duration:            {elapsed:.1f}s")
    print(f"\n--- Buy-and-Hold Benchmark ---")
    print(f"  BTC begin: ${bah['start_price']:,.2f}  →  eind: ${bah['end_price']:,.2f}")
    print(f"  B&H Return:  {bah['return_pct']:+.2f}%  |  B&H PnL: ${bah['pnl']:+,.2f}")
    diff = summary['avg_return_pct'] - bah['return_pct']
    print(f"  Agent vs B&H: {diff:+.2f}% ({'beter' if diff > 0 else 'slechter'})")

    summary_path, csv_path, txt_path = save_results(
        output_dir, episodes, summary, trade_logger, args, algo, elapsed,
        progress_log=progress_log, bah=bah
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"  Summary:  {os.path.basename(summary_path)}")
    print(f"  Report:   {os.path.basename(txt_path)}")
    print(f"  Episodes: {os.path.basename(csv_path)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
