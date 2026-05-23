"""
Plotting utilities voor training visualisatie.

Genereert automatisch plots na training:
- Loss curves (train vs val)
- Accuracy curves
- Per-class accuracy
- Confusion matrix
- Weight importance (voor DeepLOB)
- RL-specifieke plots (reward curves, portfolio value)
"""

import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (geen GUI nodig)
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def check_matplotlib():
    """Check of matplotlib beschikbaar is."""
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib niet gevonden - plots worden overgeslagen.")
        print("       Installeer met: pip install matplotlib")
        return False
    return True


# =============================================
# SUPERVISED LEARNING PLOTS (DeepLOB pretrain)
# =============================================

def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot train vs validation loss per epoch."""
    if not check_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_accuracy_curves(train_accs, val_accs, save_path):
    """Plot train vs validation accuracy per epoch."""
    if not check_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_accs) + 1)
    ax.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax.axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Random (33.3%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training vs Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_per_class_accuracy(class_accs_history, class_names, save_path):
    """
    Plot per-class accuracy over epochs.
    
    Args:
        class_accs_history: list of dicts, e.g. [{'Down': 50.0, 'Neutral': 30.0, 'Up': 45.0}, ...]
        class_names: list of class names
        save_path: pad om plot op te slaan
    """
    if not check_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(class_accs_history) + 1)
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    for i, name in enumerate(class_names):
        values = [h.get(name, 0) for h in class_accs_history]
        ax.plot(epochs, values, '-', label=name, linewidth=2,
                color=colors[i] if i < len(colors) else None)
    ax.axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix als heatmap."""
    if not check_matplotlib():
        return
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # Normalize per rij (true class)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    fig.colorbar(im, label='%')

    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm_norm[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm[i,j]:,}\n({cm_norm[i,j]:.1f}%)',
                    ha='center', va='center', color=color, fontsize=10)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Test Set)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_weight_importance(model, feature_names, save_path):
    """
    Visualiseer welke input features het zwaarst wegen.
    
    Gebruikt de L1-norm van de eerste conv layer weights als proxy
    voor feature importance.
    """
    if not check_matplotlib():
        return
    import torch

    # Zoek de eerste conv layer in het model
    first_conv = None
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            first_conv = module
            break

    if first_conv is None:
        print("  [WARN] Geen conv/linear layer gevonden voor weight importance")
        return

    with torch.no_grad():
        weights = first_conv.weight.detach().cpu()
        # L1-norm per input feature (sum over output channels en kernel)
        if weights.dim() == 3:  # Conv1d: (out_ch, in_ch, kernel)
            importance = weights.abs().sum(dim=(0, 2)).numpy()
        elif weights.dim() == 2:  # Linear: (out, in)
            importance = weights.abs().sum(dim=0).numpy()
        else:
            print("  [WARN] Onverwachte weight dimensie")
            return

    # Match feature names
    n_features = min(len(importance), len(feature_names))
    importance = importance[:n_features]
    names = feature_names[:n_features]

    # Sorteer op importance
    sorted_idx = np.argsort(importance)
    importance = importance[sorted_idx]
    names = [names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_features))
    ax.barh(range(n_features), importance, color=colors)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(names)
    ax.set_xlabel('Weight Importance (L1 norm)')
    ax.set_title('Feature Importance (First Layer Weights)')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


# =============================================
# RL PLOTS (PPO/SAC)
# =============================================

def plot_reward_curve(episode_rewards, save_path, window=100):
    """Plot episode rewards met smoothed moving average."""
    if not check_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    episodes = range(1, len(episode_rewards) + 1)

    # Raw rewards (transparant)
    ax.plot(episodes, episode_rewards, alpha=0.2, color='blue', linewidth=0.5)

    # Smoothed (moving average)
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(episode_rewards) + 1), smoothed,
                color='blue', linewidth=2, label=f'MA({window})')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_portfolio_value(portfolio_values, save_path):
    """Plot portfolio waarde over episodes."""
    if not check_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    episodes = range(1, len(portfolio_values) + 1)
    ax.plot(episodes, portfolio_values, color='green', linewidth=1.5)
    ax.axhline(y=portfolio_values[0] if portfolio_values else 1.0,
               color='gray', linestyle='--', alpha=0.5, label='Start')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Portfolio Value')
    ax.set_title('Portfolio Value Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_rl_losses(losses_dict, save_path):
    """
    Plot RL training losses.
    
    Args:
        losses_dict: dict met loss namen als keys en lijsten als values
                     e.g. {'policy_loss': [...], 'value_loss': [...]}
        save_path: pad om plot op te slaan
    """
    if not check_matplotlib():
        return
    n_plots = len(losses_dict)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    for ax, (name, values) in zip(axes, losses_dict.items()):
        steps = range(1, len(values) + 1)
        ax.plot(steps, values, linewidth=1, alpha=0.7)
        ax.set_xlabel('Update')
        ax.set_ylabel(name)
        ax.set_title(name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_learning_rate(lrs, save_path):
    """Plot learning rate schedule over epochs."""
    if not check_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(lrs) + 1), lrs, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def generate_supervised_plots(history, model, feature_names, test_preds, test_labels, plot_dir):
    """
    Genereer alle plots voor supervised training (DeepLOB pretrain).
    
    Args:
        history: dict met keys 'train_loss', 'val_loss', 'train_acc', 'val_acc', 
                 'class_accs' (list of dicts), 'lrs'
        model: het getrainde model
        feature_names: lijst van feature namen
        test_preds: numpy array van predicted labels op test set
        test_labels: numpy array van true labels op test set
        plot_dir: directory om plots op te slaan
    """
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots in {plot_dir}...")

    if 'train_loss' in history and 'val_loss' in history:
        plot_loss_curves(history['train_loss'], history['val_loss'],
                         os.path.join(plot_dir, 'loss_curves.png'))

    if 'train_acc' in history and 'val_acc' in history:
        plot_accuracy_curves(history['train_acc'], history['val_acc'],
                             os.path.join(plot_dir, 'accuracy_curves.png'))

    if 'class_accs' in history:
        plot_per_class_accuracy(history['class_accs'],
                                ['Down', 'Neutral', 'Up'],
                                os.path.join(plot_dir, 'per_class_accuracy.png'))

    if 'lrs' in history:
        plot_learning_rate(history['lrs'],
                           os.path.join(plot_dir, 'learning_rate.png'))

    if test_preds is not None and test_labels is not None:
        plot_confusion_matrix(test_labels, test_preds,
                              ['Down', 'Neutral', 'Up'],
                              os.path.join(plot_dir, 'confusion_matrix.png'))

    if model is not None and feature_names is not None:
        plot_weight_importance(model, feature_names,
                               os.path.join(plot_dir, 'weight_importance.png'))


def plot_pnl_curve(portfolio_values, initial_balance, save_path, window=50):
    """Plot profit/loss in dollar bedragen over episodes."""
    if not check_matplotlib():
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    pnl = [v - initial_balance for v in portfolio_values]
    episodes = range(1, len(pnl) + 1)

    # --- Top: PnL per episode ---
    ax = axes[0]
    colors = ['green' if p >= 0 else 'red' for p in pnl]
    ax.bar(episodes, pnl, color=colors, alpha=0.6, width=1.0)
    if len(pnl) >= window:
        smoothed = np.convolve(pnl, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(pnl) + 1), smoothed,
                color='blue', linewidth=2, label=f'MA({window})')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Profit / Loss ($)')
    ax.set_title(f'PnL per Episode (Initial Balance: ${initial_balance:,.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom: Cumulative PnL ---
    ax = axes[1]
    cum_pnl = np.cumsum(pnl)
    ax.fill_between(episodes, cum_pnl, where=[p >= 0 for p in cum_pnl], color='green', alpha=0.3)
    ax.fill_between(episodes, cum_pnl, where=[p < 0 for p in cum_pnl], color='red', alpha=0.3)
    ax.plot(episodes, cum_pnl, color='blue', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative PnL ($)')
    ax.set_title('Cumulative Profit / Loss')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_drawdown(portfolio_values, save_path):
    """Plot drawdown percentage over episodes."""
    if not check_matplotlib():
        return
    values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(values)
    drawdown = (running_max - values) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    episodes = range(1, len(drawdown) + 1)
    ax.fill_between(episodes, drawdown, color='red', alpha=0.3)
    ax.plot(episodes, drawdown, color='red', linewidth=1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(f'Drawdown from Peak (Max: {drawdown.max():.1f}%)')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_win_rate(win_rates, save_path, window=50):
    """Plot win rate over episodes."""
    if not check_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    episodes = range(1, len(win_rates) + 1)
    ax.plot(episodes, [w * 100 for w in win_rates], alpha=0.3, color='blue', linewidth=0.5)
    if len(win_rates) >= window:
        smoothed = np.convolve([w * 100 for w in win_rates], np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(win_rates) + 1), smoothed,
                color='blue', linewidth=2, label=f'MA({window})')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Trade Win Rate Over Training')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_trade_stats(trade_counts, save_path, window=50):
    """Plot trade action distribution over episodes (buy/sell/hold counts)."""
    if not check_matplotlib():
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    episodes = range(1, len(trade_counts) + 1)
    buys = [t.get('buy', 0) for t in trade_counts]
    sells = [t.get('sell', 0) for t in trade_counts]
    holds = [t.get('hold', 0) for t in trade_counts]

    # Stacked area
    ax = axes[0]
    ax.stackplot(episodes, buys, sells, holds,
                 labels=['Buy', 'Sell', 'Hold'],
                 colors=['#2ecc71', '#e74c3c', '#95a5a6'], alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Action Count')
    ax.set_title('Action Distribution per Episode')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Buy/Sell ratio
    ax = axes[1]
    total_actions = [b + s + h for b, s, h in zip(buys, sells, holds)]
    buy_pct = [b / max(t, 1) * 100 for b, t in zip(buys, total_actions)]
    sell_pct = [s / max(t, 1) * 100 for s, t in zip(sells, total_actions)]
    ax.plot(episodes, buy_pct, color='#2ecc71', alpha=0.4, linewidth=0.5)
    ax.plot(episodes, sell_pct, color='#e74c3c', alpha=0.4, linewidth=0.5)
    if len(buy_pct) >= window:
        smooth_buy = np.convolve(buy_pct, np.ones(window)/window, mode='valid')
        smooth_sell = np.convolve(sell_pct, np.ones(window)/window, mode='valid')
        x_range = range(window, len(buy_pct) + 1)
        ax.plot(x_range, smooth_buy, color='#2ecc71', linewidth=2, label=f'Buy % MA({window})')
        ax.plot(x_range, smooth_sell, color='#e74c3c', linewidth=2, label=f'Sell % MA({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Action %')
    ax.set_title('Buy / Sell Percentage Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_financial_summary(portfolio_values, initial_balance, episode_returns,
                           total_fees_list, save_path):
    """Plot financial summary dashboard."""
    if not check_matplotlib():
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    episodes = range(1, len(portfolio_values) + 1)

    # Portfolio value
    ax = axes[0, 0]
    ax.plot(episodes, portfolio_values, color='blue', linewidth=1.5)
    ax.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.5, label=f'Start ${initial_balance:,.0f}')
    final_val = portfolio_values[-1] if portfolio_values else initial_balance
    ax.set_title(f'Portfolio Value (Final: ${final_val:,.0f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Returns distribution
    ax = axes[0, 1]
    if episode_returns:
        returns_pct = [r * 100 for r in episode_returns]
        ax.hist(returns_pct, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=1)
        mean_r = np.mean(returns_pct)
        ax.axvline(x=mean_r, color='red', linestyle='--', label=f'Mean: {mean_r:.2f}%')
        ax.set_title('Return Distribution per Episode')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative fees
    ax = axes[1, 0]
    if total_fees_list:
        cum_fees = np.cumsum(total_fees_list)
        ax.plot(range(1, len(cum_fees) + 1), cum_fees, color='orange', linewidth=1.5)
        ax.set_title(f'Cumulative Fees Paid (Total: ${cum_fees[-1]:,.2f})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Fees ($)')
    ax.grid(True, alpha=0.3)

    # Return vs Episode (scatter)
    ax = axes[1, 1]
    if episode_returns:
        colors = ['green' if r >= 0 else 'red' for r in episode_returns]
        ax.scatter(range(1, len(episode_returns) + 1),
                   [r * 100 for r in episode_returns],
                   c=colors, alpha=0.3, s=5)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_title('Per-Episode Returns')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Financial Summary Dashboard', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def generate_rl_plots(episode_rewards, portfolio_values, losses_dict, plot_dir,
                      initial_balance=10000.0, win_rates=None, trade_counts=None,
                      episode_returns=None, total_fees_list=None,
                      max_drawdowns=None, sharpe_ratios=None):
    """
    Genereer alle plots voor RL training (PPO/SAC).
    
    Args:
        episode_rewards: lijst van episode rewards
        portfolio_values: lijst van portfolio values per episode
        losses_dict: dict met loss namen en waarden
        plot_dir: directory om plots op te slaan
        initial_balance: startkapitaal voor PnL berekening
        win_rates: lijst van win rates per episode
        trade_counts: lijst van dicts met buy/sell/hold counts per episode
        episode_returns: lijst van return percentages per episode
        total_fees_list: lijst van totale fees per episode
    """
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots in {plot_dir}...")

    if episode_rewards:
        plot_reward_curve(episode_rewards,
                          os.path.join(plot_dir, 'reward_curve.png'))

    if portfolio_values:
        plot_portfolio_value(portfolio_values,
                             os.path.join(plot_dir, 'portfolio_value.png'))

    if losses_dict:
        plot_rl_losses(losses_dict,
                       os.path.join(plot_dir, 'training_losses.png'))

    if portfolio_values and initial_balance:
        plot_pnl_curve(portfolio_values, initial_balance,
                       os.path.join(plot_dir, 'pnl_curve.png'))

    if portfolio_values:
        plot_drawdown(portfolio_values,
                      os.path.join(plot_dir, 'drawdown.png'))

    if win_rates:
        plot_win_rate(win_rates,
                      os.path.join(plot_dir, 'win_rate.png'))

    if trade_counts:
        plot_trade_stats(trade_counts,
                         os.path.join(plot_dir, 'trade_stats.png'))

    if portfolio_values and episode_returns:
        plot_financial_summary(portfolio_values, initial_balance,
                               episode_returns, total_fees_list or [],
                               os.path.join(plot_dir, 'financial_summary.png'))

    if episode_returns:
        plot_episode_returns(episode_returns,
                             os.path.join(plot_dir, 'episode_returns.png'))

    if episode_returns and len(episode_returns) >= 2:
        plot_sharpe_ratio(episode_returns,
                          os.path.join(plot_dir, 'sharpe_ratio.png'))

    if max_drawdowns:
        plot_max_drawdown_per_episode(max_drawdowns,
                                      os.path.join(plot_dir, 'max_drawdown_per_episode.png'))

    if trade_counts:
        plot_num_trades(trade_counts,
                        os.path.join(plot_dir, 'num_trades.png'))


# =============================================
# EPISODE-LEVEL FINANCIAL PLOTS
# =============================================


def plot_episode_returns(episode_returns, save_path, window=50):
    """Plot total return (%) per episode over training."""
    if not check_matplotlib():
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    episodes = range(1, len(episode_returns) + 1)
    returns_pct = [r * 100 for r in episode_returns]

    # Top: Return per episode
    ax = axes[0]
    colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in returns_pct]
    ax.bar(episodes, returns_pct, color=colors, alpha=0.6, width=1.0)
    if len(returns_pct) >= window:
        smoothed = np.convolve(returns_pct, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(returns_pct) + 1), smoothed,
                color='blue', linewidth=2, label=f'MA({window})')
        ax.legend()
    ax.axhline(y=0, color='black', linewidth=1)
    mean_ret = np.mean(returns_pct)
    ax.set_title(f'Total Return per Episode (Mean: {mean_ret:+.2f}%)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)

    # Bottom: Cumulative return
    ax = axes[1]
    cum_ret = np.cumsum(returns_pct)
    ax.fill_between(episodes, cum_ret, where=[c >= 0 for c in cum_ret],
                    color='green', alpha=0.3)
    ax.fill_between(episodes, cum_ret, where=[c < 0 for c in cum_ret],
                    color='red', alpha=0.3)
    ax.plot(episodes, cum_ret, color='blue', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_title(f'Cumulative Return (Total: {cum_ret[-1]:+.2f}%)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Return (%)')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_sharpe_ratio(episode_returns, save_path, window=50):
    """Plot rolling Sharpe ratio over training episodes."""
    if not check_matplotlib():
        return
    returns = np.array(episode_returns)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: Rolling Sharpe
    ax = axes[0]
    if len(returns) >= window:
        rolling_sharpe = []
        for i in range(window, len(returns) + 1):
            chunk = returns[i-window:i]
            mean_r = np.mean(chunk)
            std_r = np.std(chunk)
            sharpe = mean_r / std_r if std_r > 1e-8 else 0.0
            rolling_sharpe.append(sharpe)
        x_range = range(window, len(returns) + 1)
        ax.plot(x_range, rolling_sharpe, color='purple', linewidth=1.5)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Good (1.0)')
        ax.axhline(y=-1.0, color='red', linestyle='--', alpha=0.5, label='Bad (-1.0)')
        final_sharpe = rolling_sharpe[-1] if rolling_sharpe else 0
        ax.set_title(f'Rolling Sharpe Ratio (window={window}, Latest: {final_sharpe:.3f})')
    else:
        overall = np.mean(returns) / np.std(returns) if np.std(returns) > 1e-8 else 0
        ax.axhline(y=overall, color='purple', linewidth=2,
                    label=f'Overall Sharpe: {overall:.3f}')
        ax.set_title(f'Sharpe Ratio (Overall: {overall:.3f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: Return distribution with Sharpe annotation
    ax = axes[1]
    returns_pct = returns * 100
    ax.hist(returns_pct, bins=min(50, max(10, len(returns)//5)),
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=1)
    mean_r = np.mean(returns_pct)
    std_r = np.std(returns_pct)
    ax.axvline(x=mean_r, color='red', linestyle='--',
               label=f'Mean: {mean_r:.2f}%, Std: {std_r:.2f}%')
    overall_sharpe = mean_r / std_r if std_r > 1e-8 else 0
    ax.set_title(f'Return Distribution (Sharpe: {overall_sharpe:.3f})')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_max_drawdown_per_episode(max_drawdowns, save_path, window=50):
    """Plot max drawdown per episode over training."""
    if not check_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    episodes = range(1, len(max_drawdowns) + 1)
    dd_pct = [d * 100 for d in max_drawdowns]

    ax.bar(episodes, dd_pct, color='#e74c3c', alpha=0.5, width=1.0)
    if len(dd_pct) >= window:
        smoothed = np.convolve(dd_pct, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(dd_pct) + 1), smoothed,
                color='darkred', linewidth=2, label=f'MA({window})')
        ax.legend()
    mean_dd = np.mean(dd_pct)
    max_dd = max(dd_pct) if dd_pct else 0
    ax.set_title(f'Max Drawdown per Episode (Mean: {mean_dd:.1f}%, Worst: {max_dd:.1f}%)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Max Drawdown (%)')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_num_trades(trade_counts, save_path, window=50):
    """Plot total number of trades per episode over training."""
    if not check_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    episodes = range(1, len(trade_counts) + 1)
    totals = [t.get('buy', 0) + t.get('sell', 0) for t in trade_counts]

    ax.bar(episodes, totals, color='#3498db', alpha=0.6, width=1.0)
    if len(totals) >= window:
        smoothed = np.convolve(totals, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(totals) + 1), smoothed,
                color='darkblue', linewidth=2, label=f'MA({window})')
        ax.legend()
    mean_t = np.mean(totals)
    ax.set_title(f'Number of Trades per Episode (Mean: {mean_t:.1f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Trades')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


# =============================================
# PER-TRADE PLOTS
# =============================================

def plot_trade_pnl_scatter(trade_logger, save_path):
    """Scatter plot van winst/verlies per sell trade."""
    if not check_matplotlib():
        return
    trades = trade_logger.get_trades()
    sells = [t for t in trades if t['action'] == 'sell']
    if not sells:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: PnL per sell trade
    ax = axes[0]
    profits = [t['profit_loss'] for t in sells]
    indices = range(1, len(sells) + 1)
    colors = ['#2ecc71' if p >= 0 else '#e74c3c' for p in profits]
    ax.bar(indices, profits, color=colors, alpha=0.7, width=1.0)
    ax.axhline(y=0, color='black', linewidth=1)
    # Moving average
    if len(profits) >= 20:
        ma = np.convolve(profits, np.ones(20)/20, mode='valid')
        ax.plot(range(20, len(profits) + 1), ma, color='blue', linewidth=2,
                label='MA(20)')
        ax.legend()
    wins = sum(1 for p in profits if p > 0)
    ax.set_title(f'Profit/Loss per Sell Trade (Wins: {wins}/{len(sells)} = {wins/len(sells)*100:.1f}%)')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Profit/Loss ($)')
    ax.grid(True, alpha=0.3)

    # Bottom: Cumulative PnL
    ax = axes[1]
    cum_pnl = np.cumsum(profits)
    ax.fill_between(indices, cum_pnl, where=[p >= 0 for p in cum_pnl],
                    color='green', alpha=0.3)
    ax.fill_between(indices, cum_pnl, where=[p < 0 for p in cum_pnl],
                    color='red', alpha=0.3)
    ax.plot(indices, cum_pnl, color='blue', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=1)
    final_pnl = cum_pnl[-1] if len(cum_pnl) > 0 else 0
    ax.set_title(f'Cumulative Trade PnL (Final: ${final_pnl:+,.2f})')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative PnL ($)')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_trade_sizes(trade_logger, save_path):
    """Plot van ingezette bedragen per trade (buy en sell apart)."""
    if not check_matplotlib():
        return
    trades = trade_logger.get_trades()
    if not trades:
        return

    buys = [t for t in trades if t['action'] == 'buy']
    sells = [t for t in trades if t['action'] == 'sell']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Buy sizes over time
    ax = axes[0, 0]
    if buys:
        buy_sizes = [t['usd_value'] for t in buys]
        ax.plot(range(1, len(buy_sizes) + 1), buy_sizes, color='#2ecc71',
                alpha=0.5, linewidth=0.5)
        if len(buy_sizes) >= 20:
            ma = np.convolve(buy_sizes, np.ones(20)/20, mode='valid')
            ax.plot(range(20, len(buy_sizes) + 1), ma, color='#2ecc71',
                    linewidth=2, label='MA(20)')
            ax.legend()
        ax.set_title(f'Buy Size Over Time (Avg: ${np.mean(buy_sizes):,.2f})')
    ax.set_xlabel('Buy Trade #')
    ax.set_ylabel('USD Invested ($)')
    ax.grid(True, alpha=0.3)

    # Top right: Sell sizes over time
    ax = axes[0, 1]
    if sells:
        sell_sizes = [t['usd_value'] for t in sells]
        ax.plot(range(1, len(sell_sizes) + 1), sell_sizes, color='#e74c3c',
                alpha=0.5, linewidth=0.5)
        if len(sell_sizes) >= 20:
            ma = np.convolve(sell_sizes, np.ones(20)/20, mode='valid')
            ax.plot(range(20, len(sell_sizes) + 1), ma, color='#e74c3c',
                    linewidth=2, label='MA(20)')
            ax.legend()
        ax.set_title(f'Sell Size Over Time (Avg: ${np.mean(sell_sizes):,.2f})')
    ax.set_xlabel('Sell Trade #')
    ax.set_ylabel('USD Sold ($)')
    ax.grid(True, alpha=0.3)

    # Bottom left: Distribution histograms
    ax = axes[1, 0]
    if buys:
        ax.hist([t['usd_value'] for t in buys], bins=50, color='#2ecc71',
                alpha=0.6, label='Buy', edgecolor='black', linewidth=0.5)
    if sells:
        ax.hist([t['usd_value'] for t in sells], bins=50, color='#e74c3c',
                alpha=0.6, label='Sell', edgecolor='black', linewidth=0.5)
    ax.set_title('Trade Size Distribution')
    ax.set_xlabel('USD Value ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Fees over time
    ax = axes[1, 1]
    fees = [t['fee'] for t in trades]
    cum_fees = np.cumsum(fees)
    ax.plot(range(1, len(cum_fees) + 1), cum_fees, color='orange', linewidth=1.5)
    ax.set_title(f'Cumulative Fees (Total: ${cum_fees[-1]:,.2f})')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Fees ($)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Trade Size Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_profit_per_episode(trade_logger, save_path, window=20):
    """Plot winst/verlies geaggregeerd per episode."""
    if not check_matplotlib():
        return
    trades = trade_logger.get_trades()
    if not trades:
        return

    # Aggregeer trades per episode
    episode_pnl = {}
    episode_trades_count = {}
    for t in trades:
        ep = t['episode']
        if ep not in episode_pnl:
            episode_pnl[ep] = 0.0
            episode_trades_count[ep] = 0
        if t['action'] == 'sell':
            episode_pnl[ep] += t['profit_loss']
        episode_trades_count[ep] += 1

    if not episode_pnl:
        return

    episodes = sorted(episode_pnl.keys())
    pnls = [episode_pnl[ep] for ep in episodes]
    counts = [episode_trades_count[ep] for ep in episodes]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: PnL per episode
    ax = axes[0]
    colors = ['#2ecc71' if p >= 0 else '#e74c3c' for p in pnls]
    ax.bar(episodes, pnls, color=colors, alpha=0.6, width=1.0)
    if len(pnls) >= window:
        ma = np.convolve(pnls, np.ones(window)/window, mode='valid')
        ax.plot(range(episodes[0] + window - 1, episodes[0] + window - 1 + len(ma)),
                ma, color='blue', linewidth=2, label=f'MA({window})')
        ax.legend()
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_title('Trade Profit/Loss per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('PnL ($)')
    ax.grid(True, alpha=0.3)

    # Bottom: Number of trades per episode
    ax = axes[1]
    ax.bar(episodes, counts, color='steelblue', alpha=0.7, width=1.0)
    if len(counts) >= window:
        ma_c = np.convolve(counts, np.ones(window)/window, mode='valid')
        ax.plot(range(episodes[0] + window - 1, episodes[0] + window - 1 + len(ma_c)),
                ma_c, color='red', linewidth=2, label=f'MA({window})')
        ax.legend()
    ax.set_title(f'Trades per Episode (Avg: {np.mean(counts):.1f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('# Trades')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_trade_dashboard(trade_logger, save_path):
    """Uitgebreid trade dashboard met alle key metrics."""
    if not check_matplotlib():
        return
    summary = trade_logger.get_summary()
    trades = trade_logger.get_trades()
    if not trades:
        return

    sells = [t for t in trades if t['action'] == 'sell']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Portfolio value at trade moments
    ax = axes[0, 0]
    pv_at_trades = trade_logger.get_portfolio_values_at_trades()
    ax.plot(range(1, len(pv_at_trades) + 1), pv_at_trades, color='blue',
            linewidth=1, alpha=0.7)
    ax.set_title('Portfolio Value at Trade Moments')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)

    # 2. Profit distribution (sells only)
    ax = axes[0, 1]
    if sells:
        profits = [t['profit_loss'] for t in sells]
        ax.hist(profits, bins=50, color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=1)
        mean_p = np.mean(profits)
        ax.axvline(x=mean_p, color='red', linestyle='--',
                    label=f'Mean: ${mean_p:,.2f}')
        ax.set_title('Sell Profit Distribution')
        ax.set_xlabel('Profit/Loss ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Win/Loss pie chart
    ax = axes[0, 2]
    if summary['total_sells'] > 0:
        sizes = [summary['winning_sells'], summary['losing_sells']]
        labels = [f"Wins ({summary['winning_sells']})",
                  f"Losses ({summary['losing_sells']})"]
        colors_pie = ['#2ecc71', '#e74c3c']
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               startangle=90)
        ax.set_title(f"Win Rate: {summary['win_rate']*100:.1f}%")
    else:
        ax.text(0.5, 0.5, 'No sells yet', ha='center', va='center',
                fontsize=14)
        ax.set_title('Win/Loss Ratio')

    # 4. BTC price at trades
    ax = axes[1, 0]
    buy_prices = [t['btc_price'] for t in trades if t['action'] == 'buy']
    sell_prices = [t['btc_price'] for t in trades if t['action'] == 'sell']
    if buy_prices:
        ax.scatter(range(len(buy_prices)), buy_prices, color='#2ecc71',
                   alpha=0.5, s=10, label=f'Buy ({len(buy_prices)})')
    if sell_prices:
        ax.scatter(range(len(buy_prices), len(buy_prices) + len(sell_prices)),
                   sell_prices, color='#e74c3c', alpha=0.5, s=10,
                   label=f'Sell ({len(sell_prices)})')
    ax.set_title('BTC Price at Trade Execution')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('BTC Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Balance trajectory
    ax = axes[1, 1]
    balances = [t['balance_after'] for t in trades]
    ax.plot(range(1, len(balances) + 1), balances, color='green',
            linewidth=1, alpha=0.7)
    ax.set_title('USD Balance After Each Trade')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Balance ($)')
    ax.grid(True, alpha=0.3)

    # 6. Summary text box
    ax = axes[1, 2]
    ax.axis('off')
    text_lines = [
        f"TRADE SUMMARY",
        f"{'─'*30}",
        f"Total Trades: {summary['total_trades']:,}",
        f"Buys: {summary['total_buys']:,}  |  Sells: {summary['total_sells']:,}",
        f"Win Rate: {summary['win_rate']*100:.1f}%",
        f"",
        f"Total Profit: ${summary['total_profit']:+,.2f}",
        f"Total Loss:   ${summary['total_loss']:,.2f}",
        f"Net PnL:      ${summary['net_pnl']:+,.2f}",
        f"Total Fees:   ${summary['total_fees']:,.2f}",
        f"",
        f"Avg Buy:  ${summary['avg_buy_size_usd']:,.2f}",
        f"Avg Sell: ${summary['avg_sell_size_usd']:,.2f}",
        f"Avg Win:  ${summary['avg_profit_per_win']:+,.2f}",
        f"Avg Loss: ${summary['avg_loss_per_loss']:,.2f}",
    ]
    ax.text(0.1, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Trade Analysis Dashboard', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def generate_trade_plots(trade_logger, plot_dir):
    """
    Genereer alle per-trade plots.
    
    Args:
        trade_logger: TradeLogger instantie met alle trades
        plot_dir: directory om plots op te slaan
    """
    if not trade_logger or not trade_logger.get_trades():
        return
    
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating trade plots in {plot_dir}...")
    
    plot_trade_pnl_scatter(trade_logger,
                           os.path.join(plot_dir, 'trade_pnl.png'))
    
    plot_trade_sizes(trade_logger,
                     os.path.join(plot_dir, 'trade_sizes.png'))
    
    plot_profit_per_episode(trade_logger,
                            os.path.join(plot_dir, 'profit_per_episode.png'))
    
    plot_trade_dashboard(trade_logger,
                         os.path.join(plot_dir, 'trade_dashboard.png'))
