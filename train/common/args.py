"""
Shared Argument Parsing
=======================

Gedeelde argument parsers voor alle training scripts.
Dit voorkomt code duplicatie en zorgt voor consistente CLI interfaces.

Gebruik:
    >>> parser = argparse.ArgumentParser()
    >>> add_data_args(parser)
    >>> add_training_args(parser)
    >>> add_env_args(parser)
    >>> add_logging_args(parser)
    >>> args = parser.parse_args()

Auteur: DataDeepRL Team
"""

import argparse
from typing import Optional


def add_data_args(parser: argparse.ArgumentParser, 
                  data_dir: str = '../btc_l2_data',
                  max_files: int = 100,
                  sequence_length: int = 100) -> None:
    """
    Voeg data-gerelateerde arguments toe.
    
    Args:
        parser: ArgumentParser om arguments aan toe te voegen
        data_dir: Default data directory
        max_files: Default max aantal bestanden
        sequence_length: Default sequence lengte
    """
    group = parser.add_argument_group('Data')
    
    group.add_argument('--data_dir', type=str, default=data_dir,
                       help='Directory met BTC L2 data')
    group.add_argument('--max_files', type=int, default=max_files,
                       help='Maximum aantal data bestanden om te laden')
    group.add_argument('--sequence_length', type=int, default=sequence_length,
                       help='Lengte van input sequences (lookback window)')
    group.add_argument('--train_ratio', type=float, default=0.7,
                       help='Fractie data voor training')
    group.add_argument('--val_ratio', type=float, default=0.15,
                       help='Fractie data voor validatie')


def add_training_args(parser: argparse.ArgumentParser,
                      total_steps: int = 1_000_000,
                      batch_size: int = 256,
                      learning_rate: float = 3e-4) -> None:
    """
    Voeg training-gerelateerde arguments toe.
    
    Args:
        parser: ArgumentParser
        total_steps: Default totaal aantal training steps
        batch_size: Default batch size
        learning_rate: Default learning rate
    """
    group = parser.add_argument_group('Training')
    
    group.add_argument('--total_steps', type=int, default=total_steps,
                       help='Totaal aantal training steps')
    group.add_argument('--batch_size', type=int, default=batch_size,
                       help='Batch size voor training')
    group.add_argument('--learning_rate', '--lr', type=float, default=learning_rate,
                       help='Learning rate')
    group.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor voor toekomstige rewards')
    group.add_argument('--seed', type=int, default=42,
                       help='Random seed voor reproducibility')
    group.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device voor training (auto/cuda/cpu)')
    group.add_argument('--mixed_precision', action='store_true',
                       help='Gebruik mixed precision training (FP16)')
    group.add_argument('--random_start', action='store_true',
                       help='Random startpositie in episodes voor betere generalisatie')


def add_ppo_args(parser: argparse.ArgumentParser) -> None:
    """Voeg PPO-specifieke arguments toe."""
    group = parser.add_argument_group('PPO')
    
    group.add_argument('--n_steps', type=int, default=2048,
                       help='Steps per rollout (PPO buffer size)')
    group.add_argument('--n_epochs', type=int, default=10,
                       help='Aantal epochs per PPO update')
    group.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda parameter')
    group.add_argument('--clip_epsilon', type=float, default=0.2,
                       help='PPO clip range')
    group.add_argument('--value_coef', type=float, default=0.5,
                       help='Value loss coefficient')
    group.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy bonus coefficient')
    group.add_argument('--max_grad_norm', type=float, default=0.5,
                       help='Maximum gradient norm voor clipping')


def add_sac_args(parser: argparse.ArgumentParser) -> None:
    """Voeg SAC-specifieke arguments toe."""
    group = parser.add_argument_group('SAC')
    
    group.add_argument('--tau', type=float, default=0.005,
                       help='Soft update coefficient voor target networks')
    group.add_argument('--alpha', type=float, default=0.2,
                       help='Entropy coefficient (temperature)')
    group.add_argument('--auto_alpha', action='store_true', default=True,
                       help='Automatisch temperature tuning')
    group.add_argument('--buffer_size', type=int, default=1_000_000,
                       help='Replay buffer grootte')
    group.add_argument('--start_steps', type=int, default=10000,
                       help='Random steps aan begin van training')


def add_env_args(parser: argparse.ArgumentParser,
                 initial_balance: float = 10000.0,
                 transaction_fee: float = 0.0) -> None:
    """
    Voeg environment-gerelateerde arguments toe.
    
    Args:
        parser: ArgumentParser
        initial_balance: Default startkapitaal
        transaction_fee: Default transactie kosten
    """
    group = parser.add_argument_group('Environment')
    
    group.add_argument('--initial_balance', type=float, default=initial_balance,
                       help='Startkapitaal in USDT')
    group.add_argument('--transaction_fee', type=float, default=transaction_fee,
                       help='Transactie kosten als fractie (0.001 = 0.1%%)')
    group.add_argument('--max_position', type=float, default=1.0,
                       help='Maximum positie grootte (fractie van portfolio)')
    group.add_argument('--reward_scaling', type=float, default=1.0,
                       help='Reward schaalfactor')


def add_logging_args(parser: argparse.ArgumentParser,
                     log_dir: str = './logs') -> None:
    """
    Voeg logging-gerelateerde arguments toe.
    
    Args:
        parser: ArgumentParser
        log_dir: Default log directory
    """
    group = parser.add_argument_group('Logging')
    
    group.add_argument('--log_dir', type=str, default=log_dir,
                       help='Directory voor logs en checkpoints')
    group.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment naam (default: auto-generated)')
    group.add_argument('--log_interval', type=int, default=1000,
                       help='Interval voor logging (in steps)')
    group.add_argument('--save_freq', type=int, default=50000,
                       help='Checkpoint save frequentie (in steps)')
    group.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluatie frequentie (in steps)')
    group.add_argument('--n_eval_episodes', type=int, default=5,
                       help='Aantal episodes per evaluatie')
    group.add_argument('--no_tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    group.add_argument('--use_mlflow', action='store_true',
                       help='Enable MLflow experiment tracking')


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Voeg model architecture arguments toe."""
    group = parser.add_argument_group('Model')
    
    group.add_argument('--deeplob_hidden', type=int, default=64,
                       help='DeepLOB hidden dimension')
    group.add_argument('--lstm_hidden', type=int, default=64,
                       help='LSTM hidden dimension')
    group.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256],
                       help='MLP hidden layer dimensions')


def add_resume_args(parser: argparse.ArgumentParser) -> None:
    """Voeg resume/checkpoint arguments toe."""
    group = parser.add_argument_group('Resume')
    
    group.add_argument('--resume', type=str, default=None,
                       help='Pad naar checkpoint om training te hervatten')
    group.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory voor checkpoints')


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Maak een basis ArgumentParser met alle standaard arguments.
    
    Args:
        description: Beschrijving voor help text
        
    Returns:
        Geconfigureerde ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    
    add_data_args(parser)
    add_training_args(parser)
    add_env_args(parser)
    add_logging_args(parser)
    add_model_args(parser)
    add_resume_args(parser)
    
    return parser


def create_ppo_parser(description: str = 'Train PPO agent') -> argparse.ArgumentParser:
    """Maak parser voor PPO training."""
    parser = create_base_parser(description)
    add_ppo_args(parser)
    return parser


def create_sac_parser(description: str = 'Train SAC agent') -> argparse.ArgumentParser:
    """Maak parser voor SAC training."""
    parser = create_base_parser(description)
    add_sac_args(parser)
    return parser
