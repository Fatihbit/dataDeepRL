"""
Training Logger Module
======================

Dit module biedt uitgebreide logging functionaliteit voor het tracken
van training voortgang, metrics en model performance.

Features:
- Console logging met kleuren en progress bars
- CSV logging voor data analyse
- TensorBoard integratie
- Checkpoint management
- Training curves visualisatie

Auteur: DataDeepRL Team
"""

import os
import sys
import csv
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import numpy as np

# Optionele imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


def setup_logging(
    log_dir: str = 'logs',
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuratie voor training.
    
    Args:
        log_dir: Directory voor log bestanden
        log_level: Logging level (DEBUG, INFO, WARNING, etc.)
        log_to_file: Of logs naar bestand geschreven moeten worden
        experiment_name: Naam van het experiment (voor log bestandsnaam)
        
    Returns:
        Geconfigureerde logger
    """
    # Maak log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Experiment naam met timestamp
    if experiment_name is None:
        experiment_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Root logger configuratie
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler met formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Format: [2024-01-15 10:30:25] [INFO] [module] Message
    console_format = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_file = os.path.join(log_dir, f'{experiment_name}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


class TrainingLogger:
    """
    Uitgebreide logger voor training tracking.
    
    Houdt alle training metrics bij en biedt meerdere output formats:
    - Console output met progress bars
    - CSV bestanden voor data analyse
    - TensorBoard voor visualisatie
    - JSON config opslag
    
    Gebruik:
        >>> logger = TrainingLogger('logs/experiment_001')
        >>> logger.log_step(step=100, losses={'actor': 0.5, 'critic': 1.2})
        >>> logger.log_episode(episode=10, total_reward=150.5)
        >>> logger.save_config({'learning_rate': 0.001, 'gamma': 0.99})
        >>> logger.close()
    
    Args:
        log_dir: Basis directory voor alle logs
        experiment_name: Unieke naam voor dit experiment
        use_tensorboard: Of TensorBoard writer gemaakt moet worden
        log_interval: Hoe vaak naar console te loggen (in steps)
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        use_mlflow: bool = False,
        log_interval: int = 100
    ):
        # Experiment naam met timestamp
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.log_interval = log_interval
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        
        # Maak directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)
        
        # Python logger
        self.logger = logging.getLogger(f'TrainingLogger.{experiment_name}')
        
        # =====================================
        # MLFLOW SETUP
        # =====================================
        self.mlflow_run = None
        if self.use_mlflow:
            self._init_mlflow()
        
        # =====================================
        # TENSORBOARD SETUP
        # =====================================
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = os.path.join(self.log_dir, 'tensorboard')
            self.writer = SummaryWriter(log_dir=tb_dir)
            self.logger.info(f"TensorBoard logging to: {tb_dir}")
            self.logger.info(f"Start TensorBoard met: tensorboard --logdir={tb_dir}")
        
        # =====================================
        # CSV LOGGING
        # =====================================
        # Step metrics (losses, learning rate, etc.)
        self.step_log_file = os.path.join(self.log_dir, 'step_metrics.csv')
        self.step_log_initialized = False
        
        # Episode metrics (rewards, lengths, etc.)
        self.episode_log_file = os.path.join(self.log_dir, 'episode_metrics.csv')
        self.episode_log_initialized = False
        
        # =====================================
        # METRIC TRACKING
        # =====================================
        self.step_count = 0
        self.episode_count = 0
        self.start_time = datetime.datetime.now()
        
        # Rolling averages voor smooth logging
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Best metrics voor checkpointing
        self.best_reward = float('-inf')
        self.best_step = 0
        
        self.logger.info(f"TrainingLogger initialized: {self.log_dir}")
    
    def _init_mlflow(self):
        """
        Initialiseer MLflow experiment tracking.
        
        MLflow biedt:
        - Experiment tracking met parameters en metrics
        - Model versioning en artifact storage
        - Vergelijking tussen runs via web UI
        
        Start MLflow UI met: mlflow ui --port 5000
        """
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not available. Install with: pip install mlflow")
            self.use_mlflow = False
            return
        
        try:
            # Set tracking URI (lokale SQLite database)
            mlflow_dir = os.path.join(self.log_dir, 'mlruns')
            mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")
            
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Start run
            self.mlflow_run = mlflow.start_run(run_name=self.experiment_name)
            
            self.logger.info(f"MLflow tracking initialized")
            self.logger.info(f"Start MLflow UI met: mlflow ui --backend-store-uri file://{os.path.abspath(mlflow_dir)}")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize MLflow: {e}")
            self.use_mlflow = False
    
    def _log_to_mlflow(self, metrics: Dict[str, Any], step: int):
        """Log metrics naar MLflow."""
        if not self.use_mlflow or mlflow is None:
            return
        
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    # MLflow accepteert geen forward slashes in metric names
                    clean_key = key.replace('/', '_')
                    mlflow.log_metric(clean_key, value, step=step)
        except Exception as e:
            self.logger.debug(f"MLflow logging error: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters naar MLflow.
        
        Args:
            params: Dictionary met hyperparameters
        """
        if not self.use_mlflow or mlflow is None:
            return
        
        try:
            # MLflow params moeten strings zijn
            flat_params = {}
            for key, value in params.items():
                if isinstance(value, (list, tuple)):
                    flat_params[key] = str(value)
                elif isinstance(value, dict):
                    for k, v in value.items():
                        flat_params[f"{key}_{k}"] = str(v)
                else:
                    flat_params[key] = value
            
            mlflow.log_params(flat_params)
        except Exception as e:
            self.logger.debug(f"MLflow param logging error: {e}")
    
    def log_model(self, model, artifact_path: str = "model"):
        """
        Log PyTorch model naar MLflow.
        
        Args:
            model: PyTorch model om te loggen
            artifact_path: Pad binnen MLflow artifacts
        """
        if not self.use_mlflow or mlflow is None:
            return
        
        try:
            mlflow.pytorch.log_model(model, artifact_path)
            self.logger.info(f"Model logged to MLflow: {artifact_path}")
        except Exception as e:
            self.logger.debug(f"MLflow model logging error: {e}")

    def log_step(
        self,
        step: int,
        losses: Optional[Dict[str, float]] = None,
        metrics: Optional[Dict[str, float]] = None,
        log_to_console: bool = True
    ):
        """
        Log training step metrics.
        
        Args:
            step: Huidige training step
            losses: Dictionary met loss waarden (actor_loss, critic_loss, etc.)
            metrics: Dictionary met overige metrics (learning_rate, alpha, etc.)
            log_to_console: Of naar console gelogd moet worden
        """
        self.step_count = step
        
        # Combineer losses en metrics
        all_metrics = {}
        if losses:
            all_metrics.update({f'loss/{k}': v for k, v in losses.items()})
        if metrics:
            all_metrics.update(metrics)
        
        # TensorBoard logging
        if self.writer:
            for key, value in all_metrics.items():
                self.writer.add_scalar(key, value, step)
        
        # MLflow logging
        self._log_to_mlflow(all_metrics, step)
        
        # CSV logging
        self._log_to_csv(self.step_log_file, step, all_metrics, 'step_log_initialized')
        
        # Console logging (alleen elke log_interval steps)
        if log_to_console and step % self.log_interval == 0:
            elapsed = datetime.datetime.now() - self.start_time
            steps_per_sec = step / max(elapsed.total_seconds(), 1)
            
            # Format metrics string
            metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in all_metrics.items()])
            
            self.logger.info(
                f"Step {step:,} | {steps_per_sec:.1f} steps/s | {metrics_str}"
            )
    
    def log_episode(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        info: Optional[Dict[str, Any]] = None
    ):
        """
        Log episode resultaten.
        
        Args:
            episode: Episode nummer
            total_reward: Totale reward van de episode
            episode_length: Lengte van de episode in steps
            info: Extra informatie (portfolio_value, trades, etc.)
        """
        self.episode_count = episode
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        # Bereken rolling averages (laatste 100 episodes)
        recent_rewards = self.episode_rewards[-100:]
        avg_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        recent_lengths = self.episode_lengths[-100:]
        avg_length = np.mean(recent_lengths)
        
        # Build metrics dict
        metrics = {
            'reward': total_reward,
            'avg_reward_100': avg_reward,
            'std_reward_100': std_reward,
            'episode_length': episode_length,
            'avg_length_100': avg_length
        }
        
        if info:
            metrics.update(info)
        
        # TensorBoard logging
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'episode/{key}', value, episode)
        
        # MLflow logging (elke 10 episodes om overhead te beperken)
        if episode % 10 == 0:
            self._log_to_mlflow({f'episode/{k}': v for k, v in metrics.items() 
                                if isinstance(v, (int, float))}, episode)
        
        # CSV logging
        self._log_to_csv(self.episode_log_file, episode, metrics, 'episode_log_initialized')
        
        # Console logging
        info_str = ''
        if info:
            info_items = [f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}' 
                         for k, v in info.items() 
                         if isinstance(v, (int, float))]
            if info_items:
                info_str = ' | ' + ' | '.join(info_items[:3])  # Max 3 items
        
        self.logger.info(
            f"Episode {episode:,} | "
            f"Reward: {total_reward:,.2f} | "
            f"Avg(100): {avg_reward:,.2f} ± {std_reward:.2f} | "
            f"Length: {episode_length:,}"
            f"{info_str}"
        )
        
        # Track best
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_step = self.step_count
            self.logger.info(f"  [*] New best average reward: {avg_reward:.2f}")
    
    def log_evaluation(
        self,
        step: int,
        eval_rewards: List[float],
        eval_info: Optional[Dict[str, Any]] = None
    ):
        """
        Log evaluation resultaten.
        
        Args:
            step: Huidige training step
            eval_rewards: Lijst met rewards van evaluatie episodes
            eval_info: Extra evaluatie info
        """
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        min_reward = np.min(eval_rewards)
        max_reward = np.max(eval_rewards)
        
        metrics = {
            'eval/mean_reward': mean_reward,
            'eval/std_reward': std_reward,
            'eval/min_reward': min_reward,
            'eval/max_reward': max_reward,
            'eval/n_episodes': len(eval_rewards)
        }
        
        if eval_info:
            metrics.update({f'eval/{k}': v for k, v in eval_info.items()})
        
        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        
        self.logger.info(
            f"Evaluation @ step {step:,} | "
            f"Mean: {mean_reward:.2f} ± {std_reward:.2f} | "
            f"Min: {min_reward:.2f} | Max: {max_reward:.2f}"
        )
    
    def _log_to_csv(
        self,
        filepath: str,
        index: int,
        metrics: Dict[str, Any],
        init_flag: str
    ):
        """
        Log metrics naar CSV bestand.
        
        Args:
            filepath: Pad naar CSV bestand
            index: Index waarde (step of episode)
            metrics: Dictionary met metrics
            init_flag: Naam van initialisatie flag attribuut
        """
        # Filter alleen numerieke waarden
        numeric_metrics = {
            k: v for k, v in metrics.items() 
            if isinstance(v, (int, float, np.integer, np.floating))
        }
        
        # Initialiseer CSV met headers als dit de eerste keer is
        if not getattr(self, init_flag):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = ['index', 'timestamp'] + list(numeric_metrics.keys())
                writer.writerow(headers)
            setattr(self, init_flag, True)
        
        # Schrijf data rij
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = datetime.datetime.now().isoformat()
            values = [index, timestamp] + list(numeric_metrics.values())
            writer.writerow(values)
    
    def save_config(self, config: Dict[str, Any]):
        """
        Sla training configuratie op als JSON.
        
        Args:
            config: Dictionary met alle configuratie waarden
        """
        config_path = os.path.join(self.log_dir, 'config.json')
        
        # Converteer numpy types naar Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj
        
        config = convert(config)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Log config to MLflow
        self.log_params(config)
        
        self.logger.info(f"Config saved to: {config_path}")
    
    def get_checkpoint_path(self, name: str = 'latest') -> str:
        """
        Krijg pad voor een checkpoint bestand.
        
        Args:
            name: Naam van het checkpoint ('latest', 'best', of custom)
            
        Returns:
            Volledige pad naar checkpoint
        """
        return os.path.join(self.log_dir, 'checkpoints', f'{name}.pt')
    
    def log_text(self, tag: str, text: str, step: int):
        """
        Log tekst naar TensorBoard.
        
        Args:
            tag: Tag voor de tekst
            text: Tekst om te loggen
            step: Huidige step
        """
        if self.writer:
            self.writer.add_text(tag, text, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """
        Log histogram naar TensorBoard.
        
        Args:
            tag: Tag voor het histogram
            values: Array met waarden
            step: Huidige step
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def get_progress_bar(
        self,
        total: int,
        desc: str = 'Training',
        unit: str = 'step'
    ):
        """
        Krijg een progress bar voor training loops.
        
        Args:
            total: Totaal aantal iteraties
            desc: Beschrijving
            unit: Eenheid (step, episode, etc.)
            
        Returns:
            tqdm progress bar of range indien tqdm niet beschikbaar
        """
        if TQDM_AVAILABLE:
            return tqdm(
                range(total),
                desc=desc,
                unit=unit,
                ncols=100,
                leave=True
            )
        else:
            return range(total)
    
    def close(self):
        """
        Sluit alle logging resources.
        """
        if self.writer:
            self.writer.close()
        
        # Sluit MLflow run
        if self.use_mlflow and mlflow is not None and self.mlflow_run:
            try:
                # Log finale metrics
                mlflow.log_metric('final_best_reward', self.best_reward)
                mlflow.log_metric('total_steps', self.step_count)
                mlflow.log_metric('total_episodes', self.episode_count)
                mlflow.end_run()
            except Exception as e:
                self.logger.debug(f"MLflow close error: {e}")
        
        # Log summary
        elapsed = datetime.datetime.now() - self.start_time
        self.logger.info(f"Training completed in {elapsed}")
        self.logger.info(f"Total steps: {self.step_count:,}")
        self.logger.info(f"Total episodes: {self.episode_count:,}")
        self.logger.info(f"Best average reward: {self.best_reward:.2f} @ step {self.best_step:,}")
        self.logger.info(f"Logs saved to: {self.log_dir}")


class ProgressPrinter:
    """
    Eenvoudige progress printer voor training loops.
    
    Alternatief voor tqdm met minder dependencies.
    Print progress updates op vaste intervallen.
    
    Gebruik:
        >>> progress = ProgressPrinter(total=1000, interval=100)
        >>> for i in range(1000):
        ...     progress.update(i, loss=0.5)
        >>> progress.finish()
    """
    
    def __init__(
        self,
        total: int,
        interval: int = 100,
        desc: str = 'Progress'
    ):
        self.total = total
        self.interval = interval
        self.desc = desc
        self.start_time = datetime.datetime.now()
        self.last_update = 0
    
    def update(self, current: int, **metrics):
        """Update de progress."""
        if current - self.last_update >= self.interval or current == self.total - 1:
            self.last_update = current
            
            # Bereken progress
            pct = (current + 1) / self.total * 100
            elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
            eta = elapsed / (current + 1) * (self.total - current - 1) if current > 0 else 0
            
            # Progress bar
            bar_width = 30
            filled = int(bar_width * (current + 1) / self.total)
            bar = '█' * filled + '░' * (bar_width - filled)
            
            # Metrics string
            metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            
            print(
                f"\r{self.desc}: [{bar}] {pct:5.1f}% | "
                f"{current + 1:,}/{self.total:,} | "
                f"ETA: {int(eta)}s | {metrics_str}",
                end='', flush=True
            )
    
    def finish(self):
        """Markeer progress als voltooid."""
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        print(f"\n{self.desc} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    # Test de logger
    logger = TrainingLogger(
        log_dir='logs',
        experiment_name='test_experiment',
        use_tensorboard=True
    )
    
    # Save config
    logger.save_config({
        'learning_rate': 0.001,
        'gamma': 0.99,
        'hidden_dims': [256, 256]
    })
    
    # Simulate training
    import random
    
    episode_reward = 0
    episode_length = 0
    
    for step in range(1000):
        # Fake losses
        losses = {
            'actor_loss': random.uniform(0.1, 1.0),
            'critic_loss': random.uniform(0.5, 2.0),
            'alpha': random.uniform(0.1, 0.3)
        }
        
        logger.log_step(step, losses=losses)
        
        # Simulate episode end
        episode_reward += random.uniform(-1, 2)
        episode_length += 1
        
        if step % 100 == 99:
            logger.log_episode(
                episode=step // 100,
                total_reward=episode_reward,
                episode_length=episode_length,
                info={'portfolio_value': 10000 + episode_reward * 100}
            )
            episode_reward = 0
            episode_length = 0
    
    # Evaluation
    logger.log_evaluation(
        step=1000,
        eval_rewards=[random.uniform(50, 150) for _ in range(10)]
    )
    
    logger.close()
    print("\nLogger test completed!")
