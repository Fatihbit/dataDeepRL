"""
Training Callbacks Module
=========================

Dit module biedt callback functionaliteit voor training loops.
Callbacks worden uitgevoerd op specifieke momenten tijdens training:
- Na elke step
- Na elke episode
- Bij checkpoints
- Bij evaluatie

Standaard callbacks:
- CheckpointCallback: Regelmatig model opslaan
- EvalCallback: Periodiek evalueren en beste model opslaan
- EarlyStoppingCallback: Stoppen bij geen verbetering
- LearningRateScheduler: Learning rate aanpassen tijdens training

Auteur: DataDeepRL Team
"""

import os
import json
import datetime
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
import numpy as np

import torch


class BaseCallback(ABC):
    """
    Basis klasse voor training callbacks.
    
    Callbacks worden aangeroepen op verschillende momenten tijdens training.
    Override de relevante methodes om custom gedrag te implementeren.
    """
    
    def __init__(self, verbose: int = 1):
        """
        Args:
            verbose: Verbosity level (0=silent, 1=progress, 2=debug)
        """
        self.verbose = verbose
        self.n_calls = 0
        self.training_start_time = None
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Wordt aangeroepen aan het begin van training.
        
        Args:
            locals_: Lokale variabelen van de training functie
            globals_: Globale variabelen
        """
        self.training_start_time = datetime.datetime.now()
    
    def on_training_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """Wordt aangeroepen aan het einde van training."""
        pass
    
    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        """
        Wordt aangeroepen na elke training step.
        
        Args:
            locals_: Lokale variabelen (step, losses, etc.)
            globals_: Globale variabelen
            
        Returns:
            True om training voort te zetten, False om te stoppen
        """
        self.n_calls += 1
        return True
    
    def on_episode_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        """
        Wordt aangeroepen aan het einde van een episode.
        
        Args:
            locals_: Lokale variabelen (episode, reward, etc.)
            globals_: Globale variabelen
            
        Returns:
            True om training voort te zetten, False om te stoppen
        """
        return True


class CallbackList:
    """
    Container voor meerdere callbacks.
    
    Roept alle callbacks aan in volgorde.
    
    Gebruik:
        >>> callbacks = CallbackList([
        ...     CheckpointCallback(save_freq=1000, save_path='checkpoints'),
        ...     EvalCallback(eval_env=env, eval_freq=5000)
        ... ])
        >>> callbacks.on_step(locals_, globals_)
    """
    
    def __init__(self, callbacks: List[BaseCallback] = None):
        self.callbacks = callbacks or []
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_training_start(locals_, globals_)
    
    def on_training_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_training_end(locals_, globals_)
    
    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        """Returns False als een callback False returned."""
        continue_training = True
        for callback in self.callbacks:
            if not callback.on_step(locals_, globals_):
                continue_training = False
        return continue_training
    
    def on_episode_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        """Returns False als een callback False returned."""
        continue_training = True
        for callback in self.callbacks:
            if not callback.on_episode_end(locals_, globals_):
                continue_training = False
        return continue_training


class CheckpointCallback(BaseCallback):
    """
    Callback voor het periodiek opslaan van model checkpoints.
    
    Slaat het model op elke save_freq steps en houdt de laatste
    max_keep checkpoints.
    
    Args:
        save_freq: Hoe vaak op te slaan (in steps)
        save_path: Directory voor checkpoints
        name_prefix: Prefix voor checkpoint bestanden
        max_keep: Maximum aantal checkpoints om te bewaren
        save_replay_buffer: Of de replay buffer ook opgeslagen moet worden
        verbose: Verbosity level
    
    Gebruik:
        >>> callback = CheckpointCallback(
        ...     save_freq=10000,
        ...     save_path='checkpoints',
        ...     max_keep=5
        ... )
    """
    
    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = 'checkpoints',
        name_prefix: str = 'model',
        max_keep: int = 5,
        save_replay_buffer: bool = False,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.max_keep = max_keep
        self.save_replay_buffer = save_replay_buffer
        
        # Track saved checkpoints
        self.checkpoints = []
        
        # Maak directory
        os.makedirs(save_path, exist_ok=True)
    
    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        super().on_step(locals_, globals_)
        
        step = locals_.get('step', self.n_calls)
        
        if step > 0 and step % self.save_freq == 0:
            self._save_checkpoint(step, locals_)
        
        return True
    
    def _save_checkpoint(self, step: int, locals_: Dict[str, Any]):
        """Sla een checkpoint op."""
        # Checkpoint pad
        checkpoint_path = os.path.join(
            self.save_path, 
            f'{self.name_prefix}_step_{step}.pt'
        )
        
        # Agent ophalen uit locals
        agent = locals_.get('agent')
        if agent is None:
            if self.verbose > 0:
                print(f"Warning: No agent found in locals, cannot save checkpoint")
            return
        
        # Checkpoint data
        checkpoint = {
            'step': step,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        
        # Model state dict
        if hasattr(agent, 'state_dict'):
            checkpoint['model_state_dict'] = agent.state_dict()
        elif hasattr(agent, 'actor'):
            # SAC/PPO style agent
            checkpoint['actor_state_dict'] = agent.actor.state_dict()
            if hasattr(agent, 'critic'):
                checkpoint['critic_state_dict'] = agent.critic.state_dict()
            if hasattr(agent, 'value_net'):
                checkpoint['value_state_dict'] = agent.value_net.state_dict()
        
        # Optimizer state
        if hasattr(agent, 'actor_optimizer'):
            checkpoint['actor_optimizer'] = agent.actor_optimizer.state_dict()
        if hasattr(agent, 'critic_optimizer'):
            checkpoint['critic_optimizer'] = agent.critic_optimizer.state_dict()
        
        # Training info
        if 'episode_rewards' in locals_:
            checkpoint['episode_rewards'] = locals_['episode_rewards']
        
        # Replay buffer (optioneel, kan groot zijn)
        if self.save_replay_buffer and hasattr(agent, 'replay_buffer'):
            checkpoint['replay_buffer'] = agent.replay_buffer.get_state()
        
        # Opslaan
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        if self.verbose > 0:
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Verwijder oude checkpoints
        while len(self.checkpoints) > self.max_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                if self.verbose > 1:
                    print(f"Removed old checkpoint: {old_checkpoint}")
    
    def on_training_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """Sla finale checkpoint op."""
        step = locals_.get('step', self.n_calls)
        final_path = os.path.join(self.save_path, f'{self.name_prefix}_final.pt')
        
        # Kopieer laatste checkpoint naar final
        if self.checkpoints:
            import shutil
            shutil.copy(self.checkpoints[-1], final_path)
            if self.verbose > 0:
                print(f"Final checkpoint saved: {final_path}")


class EvalCallback(BaseCallback):
    """
    Callback voor periodieke evaluatie.
    
    Evalueert het model op een aparte environment en slaat het
    beste model op.
    
    Args:
        eval_env: Gymnasium environment voor evaluatie
        eval_freq: Hoe vaak te evalueren (in steps)
        n_eval_episodes: Aantal episodes per evaluatie
        save_best: Of het beste model opgeslagen moet worden
        best_model_save_path: Pad voor beste model
        deterministic: Of deterministic acties gebruikt moeten worden
        verbose: Verbosity level
    
    Gebruik:
        >>> eval_env = CryptoTradingEnv(data_loader, is_training=False)
        >>> callback = EvalCallback(
        ...     eval_env=eval_env,
        ...     eval_freq=10000,
        ...     n_eval_episodes=5
        ... )
    """
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_best: bool = True,
        best_model_save_path: str = 'best_model',
        deterministic: bool = True,
        log_function: Optional[Callable] = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_best = save_best
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.log_function = log_function
        
        # Track beste resultaat
        self.best_mean_reward = float('-inf')
        self.last_mean_reward = float('-inf')
        
        # Evaluatie history
        self.evaluations_results = []
        self.evaluations_steps = []
        
        # Maak directory
        if save_best:
            os.makedirs(os.path.dirname(best_model_save_path) or '.', exist_ok=True)
    
    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        super().on_step(locals_, globals_)
        
        step = locals_.get('step', self.n_calls)
        
        if step > 0 and step % self.eval_freq == 0:
            self._evaluate(step, locals_)
        
        return True
    
    def _evaluate(self, step: int, locals_: Dict[str, Any]):
        """Voer evaluatie uit."""
        agent = locals_.get('agent')
        if agent is None:
            return
        
        if self.verbose > 0:
            print(f"\n{'='*50}")
            print(f"Evaluating at step {step:,}...")
        
        # Evalueer over meerdere episodes
        episode_rewards = []
        episode_lengths = []
        episode_infos = []
        
        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Selecteer actie
                if self.deterministic and hasattr(agent, 'select_action'):
                    action = agent.select_action(obs, deterministic=True)
                elif hasattr(agent, 'select_action'):
                    action = agent.select_action(obs)
                else:
                    action = agent.actor(torch.tensor(obs).float()).argmax().item()
                
                # Step
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_infos.append(info)
        
        # Bereken statistieken
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        self.last_mean_reward = mean_reward
        self.evaluations_results.append(episode_rewards)
        self.evaluations_steps.append(step)
        
        # Log
        if self.verbose > 0:
            print(f"Eval results: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Episode lengths: {mean_length:.1f}")
            
            # Extra info van laatste episode
            if episode_infos and episode_infos[-1]:
                last_info = episode_infos[-1]
                if 'total_trades' in last_info:
                    print(f"Trades: {last_info['total_trades']}")
                if 'final_portfolio_value' in last_info:
                    print(f"Final portfolio: {last_info['final_portfolio_value']:.2f}")
        
        # Log naar externe logger
        if self.log_function:
            self.log_function(step, episode_rewards)
        
        # Sla beste model op
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            
            if self.save_best:
                self._save_best_model(step, mean_reward, agent)
        
        if self.verbose > 0:
            print(f"Best mean reward: {self.best_mean_reward:.2f}")
            print(f"{'='*50}\n")
    
    def _save_best_model(self, step: int, mean_reward: float, agent):
        """Sla het beste model op."""
        checkpoint = {
            'step': step,
            'mean_reward': mean_reward,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Model state
        if hasattr(agent, 'state_dict'):
            checkpoint['model_state_dict'] = agent.state_dict()
        elif hasattr(agent, 'actor'):
            checkpoint['actor_state_dict'] = agent.actor.state_dict()
            if hasattr(agent, 'critic'):
                checkpoint['critic_state_dict'] = agent.critic.state_dict()
        
        torch.save(checkpoint, f'{self.best_model_save_path}.pt')
        
        if self.verbose > 0:
            print(f"[*] New best model saved! Reward: {mean_reward:.2f}")


class EarlyStoppingCallback(BaseCallback):
    """
    Callback voor early stopping bij geen verbetering.
    
    Stopt training als de reward niet verbetert voor een bepaald
    aantal evaluaties. Ondersteunt zowel validation als training metrics.
    
    Args:
        patience: Aantal evaluaties zonder verbetering voordat gestopt wordt
        min_delta: Minimale verbetering om als verbetering te tellen
        check_freq: Hoe vaak te checken (in steps)
        metric_source: 'validation' (eval_mean_reward) of 'training' (episode_rewards)
        verbose: Verbosity level
        
    Gebruik met validation (aanbevolen):
        >>> callback = EarlyStoppingCallback(
        ...     patience=10,
        ...     metric_source='validation'
        ... )
        >>> # In training loop na evaluatie:
        >>> locals_dict['eval_mean_reward'] = mean_eval_reward
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        check_freq: int = 10000,
        metric_source: str = 'validation',  # 'validation' of 'training'
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self.metric_source = metric_source
        
        self.best_reward = float('-inf')
        self.wait = 0
        self._last_check_step = -1
    
    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        super().on_step(locals_, globals_)
        
        step = locals_.get('step', self.n_calls)
        
        # Check alleen op check_freq en voorkom dubbele checks
        if step > 0 and step % self.check_freq == 0 and step != self._last_check_step:
            self._last_check_step = step
            
            # Haal huidige reward op gebaseerd op metric_source
            current_reward = None
            
            if self.metric_source == 'validation':
                # Prioriteit 1: explicit eval_mean_reward
                current_reward = locals_.get('eval_mean_reward')
                
                # Prioriteit 2: eval_rewards list
                if current_reward is None:
                    eval_rewards = locals_.get('eval_rewards', [])
                    if len(eval_rewards) > 0:
                        current_reward = np.mean(eval_rewards)
            
            # Fallback naar training metrics als validation niet beschikbaar
            if current_reward is None:
                episode_rewards = locals_.get('episode_rewards', [])
                if len(episode_rewards) >= 10:
                    current_reward = np.mean(episode_rewards[-10:])
                    if self.metric_source == 'validation' and self.verbose > 1:
                        print("EarlyStopping: Using training rewards (validation not available)")
            
            if current_reward is not None:
                if current_reward > self.best_reward + self.min_delta:
                    self.best_reward = current_reward
                    self.wait = 0
                    if self.verbose > 1:
                        print(f"EarlyStopping: New best reward {current_reward:.2f}")
                else:
                    self.wait += 1
                    if self.verbose > 1:
                        print(f"EarlyStopping: No improvement, wait {self.wait}/{self.patience}")
                    
                    if self.wait >= self.patience:
                        if self.verbose > 0:
                            print(f"Early stopping triggered! No improvement for {self.patience} checks.")
                            print(f"Best reward was: {self.best_reward:.2f}")
                        return False
        
        return True
    
    def reset(self):
        """Reset early stopping state (voor hergebruik)."""
        self.best_reward = float('-inf')
        self.wait = 0
        self._last_check_step = -1


class LearningRateScheduler(BaseCallback):
    """
    Callback voor learning rate scheduling.
    
    Ondersteunt verschillende learning rate schedules:
    - Linear decay
    - Exponential decay
    - Cosine annealing
    - Custom schedule
    
    Args:
        schedule_type: Type schedule ('linear', 'exponential', 'cosine', 'custom')
        initial_lr: Initiële learning rate
        final_lr: Finale learning rate (voor linear/cosine)
        total_steps: Totaal aantal training steps
        decay_rate: Decay rate (voor exponential)
        custom_schedule: Functie (step) -> lr (voor custom)
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        schedule_type: str = 'linear',
        initial_lr: float = 3e-4,
        final_lr: float = 1e-5,
        total_steps: int = 1000000,
        decay_rate: float = 0.99999,
        custom_schedule: Optional[Callable[[int], float]] = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.decay_rate = decay_rate
        self.custom_schedule = custom_schedule
        
        self.current_lr = initial_lr
    
    def get_lr(self, step: int) -> float:
        """Bereken learning rate voor gegeven step."""
        if self.schedule_type == 'linear':
            # Lineaire interpolatie
            progress = min(step / self.total_steps, 1.0)
            return self.initial_lr + progress * (self.final_lr - self.initial_lr)
        
        elif self.schedule_type == 'exponential':
            # Exponentiële decay
            return self.initial_lr * (self.decay_rate ** step)
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing
            progress = min(step / self.total_steps, 1.0)
            return self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (1 + np.cos(np.pi * progress))
        
        elif self.schedule_type == 'custom' and self.custom_schedule:
            return self.custom_schedule(step)
        
        else:
            return self.initial_lr
    
    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        super().on_step(locals_, globals_)
        
        step = locals_.get('step', self.n_calls)
        self.current_lr = self.get_lr(step)
        
        # Update optimizer learning rates
        agent = locals_.get('agent')
        if agent:
            # SAC/PPO style
            if hasattr(agent, 'actor_optimizer'):
                for param_group in agent.actor_optimizer.param_groups:
                    param_group['lr'] = self.current_lr
            
            if hasattr(agent, 'critic_optimizer'):
                for param_group in agent.critic_optimizer.param_groups:
                    param_group['lr'] = self.current_lr
        
        return True


class ProgressCallback(BaseCallback):
    """
    Callback voor het tonen van training voortgang.
    
    Toont een progress bar en belangrijke metrics.
    
    Args:
        total_steps: Totaal aantal training steps
        log_interval: Hoe vaak te loggen (in steps)
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        total_steps: int,
        log_interval: int = 1000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.log_interval = log_interval
        
        self.episode_rewards = []
        self.start_time = None
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        super().on_training_start(locals_, globals_)
        self.start_time = datetime.datetime.now()
        
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Training started at {self.start_time}")
            print(f"Total steps: {self.total_steps:,}")
            print(f"{'='*60}\n")
    
    def on_episode_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        episode_reward = locals_.get('episode_reward', 0)
        self.episode_rewards.append(episode_reward)
        return True
    
    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        super().on_step(locals_, globals_)
        
        step = locals_.get('step', self.n_calls)
        
        if step > 0 and step % self.log_interval == 0 and self.verbose > 0:
            # Bereken metrics
            elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
            steps_per_sec = step / max(elapsed, 1)
            eta_seconds = (self.total_steps - step) / max(steps_per_sec, 1)
            
            # Progress percentage
            progress = step / self.total_steps * 100
            
            # Recent rewards
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-100:]
                mean_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards)
                reward_str = f"Reward: {mean_reward:.2f} ± {std_reward:.2f}"
            else:
                reward_str = "Reward: N/A"
            
            # Progress bar
            bar_width = 30
            filled = int(bar_width * step / self.total_steps)
            bar = '█' * filled + '░' * (bar_width - filled)
            
            # ETA formatting
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            print(
                f"[{bar}] {progress:5.1f}% | "
                f"Step {step:,}/{self.total_steps:,} | "
                f"{steps_per_sec:.1f} step/s | "
                f"ETA: {eta_str} | "
                f"{reward_str}"
            )
        
        return True
    
    def on_training_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        if self.verbose > 0:
            elapsed = datetime.datetime.now() - self.start_time
            final_step = locals_.get('step', self.n_calls)
            
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Total time: {elapsed}")
            print(f"Final step: {final_step:,}")
            
            if self.episode_rewards:
                print(f"Final avg reward (100 ep): {np.mean(self.episode_rewards[-100:]):.2f}")
                print(f"Best reward: {max(self.episode_rewards):.2f}")
            
            print(f"{'='*60}\n")


class PauseResumeCallback(BaseCallback):
    """
    Callback voor pauzeren en hervatten van training.
    
    Features:
    - Vangt Ctrl+C af en slaat checkpoint op voordat training stopt
    - Ondersteunt keyboard shortcut voor pauze (standaard: 'p')
    - Slaat automatisch checkpoint op bij pauze
    
    Args:
        checkpoint_path: Pad voor pause checkpoint
        verbose: Verbosity level
    
    Gebruik:
        >>> callback = PauseResumeCallback(checkpoint_path='checkpoints/pause.pt')
        >>> # Druk Ctrl+C om te pauzeren en checkpoint op te slaan
    """
    
    def __init__(
        self,
        checkpoint_path: str = 'checkpoints/pause_checkpoint.pt',
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.checkpoint_path = checkpoint_path
        self.paused = False
        self._original_sigint = None
        
        # Maak checkpoint directory
        os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        super().on_training_start(locals_, globals_)
        
        # Bewaar originele SIGINT handler en installeer onze eigen
        import signal
        self._original_sigint = signal.getsignal(signal.SIGINT)
        
        def pause_handler(signum, frame):
            self.paused = True
            if self.verbose > 0:
                print("\n\n[PAUSE] PAUZEREN... Checkpoint wordt opgeslagen...")
        
        signal.signal(signal.SIGINT, pause_handler)
        
        if self.verbose > 0:
            print("[TIP] Druk Ctrl+C om te pauzeren en checkpoint op te slaan")
    
    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        super().on_step(locals_, globals_)
        
        if self.paused:
            # Sla checkpoint op
            self._save_pause_checkpoint(locals_)
            
            if self.verbose > 0:
                print(f"\n[OK] Checkpoint opgeslagen: {self.checkpoint_path}")
                print(f"   Hervat later met: --resume {self.checkpoint_path}")
                print(f"{'='*60}\n")
            
            return False  # Stop training
        
        return True
    
    def _save_pause_checkpoint(self, locals_: Dict[str, Any]):
        """Sla een pause checkpoint op met alle nodige info om te hervatten."""
        agent = locals_.get('agent')
        step = locals_.get('step', self.n_calls)
        
        checkpoint = {
            'step': step,
            'timestamp': datetime.datetime.now().isoformat(),
            'paused': True,
        }
        
        # Model states
        if agent:
            if hasattr(agent, 'state_dict'):
                checkpoint['model_state_dict'] = agent.state_dict()
            elif hasattr(agent, 'actor'):
                checkpoint['actor_state_dict'] = agent.actor.state_dict()
                if hasattr(agent, 'critic'):
                    checkpoint['critic_state_dict'] = agent.critic.state_dict()
                if hasattr(agent, 'critic_target'):
                    checkpoint['critic_target_state_dict'] = agent.critic_target.state_dict()
                if hasattr(agent, 'value_net'):
                    checkpoint['value_state_dict'] = agent.value_net.state_dict()
            
            # Optimizers
            if hasattr(agent, 'optimizer'):
                checkpoint['optimizer_state_dict'] = agent.optimizer.state_dict()
            if hasattr(agent, 'actor_optimizer'):
                checkpoint['actor_optimizer'] = agent.actor_optimizer.state_dict()
            if hasattr(agent, 'critic_optimizer'):
                checkpoint['critic_optimizer'] = agent.critic_optimizer.state_dict()
            
            # SAC specifiek
            if hasattr(agent, 'log_alpha'):
                checkpoint['log_alpha'] = agent.log_alpha.item()
            if hasattr(agent, 'alpha_optimizer'):
                checkpoint['alpha_optimizer'] = agent.alpha_optimizer.state_dict()
            
            # Replay buffer (voor off-policy)
            if hasattr(agent, 'replay_buffer') and hasattr(agent.replay_buffer, 'get_state'):
                checkpoint['replay_buffer'] = agent.replay_buffer.get_state()
        
        # Training stats
        if 'episode_rewards' in locals_:
            checkpoint['episode_rewards'] = locals_['episode_rewards']
        if 'episode_count' in locals_:
            checkpoint['episode_count'] = locals_['episode_count']
        if 'update_count' in locals_:
            checkpoint['update_count'] = locals_['update_count']
        if 'best_eval_reward' in locals_:
            checkpoint['best_eval_reward'] = locals_['best_eval_reward']
        
        torch.save(checkpoint, self.checkpoint_path)
    
    def on_training_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Herstel originele SIGINT handler
        import signal
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)


def load_checkpoint(checkpoint_path: str, agent, device: str = 'cpu') -> Dict[str, Any]:
    """
    Laad een checkpoint en herstel agent state.
    
    Args:
        checkpoint_path: Pad naar checkpoint bestand
        agent: De agent om te herstellen
        device: Device om tensors naar te laden
    
    Returns:
        Dictionary met training state (step, episode_count, etc.)
    
    Gebruik:
        >>> checkpoint_info = load_checkpoint('checkpoints/pause.pt', agent)
        >>> start_step = checkpoint_info.get('step', 0)
        >>> episode_rewards = checkpoint_info.get('episode_rewards', [])
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint niet gevonden: {checkpoint_path}")
    
    print(f"[LOAD] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Herstel model states
    if hasattr(agent, 'load_state_dict') and 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        # PPO style agent (policy + value_net)
        if hasattr(agent, 'policy') and 'policy_state_dict' in checkpoint:
            agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        if hasattr(agent, 'value_net') and 'value_state_dict' in checkpoint:
            agent.value_net.load_state_dict(checkpoint['value_state_dict'])
        
        # SAC style agent (actor + critic)
        if hasattr(agent, 'actor') and 'actor_state_dict' in checkpoint:
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        if hasattr(agent, 'critic') and 'critic_state_dict' in checkpoint:
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        if hasattr(agent, 'critic_target') and 'critic_target_state_dict' in checkpoint:
            agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    
    # Herstel optimizers
    if hasattr(agent, 'optimizer') and 'optimizer_state_dict' in checkpoint:
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if hasattr(agent, 'actor_optimizer') and 'actor_optimizer' in checkpoint:
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    if hasattr(agent, 'critic_optimizer') and 'critic_optimizer' in checkpoint:
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    
    # SAC specifiek
    if hasattr(agent, 'log_alpha') and 'log_alpha' in checkpoint:
        agent.log_alpha.data.fill_(checkpoint['log_alpha'])
    if hasattr(agent, 'alpha_optimizer') and 'alpha_optimizer' in checkpoint:
        agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
    
    # Herstel replay buffer
    if hasattr(agent, 'replay_buffer') and 'replay_buffer' in checkpoint:
        if hasattr(agent.replay_buffer, 'set_state'):
            agent.replay_buffer.set_state(checkpoint['replay_buffer'])
    
    step = checkpoint.get('step', 0)
    timestamp = checkpoint.get('timestamp', 'unknown')
    print(f"[OK] Checkpoint loaded! Resuming from step {step:,} (saved: {timestamp})")
    
    return {
        'step': step,
        'episode_rewards': checkpoint.get('episode_rewards', []),
        'episode_count': checkpoint.get('episode_count', 0),
        'update_count': checkpoint.get('update_count', 0),
        'best_eval_reward': checkpoint.get('best_eval_reward', float('-inf')),
    }


if __name__ == "__main__":
    # Test callbacks
    print("Testing callbacks...")
    
    # Maak dummy agent
    class DummyAgent:
        def __init__(self):
            self.actor = torch.nn.Linear(10, 3)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
    
    agent = DummyAgent()
    
    # Maak callbacks
    callbacks = CallbackList([
        ProgressCallback(total_steps=1000, log_interval=200),
        LearningRateScheduler(
            schedule_type='linear',
            initial_lr=0.001,
            final_lr=0.0001,
            total_steps=1000
        )
    ])
    
    # Simulate training
    locals_ = {'agent': agent, 'step': 0}
    callbacks.on_training_start(locals_, {})
    
    import random
    episode_reward = 0
    episode_count = 0
    
    for step in range(1000):
        locals_['step'] = step
        
        # Simulate episode
        episode_reward += random.uniform(-1, 2)
        
        if step % 50 == 49:
            locals_['episode_reward'] = episode_reward
            callbacks.on_episode_end(locals_, {})
            episode_reward = 0
            episode_count += 1
        
        if not callbacks.on_step(locals_, {}):
            print("Training stopped by callback")
            break
    
    callbacks.on_training_end(locals_, {})
    
    print("\nCallbacks test completed!")
