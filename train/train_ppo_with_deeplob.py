"""
Training Script: PPO met Pre-trained DeepLOB
============================================

Dit script traint een Proximal Policy Optimization (PPO) agent met een 
voorgetrainde DeepLOB feature extractor voor cryptocurrency trading.

WAT IS PPO?
-----------
PPO is een on-policy reinforcement learning algoritme dat:
- Stabiel leert door een "clipped" objective functie
- Meerdere gradient updates doet per batch data
- Goed werkt zonder veel hyperparameter tuning

WAT IS DEEPLOB?
---------------
DeepLOB is een neural network architectuur die:
- Order book data verwerkt met CNN layers
- Temporele patronen leert met LSTM
- Werd voorgetraind op price direction prediction

WORKFLOW:
---------
1. Train eerst DeepLOB (supervised): 
   python train_deeplob_pretrain.py
   
2. Gebruik die voor PPO (reinforcement learning):
   python train_ppo_with_deeplob.py --deeplob_model ./models/deeplob_pretrained.pt

OPTIES:
-------
--freeze_deeplob: DeepLOB weights blijven vast (sneller, stabiel)
(default): DeepLOB wordt fine-tuned samen met PPO (kan beter worden)

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
import torch.nn as nn
import torch.nn.functional as F

# Voeg project root toe aan Python path zodat we src kunnen importeren
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.trading_env import CryptoTradingEnv
from src.envs.vec_env import VectorizedTradingEnv
from src.models.deeplob import DeepLOB
from src.utils.logger import TrainingLogger, setup_logging
from src.utils.trade_logger import TradeLogger
from train.common.setup import load_coredata_streaming

# Onderdruk warnings voor schonere output
warnings.filterwarnings('ignore')


# =============================================================================
# PPO NETWORK
# =============================================================================

class PPONetwork(nn.Module):
    """
    Gecombineerd Policy en Value Network voor PPO.
    
    Dit netwerk heeft:
    - Shared layers: Gedeelde feature processing (efficiënter)
    - Policy head: Output actie probabiliteiten (welke actie te nemen)
    - Value head: Schat verwachte toekomstige rewards (hoe goed is huidige staat)
    
    De shared layers zorgen ervoor dat beide heads dezelfde representaties
    leren, wat training efficiënter maakt.
    
    Architectuur:
        Input features
            │
            ▼
        ┌─────────────┐
        │ Shared MLP  │  ← Gedeelde hidden layers
        │ (256 → 256) │
        └─────────────┘
            │
        ┌───┴───┐
        ▼       ▼
    ┌───────┐ ┌───────┐
    │Policy │ │ Value │
    │ head  │ │ head  │
    └───────┘ └───────┘
        │       │
        ▼       ▼
    Actie    Waarde
    probs    schatting
    
    Args:
        feature_dim: Dimensie van input features (van DeepLOB + portfolio)
        action_dim: Aantal mogelijke acties (3: buy, sell, hold)
        hidden_dims: Tuple met hidden layer groottes
    """
    
    def __init__(self, feature_dim: int, action_dim: int, hidden_dims: tuple = (256, 256)):
        super().__init__()
        
        # =====================================
        # SHARED LAYERS
        # =====================================
        # Deze layers worden gedeeld door policy en value networks.
        # Dit is efficiënter dan aparte networks en helpt met feature learning.
        layers = []
        prev_dim = feature_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),  # Lineaire transformatie
                nn.ReLU(),                  # Non-lineaire activatie
            ])
            prev_dim = dim
        self.shared = nn.Sequential(*layers)
        
        # =====================================
        # POLICY HEAD (Actor)
        # =====================================
        # Output: logits voor elke actie (worden omgezet naar probabiliteiten)
        # Voor trading: 3 outputs = [buy, sell, hold]
        self.policy = nn.Linear(prev_dim, action_dim)
        
        # =====================================
        # VALUE HEAD (Critic)
        # =====================================
        # Output: geschatte waarde van huidige staat
        # Dit is de verwachte totale toekomstige reward
        self.value = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        """
        Forward pass door het netwerk.
        
        Args:
            x: Input features tensor (batch_size, feature_dim)
            
        Returns:
            policy_logits: Raw scores voor elke actie (batch_size, action_dim)
            value: Geschatte waarde (batch_size, 1)
        """
        shared = self.shared(x)  # Gedeelde feature processing
        return self.policy(shared), self.value(shared)
    
    def get_action_probs(self, x):
        """
        Bereken actie probabiliteiten.
        
        Softmax converteert logits naar probabiliteiten die optellen tot 1.
        
        Args:
            x: Input features
            
        Returns:
            Probabiliteiten voor elke actie (sommeren tot 1.0)
        """
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)  # Softmax voor probabiliteiten
    
    def get_value(self, x):
        """
        Haal alleen de value schatting op.
        
        Args:
            x: Input features
            
        Returns:
            Geschatte waarde van de staat
        """
        _, value = self.forward(x)
        return value


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """
    Buffer voor het opslaan van on-policy ervaringen.
    
    PPO is een on-policy algoritme, wat betekent dat het leert van
    ervaringen die zijn verzameld met het HUIDIGE beleid. Dit is anders
    dan off-policy (zoals SAC) dat oude ervaringen hergebruikt.
    
    De buffer slaat op:
    - observations: Wat de agent zag
    - actions: Welke actie werd genomen
    - rewards: Hoeveel reward werd ontvangen
    - dones: Of de episode eindigde
    - values: Geschatte waarde (van critic)
    - log_probs: Log probability van de genomen actie
    
    Na het verzamelen wordt GAE (Generalized Advantage Estimation) 
    berekend om te bepalen hoe goed elke actie was ten opzichte van
    de verwachting.
    
    Supports vectorized environments: stores data in (n_steps, num_envs) layout
    and computes GAE independently per environment.
    """
    
    def __init__(self, num_envs: int = 1):
        """Initialiseer lege lijsten voor alle data."""
        self.num_envs = num_envs
        # Per-step storage: each entry is a list of num_envs items
        self.observations = []   # List of lists: [step][env] = obs dict
        self.actions = []        # List of lists: [step][env] = action
        self.rewards = []        # List of lists: [step][env] = reward
        self.dones = []          # List of lists: [step][env] = done
        self.values = []         # List of lists: [step][env] = value
        self.log_probs = []      # List of lists: [step][env] = log_prob
        self.advantages = []     # Flattened after computation
        self.returns = []        # Flattened after computation
        self._flat_observations = None  # Flattened for get_batches
    
    def add_batch(self, obs_list, actions, rewards, dones, values, log_probs):
        """
        Voeg een batch van ervaringen toe (één per environment).
        
        Args:
            obs_list: Lijst van observatie dicts (num_envs,)
            actions: Lijst van acties (num_envs,)
            rewards: Lijst van rewards (num_envs,)
            dones: Lijst van done flags (num_envs,)
            values: Lijst van value schattingen (num_envs,)
            log_probs: Lijst van log probabilities (num_envs,)
        """
        self.observations.append(list(obs_list))
        self.actions.append(list(actions))
        self.rewards.append(list(rewards))
        self.dones.append(list(dones))
        self.values.append(list(values))
        self.log_probs.append(list(log_probs))
    
    def add(self, obs, action, reward, done, value, log_prob):
        """Single-env compatibility: wraps as batch of 1."""
        self.add_batch([obs], [action], [reward], [done], [value], [log_prob])
    
    def compute_returns_and_advantages(self, last_values, gamma: float, gae_lambda: float):
        """
        Bereken returns en advantages met GAE per environment.
        
        Args:
            last_values: List of value estimates for the last state per env (num_envs,).
                         For single-env, can be a float.
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        if isinstance(last_values, (int, float)):
            last_values = [last_values]
        
        n_steps = len(self.rewards)
        num_envs = self.num_envs
        
        # Pre-allocate advantages and returns as 2D arrays (n_steps, num_envs)
        adv = np.zeros((n_steps, num_envs), dtype=np.float32)
        ret = np.zeros((n_steps, num_envs), dtype=np.float32)
        
        # Compute GAE backwards per environment
        gae = np.zeros(num_envs, dtype=np.float32)
        
        for t in reversed(range(n_steps)):
            for e in range(num_envs):
                next_value = last_values[e] if t == n_steps - 1 else self.values[t + 1][e]
                delta = self.rewards[t][e] + gamma * next_value * (1 - self.dones[t][e]) - self.values[t][e]
                gae[e] = delta + gamma * gae_lambda * (1 - self.dones[t][e]) * gae[e]
                adv[t, e] = gae[e]
                ret[t, e] = gae[e] + self.values[t][e]
        
        # Flatten to 1D for training: (n_steps * num_envs,)
        self.advantages = adv.flatten().tolist()
        self.returns = ret.flatten().tolist()
        
        # Flatten observations for training
        self._flat_observations = []
        self._flat_actions = []
        self._flat_log_probs = []
        for t in range(n_steps):
            for e in range(num_envs):
                self._flat_observations.append(self.observations[t][e])
                self._flat_actions.append(self.actions[t][e])
                self._flat_log_probs.append(self.log_probs[t][e])
    
    def get_batches(self, batch_size: int, device: torch.device):
        """
        Genereer random mini-batches voor training.
        """
        n = len(self._flat_observations)
        
        # Random shuffle indices
        indices = np.random.permutation(n)
        
        # Converteer naar tensors
        obs = self._flat_observations
        actions = torch.LongTensor(self._flat_actions).to(device)
        log_probs = torch.FloatTensor(self._flat_log_probs).to(device)
        advantages = torch.FloatTensor(self.advantages).to(device)
        returns = torch.FloatTensor(self.returns).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Genereer batches
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            yield {
                'observations': [obs[i] for i in batch_indices],
                'actions': actions[batch_indices],
                'log_probs': log_probs[batch_indices],
                'advantages': advantages[batch_indices],
                'returns': returns[batch_indices]
            }
    
    def clear(self):
        """
        Leeg de buffer na een update.
        """
        num_envs = self.num_envs
        self.__init__(num_envs=num_envs)


class PPOWithPretrainedDeepLOB:
    """
    PPO Agent met Pre-trained DeepLOB Feature Extractor.
    
    Dit is de hoofdklasse die alles combineert:
    1. Pre-trained DeepLOB: Extraheert features uit order book data
    2. Portfolio encoder: Verwerkt portfolio informatie (balans, positie)
    3. PPO Network: Leert trading beslissingen
    
    ARCHITECTUUR:
    -------------
                Order Book Data          Portfolio Info
                     │                        │
                     ▼                        ▼
              ┌─────────────┐         ┌──────────────┐
              │  Pre-trained│         │   Portfolio  │
              │   DeepLOB   │         │   Encoder    │
              │ (CNN+LSTM)  │         │   (MLP)      │
              └─────────────┘         └──────────────┘
                     │                        │
                     └────────┬───────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   PPO Network   │
                     │ (Policy+Value)  │
                     └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Actie (buy/sell/hold)  Value schatting
    
    FREEZE VS FINE-TUNE:
    --------------------
    - freeze_deeplob=True:  DeepLOB weights blijven vast
                            → Snellere training, stabiele features
                            → Goed als DeepLOB al goed getraind is
    
    - freeze_deeplob=False: DeepLOB wordt mee-getraind met PPO
                            → Kan betere features leren voor trading
                            → Maar trager en kan overfitting veroorzaken
    
    Args:
        deeplob_model_path: Pad naar opgeslagen DeepLOB model
        portfolio_dim: Aantal portfolio features (default: 4)
                       [balance, position, unrealized_pnl, realized_pnl]
        action_dim: Aantal acties (default: 3 voor buy/sell/hold)
        freeze_deeplob: Of DeepLOB weights vastgezet worden
        hidden_dims: Hidden layer groottes voor PPO network
        lr: Learning rate (hoe snel het model leert)
        gamma: Discount factor (belang van toekomstige rewards)
        gae_lambda: GAE lambda voor advantage estimation
        clip_epsilon: PPO clipping parameter (voorkomt te grote updates)
        value_coef: Gewicht van value loss
        entropy_coef: Gewicht van entropy bonus (bevordert exploratie)
        max_grad_norm: Maximum gradient norm (voorkomt exploding gradients)
        n_epochs: Aantal keer door de data per update
        batch_size: Mini-batch grootte
        device: 'cuda', 'cpu', of 'auto'
    """
    
    def __init__(
        self,
        deeplob_model_path: str,
        portfolio_dim: int = 4,
        action_dim: int = 3,
        freeze_deeplob: bool = False,
        hidden_dims: tuple = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = 'auto'
    ):
        # =====================================
        # DEVICE SETUP
        # =====================================
        # Kies automatisch GPU als beschikbaar voor snellere training
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Sla hyperparameters op
        self.portfolio_dim = portfolio_dim
        self.action_dim = action_dim
        self.freeze_deeplob = freeze_deeplob
        self.gamma = gamma              # Discount voor toekomstige rewards
        self.gae_lambda = gae_lambda    # GAE smoothing parameter
        self.clip_epsilon = clip_epsilon # PPO clipping (stabilitiet)
        self.value_coef = value_coef    # Gewicht van value loss
        self.entropy_coef = entropy_coef # Gewicht van entropy (exploratie)
        self.max_grad_norm = max_grad_norm # Gradient clipping
        self.n_epochs = n_epochs        # Epochs per update
        self.batch_size = batch_size    # Mini-batch grootte
        
        # =====================================
        # LOAD PRE-TRAINED DEEPLOB
        # =====================================
        # Laad het voorgetrainde DeepLOB model dat getraind is op
        # price direction prediction (supervised learning)
        print(f"Loading pre-trained DeepLOB from: {deeplob_model_path}")
        
        checkpoint = torch.load(deeplob_model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        # Rebuild DeepLOB met dezelfde configuratie als tijdens pre-training
        self.deeplob = DeepLOB(
            input_dim=config['input_dim'],      # Aantal input features
            hidden_dim=config['hidden_dim'],     # CNN hidden dim
            lstm_hidden=config['lstm_hidden'],   # LSTM hidden dim
            output_dim=config['lstm_hidden'] * 2, # Output features (bidirectional)
            dropout=config['dropout']
        ).to(self.device)
        
        # Laad de getrainde weights
        self.deeplob.load_state_dict(checkpoint['deeplob_state_dict'])
        print(f"  Loaded from epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.2f}%")
        
        # =====================================
        # FREEZE DEEPLOB (OPTIONEEL)
        # =====================================
        if freeze_deeplob:
            # Zet alle parameters vast - geen gradient updates
            print("  DeepLOB is FROZEN")
            for param in self.deeplob.parameters():
                param.requires_grad = False
            # Zet in eval mode (belangrijk voor dropout/batchnorm)
            self.deeplob.eval()
        else:
            print("  DeepLOB will be FINE-TUNED")
        
        # Bewaar dimensies voor later
        self.deeplob_output_dim = config['lstm_hidden'] * 2  # Bidirectional LSTM
        self.sequence_length = config['sequence_length']      # Input window grootte
        self.num_features = config['input_dim']               # Features per tijdstap
        
        # =====================================
        # PORTFOLIO ENCODER
        # =====================================
        # Simpel MLP dat portfolio informatie verwerkt:
        # - Current balance (hoeveel geld)
        # - Current position (hoeveel BTC)
        # - Unrealized P&L (papieren winst/verlies)
        # - Realized P&L (gerealiseerde winst/verlies)
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(portfolio_dim, 32),  # 4 → 32
            nn.ReLU(),
            nn.Linear(32, 32),              # 32 → 32
            nn.ReLU()
        ).to(self.device)
        
        # Totale feature dimensie voor PPO network
        # = DeepLOB output + portfolio encoding
        combined_dim = self.deeplob_output_dim + 32
        
        # =====================================
        # PPO NETWORK
        # =====================================
        # Het policy/value network dat trading beslissingen maakt
        self.network = PPONetwork(combined_dim, action_dim, hidden_dims).to(self.device)
        
        # =====================================
        # OPTIMIZER
        # =====================================
        # Adam optimizer met alle trainbare parameters
        params = list(self.portfolio_encoder.parameters()) + list(self.network.parameters())
        if not freeze_deeplob:
            # Als DeepLOB niet frozen is, train die ook mee
            params += list(self.deeplob.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        
        # =====================================
        # ROLLOUT BUFFER
        # =====================================
        # Buffer voor het verzamelen van ervaringen
        # num_envs will be set later via set_num_envs()
        self.num_envs = 1
        self.rollout_buffer = RolloutBuffer(num_envs=1)
        
        # Statistieken
        self.total_steps = 0
        
        print(f"PPO initialized: combined_dim={combined_dim}, action_dim={action_dim}")
    
    def _extract_features(self, obs_dict: dict) -> torch.Tensor:
        """
        Extraheer features uit een observatie dictionary.
        
        Dit combineert:
        1. DeepLOB features uit order book data
        2. Portfolio features uit account informatie
        
        Args:
            obs_dict: Dictionary met 'orderbook' en 'features' keys
            
        Returns:
            Gecombineerde feature tensor
        """
        # =====================================
        # PROCESS ORDER BOOK
        # =====================================
        orderbook = obs_dict['features']
        
        # Converteer naar tensor indien nodig
        if not isinstance(orderbook, torch.Tensor):
            orderbook = torch.FloatTensor(orderbook)
        
        # Voeg batch dimensie toe als nodig (seq_len, features) → (1, seq_len, features)
        if orderbook.dim() == 2:
            orderbook = orderbook.unsqueeze(0)
        
        orderbook = orderbook.to(self.device)
        
        # Haal DeepLOB features op
        # Als frozen: geen gradients berekenen (sneller)
        if self.freeze_deeplob:
            with torch.no_grad():
                lob_features = self.deeplob(orderbook)
        else:
            lob_features = self.deeplob(orderbook)
        
        # =====================================
        # PROCESS PORTFOLIO
        # =====================================
        portfolio = obs_dict['portfolio']
        
        if not isinstance(portfolio, torch.Tensor):
            portfolio = torch.FloatTensor(portfolio)
        if portfolio.dim() == 1:
            portfolio = portfolio.unsqueeze(0)
        portfolio = portfolio.to(self.device)
        
        portfolio_features = self.portfolio_encoder(portfolio)
        
        # =====================================
        # COMBINE
        # =====================================
        # Concatenate DeepLOB en portfolio features
        return torch.cat([lob_features, portfolio_features], dim=1)
    
    def _extract_features_batch(self, obs_list: list):
        """
        Batched feature extraction - verwerkt meerdere observaties tegelijk.
        Veel sneller dan _extract_features in een loop.
        """
        # Stack orderbook data into single batch tensor
        orderbooks = []
        portfolios = []
        for obs in obs_list:
            ob = obs['features']
            if not isinstance(ob, torch.Tensor):
                ob = torch.FloatTensor(ob)
            if ob.dim() == 2:
                ob = ob.unsqueeze(0)
            orderbooks.append(ob)
            
            p = obs['portfolio']
            if not isinstance(p, torch.Tensor):
                p = torch.FloatTensor(p)
            if p.dim() == 1:
                p = p.unsqueeze(0)
            portfolios.append(p)
        
        # Single GPU transfer for entire batch
        orderbook_batch = torch.cat(orderbooks, dim=0).to(self.device)
        portfolio_batch = torch.cat(portfolios, dim=0).to(self.device)
        
        # Single DeepLOB forward pass for entire batch
        if self.freeze_deeplob:
            with torch.no_grad():
                lob_features = self.deeplob(orderbook_batch)
        else:
            lob_features = self.deeplob(orderbook_batch)
        
        portfolio_features = self.portfolio_encoder(portfolio_batch)
        return torch.cat([lob_features, portfolio_features], dim=1)
    
    def select_action(self, obs: dict, deterministic: bool = False):
        """
        Selecteer een actie gegeven een observatie.
        
        STOCHASTISCH (deterministic=False):
        - Sample actie uit de probability distributie
        - Gebruikt tijdens training voor exploratie
        
        DETERMINISTISCH (deterministic=True):
        - Kies actie met hoogste probability
        - Gebruikt tijdens evaluatie
        
        Args:
            obs: Observatie dictionary
            deterministic: Of deterministisch gekozen moet worden
            
        Returns:
            action: Geselecteerde actie (0=buy, 1=sell, 2=hold)
            log_prob: Log probability van de actie
            value: Value schatting van de staat
        """
        # Geen gradients nodig voor actie selectie
        with torch.no_grad():
            # Extraheer features
            features = self._extract_features(obs)
            
            # Haal actie probabilities en value schatting op
            action_probs = self.network.get_action_probs(features)
            value = self.network.get_value(features)
            
            if deterministic:
                # Kies actie met hoogste probability
                action = action_probs.argmax(dim=1).item()
                log_prob = torch.log(action_probs[0, action] + 1e-8).item()
            else:
                # Sample uit categorische distributie
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        
        return action, log_prob, value.item()
    
    def select_actions_batch(self, obs_list: list, deterministic: bool = False):
        """
        Selecteer acties voor een batch observaties (vectorized environments).
        
        Één enkele GPU forward pass voor alle environments tegelijk.
        
        Args:
            obs_list: Lijst van observatie dictionaries (1 per env)
            deterministic: Of deterministisch gekozen moet worden
            
        Returns:
            actions: Lijst van acties
            log_probs: Lijst van log probabilities
            values: Lijst van value schattingen
        """
        with torch.no_grad():
            # Single batched GPU forward pass for all envs
            features = self._extract_features_batch(obs_list)
            
            action_probs = self.network.get_action_probs(features)
            values = self.network.get_value(features)
            
            if deterministic:
                actions_t = action_probs.argmax(dim=1)
                log_probs_t = torch.log(action_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-8)
            else:
                dist = torch.distributions.Categorical(action_probs)
                actions_t = dist.sample()
                log_probs_t = dist.log_prob(actions_t)
            
            actions = actions_t.cpu().tolist()
            log_probs = log_probs_t.cpu().tolist()
            values_list = values.squeeze(1).cpu().tolist()
        
        return actions, log_probs, values_list

    def store_transition(self, obs, action, reward, done, value, log_prob):
        """
        Sla een transitie op in de rollout buffer.
        
        Wordt aangeroepen na elke environment stap.
        
        Args:
            obs: Observatie (staat)
            action: Genomen actie
            reward: Ontvangen reward
            done: Of episode klaar is
            value: Value schatting
            log_prob: Log probability van actie
        """
        self.rollout_buffer.add(obs, action, reward, done, value, log_prob)
        self.total_steps += 1

    def store_transitions_batch(self, obs_list, actions, rewards, dones, values, log_probs):
        """
        Sla een stap van alle environments tegelijk op (vectorized).
        """
        self.rollout_buffer.add_batch(obs_list, actions, rewards, dones, values, log_probs)
    
    def set_num_envs(self, num_envs: int):
        """Set the number of parallel environments for the rollout buffer."""
        self.num_envs = num_envs
        self.rollout_buffer = RolloutBuffer(num_envs=num_envs)
    
    def update(self) -> dict:
        """
        Voer een PPO update uit.
        
        Dit is waar het leren gebeurt. PPO doet:
        1. Bereken advantages (hoe goed was elke actie)
        2. Loop meerdere epochs over de data
        3. Update policy met clipped objective
        4. Update value network met MSE loss
        
        PPO CLIPPING:
        -------------
        De "clipped" objective voorkomt te grote policy updates:
        
        L_CLIP = min(r * A, clip(r, 1-ε, 1+ε) * A)
        
        Waar:
        - r = π_new(a|s) / π_old(a|s) (probability ratio)
        - A = advantage
        - ε = clip epsilon (bijv. 0.2)
        
        Dit zorgt ervoor dat de nieuwe policy niet te veel
        afwijkt van de oude, wat training stabiliseert.
        
        Returns:
            Dictionary met losses voor logging
        """
        # =====================================
        # STAP 1: BEREKEN ADVANTAGES
        # =====================================
        # Haal value van laatste observaties voor bootstrapping (per env)
        last_obs_list = self.rollout_buffer.observations[-1]  # List of num_envs obs dicts
        with torch.no_grad():
            features = self._extract_features_batch(last_obs_list)
            last_values = self.network.get_value(features).squeeze(1).cpu().tolist()
        
        # Bereken returns en advantages met GAE
        self.rollout_buffer.compute_returns_and_advantages(
            last_values, self.gamma, self.gae_lambda
        )
        
        # =====================================
        # STAP 2: PPO UPDATES
        # =====================================
        # Trackers voor logging
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # Meerdere epochs over dezelfde data
        for epoch in range(self.n_epochs):
            # Random mini-batches per epoch
            for batch in self.rollout_buffer.get_batches(self.batch_size, self.device):
                # ---------------------------------
                # Batched feature extraction (single GPU pass)
                # ---------------------------------
                features = self._extract_features_batch(batch['observations'])
                
                # Pak batch data
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                
                # ---------------------------------
                # Forward pass
                # ---------------------------------
                action_probs = self.network.get_action_probs(features)
                values = self.network.get_value(features).squeeze()
                
                # Bereken nieuwe log probs en entropy
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()  # Gemiddelde entropy (exploratie)
                
                # ---------------------------------
                # POLICY LOSS (Clipped Surrogate)
                # ---------------------------------
                # Probability ratio: hoe veel is de policy veranderd?
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Twee termen:
                # surr1: standaard policy gradient
                # surr2: geclipte versie
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                
                # Neem het minimum (pessimistisch)
                # - zorgt voor stabiele updates
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # ---------------------------------
                # VALUE LOSS
                # ---------------------------------
                # MSE tussen predicted en target returns
                value_loss = F.mse_loss(values, returns)
                
                # ---------------------------------
                # TOTAL LOSS
                # ---------------------------------
                # Combineer alle loss componenten:
                # - Policy loss: leer betere acties
                # - Value loss: leer betere value schattingen
                # - Entropy bonus: bevorder exploratie (- omdat we maximaliseren)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # ---------------------------------
                # BACKPROPAGATION
                # ---------------------------------
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping voorkomt exploding gradients
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track voor logging
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        # =====================================
        # STAP 3: CLEAR BUFFER
        # =====================================
        # On-policy: oude data is niet meer bruikbaar
        self.rollout_buffer.clear()
        
        # Return gemiddelde losses
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1)
        }
    
    def save(self, path: str):
        """
        Sla het complete model op.
        
        Slaat alle netwerk weights en optimizer state op
        zodat training hervat kan worden.
        
        Args:
            path: Pad voor het checkpoint bestand
        """
        torch.save({
            'deeplob_state_dict': self.deeplob.state_dict(),
            'portfolio_encoder_state_dict': self.portfolio_encoder.state_dict(),
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps
        }, path)
    
    def load(self, path: str):
        """
        Laad een opgeslagen model.
        
        Args:
            path: Pad naar checkpoint bestand
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.deeplob.load_state_dict(checkpoint['deeplob_state_dict'])
        self.portfolio_encoder.load_state_dict(checkpoint['portfolio_encoder_state_dict'])
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO with pre-trained DeepLOB')
    
    parser.add_argument('--deeplob_model', type=str, default='./models/deeplob_pretrained.pt')
    parser.add_argument('--freeze_deeplob', action='store_true', default=True,
                        help='Freeze DeepLOB weights (default: True). Use --no_freeze_deeplob to fine-tune.')
    parser.add_argument('--no_freeze_deeplob', action='store_true',
                        help='Fine-tune DeepLOB weights during RL training (not recommended initially)')
    
    parser.add_argument('--data_dir', type=str, default='./coreData')
    parser.add_argument('--max_files', type=int, default=100)
    parser.add_argument('--max_rows', type=int, default=90_000_000)
    
    parser.add_argument('--total_steps', type=int, default=10_000_000)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_epochs', type=int, default=14)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--gae_lambda', type=float, default=0.99)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.003,
                        help='Entropy bonus coefficient. Higher = more exploration')
    parser.add_argument('--value_coef', type=float, default=0.25)

    parser.add_argument('--num_envs', type=int, default=256,
                        help='Number of parallel environments (higher = better GPU/CPU utilization)')

    parser.add_argument('--initial_balance', type=float, default=100000.0)
    parser.add_argument('--transaction_fee', type=float, default=0.0)
    parser.add_argument('--flat_fee', type=float, default=0.0, help='Flat fee per trade in USDT')
    parser.add_argument('--max_episode_steps', type=int, default=3600,
                        help='Max steps per training episode (0=unlimited)')

    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--n_eval_episodes', type=int, default=5)
    parser.add_argument('--max_eval_steps', type=int, default=2000, help='Max steps per eval episode (0=unlimited)')
    
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=10)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None,
                        help='Pad naar checkpoint om training te hervatten')

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle freeze/no_freeze flags
    if args.no_freeze_deeplob:
        args.freeze_deeplob = False
    
    if args.experiment_name is None:
        freeze_str = "_frozen" if args.freeze_deeplob else "_finetune"
        args.experiment_name = f"ppo_deeplob{freeze_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    setup_logging(log_dir=args.log_dir, experiment_name=args.experiment_name)
    
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    
    # GPU optimizations
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
        torch.set_float32_matmul_precision('high')  # Use TF32 on Ampere+
    
    print(f"\n{'='*60}")
    print(f"PPO + Pre-trained DeepLOB")
    print(f"{'='*60}")
    print(f"DeepLOB model: {args.deeplob_model}")
    print(f"DeepLOB frozen: {args.freeze_deeplob}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"cudnn.benchmark: True")
    print(f"{'='*60}\n")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load DeepLOB config
    deeplob_checkpoint = torch.load(args.deeplob_model, map_location='cpu', weights_only=False)
    sequence_length = deeplob_checkpoint['config']['sequence_length']
    
    # Data
    print("Loading data...")
    train_features, train_prices, val_features, val_prices, _, _ = load_coredata_streaming(
        data_dir=args.data_dir,
        sequence_length=sequence_length,
        max_rows=args.max_rows,
    )
    expected_dim = deeplob_checkpoint['config']['input_dim']
    actual_dim = train_features.shape[1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Feature mismatch: coreData has {actual_dim} features but "
            f"DeepLOB checkpoint expects input_dim={expected_dim}. "
            f"Re-train DeepLOB with matching features or adjust data."
        )

    # Environments
    num_envs = args.num_envs
    print(f"Creating {num_envs} parallel training environments...")

    train_env_kwargs = dict(
        raw_features=train_features,
        prices=train_prices,
        window_size=sequence_length,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        flat_fee=args.flat_fee,
        max_episode_steps=args.max_episode_steps,
        random_start=True,
        random_start_range=1.0,
    )
    eval_env = CryptoTradingEnv(
        raw_features=val_features,
        prices=val_prices,
        window_size=sequence_length,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        flat_fee=args.flat_fee,
        max_episode_steps=args.max_eval_steps,
        random_start=True,
        random_start_range=0.5,
    )

    train_vec_env = VectorizedTradingEnv(num_envs=num_envs, env_kwargs=train_env_kwargs)
    
    # Agent
    agent = PPOWithPretrainedDeepLOB(
        deeplob_model_path=args.deeplob_model,
        freeze_deeplob=args.freeze_deeplob,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device
    )
    agent.set_num_envs(num_envs)
    
    # Logger
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
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
    best_eval = float('-inf')
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
    last_eval_sharpe = 0.0
    last_eval_return = 0.0
    last_eval_drawdown = 0.0

    save_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    resume_ckpt_path = os.path.join(save_dir, 'resume_checkpoint.pt')
    
    # Episode finance CSV (written at episode end)
    episode_csv_path = os.path.join(save_dir, 'episode_finance.csv')
    if not os.path.exists(episode_csv_path):
        with open(episode_csv_path, 'w') as f:
            f.write('episode,reward,portfolio_value,pnl,return_pct,win_rate,total_trades,fees,max_drawdown,money_gained,money_lost,composite_score\n')

    # Update log CSV (written every update - progress line data)
    update_log_path = os.path.join(save_dir, 'update_log.csv')
    if not os.path.exists(update_log_path):
        with open(update_log_path, 'w') as f:
            f.write('update,total_steps,progress_pct,steps_per_sec,episodes,'
                    'avg_reward,avg_pv,avg_return_pct,policy_loss,value_loss,eta_min\n')

    # Training monitor CSV (written every eval period - comprehensive diagnostics)
    monitor_csv_path = os.path.join(save_dir, 'training_monitor.csv')
    if not os.path.exists(monitor_csv_path):
        with open(monitor_csv_path, 'w') as f:
            f.write('step,episode,timestamp,reward_avg100,portfolio_value,'
                    'total_trades,buys,sells,winning_sells,losing_sells,win_pct,'
                    'total_profit,total_loss,net_pnl,total_fees,'
                    'avg_buy_size,avg_sell_size,avg_profit_per_win,avg_loss_per_loss,'
                    'trade_freq,policy_loss,value_loss,entropy,eval_composite_score,'
                    'eval_sharpe,eval_return,eval_drawdown\n')
    
    # Per-trade logger
    trade_logger = TradeLogger(save_dir)
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, weights_only=False, map_location=device)
            if 'deeplob_state_dict' in ckpt:
                agent.deeplob.load_state_dict(ckpt['deeplob_state_dict'])
            if 'portfolio_encoder_state_dict' in ckpt:
                agent.portfolio_encoder.load_state_dict(ckpt['portfolio_encoder_state_dict'])
            if 'network_state_dict' in ckpt:
                agent.network.load_state_dict(ckpt['network_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            total_steps = ckpt.get('total_steps', 0)
            update_count = ckpt.get('update_count', 0)
            episode_count = ckpt.get('episode_count', 0)
            episode_rewards = ckpt.get('episode_rewards', [])
            portfolio_values = ckpt.get('portfolio_values', [])
            all_policy_losses = ckpt.get('all_policy_losses', [])
            all_value_losses = ckpt.get('all_value_losses', [])
            best_eval = ckpt.get('best_eval_reward', float('-inf'))
            win_rates = ckpt.get('win_rates', [])
            trade_counts = ckpt.get('trade_counts', [])
            episode_returns = ckpt.get('episode_returns', [])
            total_fees_list = ckpt.get('total_fees_list', [])
            total_money_lost = ckpt.get('total_money_lost', 0.0)
            total_money_gained = ckpt.get('total_money_gained', 0.0)
            total_fees_paid = ckpt.get('total_fees_paid', 0.0)
            max_drawdowns = ckpt.get('max_drawdowns', [])
            trade_logger.cumulative_pnl = ckpt.get('cumulative_pnl', 0.0)
            print(f"  Continuing from step {total_steps:,}, updates: {update_count}, episodes: {episode_count}")
            print(f"  Total gained: ${total_money_gained:,.2f}, Total lost: ${total_money_lost:,.2f}, Fees: ${total_fees_paid:,.2f}")
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
    print(f"\nStarting training... (total_steps target: {args.total_steps}, n_steps: {args.n_steps})")
    print(f"  Train data: {len(train_vec_env.prices):,}, Val data: {len(eval_env.prices):,}")
    print(f"  Parallel envs: {num_envs}, transitions per rollout: {args.n_steps * num_envs:,}")
    print(f"  Eval freq: every {args.eval_freq} updates, {args.n_eval_episodes} episodes, max {args.max_eval_steps} steps each")
    start_time = time.time()
    
    # Episode state persists across rollouts — per-environment tracking
    obs_list = train_vec_env.reset()
    # Per-env episode trackers
    ep_rewards = [0.0] * num_envs
    ep_lengths = [0] * num_envs
    ep_buy_counts = [0] * num_envs
    ep_sell_counts = [0] * num_envs
    ep_hold_counts = [0] * num_envs
    ep_fees_list = [0.0] * num_envs
    prev_balances = [args.initial_balance] * num_envs
    prev_btc_helds = [0.0] * num_envs
    
    try:
        while total_steps < actual_max_steps:
            # Collect rollout from all parallel environments
            for step in range(args.n_steps):
                # Batched action selection — single GPU forward pass for all envs
                actions, log_probs, values = agent.select_actions_batch(obs_list)
                
                # Step all environments
                next_obs_list, rewards, dones, infos = train_vec_env.step(actions)
                
                # Store all transitions
                agent.store_transitions_batch(obs_list, actions, rewards.tolist(), dones.tolist(), values, log_probs)
                total_steps += num_envs  # num_envs transitions collected per step
                
                # Per-environment episode tracking
                for i in range(num_envs):
                    ep_rewards[i] += rewards[i]
                    ep_lengths[i] += 1
                    
                    # Track action distribution
                    if actions[i] == 1:
                        ep_buy_counts[i] += 1
                    elif actions[i] == 2:
                        ep_sell_counts[i] += 1
                    else:
                        ep_hold_counts[i] += 1
                    
                    # Per-trade logging
                    info = infos[i]
                    trade_info = info.get('trade_info', {})
                    if trade_info.get('executed', False):
                        ep_fees_list[i] += trade_info.get('fee', 0.0)
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
                        # Get financial info (from terminal info before auto-reset)
                        terminal_info = info.get('terminal_info', info)
                        pv = terminal_info.get('portfolio_value', args.initial_balance)
                        ep_return = terminal_info.get('total_return', 0.0)
                        ep_pnl = pv - args.initial_balance
                        ep_win_rate = terminal_info.get('win_rate', 0.0)
                        ep_trades = terminal_info.get('total_trades', 0)
                        ep_drawdown = terminal_info.get('max_drawdown', 0.0)
                        ep_sharpe = terminal_info.get('sharpe_ratio', 0.0)
                        ep_composite = 0.5 * np.clip(ep_sharpe, -5, 5) / 5 + 0.5 * np.clip(ep_return, -1, 1) - 0.2 * ep_drawdown

                        # Track money gained/lost
                        if ep_pnl >= 0:
                            total_money_gained += ep_pnl
                        else:
                            total_money_lost += abs(ep_pnl)
                        total_fees_paid += ep_fees_list[i]
                        
                        logger.log_episode(episode_count, ep_rewards[i], ep_lengths[i])
                        episode_rewards.append(ep_rewards[i])
                        portfolio_values.append(pv)
                        win_rates.append(ep_win_rate)
                        trade_counts.append({'buy': ep_buy_counts[i], 'sell': ep_sell_counts[i], 'hold': ep_hold_counts[i]})
                        episode_returns.append(ep_return)
                        total_fees_list.append(ep_fees_list[i])
                        max_drawdowns.append(ep_drawdown)
                        episode_count += 1
                        
                        # Write episode metrics to CSV
                        with open(episode_csv_path, 'a') as f:
                            f.write(f'{episode_count},{ep_rewards[i]:.4f},{pv:.2f},{ep_pnl:.2f},'
                                    f'{ep_return*100:.4f},{ep_win_rate:.4f},{ep_trades},{ep_fees_list[i]:.4f},'
                                    f'{ep_drawdown:.4f},{total_money_gained:.2f},{total_money_lost:.2f},{ep_composite:.6f}\n')
                        
                        # Detailed financial logging every 10 episodes
                        if episode_count % 10 == 0:
                            net_pnl = total_money_gained - total_money_lost - total_fees_paid
                            print(
                                f"  [FINANCE] Ep {episode_count} | "
                                f"PnL: ${ep_pnl:+,.2f} | "
                                f"Portfolio: ${pv:,.2f} | "
                                f"Return: {ep_return*100:+.2f}% | "
                                f"Win Rate: {ep_win_rate*100:.0f}% | "
                                f"Trades: {ep_trades} (B:{ep_buy_counts[i]} S:{ep_sell_counts[i]} H:{ep_hold_counts[i]}) | "
                                f"Fees: ${ep_fees_list[i]:.2f} | "
                                f"Drawdown: {ep_drawdown*100:.1f}%"
                            )
                            print(
                                f"  [CUMULATIVE] Gained: ${total_money_gained:,.2f} | "
                                f"Lost: ${total_money_lost:,.2f} | "
                                f"Fees: ${total_fees_paid:,.2f} | "
                                f"Net: ${net_pnl:+,.2f}"
                            )
                        
                        # Reset per-env episode state (env already auto-reset by VecEnv)
                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0
                        ep_buy_counts[i] = 0
                        ep_sell_counts[i] = 0
                        ep_hold_counts[i] = 0
                        ep_fees_list[i] = 0.0
                        prev_balances[i] = args.initial_balance
                        prev_btc_helds[i] = 0.0
                
                obs_list = next_obs_list
            
            # Flush trade log to CSV (buffered, not per-trade)
            trade_logger.flush_to_csv()

            # Update
            losses = agent.update()
            update_count += 1
            all_policy_losses.append(losses.get('policy_loss', 0))
            all_value_losses.append(losses.get('value_loss', 0))
            last_entropy = losses.get('entropy', 0)
            
            logger.log_step(step=total_steps, losses=losses)
            
            # Progress logging
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
                f"Policy: {losses['policy_loss']:.4f} | "
                f"Value: {losses['value_loss']:.4f} | "
                f"ETA: {remaining/60:.1f}min"
            )
            with open(update_log_path, 'a') as f:
                f.write(f"{update_count},{total_steps},{progress:.2f},{steps_per_sec:.0f},"
                        f"{episode_count},{avg_reward:.4f},{avg_pv:.2f},{avg_return:.4f},"
                        f"{losses['policy_loss']:.6f},{losses['value_loss']:.6f},{remaining/60:.1f}\n")
            
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
                    eval_buys = 0
                    eval_sells = 0
                    eval_holds = 0
                    eval_cum_profit = 0.0
                    eval_cum_loss = 0.0
                    eval_info = {}
                    while not eval_done:
                        eval_action, _, _ = agent.select_action(eval_obs, deterministic=True)
                        if eval_action == 0: eval_holds += 1
                        elif eval_action == 1: eval_buys += 1
                        elif eval_action == 2: eval_sells += 1
                        eval_obs, r, term, trunc, eval_info = eval_env.step(eval_action)
                        eval_done = term or trunc
                        eval_reward += r
                        eval_steps += 1
                        trade_info = eval_info.get('trade_info', {})
                        if trade_info and trade_info.get('executed', False) and trade_info.get('type') == 'sell':
                            p = trade_info.get('profit', 0.0)
                            if p > 0: eval_cum_profit += p
                            else: eval_cum_loss += abs(p)
                        if args.max_eval_steps > 0 and eval_steps >= args.max_eval_steps:
                            break
                    eval_pv = eval_info.get('portfolio_value', args.initial_balance)
                    eval_pnl = eval_pv - args.initial_balance
                    eval_net = eval_cum_profit - eval_cum_loss
                    eval_rewards.append(eval_reward)
                    eval_infos.append(eval_info)
                    print(f"    Eval ep {ep_i+1}/{args.n_eval_episodes}: {eval_steps} steps | "
                          f"PV=${eval_pv:,.2f} | PnL=${eval_pnl:+,.2f} | "
                          f"Realized=${eval_net:+,.2f} (W:${eval_cum_profit:,.2f} L:${eval_cum_loss:,.2f}) | "
                          f"Trades={eval_buys+eval_sells} (B:{eval_buys}/S:{eval_sells}/H:{eval_holds})")
                _c_scores = [0.5 * np.clip(i.get('sharpe_ratio', 0.0), -5, 5) / 5
                             + 0.5 * np.clip(i.get('total_return', 0.0), -1, 1)
                             - 0.2 * i.get('max_drawdown', 0.0) for i in eval_infos]
                last_eval_composite = float(np.mean(_c_scores)) if _c_scores else 0.0
                last_eval_sharpe = float(np.mean([i.get('sharpe_ratio', 0.0) for i in eval_infos])) if eval_infos else 0.0
                last_eval_return = float(np.mean([i.get('total_return', 0.0) for i in eval_infos])) if eval_infos else 0.0
                last_eval_drawdown = float(np.mean([i.get('max_drawdown', 0.0) for i in eval_infos])) if eval_infos else 0.0

                mean_eval = np.mean(eval_rewards)
                logger.log_evaluation(total_steps, eval_rewards)
                print(f"  Eval: {mean_eval:.2f} ± {np.std(eval_rewards):.2f}")
                
                is_improving = mean_eval > best_eval

                if is_improving:
                    best_eval = mean_eval
                    best_path = os.path.join(save_dir, 'best_model.pt')
                    agent.save(best_path)

                    # Backup met prestatie-info
                    backup_dir = os.path.join(save_dir, 'backups')
                    os.makedirs(backup_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_name = f"best_model_step{total_steps}_r{mean_eval:.1f}_{ts}.pt"
                    agent.save(os.path.join(backup_dir, backup_name))
                    print(f"  [*] New best!")
                    print(f"  [BACKUP] {backup_name}")

                # Save snapshot
                try:
                    snapshot_dir = os.path.join(save_dir, 'snapshots')
                    os.makedirs(snapshot_dir, exist_ok=True)
                    avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    net_pnl = total_money_gained - total_money_lost - total_fees_paid
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
                        pf.write(f"Sequence length:   {agent.sequence_length}\n")
                        pf.write(f"Total steps:       {args.total_steps:,}\n")
                        pf.write(f"Initial balance:   {args.initial_balance}\n")
                        pf.write(f"Transaction fee:   {args.transaction_fee}\n")
                        pf.write(f"DeepLOB model:     {args.deeplob_model}\n")
                        pf.write(f"Freeze DeepLOB:    {args.freeze_deeplob}\n")
                        pf.write(f"Seed:              {args.seed}\n")
                        pf.write(f"\n--- Training Progress ---\n")
                        pf.write(f"Current step:      {total_steps:,}\n")
                        pf.write(f"Updates:           {update_count}\n")
                        pf.write(f"Episodes:          {episode_count}\n")
                        pf.write(f"Avg reward (100):  {avg_r:.2f}\n")
                        pf.write(f"Eval reward:       {mean_eval:.2f}\n")
                        pf.write(f"Best eval reward:  {best_eval:.2f}\n")
                        pf.write(f"\n--- Financial Summary ---\n")
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
                last_ent = locals().get('last_entropy', 0)
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
                            f"{trade_freq:.6f},{last_pl:.6f},{last_vl:.6f},{last_ent:.6f},{last_eval_composite:.6f},"
                            f"{last_eval_sharpe:.6f},{last_eval_return:.6f},{last_eval_drawdown:.6f}\n")
            except Exception as e:
                print(f"  [WARN] Could not write monitor CSV: {e}")
            
            # Save resume checkpoint
            if update_count % args.save_freq == 0:
                _save_ppo_deeplob_checkpoint(
                    agent, total_steps, update_count, episode_count,
                    episode_rewards, portfolio_values,
                    all_policy_losses, all_value_losses,
                    best_eval, resume_ckpt_path,
                    win_rates=win_rates, trade_counts=trade_counts,
                    episode_returns=episode_returns, total_fees_list=total_fees_list,
                    total_money_lost=total_money_lost, total_money_gained=total_money_gained,
                    total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
                    max_drawdowns=max_drawdowns
                )
                trade_logger.flush_to_csv()
                print(f"  [SAVE] Checkpoint saved at step {total_steps:,}")
            
            # Check pause
            if _pause_requested:
                _save_ppo_deeplob_checkpoint(
                    agent, total_steps, update_count, episode_count,
                    episode_rewards, portfolio_values,
                    all_policy_losses, all_value_losses,
                    best_eval, resume_ckpt_path,
                    win_rates=win_rates, trade_counts=trade_counts,
                    episode_returns=episode_returns, total_fees_list=total_fees_list,
                    total_money_lost=total_money_lost, total_money_gained=total_money_gained,
                    total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
                    max_drawdowns=max_drawdowns
                )
                print(f"\n[PAUSE] Training paused at step {total_steps:,}")
                print(f"  Resume: python train_ppo_with_deeplob.py --deeplob_model {args.deeplob_model} --resume {resume_ckpt_path} [other args]")
                break
    
    finally:
        # Save final model
        _save_ppo_deeplob_checkpoint(
            agent, total_steps, update_count, episode_count,
            episode_rewards, portfolio_values,
            all_policy_losses, all_value_losses,
            best_eval,
            os.path.join(save_dir, 'final_model.pt'),
            win_rates=win_rates, trade_counts=trade_counts,
            episode_returns=episode_returns, total_fees_list=total_fees_list,
            total_money_lost=total_money_lost, total_money_gained=total_money_gained,
            total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
            max_drawdowns=max_drawdowns
        )
        signal.signal(signal.SIGINT, _original_sigint)
        logger.close()
        
        # Generate final summary
        try:
            final_dir = os.path.join(save_dir, 'final')
            os.makedirs(final_dir, exist_ok=True)
            trade_logger.flush_to_csv()
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
                pf.write(f"Sequence length:   {agent.sequence_length}\n")
                pf.write(f"Total steps:       {args.total_steps:,}\n")
                pf.write(f"Initial balance:   {args.initial_balance}\n")
                pf.write(f"Transaction fee:   {args.transaction_fee}\n")
                pf.write(f"DeepLOB model:     {args.deeplob_model}\n")
                pf.write(f"Freeze DeepLOB:    {args.freeze_deeplob}\n")
                pf.write(f"Seed:              {args.seed}\n")
                pf.write(f"\n--- Final Results ---\n")
                pf.write(f"Total steps:       {total_steps:,}\n")
                pf.write(f"Updates:           {update_count}\n")
                pf.write(f"Episodes:          {episode_count}\n")
                pf.write(f"Avg reward (100):  {avg_r:.2f}\n")
                pf.write(f"Best eval reward:  {best_eval:.2f}\n")
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
        print(f"  Best eval reward:   {best_eval:.2f}")
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
                        '--algo', 'ppo_deeplob',
                        '--split', split,
                        '--deeplob_model', args.deeplob_model,
                        '--data_dir', args.data_dir,
                        '--n_episodes', '10',
                        '--max_steps', '2000',
                        '--initial_balance', str(args.initial_balance),
                        '--transaction_fee', str(args.transaction_fee),
                        '--flat_fee', str(args.flat_fee),
                        '--device', str(device),
                    ], check=False)
            except Exception as e:
                print(f"[WARN] Auto-evaluation failed: {e}")


def _save_ppo_deeplob_checkpoint(agent, total_steps, update_count, episode_count,
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
        'total_steps': total_steps,
        'update_count': update_count,
        'episode_count': episode_count,
        'episode_rewards': episode_rewards,
        'portfolio_values': portfolio_values,
        'all_policy_losses': all_policy_losses,
        'all_value_losses': all_value_losses,
        'best_eval_reward': best_eval_reward,
        'win_rates': win_rates or [],
        'trade_counts': trade_counts or [],
        'episode_returns': episode_returns or [],
        'total_fees_list': total_fees_list or [],
        'total_money_lost': total_money_lost,
        'total_money_gained': total_money_gained,
        'total_fees_paid': total_fees_paid,
        'cumulative_pnl': (trade_history[-1]['cumulative_pnl'] if trade_history else 0.0),
        'max_drawdowns': max_drawdowns or [],
        'deeplob_state_dict': agent.deeplob.state_dict(),
        'portfolio_encoder_state_dict': agent.portfolio_encoder.state_dict(),
        'network_state_dict': agent.network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }
    torch.save(ckpt, path)


if __name__ == '__main__':
    main()
