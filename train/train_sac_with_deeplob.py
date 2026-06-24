"""
Training Script: SAC met Pre-trained DeepLOB
============================================

Dit script traint een Soft Actor-Critic (SAC) agent die gebruikmaakt van
een voorgetraind DeepLOB model als feature extractor voor order book data.

WAT IS SAC?
-----------
SAC (Soft Actor-Critic) is een state-of-the-art off-policy RL algoritme dat:

1. Off-Policy is:
   - Leert van ervaringen opgeslagen in een replay buffer
   - Kan oude data hergebruiken (sample efficient)
   - Elke transitie kan meerdere keren gebruikt worden
   
2. Maximum Entropy RL gebruikt:
   - Maximaliseert reward + entropy
   - Entropy: maat voor willekeurigheid in de policy
   - Bevordert exploratie en robuustheid
   
3. Actor-Critic methode is:
   - Actor: leert de policy (welke actie)
   - Critic: schat de value (hoe goed is actie)

SAC FORMULE:
------------
De objective die SAC maximaliseert is:

    J(π) = Σ E[r(s,a) + α * H(π(·|s))]

Waar:
    - r(s,a): reward voor state-action pair
    - H(π): entropy van de policy
    - α (alpha): temperature parameter, bepaalt exploratie vs exploitatie

SAC VS PPO:
-----------
SAC:                              PPO:
- Off-policy                      - On-policy
- Replay buffer (1M+ transitions) - Rollout buffer (klein, vers)
- Sample efficient               - Minder sample efficient
- Continuous actions native      - Discrete actions native
- Automatic temperature tuning   - Manual entropy coefficient

WORKFLOW:
---------
1. Train eerst DeepLOB (supervised):
   python train_deeplob_pretrain.py
   
2. Gebruik voor SAC (RL):
   python train_sac_with_deeplob.py --deeplob_model ./models/deeplob_pretrained.pt

OPTIES:
-------
- --freeze_deeplob: Houd DeepLOB weights vast (snellere training)
- Zonder flag: Fine-tune DeepLOB samen met SAC (potentieel beter)

Auteur: DataDeepRL Team
"""

import os
import sys
import signal
import argparse
import datetime
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Voeg src toe aan path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.trading_env import CryptoTradingEnv
from src.envs.vec_env import VectorizedTradingEnv
from src.models.deeplob import DeepLOB
from src.models.sac import ReplayBuffer
from src.utils.logger import TrainingLogger, setup_logging
from src.utils.trade_logger import TradeLogger
from train.common.setup import load_coredata_streaming

warnings.filterwarnings('ignore')


class DiscreteActor(nn.Module):
    """Actor network voor discrete acties (softmax policy)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple = (256, 256)):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.01)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class DiscreteCritic(nn.Module):
    """Twin Q-network voor discrete acties (output Q per actie)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple = (256, 256)):
        super().__init__()
        self.q1 = self._build_network(obs_dim, action_dim, hidden_dims)
        self.q2 = self._build_network(obs_dim, action_dim, hidden_dims)
    
    def _build_network(self, obs_dim, action_dim, hidden_dims):
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.01)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        return nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, action=None, return_all: bool = False):
        q1 = self.q1(obs)
        q2 = self.q2(obs)
        if return_all:
            return q1, q2
        if action is not None:
            q1 = q1.gather(1, action.long())
            q2 = q2.gather(1, action.long())
        return q1, q2


class SACWithPretrainedDeepLOB:
    """
    SAC Agent met Pre-trained DeepLOB Feature Extractor.
    
    Dit is de hoofdklasse voor SAC trading. Het combineert:
    
    1. Pre-trained DeepLOB:
       - Getraind met supervised learning op prijsvoorspelling
       - Extraheert relevante features uit order book data
       - Kan frozen of fine-tuned worden
       
    2. Portfolio Encoder:
       - Verwerkt account informatie (balans, positie, etc.)
       - Klein MLP netwerk
       
    3. SAC Components:
       - Actor: Kiest acties (soft policy met entropy)
       - Critic: Schat Q-values (twin networks)
       - Temperature α: Balanceert exploration/exploitation
    
    ARCHITECTUUR:
    -------------
              Order Book Data              Portfolio Info
                   │                            │
                   ▼                            ▼
            ┌─────────────┐              ┌──────────────┐
            │  DeepLOB    │              │   Portfolio  │
            │ (CNN+LSTM)  │              │   Encoder    │
            └─────────────┘              └──────────────┘
                   │                            │
                   └──────────┬─────────────────┘
                              │
                              ▼
                     ┌───────────────────┐
                     │  Combined Features │
                     └───────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               │               ▼
        ┌─────────┐          │          ┌─────────┐
        │  Actor  │          │          │ Critics │
        │ (Policy)│          │          │ Q1 + Q2 │
        └─────────┘          │          └─────────┘
              │              │
              └──────────────┘
                              │
                              ▼
                        Trade Actie
    
    SAC SPECIFIEKE COMPONENTEN:
    ---------------------------
    
    1. Twin Q-Networks (Critic):
       - Twee identieke Q-networks (Q1, Q2)
       - Nemen het minimum van beide voor targets
       - Vermindert overestimation bias
    
    2. Target Networks:
       - Kopie van critics die langzaam ge-update wordt
       - Soft update: θ_target = τ*θ + (1-τ)*θ_target
       - Stabiliseert training
    
    3. Automatic Temperature Tuning:
       - α wordt automatisch aangepast
       - Target entropy gebaseerd op action space
       - Meer exploratie in begin, minder later
    
    4. Replay Buffer:
       - Slaat transitions op: (s, a, r, s', done)
       - Random sampling breekt correlaties
       - Verbetert sample efficiency
    
    Args:
        deeplob_model_path: Pad naar pre-trained DeepLOB checkpoint
        portfolio_dim: Dimensie van portfolio features (default: 4)
                       [balance, position, unrealized_pnl, portfolio_ratio]
        action_dim: Aantal acties (3 voor discrete buy/sell/hold)
        freeze_deeplob: Of DeepLOB weights frozen moeten blijven
        hidden_dims: Hidden dimensions voor actor/critic MLPs
        lr: Learning rate voor alle optimizers
        gamma: Discount factor (belang van toekomstige rewards)
        tau: Soft update coefficient (hoe snel targets updaten)
        alpha: Initiële entropy temperature
        auto_alpha: Automatisch α tunen (aanbevolen: True)
        buffer_size: Grootte van replay buffer
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
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 1_000_000,
        device: str = 'auto'
    ):
        # =====================================
        # DEVICE SETUP
        # =====================================
        # Selecteer automatisch GPU als beschikbaar voor snellere training
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # =====================================
        # HYPERPARAMETERS OPSLAAN
        # =====================================
        self.portfolio_dim = portfolio_dim
        self.action_dim = action_dim
        self.freeze_deeplob = freeze_deeplob
        self.gamma = gamma          # Discount factor: hoe belangrijk zijn toekomstige rewards
        self.tau = tau              # Soft update rate: hoe snel volgt target network
        
        # =====================================
        # LOAD PRE-TRAINED DEEPLOB
        # =====================================
        # Laad het DeepLOB model dat getraind is met supervised learning
        # Dit model heeft geleerd om prijsbewegingen te voorspellen
        print(f"Loading pre-trained DeepLOB from: {deeplob_model_path}")
        
        checkpoint = torch.load(deeplob_model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        # Herbouw DeepLOB met dezelfde configuratie als tijdens training
        self.deeplob = DeepLOB(
            input_dim=config['input_dim'],       # Aantal features per timestep
            hidden_dim=config['hidden_dim'],      # CNN hidden dimension
            lstm_hidden=config['lstm_hidden'],    # LSTM hidden dimension
            output_dim=config['lstm_hidden'] * 2, # Output dim (bidirectional)
            dropout=config['dropout']
        ).to(self.device)
        
        # Laad de getrainde weights
        self.deeplob.load_state_dict(checkpoint['deeplob_state_dict'])
        print(f"  Loaded from epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.2f}%")
        
        # =====================================
        # FREEZE DEEPLOB (OPTIONEEL)
        # =====================================
        # Freeze = geen gradient updates = snellere training
        # Fine-tune = gradients door DeepLOB = potentieel betere features
        if freeze_deeplob:
            print("  DeepLOB is FROZEN (no fine-tuning)")
            for param in self.deeplob.parameters():
                param.requires_grad = False  # Geen gradient berekening
            self.deeplob.eval()  # Zet in evaluation mode (belangrijk voor dropout)
        else:
            print("  DeepLOB will be FINE-TUNED")
        
        # Bewaar dimensie informatie
        self.deeplob_output_dim = config['lstm_hidden'] * 2  # Bidirectional
        self.sequence_length = config['sequence_length']
        self.num_features = config['input_dim']
        
        # =====================================
        # PORTFOLIO ENCODER
        # =====================================
        # Encode portfolio state naar een representation
        # Input: [balance_ratio, btc_ratio, unrealized_pnl, portfolio_ratio]
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(portfolio_dim, 32),   # 4 → 32
            nn.LayerNorm(32),               # Normalisatie voor stabiliteit
            nn.ReLU(),                      # Non-lineaire activatie
            nn.Linear(32, 32),              # 32 → 32
            nn.ReLU()
        ).to(self.device)
        
        # Totale feature dimensie voor SAC networks
        combined_dim = self.deeplob_output_dim + 32
        
        # =====================================
        # SAC NETWORKS
        # =====================================
        # Actor: leert de policy π(a|s) - kiest acties (discrete softmax)
        self.actor = DiscreteActor(combined_dim, action_dim, hidden_dims).to(self.device)
        
        # Critic: leert Q(s,a) - schat waarde van state-action pairs
        # We gebruiken TWIN Q-networks (Q1 en Q2) voor stabiliteit
        self.critic = DiscreteCritic(combined_dim, action_dim, hidden_dims).to(self.device)
        
        # Target Critic: langzaam ge-update kopie voor stabiele targets
        # Dit is cruciaal voor SAC stabiliteit
        self.critic_target = DiscreteCritic(combined_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # Kopieer weights
        
        # =====================================
        # OPTIMIZERS
        # =====================================
        # Actor optimizer: update policy + feature extractors
        actor_params = list(self.portfolio_encoder.parameters()) + list(self.actor.parameters())
        if not freeze_deeplob:
            # Als niet frozen, train DeepLOB mee met actor
            actor_params += list(self.deeplob.parameters())
        
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr)
        
        # Critic optimizer: update alleen critic network
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # =====================================
        # ENTROPY TEMPERATURE (α)
        # =====================================
        # α bepaalt de trade-off:
        # - Hoge α: meer exploratie (meer willekeurige acties)
        # - Lage α: meer exploitatie (greedy naar beste actie)
        
        self.auto_alpha = auto_alpha
        
        if auto_alpha:
            # Automatic tuning van α
            # Target entropy is een heuristiek: -dim(A) * factor
            self.target_entropy = -action_dim * 0.5  # Aangepast voor discrete
            
            # log_alpha is leerbaar (niet α direct voor numerieke stabiliteit)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            # Vaste α waarde
            self.alpha = alpha
        
        # =====================================
        # REPLAY BUFFER
        # =====================================
        # SAC is off-policy: we slaan ervaringen op en samplen later
        # Dit maakt het zeer sample efficient
        flat_obs_dim = self.sequence_length * self.num_features + portfolio_dim
        self.buffer = ReplayBuffer(
            capacity=buffer_size,              # Max 1M transitions
            obs_shape=(flat_obs_dim,),         # Flat observation shape
            action_dim=1                       # Discrete action stored as single int
        )
        
        # =====================================
        # TRAINING STATS
        # =====================================
        self.total_steps = 0
        
        print(f"SAC initialized: combined_dim={combined_dim}, action_dim={action_dim}")
    
    def _extract_features(self, obs_dict: dict) -> torch.Tensor:
        """
        Extract features van een observatie dictionary.
        
        Dit combineert:
        1. DeepLOB features uit order book data (market microstructure)
        2. Portfolio features uit account informatie (portfolio state)
        
        De gecombineerde features worden gebruikt door actor en critic.
        
        Args:
            obs_dict: Dictionary met:
                - 'features': (seq_len, features) order book data
                - 'portfolio': portfolio state vector
                
        Returns:
            Gecombineerde feature tensor (batch, combined_dim)
        """
        # =====================================
        # PROCESS ORDER BOOK MET DEEPLOB
        # =====================================
        orderbook = obs_dict['features']
        
        # Converteer naar tensor indien nodig
        if not isinstance(orderbook, torch.Tensor):
            orderbook = torch.FloatTensor(orderbook)
        
        # Voeg batch dimensie toe: (seq, feat) → (1, seq, feat)
        if orderbook.dim() == 2:
            orderbook = orderbook.unsqueeze(0)
        
        orderbook = orderbook.to(self.device)
        
        # DeepLOB forward pass
        # Als frozen: geen gradients (sneller, minder memory)
        if self.freeze_deeplob:
            with torch.no_grad():
                lob_features = self.deeplob(orderbook)
        else:
            lob_features = self.deeplob(orderbook)
        
        # =====================================
        # PROCESS PORTFOLIO STATE
        # =====================================
        portfolio = obs_dict['portfolio']
        
        if not isinstance(portfolio, torch.Tensor):
            portfolio = torch.FloatTensor(portfolio)
        if portfolio.dim() == 1:
            portfolio = portfolio.unsqueeze(0)  # Voeg batch dim toe
        portfolio = portfolio.to(self.device)
        portfolio_features = self.portfolio_encoder(portfolio)
        
        # =====================================
        # COMBINEER FEATURES
        # =====================================
        # Concatenate langs feature dimensie
        return torch.cat([lob_features, portfolio_features], dim=1)
    
    def _extract_features_from_flat(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract features van een platte (flattened) observatie.
        
        Dit is nodig voor de replay buffer, omdat we daar platte
        tensors opslaan voor efficiëntie.
        
        Args:
            obs: Platte observatie tensor (batch, flat_dim)
            
        Returns:
            Gecombineerde features (batch, combined_dim)
        """
        batch_size = obs.size(0)
        
        # Split platte tensor terug in onderdelen
        # Eerste deel: orderbook (geflattend)
        # Laatste deel: portfolio state
        orderbook_flat = obs[:, :-self.portfolio_dim]
        portfolio = obs[:, -self.portfolio_dim:]
        
        # Reshape orderbook: (batch, flat) → (batch, seq_len, features)
        orderbook = orderbook_flat.view(batch_size, self.sequence_length, self.num_features)
        
        # DeepLOB features
        if self.freeze_deeplob:
            with torch.no_grad():
                lob_features = self.deeplob(orderbook)
        else:
            lob_features = self.deeplob(orderbook)
        
        # Portfolio features
        portfolio_features = self.portfolio_encoder(portfolio)
        
        # Combineer
        return torch.cat([lob_features, portfolio_features], dim=1)
    
    def select_action(self, obs: dict, deterministic: bool = False) -> int:
        """
        Selecteer een discrete actie voor de gegeven observatie.
        
        STOCHASTISCH (deterministic=False):
        - Sample actie uit softmax distributie over Q-values
        - Gebruikt tijdens training voor exploratie
        - Bevordert entropy (willekeurigheid)
        
        DETERMINISTISCH (deterministic=True):
        - Kies actie met hoogste Q-value (greedy)
        - Gebruikt tijdens evaluatie
        
        Args:
            obs: Observatie dictionary
            deterministic: Of greedy gekozen moet worden
            
        Returns:
            Discrete actie (0=hold, 1=buy, 2=sell)
        """
        with torch.no_grad():
            # Extract features uit observatie
            features = self._extract_features(obs)
            
            if deterministic:
                # Greedy selectie: actie met hoogste Q-value
                # Neem minimum van Q1 en Q2 voor conservatieve schatting
                q1, q2 = self.critic(features, None, return_all=True)
                q = torch.min(q1, q2)
                action = q.argmax(dim=1).item()
            else:
                # Sample van softmax policy
                # Dit voegt stochasticiteit toe voor exploratie
                action_probs = self.actor.get_action_probs(features)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
        
        return action
    
    def store_transition(self, obs: dict, action: int, reward: float, next_obs: dict, done: bool):
        """
        Sla een transition op in de replay buffer.
        
        Een transition is een tuple (s, a, r, s', done) die één stap
        in de environment representeert. De replay buffer slaat
        deze op zodat we er later van kunnen leren.
        
        Args:
            obs: Huidige observatie (state s)
            action: Genomen actie (a)
            reward: Ontvangen reward (r)
            next_obs: Volgende observatie (state s')
            done: Of de episode klaar is
        """
        # Flatten observations voor opslag
        # Dit is efficiënter dan dict opslag
        orderbook = obs['features'].flatten()
        portfolio = obs['portfolio']
        flat_obs = np.concatenate([orderbook, portfolio])
        
        next_orderbook = next_obs['features'].flatten()
        next_portfolio = next_obs['portfolio']
        flat_next_obs = np.concatenate([next_orderbook, next_portfolio])
        
        # Voeg toe aan buffer
        self.buffer.add(flat_obs, np.array([action]), reward, flat_next_obs, done)
        self.total_steps += 1
    
    def select_actions_batch(self, obs_list: list, deterministic: bool = False) -> np.ndarray:
        """Batched action selection for N parallel environments."""
        n = len(obs_list)
        with torch.no_grad():
            # Stack orderbook data: (N, seq_len, num_features)
            orderbooks = np.stack([o['features'] for o in obs_list])
            portfolios = np.stack([o['portfolio'] for o in obs_list])
            
            ob_t = torch.FloatTensor(orderbooks).to(self.device)
            pf_t = torch.FloatTensor(portfolios).to(self.device)
            
            if self.freeze_deeplob:
                lob_features = self.deeplob(ob_t)
            else:
                lob_features = self.deeplob(ob_t)
            portfolio_features = self.portfolio_encoder(pf_t)
            features = torch.cat([lob_features, portfolio_features], dim=1)
            
            if deterministic:
                q1, q2 = self.critic(features, None, return_all=True)
                q = torch.min(q1, q2)
                actions = q.argmax(dim=1).cpu().numpy()
            else:
                action_probs = self.actor.get_action_probs(features)
                action_dist = torch.distributions.Categorical(action_probs)
                actions = action_dist.sample().cpu().numpy()
        return actions
    
    def store_transitions_batch(self, obs_list, actions, rewards, next_obs_list, dones):
        """Store N transitions from parallel environments into replay buffer."""
        for i in range(len(obs_list)):
            orderbook = obs_list[i]['features'].flatten()
            portfolio = obs_list[i]['portfolio']
            flat_obs = np.concatenate([orderbook, portfolio])
            
            next_orderbook = next_obs_list[i]['features'].flatten()
            next_portfolio = next_obs_list[i]['portfolio']
            flat_next_obs = np.concatenate([next_orderbook, next_portfolio])
            
            self.buffer.add(flat_obs, np.array([actions[i]]), rewards[i], flat_next_obs, float(dones[i]))
            self.total_steps += 1
    
    def update(self, batch_size: int = 256) -> dict:
        """
        Voer één SAC update stap uit.
        
        Dit is het hart van SAC training. De update bestaat uit:
        
        1. CRITIC UPDATE (Q-learning):
           - Sample batch uit replay buffer
           - Bereken target: r + γ * (min(Q1', Q2') - α * log(π))
           - Minimaliseer MSE loss voor beide Q-networks
        
        2. ACTOR UPDATE (Policy Gradient):
           - Maximaliseer: Q(s, a) - α * log(π(a|s))
           - Equivalent: minimaliseer α * log(π) - Q
        
        3. TEMPERATURE UPDATE (α tuning):
           - Past α aan zodat entropy ≈ target_entropy
           - Meer exploratie als entropy te laag
        
        4. TARGET NETWORK UPDATE:
           - Soft update: θ' = τ*θ + (1-τ)*θ'
           - Zorgt voor stabiele targets
        
        SAC LOSS FUNCTIES:
        ------------------
        
        Critic Loss (TD-learning):
            L_critic = E[(Q(s,a) - (r + γ * V(s')))²]
            
        Waar V(s') = E[Q(s',a') - α * log π(a'|s')]
        
        Actor Loss (Maximum Entropy):
            L_actor = E[α * log π(a|s) - Q(s,a)]
            
        We willen hoge Q en hoge entropy (lage log π).
        
        Temperature Loss (Entropy Constraint):
            L_α = E[-α * (log π(a|s) + target_entropy)]
            
        Args:
            batch_size: Aantal samples per update
            
        Returns:
            Dictionary met loss waarden voor logging
        """
        # Check of we genoeg samples hebben
        if len(self.buffer) < batch_size:
            return {}
        
        # =====================================
        # SAMPLE BATCH
        # =====================================
        # Random sampling uit replay buffer
        batch = self.buffer.sample(batch_size, self.device)
        obs = batch['observations']
        actions = batch['actions'].long()  # Discrete acties als long tensor
        rewards = batch['rewards']
        next_obs = batch['next_observations']
        dones = batch['dones']
        
        # =====================================
        # EXTRACT FEATURES
        # =====================================
        features = self._extract_features_from_flat(obs)
        
        # =====================================
        # CRITIC UPDATE
        # =====================================
        with torch.no_grad():
            # Features voor next state
            next_features = self._extract_features_from_flat(next_obs)
            
            # Actie distributie voor next state
            next_probs = self.actor.get_action_probs(next_features)
            next_log_probs = torch.log(next_probs + 1e-8)  # +epsilon voor stabiliteit
            
            # Target Q-values van target networks
            next_q1, next_q2 = self.critic_target(next_features, None, return_all=True)
            next_q = torch.min(next_q1, next_q2)  # Pessimistisch (voorkom overestimatie)
            
            # V(s') = E_a'[Q(s',a') - α * log π(a'|s')]
            # Expectation over alle acties (gewogen naar probability)
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            
            # TD Target: r + γ * (1 - done) * V(s')
            target_q = rewards + self.gamma * (1 - dones) * next_v
        
        # Huidige Q-values
        current_q1, current_q2 = self.critic(features.detach(), None, return_all=True)
        current_q1 = current_q1.gather(1, actions)  # Q voor gekozen actie
        current_q2 = current_q2.gather(1, actions)
        
        # MSE Loss voor beide critics
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Backprop critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Gradient clipping
        self.critic_optimizer.step()
        
        # =====================================
        # ACTOR UPDATE
        # =====================================
        # Bereken actie probabilities met gradient
        action_probs = self.actor.get_action_probs(features)
        log_probs = torch.log(action_probs + 1e-8)
        
        # Q-values voor actor update (zonder gradient door critic)
        with torch.no_grad():
            q1, q2 = self.critic(features, None, return_all=True)
            q = torch.min(q1, q2)
        
        # Actor loss: E[α * log π - Q]
        # We willen hoge Q en lage log π (hoge entropy)
        actor_loss = (action_probs * (self.alpha * log_probs - q)).sum(dim=1).mean()
        
        # Backprop actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        if not self.freeze_deeplob:
            torch.nn.utils.clip_grad_norm_(self.deeplob.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # =====================================
        # ALPHA (TEMPERATURE) UPDATE
        # =====================================
        alpha_loss = 0.0
        
        if self.auto_alpha:
            # Bereken huidige entropy
            with torch.no_grad():
                entropy = -(action_probs * log_probs).sum(dim=1).mean()
            
            # Als entropy < target: verhoog α (meer exploratie)
            # Als entropy > target: verlaag α (meer exploitatie)
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update α waarde
            self.alpha = self.log_alpha.exp().item()
        
        # =====================================
        # SOFT UPDATE TARGET NETWORKS
        # =====================================
        # θ_target = τ * θ + (1 - τ) * θ_target
        # Dit zorgt voor langzame, stabiele target updates
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Return losses voor logging
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item() if self.auto_alpha else 0.0
        }
    
    def save(self, path: str):
        """
        Sla het complete model op.
        
        Slaat alle network weights, optimizer states, en training
        statistieken op zodat training hervat kan worden.
        
        Args:
            path: Pad voor het checkpoint bestand
        """
        torch.save({
            'deeplob_state_dict': self.deeplob.state_dict(),
            'portfolio_encoder_state_dict': self.portfolio_encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'alpha': self.alpha,
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
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.alpha = checkpoint['alpha']
        self.total_steps = checkpoint['total_steps']


def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC with pre-trained DeepLOB')
    
    # DeepLOB
    parser.add_argument('--deeplob_model', type=str, default='./models/deeplob_pretrained.pt',
                        help='Path naar pre-trained DeepLOB model')
    parser.add_argument('--freeze_deeplob', action='store_true', default=True,
                        help='Freeze DeepLOB weights (geen fine-tuning)')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./coreData')
    parser.add_argument('--max_files', type=int, default=100)
    parser.add_argument('--max_rows', type=int, default=90_000_000)

    # Training
    parser.add_argument('--total_steps', type=int, default=10_000_000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=7e-5)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto_alpha', action='store_true', default=True)

    # Environment
    parser.add_argument('--initial_balance', type=float, default=100000.0)
    parser.add_argument('--transaction_fee', type=float, default=0.0)
    parser.add_argument('--flat_fee', type=float, default=0.0, help='Flat fee per trade in USDT')
    parser.add_argument('--max_episode_steps', type=int, default=3600,
                        help='Max steps per training episode (moet <= n_steps zijn, 0=unlimited)')
    parser.add_argument('--num_envs', type=int, default=256,
                        help='Number of parallel environments')

    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=1, help='Eval na elke N updates')
    parser.add_argument('--n_eval_episodes', type=int, default=5)
    parser.add_argument('--max_eval_steps', type=int, default=2000, help='Max steps per eval episode (0=unlimited)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=50000)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None,
                        help='Pad naar checkpoint om training te hervatten')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    if args.experiment_name is None:
        freeze_str = "_frozen" if args.freeze_deeplob else "_finetune"
        args.experiment_name = f"sac_deeplob{freeze_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    setup_logging(log_dir=args.log_dir, experiment_name=args.experiment_name)
    
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    
    # GPU optimizations
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    
    print(f"\n{'='*60}")
    print(f"SAC + Pre-trained DeepLOB")
    print(f"{'='*60}")
    print(f"DeepLOB model: {args.deeplob_model}")
    print(f"DeepLOB frozen: {args.freeze_deeplob}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"{'='*60}\n")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # =====================================
    # LOAD DEEPLOB CONFIG
    # =====================================
    deeplob_checkpoint = torch.load(args.deeplob_model, map_location='cpu', weights_only=False)
    deeplob_config = deeplob_checkpoint['config']
    sequence_length = deeplob_config['sequence_length']
    
    # =====================================
    # DATA
    # =====================================
    print("Loading data...")
    train_features, train_prices, val_features, val_prices, _, _ = load_coredata_streaming(
        data_dir=args.data_dir,
        sequence_length=sequence_length,
        max_rows=args.max_rows,
    )
    expected_dim = deeplob_config['input_dim']
    actual_dim = train_features.shape[1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Feature mismatch: coreData has {actual_dim} features but "
            f"DeepLOB checkpoint expects input_dim={expected_dim}. "
            f"Re-train DeepLOB with matching features or adjust data."
        )
    print(f"Train: {len(train_prices):,}, Val: {len(val_prices):,}")

    # =====================================
    # ENVIRONMENTS
    # =====================================
    num_envs = args.num_envs
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
    )

    train_vec_env = VectorizedTradingEnv(
        num_envs=num_envs,
        env_kwargs=train_env_kwargs,
        env_class=CryptoTradingEnv
    )
    
    # =====================================
    # AGENT
    # =====================================
    agent = SACWithPretrainedDeepLOB(
        deeplob_model_path=args.deeplob_model,
        portfolio_dim=4,
        action_dim=3,
        freeze_deeplob=args.freeze_deeplob,
        lr=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        device=device
    )
    
    # =====================================
    # LOGGER
    # =====================================
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        log_interval=args.log_interval
    )
    logger.save_config(vars(args))
    
    # =====================================
    # TRAINING
    # =====================================
    print("\nStarting training...")
    
    episode_count = 0
    episode_rewards = []
    portfolio_values = []
    all_critic_losses = []
    all_actor_losses = []
    best_eval = float('-inf')
    start_step = 0
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
    
    # Training monitor CSV (written every eval period - comprehensive diagnostics)
    monitor_csv_path = os.path.join(save_dir, 'training_monitor.csv')
    if not os.path.exists(monitor_csv_path):
        with open(monitor_csv_path, 'w') as f:
            f.write('step,episode,timestamp,reward_avg100,portfolio_value,'
                    'total_trades,buys,sells,winning_sells,losing_sells,win_pct,'
                    'total_profit,total_loss,net_pnl,total_fees,'
                    'avg_buy_size,avg_sell_size,avg_profit_per_win,avg_loss_per_loss,'
                    'trade_freq,critic_loss,actor_loss,eval_composite_score,'
                    'eval_sharpe,eval_return,eval_drawdown\n')
    
    # Per-trade logger
    trade_logger = TradeLogger(save_dir)
    
    # Resume from checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            if 'agent_path' in ckpt:
                agent.load(ckpt['agent_path'])
            else:
                # Inline restore agent state
                if 'deeplob_state_dict' in ckpt:
                    agent.deeplob.load_state_dict(ckpt['deeplob_state_dict'])
                if 'portfolio_encoder_state_dict' in ckpt:
                    agent.portfolio_encoder.load_state_dict(ckpt['portfolio_encoder_state_dict'])
                if 'actor_state_dict' in ckpt:
                    agent.actor.load_state_dict(ckpt['actor_state_dict'])
                if 'critic_state_dict' in ckpt:
                    agent.critic.load_state_dict(ckpt['critic_state_dict'])
                if 'critic_target_state_dict' in ckpt:
                    agent.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
                if 'alpha' in ckpt:
                    agent.alpha = ckpt['alpha']
                if 'total_steps' in ckpt:
                    agent.total_steps = ckpt['total_steps']
            start_step = ckpt.get('step', 0) + 1
            episode_count = ckpt.get('episode_count', 0)
            episode_rewards = ckpt.get('episode_rewards', [])
            portfolio_values = ckpt.get('portfolio_values', [])
            all_critic_losses = ckpt.get('all_critic_losses', [])
            all_actor_losses = ckpt.get('all_actor_losses', [])
            best_eval = ckpt.get('best_eval', float('-inf'))
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
            print(f"  Continuing from step {start_step:,}, episodes: {episode_count}, best eval: {best_eval:.2f}")
            print(f"  Total gained: ${total_money_gained:,.2f}, Total lost: ${total_money_lost:,.2f}, Fees: ${total_fees_paid:,.2f}")
            print(f"  Restored {len(saved_trades)} trades from checkpoint")
        else:
            print(f"Warning: Checkpoint not found: {args.resume}")
    
    # Pause signal handler
    _pause_requested = False
    _original_sigint = signal.getsignal(signal.SIGINT)
    
    def _signal_handler(signum, frame):
        nonlocal _pause_requested
        if _pause_requested:
            print("\nForce quit!")
            sys.exit(1)
        _pause_requested = True
        print("\n[PAUSE] Pause requested! Saving checkpoint after current step...")
    
    signal.signal(signal.SIGINT, _signal_handler)
    
    actual_max_steps = args.total_steps

    training_start = time.time()
    print(f"\nStarting training... (total_steps: {args.total_steps:,}, from step {start_step:,})")
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

    update_count = 0
    step = start_step
    try:
        for step in range(start_step, actual_max_steps, num_envs):
            # Batched action selection
            actions = agent.select_actions_batch(obs_list)
            
            # Step all environments
            next_obs_list, rewards, dones, infos = train_vec_env.step(actions.tolist())
            
            # Store N transitions
            agent.store_transitions_batch(obs_list, actions, rewards, next_obs_list, dones)
            
            # Per-env tracking
            for i in range(num_envs):
                info = infos[i]
                trade_info = info.get('trade_info', {})
                if trade_info and trade_info.get('executed', False):
                    trade_type = trade_info.get('type', 'unknown')
                    ep_fees_env[i] += trade_info.get('fee', 0.0)
                    
                    if trade_type == 'buy':
                        ep_buy_counts[i] += 1
                    elif trade_type == 'sell':
                        ep_sell_counts[i] += 1
                        profit = trade_info.get('profit', 0)
                        if profit > 0:
                            total_money_gained += profit
                        else:
                            total_money_lost += abs(profit)
                    total_fees_paid += trade_info.get('fee', 0.0)
                    
                    trade_logger.log_trade(
                        episode=episode_count,
                        step=step + i,
                        trade_info=trade_info,
                        balance_before=prev_balances[i],
                        balance_after=info.get('balance', prev_balances[i]),
                        btc_held_before=prev_btc_helds[i],
                        btc_held_after=info.get('btc_held', prev_btc_helds[i]),
                        portfolio_value=info.get('portfolio_value', args.initial_balance),
                    )
                else:
                    ep_hold_counts[i] += 1
                
                prev_balances[i] = info.get('balance', prev_balances[i])
                prev_btc_helds[i] = info.get('btc_held', prev_btc_helds[i])
                ep_rewards[i] += rewards[i]
                ep_lengths[i] += 1
                
                # Episode end for env i
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

                    logger.log_episode(
                        episode=episode_count,
                        total_reward=ep_rewards[i],
                        episode_length=ep_lengths[i],
                        info={'portfolio': pv}
                    )
                    episode_rewards.append(ep_rewards[i])
                    portfolio_values.append(pv)
                    win_rates.append(ep_win_rate)
                    trade_counts.append({'buy': ep_buy_counts[i], 'sell': ep_sell_counts[i], 'hold': ep_hold_counts[i]})
                    episode_returns.append(ep_return)
                    total_fees_list.append(ep_fees_env[i])
                    max_drawdowns.append(ep_drawdown)
                    episode_count += 1
                    
                    with open(episode_csv_path, 'a') as f:
                        f.write(f'{episode_count},{ep_rewards[i]:.4f},{pv:.2f},{ep_pnl:.2f},'
                                f'{ep_return*100:.4f},{ep_win_rate:.4f},{ep_trades},{ep_fees_env[i]:.4f},'
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
                            f"Fees: ${ep_fees_env[i]:.2f} | "
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
                    ep_fees_env[i] = 0.0
                    prev_balances[i] = args.initial_balance
                    prev_btc_helds[i] = 0.0
            
            obs_list = next_obs_list
            
            # SAC update (already batched from replay buffer)
            if agent.total_steps >= args.batch_size:
                losses = agent.update(args.batch_size)
                if losses:
                    all_critic_losses.append(losses.get('critic_loss', 0))
                    all_actor_losses.append(losses.get('actor_loss', 0))
                    update_count += 1
            
            if step % args.log_interval < num_envs:
                elapsed = time.time() - training_start
                steps_done = step - start_step + num_envs
                steps_per_sec = steps_done / max(elapsed, 1)
                progress = (step + num_envs) / actual_max_steps * 100
                remaining = (actual_max_steps - step - num_envs) / max(steps_per_sec, 0.01)
                
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                avg_pv = np.mean(portfolio_values[-100:]) if portfolio_values else args.initial_balance
                avg_return = np.mean(episode_returns[-100:]) * 100 if episode_returns else 0
                print(
                    f"Step {step+num_envs:>8,}/{actual_max_steps:,} ({progress:5.1f}%) | "
                    f"{steps_per_sec:.0f} steps/s | "
                    f"Ep: {episode_count} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg PV: ${avg_pv:,.0f} | "
                    f"Avg Return: {avg_return:+.2f}% | "
                    f"ETA: {remaining/60:.1f}min"
                )
            
            # Evaluation
            if update_count > 0 and update_count % args.eval_freq == 0:
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
                        eval_action = agent.select_action(eval_obs, deterministic=True)
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
                last_eval_sharpe = float(np.mean([i.get('sharpe_ratio', 0.0) for i in eval_infos])) if eval_infos else 0.0
                last_eval_return = float(np.mean([i.get('total_return', 0.0) for i in eval_infos])) if eval_infos else 0.0
                last_eval_drawdown = float(np.mean([i.get('max_drawdown', 0.0) for i in eval_infos])) if eval_infos else 0.0

                mean_eval = np.mean(eval_rewards)
                logger.log_evaluation(step, eval_rewards)
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
                    backup_name = f"best_model_step{step}_r{mean_eval:.1f}_{ts}.pt"
                    agent.save(os.path.join(backup_dir, backup_name))
                    print(f"[*] New best: {mean_eval:.2f}")
                    print(f"  [BACKUP] {backup_name}")

                # Save snapshot
                try:
                    snapshot_dir = os.path.join(save_dir, 'plots', f'step_{step:07d}')
                    os.makedirs(snapshot_dir, exist_ok=True)
                    avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    net_pnl = total_money_gained - total_money_lost - total_fees_paid
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
                        pf.write(f"Sequence length:   {sequence_length}\n")
                        pf.write(f"Total steps:       {args.total_steps:,}\n")
                        pf.write(f"Initial balance:   {args.initial_balance}\n")
                        pf.write(f"Transaction fee:   {args.transaction_fee}\n")
                        pf.write(f"DeepLOB model:     {args.deeplob_model}\n")
                        pf.write(f"Freeze DeepLOB:    {args.freeze_deeplob}\n")
                        pf.write(f"Seed:              {args.seed}\n")
                        pf.write(f"\n--- Training Progress ---\n")
                        pf.write(f"Current step:      {step:,}\n")
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

                # Write training monitor CSV (at eval time only)
                try:
                    ts = trade_logger.get_summary()
                    trade_freq = ts['total_trades'] / max(step, 1)
                    last_cl = all_critic_losses[-1] if all_critic_losses else 0
                    last_al = all_actor_losses[-1] if all_actor_losses else 0
                    avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    pv = portfolio_values[-1] if portfolio_values else args.initial_balance
                    import datetime as _dt
                    with open(monitor_csv_path, 'a') as f:
                        f.write(f"{step},{episode_count},{_dt.datetime.now().isoformat()},"
                                f"{avg_r:.4f},{pv:.2f},"
                                f"{ts['total_trades']},{ts['total_buys']},{ts['total_sells']},"
                                f"{ts['winning_sells']},{ts['losing_sells']},{ts['win_rate']*100:.2f},"
                                f"{ts['total_profit']:.2f},{ts['total_loss']:.2f},{ts['net_pnl']:.2f},{ts['total_fees']:.2f},"
                                f"{ts['avg_buy_size_usd']:.2f},{ts['avg_sell_size_usd']:.2f},"
                                f"{ts['avg_profit_per_win']:.2f},{ts['avg_loss_per_loss']:.2f},"
                                f"{trade_freq:.6f},{last_cl:.6f},{last_al:.6f},{last_eval_composite:.6f},"
                                f"{last_eval_sharpe:.6f},{last_eval_return:.6f},{last_eval_drawdown:.6f}\n")
                except Exception as e:
                    print(f"  [WARN] Could not write monitor CSV: {e}")

            # Checkpoint
            if step > 0 and step % args.save_freq == 0:
                _save_sac_deeplob_checkpoint(
                    agent, step, episode_count, episode_rewards,
                    portfolio_values, all_critic_losses, all_actor_losses,
                    best_eval, resume_ckpt_path,
                    win_rates=win_rates, trade_counts=trade_counts,
                    episode_returns=episode_returns, total_fees_list=total_fees_list,
                    total_money_lost=total_money_lost, total_money_gained=total_money_gained,
                    total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
                    max_drawdowns=max_drawdowns
                )
                print(f"  [SAVE] Checkpoint saved at step {step:,}")
            
            # Check pause
            if _pause_requested:
                _save_sac_deeplob_checkpoint(
                    agent, step, episode_count, episode_rewards,
                    portfolio_values, all_critic_losses, all_actor_losses,
                    best_eval, resume_ckpt_path,
                    win_rates=win_rates, trade_counts=trade_counts,
                    episode_returns=episode_returns, total_fees_list=total_fees_list,
                    total_money_lost=total_money_lost, total_money_gained=total_money_gained,
                    total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
                    max_drawdowns=max_drawdowns
                )
                print(f"\n[PAUSE] Training paused at step {step:,}")
                print(f"  Resume: python train_sac_with_deeplob.py --deeplob_model {args.deeplob_model} --resume {resume_ckpt_path} [other args]")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted - saving checkpoint...")
        _save_sac_deeplob_checkpoint(
            agent, step, episode_count, episode_rewards,
            portfolio_values, all_critic_losses, all_actor_losses,
            best_eval, resume_ckpt_path,
            win_rates=win_rates, trade_counts=trade_counts,
            episode_returns=episode_returns, total_fees_list=total_fees_list,
            total_money_lost=total_money_lost, total_money_gained=total_money_gained,
            total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades(),
            max_drawdowns=max_drawdowns
        )
    
    finally:
        _save_sac_deeplob_checkpoint(
            agent, step, episode_count, episode_rewards,
            portfolio_values, all_critic_losses, all_actor_losses,
            best_eval, os.path.join(save_dir, 'final_checkpoint.pt'),
            win_rates=win_rates, trade_counts=trade_counts,
            episode_returns=episode_returns, total_fees_list=total_fees_list,
            total_money_lost=total_money_lost, total_money_gained=total_money_gained,
            total_fees_paid=total_fees_paid, trade_history=trade_logger.get_trades()
        )
        agent.save(os.path.join(save_dir, 'final_model.pt'))
        signal.signal(signal.SIGINT, _original_sigint)
        logger.close()
        
        # Generate final plots
        try:
            final_dir = os.path.join(save_dir, 'plots', 'final')
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
                pf.write(f"Sequence length:   {sequence_length}\n")
                pf.write(f"Total steps:       {args.total_steps:,}\n")
                pf.write(f"Initial balance:   {args.initial_balance}\n")
                pf.write(f"Transaction fee:   {args.transaction_fee}\n")
                pf.write(f"DeepLOB model:     {args.deeplob_model}\n")
                pf.write(f"Freeze DeepLOB:    {args.freeze_deeplob}\n")
                pf.write(f"Seed:              {args.seed}\n")
                pf.write(f"\n--- Final Results ---\n")
                pf.write(f"Total steps:       {step:,}\n")
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
        
        elapsed = time.time() - training_start
        gross_pnl_final = total_money_gained - total_money_lost
        net_pnl_final = gross_pnl_final - total_fees_paid
        print(f"\n{'='*60}")
        print(f"Training completed! Total time: {elapsed/60:.1f}min")
        print(f"  Steps: {step:,} | Episodes: {episode_count}")
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
                        '--algo', 'sac_deeplob',
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


def _save_sac_deeplob_checkpoint(agent, step, episode_count, episode_rewards,
                                 portfolio_values, all_critic_losses, all_actor_losses,
                                 best_eval, path,
                                 win_rates=None, trade_counts=None,
                                 episode_returns=None, total_fees_list=None,
                                 total_money_lost=0.0, total_money_gained=0.0,
                                 total_fees_paid=0.0, trade_history=None,
                                 max_drawdowns=None):
    """Save a checkpoint with all state needed for resuming."""
    ckpt = {
        'step': step,
        'episode_count': episode_count,
        'episode_rewards': episode_rewards,
        'portfolio_values': portfolio_values,
        'all_critic_losses': all_critic_losses,
        'all_actor_losses': all_actor_losses,
        'best_eval': best_eval,
        'win_rates': win_rates or [],
        'trade_counts': trade_counts or [],
        'episode_returns': episode_returns or [],
        'total_fees_list': total_fees_list or [],
        'total_money_lost': total_money_lost,
        'total_money_gained': total_money_gained,
        'total_fees_paid': total_fees_paid,
        'trade_history': trade_history or [],
        'max_drawdowns': max_drawdowns or [],
        'deeplob_state_dict': agent.deeplob.state_dict(),
        'portfolio_encoder_state_dict': agent.portfolio_encoder.state_dict(),
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'alpha': agent.alpha,
        'total_steps': agent.total_steps,
    }
    torch.save(ckpt, path)


if __name__ == '__main__':
    main()
