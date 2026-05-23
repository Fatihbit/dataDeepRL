# DataDeepRL Architectuur Documentatie

## Overzicht

DataDeepRL is een reinforcement learning framework voor cryptocurrency trading met Bitcoin (BTC) order book data. Het project combineert deep learning feature extraction (DeepLOB) met state-of-the-art RL algoritmes (PPO en SAC).

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATAFLOW ARCHITECTUUR                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Binance L2   │───>│  DataLoader  │───>│  Trading Env     │  │
│  │  Order Book   │    │  (Sequences) │    │  (Gymnasium)     │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                   │             │
│                                                   ▼             │
│                           ┌─────────────────────────────────┐   │
│                           │         RL AGENT                │   │
│                           │  ┌─────────────────────────┐    │   │
│                           │  │      DeepLOB            │    │   │
│                           │  │  (Feature Extractor)    │    │   │
│                           │  │  CNN + LSTM + Attention │    │   │
│                           │  └────────────┬────────────┘    │   │
│                           │               │                  │   │
│                           │               ▼                  │   │
│                           │  ┌─────────────────────────┐    │   │
│                           │  │   PPO / SAC Networks    │    │   │
│                           │  │  Policy + Value/Critic  │    │   │
│                           │  └─────────────────────────┘    │   │
│                           └─────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structuur

```
dataDeepRL/
├── src/                        # Broncode
│   ├── data/                   # Data laden en verwerken
│   │   └── dataloader.py       # BTCDataLoader voor order book data
│   │
│   ├── envs/                   # Gymnasium environments
│   │   └── trading_env.py      # CryptoTradingEnv voor RL training
│   │
│   ├── models/                 # Neural network modellen
│   │   ├── deeplob.py          # DeepLOB feature extractor
│   │   ├── mlp.py              # MLP networks voor PPO/SAC
│   │   ├── ppo.py              # PPO agent implementatie
│   │   └── sac.py              # SAC agent implementatie
│   │
│   └── utils/                  # Utility modules
│       ├── callbacks.py        # Training callbacks
│       ├── logger.py           # Training logger
│       └── mixed_precision.py  # AMP training support
│
├── train/                      # Training scripts
│   ├── common/                 # Gedeelde training utilities
│   │   ├── args.py             # Argument parsers
│   │   └── setup.py            # Setup functies
│   │
│   ├── train_ppo_deeplob.py    # PPO + DeepLOB training
│   ├── train_sac_deeplob.py    # SAC + DeepLOB training
│   └── ...                     # Overige training scripts
│
├── configs/                    # YAML configuratie bestanden
│   └── default.yaml            # Default hyperparameters
│
├── btc_l2_data/               # Data directory (niet in git)
├── logs/                       # Training logs
├── checkpoints/                # Model checkpoints
│
├── tune_hyperparams.py         # Optuna hyperparameter tuning
└── requirements.txt            # Python dependencies
```

---

## Module Beschrijvingen

### 1. Data Module (`src/data/`)

#### BTCDataLoader (`dataloader.py`)

**Doel**: Laad en prepareer Binance L2 order book data voor training.

**Data Pipeline**:
```
Parquet Files ──> Load ──> Feature Engineering ──> Sequences ──> Train/Val/Test Split
```

**Belangrijke Features**:
- OHLCV data (Open, High, Low, Close, Volume)
- Bid/Ask prices en volumes
- Technische indicatoren (RSI, MACD, ATR, etc.)
- Order flow imbalance metrics

**Normalisatie**:
- StandardScaler: `(x - μ) / σ` - waarden rond 0
- MinMaxScaler: `(x - min) / (max - min)` - waarden 0-1

**Data Split**:
```
[========== 70% Train ==========][=== 15% Val ===][=== 15% Test ===]
                                   ↑
                      Chronologische split (geen shuffle!)
```

---

### 2. Environment Module (`src/envs/`)

#### CryptoTradingEnv (`trading_env.py`)

**Doel**: Gymnasium environment voor trading simulatie.

**State Space**:
```python
observation = {
    'features': np.array(shape=(window_size, num_features)),  # Market data
    'portfolio': np.array([balance_ratio, btc_ratio, unrealized_pnl, portfolio_ratio])
}
```

**Action Space**:
- Discrete (default): `{0: Hold, 1: Buy, 2: Sell}`
- Continue: `[-1.0, +1.0]` waar -1=sell, 0=hold, +1=buy

**Reward Functie**:
```python
reward = (portfolio_change_pct * 100 * scaling)    # Basis: PnL
       - (transaction_fee * 10)                     # Fee penalty
       - (drawdown_penalty if drawdown > 10%)       # Risk penalty
```

**Episode Terminatie**:
- Einde data (`truncated=True`)
- Failliet: portfolio < 10% van start (`terminated=True`)

---

### 3. Models Module (`src/models/`)

#### DeepLOB (`deeplob.py`)

**Doel**: Feature extraction uit order book sequences.

**Architectuur**:
```
Input: (batch, window_size, num_features)
              │
    ┌─────────▼─────────┐
    │   Conv1D Block 1   │   Filter: hidden_dim, kernel: 3
    │   Conv1D Block 2   │   BatchNorm + LeakyReLU
    │   MaxPool1d (2)    │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   Conv1D Block 3   │   Filter: hidden_dim * 2
    │   Conv1D Block 4   │
    │   MaxPool1d (2)    │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │  Inception Module  │   Multi-scale features (1x1, 3x3, 5x5)
    │  4 parallel branches│
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   Bidirectional    │   hidden: lstm_hidden
    │       LSTM         │   Captures temporal dependencies
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   Attention Layer  │   Weighted pooling over time
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   Output FC Layer  │   output_dim features
    └─────────┴─────────┘

Output: (batch, output_dim)
```

---

#### PPO Agent (`ppo.py`)

**Doel**: On-policy reinforcement learning met clipped surrogate objective.

**Algoritme Overzicht**:
```
1. Collect rollout (n_steps transitions)
2. Compute advantages met GAE
3. Multiple epochs van mini-batch updates
4. Clipped policy gradient update
5. Value function update
6. Reset buffer, repeat
```

**Kernconcepten**:

**GAE (Generalized Advantage Estimation)**:
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)     # TD error
A_t = Σ (γλ)^l δ_{t+l}                # GAE
```

**Clipped Surrogate Loss**:
```
r(θ) = π_θ(a|s) / π_θ_old(a|s)        # Probability ratio
L_CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
```

**Hyperparameters**:
| Parameter | Default | Beschrijving |
|-----------|---------|--------------|
| lr | 3e-4 | Learning rate |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE lambda |
| clip_range | 0.2 | PPO clip ε |
| value_coef | 0.5 | Value loss weight |
| entropy_coef | 0.01 | Entropy bonus |
| n_epochs | 10 | Updates per rollout |

---

#### SAC Agent (`sac.py`)

**Doel**: Off-policy maximum entropy RL.

**Algoritme Overzicht**:
```
1. Sample action met actor
2. Store transition in replay buffer
3. Sample batch from buffer
4. Update critics met TD target
5. Update actor met policy gradient
6. Update temperature α (auto-tune)
7. Soft update target networks
```

**Kernconcepten**:

**Maximum Entropy RL**:
```
J(π) = Σ E[r + γV(s') - α·log(π(a|s))]
```
Hogere α = meer exploratie door entropy bonus.

**Twin Q-Networks**:
```
Q_target = min(Q1, Q2) - α·log(π)    # Prevent overestimation
```

**Soft Update**:
```
θ_target = τ·θ + (1-τ)·θ_target      # Smooth target update
```

**Hyperparameters**:
| Parameter | Default | Beschrijving |
|-----------|---------|--------------|
| lr | 3e-4 | Learning rate |
| gamma | 0.99 | Discount factor |
| tau | 0.005 | Soft update coef |
| alpha | 0.2 | Initial temperature |
| auto_alpha | True | Auto-tune α |
| buffer_size | 1M | Replay buffer capacity |
| start_steps | 10000 | Random warmup steps |

---

### 4. Utils Module (`src/utils/`)

#### TrainingLogger (`logger.py`)

**Features**:
- Console logging met progress
- CSV metrics export
- TensorBoard integration
- MLflow experiment tracking (optioneel)

**Log Locaties**:
```
logs/{experiment_name}/
├── config.json           # Training configuratie
├── step_metrics.csv      # Per-step losses
├── episode_metrics.csv   # Per-episode rewards
├── tensorboard/          # TensorBoard logs
├── mlruns/               # MLflow tracking (indien enabled)
└── checkpoints/          # Model checkpoints
```

#### Callbacks (`callbacks.py`)

| Callback | Functie |
|----------|---------|
| CheckpointCallback | Periodiek model opslaan |
| EvalCallback | Evaluatie op validation env |
| EarlyStoppingCallback | Stop bij geen verbetering |
| LearningRateScheduler | LR schedule (linear/cosine) |
| ProgressCallback | Progress bar |
| PauseResumeCallback | Ctrl+C voor pause + checkpoint |

---

## Training Flow

### 1. PPO Training Loop

```
┌────────────────────────────────────────────────────────────────┐
│                     PPO TRAINING LOOP                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                               │
│  │  1. Reset   │                                               │
│  │   Rollout   │                                               │
│  │   Buffer    │                                               │
│  └──────┬──────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────┐                   │
│  │  2. Collect Rollout (n_steps)            │◄────┐            │
│  │     for step in n_steps:                 │     │            │
│  │       action = agent.select_action(obs)  │     │            │
│  │       obs', r, done = env.step(action)   │     │            │
│  │       buffer.add(obs, action, r, done)   │     │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    ▼                              │            │
│  ┌─────────────────────────────────────────┐     │            │
│  │  3. Compute Advantages (GAE)             │     │            │
│  │     returns, advantages = compute_gae()  │     │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    ▼                              │            │
│  ┌─────────────────────────────────────────┐     │            │
│  │  4. PPO Update (n_epochs)                │     │            │
│  │     for epoch in n_epochs:               │     │            │
│  │       for batch in batches:              │     │            │
│  │         compute_losses()                 │     │            │
│  │         optimizer.step()                 │     │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    ▼                              │            │
│  ┌─────────────────────────────────────────┐     │            │
│  │  5. Evaluation (periodiek)               │     │            │
│  │     eval_reward = evaluate(eval_env)     │     │            │
│  │     if best: save_model()                │     │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    └──────────────────────────────┘            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 2. SAC Training Loop

```
┌────────────────────────────────────────────────────────────────┐
│                     SAC TRAINING LOOP                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────┐                   │
│  │  1. Warmup Phase (start_steps)           │                  │
│  │     Random actions to fill buffer        │                  │
│  └─────────────────┬───────────────────────┘                   │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │  2. Collect Experience                   │◄────┐            │
│  │     action = agent.select_action(obs)    │     │            │
│  │     obs', r, done = env.step(action)     │     │            │
│  │     buffer.add(obs, action, r, obs', done)│    │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    ▼                              │            │
│  ┌─────────────────────────────────────────┐     │            │
│  │  3. Sample Batch                         │     │            │
│  │     batch = buffer.sample(batch_size)    │     │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    ▼                              │            │
│  ┌─────────────────────────────────────────┐     │            │
│  │  4. Update Critics                       │     │            │
│  │     target_Q = r + γ*(min(Q1',Q2')-α*logπ)│    │            │
│  │     critic_loss = MSE(Q, target_Q)       │     │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    ▼                              │            │
│  ┌─────────────────────────────────────────┐     │            │
│  │  5. Update Actor                         │     │            │
│  │     actor_loss = α*logπ - Q1(s, π(s))    │     │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    ▼                              │            │
│  ┌─────────────────────────────────────────┐     │            │
│  │  6. Update Temperature (if auto_alpha)   │     │            │
│  │     α_loss = -α*(logπ + target_entropy)  │     │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    ▼                              │            │
│  ┌─────────────────────────────────────────┐     │            │
│  │  7. Soft Update Targets                  │     │            │
│  │     θ_target = τ*θ + (1-τ)*θ_target      │     │            │
│  └─────────────────┬───────────────────────┘     │            │
│                    │                              │            │
│                    └──────────────────────────────┘            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Hyperparameter Tuning

### Optuna Integratie (`tune_hyperparams.py`)

**Ondersteunde Parameters**:
```python
{
    'learning_rate': [1e-5, 1e-3],      # Log uniform
    'gamma': [0.9, 0.9999],             # Uniform
    'batch_size': [32, 64, 128, 256],   # Categorical
    'hidden_dims': [(64,64), (128,128), (256,256)],
    'entropy_coef': [0.001, 0.1],       # Log uniform (PPO)
    'tau': [0.001, 0.01],               # Uniform (SAC)
}
```

**Pruning**:
Optuna stopt slecht presterende trials vroeg via MedianPruner.

**Gebruik**:
```bash
python tune_hyperparams.py --algo ppo --n_trials 100 --n_jobs 4
```

---

## Experiment Tracking

### MLflow (Optioneel)

**Enable**:
```python
logger = TrainingLogger(..., use_mlflow=True)
```

**Start UI**:
```bash
mlflow ui --backend-store-uri file:///path/to/logs/mlruns
```

**Getrackte Metrics**:
- Training losses (actor, critic, value)
- Episode rewards
- Evaluation metrics
- Hyperparameters

### TensorBoard

**Start**:
```bash
tensorboard --logdir logs/{experiment}/tensorboard
```

**Beschikbare Plots**:
- Loss curves
- Reward trajectories
- Learning rate
- Episode lengths

---

## ONNX Export

**Voor Production Inference**:
```python
agent.export_onnx('model.onnx')

# Later in productie:
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
action_probs = session.run(None, {'observation': obs})
```

---

## Best Practices

### 1. Training

1. **Start met kleine dataset** om bugs te vinden
2. **Monitor TensorBoard** voor training curves
3. **Gebruik checkpoints** voor lange training runs
4. **Evalueer regelmatig** op validation set
5. **Track experiments** met MLflow

### 2. Hyperparameters

1. **Learning rate**: Start met 3e-4, tune indien nodig
2. **Batch size**: Groter = stabieler, maar langzamer
3. **Buffer size (SAC)**: Meer = beter off-policy learning
4. **Entropy coef (PPO)**: Hogere waarde = meer exploratie

### 3. Debugging

1. **Check rewards**: Moeten over tijd stijgen
2. **Monitor losses**: Value loss moet dalen
3. **Check action distribution**: Niet te deterministisch vroeg in training
4. **Visualize trades**: Kijk of agent rationeel handelt

---

## Troubleshooting

| Probleem | Mogelijke Oorzaak | Oplossing |
|----------|-------------------|-----------|
| Reward stijgt niet | LR te hoog/laag | Tune learning rate |
| Value loss explodeert | Gradients te groot | Verlaag max_grad_norm |
| Agent doet niets | Entropy te laag | Verhoog entropy_coef |
| Out of memory | Batch/buffer te groot | Verklein batch_size |
| Training onstabiel | Clip range te groot | Verlaag clip_range |

---

## Referenties

- [DeepLOB Paper](https://arxiv.org/abs/1905.05514) - Zhang et al., 2019
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [SAC Paper](https://arxiv.org/abs/1812.05905) - Haarnoja et al., 2018
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
