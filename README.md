# DataDeepRL - Cryptocurrency Trading met Deep Reinforcement Learning

Dit project implementeert deep reinforcement learning agents voor cryptocurrency trading met BTC/USDT data.

## 📁 Project Structuur

```
dataDeepRL/
├── btc_l2_data/                 # BTC Level 2 order book data
├── models/                      # Opgeslagen modellen
├── logs/                        # Training logs en TensorBoard
├── src/
│   ├── data/
│   │   └── dataloader.py        # Data loading en feature engineering
│   ├── envs/
│   │   └── trading_env.py       # Gymnasium trading environment
│   ├── models/
│   │   ├── deeplob.py           # DeepLOB CNN+LSTM architectuur
│   │   ├── mlp.py               # MLP feature extractors
│   │   ├── sac.py               # Soft Actor-Critic
│   │   └── ppo.py               # Proximal Policy Optimization
│   └── utils/
│       ├── logger.py            # Logging en TensorBoard
│       └── callbacks.py         # Training callbacks
│
├── train/
│   ├── train_deeplob_pretrain.py # 1️⃣ Pre-train DeepLOB (supervised)
│   ├── train_sac_with_deeplob.py  # 2️⃣ SAC met pre-trained DeepLOB
│   ├── train_ppo_with_deeplob.py  # 2️⃣ PPO met pre-trained DeepLOB
│   ├── train_sac_only.py          # SAC zonder DeepLOB (MLP only)
│   ├── train_ppo_only.py          # PPO zonder DeepLOB (MLP only)
│   ├── train_sac_deeplob.py       # SAC + DeepLOB (end-to-end)
│   └── train_ppo_deeplob.py       # PPO + DeepLOB (end-to-end)
│
├── dataVerwerken/                   # Data preprocessing scripts
│   ├── preprocess_data.py
│   ├── create_core_data.py
│   └── parquet_to_csv.py
│
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installeer Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download/Prepareer Data

```bash
python binance_l2.py
```

## 🧠 Training Workflow

### Optie A: DeepLOB + RL (Aanbevolen)

**Stap 1: Pre-train DeepLOB** (leert order book patterns herkennen)
```bash
python train/train_deeplob_pretrain.py --data_dir ./btc_l2_data --epochs 50
```
Dit traint DeepLOB supervised op price direction prediction en slaat het model op in `./models/deeplob_pretrained.pt`.

**Stap 2: Train SAC of PPO met pre-trained DeepLOB**
```bash
# SAC met pre-trained DeepLOB (fine-tune)
python train/train_sac_with_deeplob.py --deeplob_model ./models/deeplob_pretrained.pt

# Of PPO met pre-trained DeepLOB
python train/train_ppo_with_deeplob.py --deeplob_model ./models/deeplob_pretrained.pt

# Optioneel: freeze DeepLOB weights (snellere training)
python train/train_sac_with_deeplob.py --deeplob_model ./models/deeplob_pretrained.pt --freeze_deeplob
```

### Optie B: Alleen RL (Simpeler, Sneller)

Train SAC of PPO direct met MLP feature extraction:
```bash
# SAC alleen
python train/train_sac_only.py --data_dir ./btc_l2_data --total_steps 500000

# PPO alleen
python train/train_ppo_only.py --data_dir ./btc_l2_data --total_steps 500000
```

### 3. Monitor Training

```bash
tensorboard --logdir ./logs
```
Open http://localhost:6006

## 🧠 Model Architecturen

### DeepLOB (CNN + LSTM)

DeepLOB is gebaseerd op het paper ["DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"](https://arxiv.org/abs/1808.03668).

```
Input (sequence_length, features)
    │
    ▼
Conv1D Blocks (feature extraction)
    │
    ▼
Inception Module (multi-scale patterns)
    │
    ▼
Bidirectional LSTM (temporal dependencies)
    │
    ▼
Attention Pooling
    │
    ▼
Output Features
```

### SAC (Soft Actor-Critic)

Off-policy RL met maximum entropy:
- **Actor**: Leert stochastisch beleid met entropy bonus
- **Critic**: Twin Q-networks voor stabiele value estimation
- **Automatic Temperature**: Automatische α (entropy coefficient) tuning

Voordelen:
- Sample efficient (replay buffer)
- Exploration via entropy
- Stabiele training

### PPO (Proximal Policy Optimization)

On-policy RL met clipped surrogate:
- **Policy**: Categorische distributie voor discrete acties
- **Value Network**: Baseline voor advantage estimation
- **GAE**: Generalized Advantage Estimation

Voordelen:
- Stabiel en robuust
- Werkt goed met weinig hyperparameter tuning
- Goede performance op diverse taken

## ⚙️ Hyperparameters

### SAC Defaults
| Parameter | Waarde | Beschrijving |
|-----------|--------|--------------|
| learning_rate | 3e-4 | Learning rate |
| gamma | 0.99 | Discount factor |
| tau | 0.005 | Soft update coefficient |
| alpha | 0.2 | Entropy coefficient |
| batch_size | 256 | Batch size |

### PPO Defaults
| Parameter | Waarde | Beschrijving |
|-----------|--------|--------------|
| learning_rate | 3e-4 | Learning rate |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE lambda |
| clip_epsilon | 0.2 | PPO clipping |
| n_epochs | 10 | Epochs per update |
| n_steps | 2048 | Steps per rollout |

## 📊 Trading Environment

De trading environment ondersteunt:

- **Acties**: Buy (0), Sell (1), Hold (2)
- **Observaties**: 
  - Order book data (bid/ask prices & volumes)
  - Technical indicators (RSI, MACD, SMA, EMA, ATR)
  - Portfolio state (position, balance)

- **Reward**: 
  - Profit/loss van trades
  - Sharpe ratio component
  - Transaction cost penalty

## 📈 Features

- **Data Loading**: Automatisch laden van Parquet files met ZSTD compression
- **Feature Engineering**: 
  - Returns, log returns
  - SMA, EMA (meerdere periodes)
  - RSI, MACD, ATR
  - Order flow imbalance
- **Logging**: 
  - TensorBoard integratie
  - CSV exports
  - Checkpoints
- **Callbacks**:
  - Early stopping
  - Learning rate scheduling
  - Model checkpointing

## 🔧 Command Line Arguments

Alle training scripts ondersteunen dezelfde basis arguments:

```bash
python train/train_sac_deeplob.py \
    --data_dir ./btc_l2_data \    # Data directory
    --total_steps 1000000 \        # Totaal training steps
    --batch_size 256 \             # Batch size
    --learning_rate 3e-4 \         # Learning rate
    --gamma 0.99 \                 # Discount factor
    --initial_balance 10000 \      # Start kapitaal
    --transaction_fee 0.001 \      # Transactie kosten (0.1%)
    --eval_freq 10000 \            # Evaluatie frequentie
    --log_dir ./logs \             # Log directory
    --device auto \                # cuda/cpu/auto
    --seed 42                      # Random seed
```

## 📝 Licentie

MIT License

## 🤝 Contributing

Pull requests zijn welkom! Voor grote wijzigingen, open eerst een issue om te bespreken.
