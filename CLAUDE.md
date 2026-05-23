# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataDeepRL is a Deep Reinforcement Learning framework for BTC/USDT cryptocurrency trading using Binance Level 2 (L2) order book data. It combines a DeepLOB (CNN+LSTM) feature extractor with PPO and SAC RL agents.

## Setup

```bash
pip install -r requirements.txt
```

## Common Commands

### Data Pipeline (run in order)
```bash
python binance_l2.py                          # Download Binance L2 data → btc_l2_data/
python dataVerwerken/preprocess_data.py       # Feature engineering + normalization → DataNorm/
python dataVerwerken/create_core_data.py      # Train/val/test split (70/15/15) → coreData/
```

### Training
```bash
# Step 1: Pre-train DeepLOB (supervised, price direction prediction)
python train/train_deeplob_pretrain.py --data_dir btc_l2_data --epochs 50

# Step 2: Train RL agent with frozen pre-trained DeepLOB (recommended)
python train/train_ppo_with_deeplob.py --deeplob_model models/deeplob_pretrained.pt --freeze_deeplob --total_steps 500000
python train/train_sac_with_deeplob.py --deeplob_model models/deeplob_pretrained.pt --freeze_deeplob --total_steps 500000

# Baseline (no DeepLOB, MLP only)
python train/train_ppo_only.py --total_steps 100000
python train/train_sac_only.py --total_steps 100000
```

### Monitoring
```bash
tensorboard --logdir logs/
mlflow ui --backend-store-uri logs/mlruns
```

## Architecture

### Data Flow
```
Binance L2 parquet → BTCDataLoader → CryptoTradingEnv → RL Agent → Action → Reward
```

### Key Components

**`src/data/dataloader.py` — BTCDataLoader**
- Loads parquet files, engineers features (returns, SMA/EMA, RSI, MACD, ATR, bid/ask spread, order flow imbalance)
- Applies StandardScaler or MinMaxScaler normalization
- Chronological train/val/test split

**`src/envs/trading_env.py` — CryptoTradingEnv**
- Gymnasium-compatible environment
- Discrete actions: Hold=0, Buy=1, Sell=2 (or continuous -1 to +1)
- Dict observation: `features` (window of market data) + `portfolio` state
- Reward: portfolio PnL change minus transaction fee penalty; extra penalty when drawdown > 10%
- Episode ends at data exhaustion; bankruptcy (balance < 10% of initial) resets balance to initial_balance

**`src/models/deeplob.py` — DeepLOB**
- Feature extractor implementing Zhang et al. (2019) "DeepLOB" paper
- Pipeline: Conv1D blocks → Inception module (1×1, 3×3, 5×5 kernels) → BiLSTM → Attention pooling → FC output
- Input: `(batch, window_size, input_dim)` e.g. `(32, 100, 20)` → Output: `(batch, 64)`
- Used as a frozen backbone after supervised pre-training

**`src/models/ppo.py` — PPO Agent**
- On-policy; uses RolloutBuffer, GAE, clipped objective
- Policy network outputs Categorical distribution over discrete actions
- Default: lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2

**`src/models/sac.py` — SAC Agent**
- Off-policy; uses ReplayBuffer (1M), twin Q-networks, auto-tuned temperature α
- Actor outputs Normal distribution (stochastic policy)
- Default: lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2

**`src/models/mlp.py`** — Baseline MLP feature extractor (linear + LayerNorm + LeakyReLU), used in `*_only` training scripts.

**`src/utils/`**
- `logger.py`: TrainingLogger with TensorBoard, MLflow, and CSV export
- `callbacks.py`: CheckpointCallback, EvalCallback, EarlyStoppingCallback, LearningRateScheduler, ProgressCallback
- `mixed_precision.py`: AMP (Automatic Mixed Precision) support
- `trade_logger.py`: Per-trade metrics logging

### Training Paradigms
1. **Supervised pre-training**: DeepLOB trained on price-direction labels
2. **Transfer learning (recommended)**: Frozen pre-trained DeepLOB as backbone for RL agent
3. **End-to-end RL**: MLP baseline without DeepLOB (faster iteration, lower performance)

### Configuration
`configs/default.yaml` — Central hyperparameter config:
- `window_size: 100`, `sequence_length: 100`
- `initial_balance: 100000`, `transaction_fee: 0.0`, `flat_fee: 1.0`
- `total_steps: 1000000`, `device: auto`
- `eval_freq: 10000`, `log_dir: ./logs`
