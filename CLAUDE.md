# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

DataDeepRL is a Deep RL framework for BTC/USDT trading using Binance L2 order
book data. It combines a DeepLOB (CNN + Inception + BiLSTM) feature extractor
with PPO and SAC agents.

## Setup

```bash
pip install -r requirements.txt
```

## Common commands

### Data pipeline (run once per dataset)

```bash
python binance_l2.py                          # raw download → btc_l2_data/
python dataVerwerken/preprocess_data.py       # features + z-score → DataNorm/
python dataVerwerken/create_core_data.py      # 80/10/10 split → coreData/
```

After `create_core_data.py`, only `coreData/` is needed for training.
`DataNorm/` is intermediate scratch and can be deleted.

### Training (all use coreData/)

```bash
# Step 1: Pretrain DeepLOB (supervised, price direction)
python -m train.train_deeplob_pretrain --max_rows 20000000 --epochs 30

# Step 2: RL agent with frozen pretrained DeepLOB (recommended)
python -m train.train_ppo_with_deeplob --deeplob_model ./models/deeplob_pretrained.pt --freeze_deeplob
python -m train.train_sac_with_deeplob --deeplob_model ./models/deeplob_pretrained.pt --freeze_deeplob

# MLP baseline (no DeepLOB)
python -m train.train_ppo_only
python -m train.train_sac_only
```

### Monitoring & evaluation

```bash
tensorboard --logdir ./logs
python evaluate.py --model_path logs/<run>/best_model.pt --algo ppo_deeplob \
                   --deeplob_model models/deeplob_pretrained.pt --split test
```

## Architecture

### Data flow

```
coreData/{train,val,test}.parquet
   → load_coredata_streaming()  (raw features + denormalised prices)
   → CryptoTradingEnv           (sliding windows on-the-fly, zero-copy)
   → RL agent                   (PPO or SAC, optionally with DeepLOB backbone)
```

Training data is **pre-normalized** in `coreData/`. The env streams raw
features (N × num_features) and creates windowed sequences on demand via
numpy slicing — avoids materialising a (N × seq_len × num_features) array.
Prices are denormalized back to USD by the loader (using `normalization_stats.json`).

### Key components

**`src/envs/trading_env.py` — CryptoTradingEnv**
- Gymnasium-compatible. Dict obs (`features`, `portfolio`).
- Actions: Hold = 0, Buy = 1, Sell = 2 (discrete) or Box(-1, 1) continuous.
- Reward: portfolio PnL change minus fees, extra penalty when drawdown > 15%.
- `FlatCryptoTradingEnv` is the same env with a flat observation for MLP agents.

**`src/envs/vec_env.py` — VectorizedTradingEnv**
- Runs N parallel envs in subprocesses for sample throughput.

**`src/models/deeplob.py` — DeepLOB**
- Zhang et al. (2019) DeepLOB: Conv1D → Inception (1×1, 3×3, 5×5) → BiLSTM
  → attention pooling. Used as frozen backbone after supervised pretraining.

**`src/models/ppo.py` — PPOAgent, `src/models/sac.py` — SACAgent**
- MLP-based agents (used by `train_*_only.py`).

**`train/train_ppo_with_deeplob.py` and `train_sac_with_deeplob.py`**
- Contain their own `PPOWithPretrainedDeepLOB` / `SACWithPretrainedDeepLOB`
  classes (PPO/SAC algorithm + DeepLOB backbone + portfolio encoder).
- Core algorithm code overlaps with `src/models/{ppo,sac}.py` but the two
  variants are intentionally kept separate.

**`train/common/setup.py`**
- `load_coredata_streaming()` — memory-efficient data loader.
- `STATIONARY_FEATURES` — list of feature columns used as model input.

## Hyperparameters

Edited inline in each training script's `parse_args()` defaults. The model
classes (e.g. `PPOAgent.__init__`) also carry sensible defaults. There is no
external tuning framework — tune by editing the script you are running.

## Notes

- Always reach training via `python -m train.<script>` (module form) so
  `from train.common.setup import ...` resolves.
- After `coreData/` exists, you do not need `btc_l2_data/` or `DataNorm/` for
  training.
- `normalization_stats.json` in `coreData/` is required at training time —
  the env uses it to denormalize the price column back to USD.
