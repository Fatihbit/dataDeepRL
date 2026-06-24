# DataDeepRL — Crypto Trading met Deep RL

Deep reinforcement learning agents (PPO / SAC) voor BTC/USDT trading op basis
van Binance L2 order-book data, met een optionele DeepLOB feature extractor.

## Projectstructuur

```
dataDeepRL/
├── btc_l2_data/                Ruwe Binance L2 parquet bestanden (per dag)
├── coreData/                   Genormaliseerde + gesplitste train/val/test data
├── models/                     Opgeslagen modellen (DeepLOB pretrained etc.)
├── logs/                       TensorBoard / training runs
│
├── src/
│   ├── envs/
│   │   ├── trading_env.py      Gymnasium trading environment
│   │   └── vec_env.py          Vectorized env wrapper
│   ├── models/
│   │   ├── deeplob.py          DeepLOB (CNN + Inception + BiLSTM + attention)
│   │   ├── mlp.py              MLP feature extractors
│   │   ├── ppo.py              PPO agent (MLP)
│   │   └── sac.py              SAC agent (MLP)
│   └── utils/                  Logger, callbacks, trade logger, mixed precision
│
├── train/
│   ├── common/setup.py         load_coredata_streaming + STATIONARY_FEATURES
│   ├── train_deeplob_pretrain.py
│   ├── train_ppo_only.py        PPO + MLP
│   ├── train_sac_only.py        SAC + MLP
│   ├── train_ppo_with_deeplob.py PPO + pretrained DeepLOB
│   └── train_sac_with_deeplob.py SAC + pretrained DeepLOB
│
├── dataVerwerken/              Data preprocessing pipeline
│   ├── preprocess_data.py      Feature engineering + z-score normalisatie
│   └── create_core_data.py     80/10/10 train/val/test split
│
├── binance_l2.py               Download script voor Binance aggTrades
├── evaluate.py                 Evalueer getrainde modellen op val/test
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data pipeline

Drie stappen, eenmalig per dataset:

```bash
# 1. Download ruwe data van Binance (pas START_DATE / END_DATE aan in script)
python binance_l2.py

# 2. Feature engineering + z-score normalisatie
python dataVerwerken/preprocess_data.py

# 3. Train/val/test split (80/10/10)
python dataVerwerken/create_core_data.py
```

Na stap 3 staat alle benodigde data in `coreData/`:
`train.parquet`, `val.parquet`, `test.parquet`, `normalization_stats.json`.
De tussenproducten in `DataNorm/` kun je daarna weggooien.

## Training

### Stap 1 — Pretrain DeepLOB (supervised)

```bash
python -m train.train_deeplob_pretrain --max_rows 20000000 --epochs 30
```

Schrijft model naar `models/deeplob_pretrained.pt`.

### Stap 2 — RL agent met DeepLOB backbone (aanbevolen)

```bash
python -m train.train_ppo_with_deeplob --deeplob_model ./models/deeplob_pretrained.pt --freeze_deeplob
python -m train.train_sac_with_deeplob --deeplob_model ./models/deeplob_pretrained.pt --freeze_deeplob
```

### Baseline — RL agent zonder DeepLOB

```bash
python -m train.train_ppo_only
python -m train.train_sac_only
```

### Hervatten

Alle training scripts ondersteunen `--resume <pad/naar/resume_checkpoint.pt>`.

## Hyperparameters

Worden bewerkt in de training scripts zelf. Elk script heeft een
`parse_args()` block met `--learning_rate`, `--gamma`, `--batch_size` etc.;
die defaults zijn de "ingebakken" hyperparameters. CLI-overrides werken
voor losse experimenten.

## Evaluatie

```bash
python evaluate.py --model_path ./logs/<run>/best_model.pt --algo ppo_deeplob \
                   --deeplob_model ./models/deeplob_pretrained.pt --split test
```

Algoritmes: `ppo_deeplob`, `sac_deeplob`, `ppo_only`, `sac_only`.

## Monitoring

```bash
tensorboard --logdir ./logs
```

## Trading environment

- **Acties (discreet):** Hold = 0, Buy = 1, Sell = 2
- **Observatie:** `features` (window van market data) + `portfolio` (balans, BTC, PnL)
- **Reward:** PnL-verandering minus fees; extra penalty bij drawdown > 15%
- **Episode einde:** data op of bankruptcy (portfolio < 10% van start → reset)
