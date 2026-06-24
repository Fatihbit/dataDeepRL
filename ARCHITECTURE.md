# Software Architectuur — DataDeepRL

## 1. Overzicht

DataDeepRL is een Deep Reinforcement Learning framework voor cryptocurrency trading.
Het systeem combineert een **DeepLOB feature extractor** (CNN + Inception + BiLSTM)
met **PPO** en **SAC** RL-agenten om handelsbeslissingen te nemen op basis van
Binance BTC/USDT L2 order book data.

---

## 2. Systeemoverzicht

```mermaid
flowchart TD
    subgraph DATAPIPELINE["📦 Datapipeline (eenmalig)"]
        A[Binance API] -->|aggTrades per dag| B[binance_l2.py]
        B --> C[(btc_l2_data/\n.parquet per dag)]
        C --> D[preprocess_data.py\nfeature engineering + z-score]
        D --> E[(DataNorm/\ngenormaliseerde features)]
        E --> F[create_core_data.py\n80/10/10 split]
        F --> G[(coreData/\ntrain · val · test\n+ normalization_stats.json)]
    end

    subgraph TRAINING["🧠 Training"]
        G -->|load_coredata_streaming| H[CryptoTradingEnv]
        H -->|obs, reward, done| I{Agent}
        I -->|PPO + DeepLOB| J[train_ppo_with_deeplob.py]
        I -->|SAC + DeepLOB| K[train_sac_with_deeplob.py]
        I -->|PPO + MLP| L[train_ppo_only.py]
        I -->|SAC + MLP| M[train_sac_only.py]
        J & K & L & M --> N[(logs/\nbest_model.pt\ntraining_monitor.csv)]
    end

    subgraph EVALUATIE["📊 Evaluatie"]
        N --> O[evaluate.py]
        G -->|test.parquet| O
        O --> P[Sharpe · Drawdown · Return\nComposite Score\nBuy-and-Hold vergelijking]
    end
```

---

## 3. Datapipeline

```mermaid
flowchart LR
    RAW["btc_l2_data/\n(ruwe parquet)"]
    NORM["DataNorm/\n(z-score features)"]
    CORE["coreData/"]

    RAW -->|"feature engineering\nper maand chunk"| NORM
    NORM -->|"chronologische split"| CORE

    CORE --> TR["train.parquet\n80%"]
    CORE --> VA["val.parquet\n10%"]
    CORE --> TE["test.parquet\n10%"]
    CORE --> ST["normalization_stats.json"]
```

**Features (15 stuks — `STATIONARY_FEATURES`):**

| Categorie | Features |
|---|---|
| Spread | `spread`, `spread_pct` |
| Order flow | `buy_ratio`, `order_imbalance`, `volume_ratio` |
| Returns | `return_5s`, `return_10s`, `return_30s`, `return_60s` |
| Volatiliteit | `volatility_10`, `volatility_30`, `volatility_60` |
| Momentum | `momentum_10`, `momentum_30` |
| Technisch | `rsi_14` |

---

## 4. DeepLOB Architectuur

```mermaid
flowchart TD
    IN["Input\n(batch × window_size × 15 features)"]

    IN --> CB1
    CB1["ConvBlock 1\nConv1D → BatchNorm → LeakyReLU"]
    CB1 --> CB2
    CB2["ConvBlock 2\nConv1D → BatchNorm → LeakyReLU"]
    CB2 --> INC

    subgraph INC["Inception Module (multi-scale)"]
        direction LR
        P1["Conv 1×1"]
        P2["Conv 3×3"]
        P3["Conv 5×5"]
    end

    INC -->|concat| LSTM
    LSTM["BiLSTM\nbidirectionele LSTM\ntemporele afhankelijkheden"]
    LSTM --> ATT
    ATT["Attention Pooling\ngewogen gemiddelde\nover tijdsdimensie"]
    ATT --> OUT

    subgraph PRETRAIN["Pretraining (supervised)"]
        OUT["Feature vector\n(feature_dim,)"]
        OUT -->|CrossEntropyLoss| CLS["Classificatie\nomhoog / vlak / omlaag"]
    end

    OUT -->|bevroren gewichten| RL["RL Agent backbone"]
```

> Na pretraining worden de gewichten **bevroren** (`--freeze_deeplob`).
> De DeepLOB fungeert daarna puur als feature extractor voor de RL-agent.

---

## 5. RL Agent — Beslissingslus

```mermaid
sequenceDiagram
    participant E as CryptoTradingEnv
    participant A as RL Agent (PPO/SAC)
    participant D as DeepLOB (optioneel)
    participant B as Buffer

    loop Elke tijdstap
        E->>A: obs {features (100×15), portfolio (3,)}
        A->>D: features window
        D->>A: feature vector
        A->>E: actie (Hold / Buy / Sell)
        E->>A: reward + volgende obs + done
        A->>B: sla transitie op
    end

    Note over A,B: PPO: RolloutBuffer → update na N stappen (on-policy)
    Note over A,B: SAC: ReplayBuffer → update per stap (off-policy)
```

---

## 6. PPO vs SAC — Algoritme vergelijking

```mermaid
flowchart LR
    subgraph PPO["PPO (On-policy)"]
        direction TB
        P1["Verzamel rollout\n(N stappen)"]
        P2["Bereken GAE-voordelen"]
        P3["Update policy\nmeerdere epochs\n— clipped surrogate loss\n— value loss\n— entropy bonus"]
        P4["Gooi data weg"]
        P1 --> P2 --> P3 --> P4 --> P1
    end

    subgraph SAC["SAC (Off-policy)"]
        direction TB
        S1["Voer stap uit\nmet actor"]
        S2["Sla op in\nReplay Buffer"]
        S3["Sample random\nbatch"]
        S4["Update Twin Q-networks\n+ Actor\n+ temperatuur α"]
        S1 --> S2 --> S3 --> S4 --> S1
    end
```

| | PPO | SAC |
|---|---|---|
| Type | On-policy | Off-policy |
| Buffer | RolloutBuffer (weggooi) | ReplayBuffer (bewaar) |
| Sample efficiency | Lager | Hoger |
| Stabiliteit | Hoog | Hoog |
| Exploratie | Entropy bonus | Maximum entropy principe |

---

## 7. Trainingsflow

```mermaid
sequenceDiagram
    participant TR as train.parquet
    participant VA as val.parquet
    participant AG as RL Agent
    participant CK as best_model.pt

    Note over TR,CK: Stap 1 — Pretrain DeepLOB (supervised)
    TR->>AG: batches prijsrichting labels
    VA->>AG: val_loss (early stopping)
    AG->>CK: sla op als deeplob_pretrained.pt

    Note over TR,CK: Stap 2 — RL Training
    loop Elke update
        TR->>AG: stream van marktdata
        AG->>AG: trade, collect reward
    end

    loop Elke eval_freq stappen
        VA->>AG: evalueer N episodes
        AG->>AG: bereken composite_score
        AG-->>CK: sla op als beste score beter is
    end

    Note over TR,CK: Stap 3 — Eindoordeel (evaluate.py)
    Note over TR,CK: test.parquet (nooit eerder gezien)
```

---

## 8. Data-splitsing en rol per fase

```mermaid
gantt
    title Dataset gebruik per fase
    dateFormat X
    axisFormat %s%%

    section train.parquet (80%)
    RL agent leert strategie       :0, 80

    section val.parquet (10%)
    Checkpoint selectie tijdens training :80, 90

    section test.parquet (10%)
    Eerlijk eindoordeel (scriptie) :90, 100
```

> **Testdata wordt nooit gebruikt tijdens training of modelkeuze.**
> Alleen de testsplit geeft een onbevooroordeeld eindresultaat voor de scriptie.

---

## 9. Evaluatiemetrieken

```mermaid
flowchart TD
    M[evaluate.py\nbest_model.pt + test.parquet]

    M --> SR["Sharpe Ratio\nrisicogecorrigeerd rendement\nSharpe > 1 = goed"]
    M --> MD["Max Drawdown\nmaximale piekdaling\nlager = beter"]
    M --> RT["Total Return %\nprocentuele eindwinst/-verlies"]
    M --> PL["Total Profit / Total Loss\nsom winnende vs verliezende trades"]
    M --> CS["Composite Score\n0.5 × clip(Sharpe,-5,5)/5\n+ 0.5 × clip(Return,-1,1)\n− 0.2 × Drawdown"]
    M --> BH["Buy-and-Hold Benchmark\npassieve strategie ter vergelijking"]

    CS -->|"hogere score = beter model"| VERDICT[Eindoordeel]
    BH --> VERDICT
```

---

## 10. Ontwerpkeuzes

| Keuze | Motivatie |
|---|---|
| Gesplitste trainingsscripts per variant | Expliciete controle; geen hidden flags; eenvoudiger debuggen |
| Bevroren DeepLOB tijdens RL-training | Voorkomt dat RL de feature-representaties overschrijft; stabielere training |
| On-the-fly windowing in de env | Vermijdt materialisatie van een (N × seq_len × features) array; past in RAM |
| Z-score normalisatie in preprocessing | Stationariteit; vereist door LSTM-gebaseerde modellen |
| Composite score als checkpoint-criterium | Balanceert Sharpe, return én drawdown in één getal voor modelselectie |
| Chronologische split (geen random) | Voorkomt data leakage; respecteert temporele afhankelijkheid van financiële tijdreeksen |
| Inception module (multi-scale conv) | Vangt patronen op korte én lange tijdschalen tegelijk op in het LOB |
| BiLSTM i.p.v. LSTM | Ziet zowel de vorige als de komende context; betere representatie |

---

## 11. Mappenstructuur

```
dataDeepRL/
├── btc_l2_data/            Ruwe Binance L2 parquet (per dag)
├── coreData/               Genormaliseerde train/val/test + normalization_stats.json
├── models/                 Opgeslagen modellen (deeplob_pretrained.pt)
├── logs/                   TensorBoard runs + training_monitor.csv per run
│
├── src/
│   ├── envs/
│   │   ├── trading_env.py  CryptoTradingEnv + FlatCryptoTradingEnv
│   │   └── vec_env.py      VectorizedTradingEnv (N parallelle subprocessen)
│   ├── models/
│   │   ├── deeplob.py      DeepLOB (ConvBlock, Inception, BiLSTM, Attention)
│   │   ├── mlp.py          MLP netwerken (policy, value, actor, critic)
│   │   ├── ppo.py          PPOAgent + RolloutBuffer
│   │   └── sac.py          SACAgent + ReplayBuffer
│   └── utils/
│       ├── logger.py       Training logger
│       ├── callbacks.py    Eval callbacks
│       ├── trade_logger.py Trade-level logging
│       └── mixed_precision.py  AMP support
│
├── train/
│   ├── common/
│   │   └── setup.py        load_coredata_streaming() + STATIONARY_FEATURES
│   ├── train_deeplob_pretrain.py
│   ├── train_ppo_only.py
│   ├── train_sac_only.py
│   ├── train_ppo_with_deeplob.py
│   └── train_sac_with_deeplob.py
│
├── dataVerwerken/
│   ├── preprocess_data.py  Feature engineering + z-score
│   └── create_core_data.py 80/10/10 split
│
├── binance_l2.py           Binance download script
├── evaluate.py             Evalueer model op val/test split
├── tune.py                 Optuna hyperparameter tuning
├── ARCHITECTURE.md         Dit document
└── README.md               Installatie + gebruik
```
