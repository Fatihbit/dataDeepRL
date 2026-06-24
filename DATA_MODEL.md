# Datamodel en Data-analyse — DataDeepRL

## 1. Inleiding

Dit document beschrijft het volledige datamodel van DataDeepRL: welke data er
binnenkomt, hoe het wordt verwerkt, welke features worden berekend, hoe de data
wordt opgeslagen en hoe het door het model wordt gebruikt.

---

## 2. Data-entiteiten overzicht

```mermaid
erDiagram
    RAW_TRADE {
        int64   timestamp       "Unix ms — tijdstip van trade"
        float64 price           "Handelsprijs in USDT"
        float64 quantity        "Verhandelde hoeveelheid BTC"
        bool    is_buyer_maker  "True = verkoper initieerde"
        string  date            "YYYY-MM-DD (partitie)"
    }

    ORDER_BOOK_SNAPSHOT {
        int64   timestamp   "Unix ms"
        float64 bid_price   "Beste biedprijs"
        float64 bid_volume  "Volume op beste bid"
        float64 ask_price   "Beste vraagprijs"
        float64 ask_volume  "Volume op beste ask"
        float64 close       "Laatste handelsprijs"
        float64 volume      "Volume in tijdvenster"
    }

    ENGINEERED_FEATURES {
        int64   timestamp       "Tijdstip"
        float64 close           "Prijs (later gedenormaliseerd)"
        float64 spread          "ask - bid in USDT"
        float64 spread_pct      "spread / close × 100"
        float64 buy_ratio       "aandeel buyer-initiated trades"
        float64 return_5s       "prijsverandering 5 seconden"
        float64 return_10s      "prijsverandering 10 seconden"
        float64 return_30s      "prijsverandering 30 seconden"
        float64 return_60s      "prijsverandering 60 seconden"
        float64 volatility_10   "std(return_1s, window=10)"
        float64 volatility_30   "std(return_1s, window=30)"
        float64 volatility_60   "std(return_1s, window=60)"
        float64 momentum_10     "close - close_10s_geleden"
        float64 momentum_30     "close - close_30s_geleden"
        float64 rsi_14          "Relative Strength Index (14)"
        float64 volume_ratio    "volume / sma_volume_10"
        float64 order_imbalance "(bid_vol - ask_vol) / totaal_vol"
    }

    NORMALIZED_DATA {
        int64   timestamp   "Tijdstip"
        float32 close       "z-score genormaliseerd"
        float32 spread      "z-score genormaliseerd"
        float32 return_5s   "z-score genormaliseerd"
        string  ELKE_FEATURE "z = (x - mean) / std"
    }

    CORE_SPLIT {
        string  split       "train / val / test"
        float32 features    "15 STATIONARY_FEATURES (z-score)"
        float64 close       "denorm. USD prijs (via stats.json)"
    }

    NORMALIZATION_STATS {
        string  column  "kolomnaam"
        float64 mean    "gemiddelde over trainingsdata"
        float64 std     "standaarddeviatie"
        float64 min     "minimum waarde"
        float64 max     "maximum waarde"
    }

    RAW_TRADE         ||--o{ ORDER_BOOK_SNAPSHOT : "aggregeert naar"
    ORDER_BOOK_SNAPSHOT ||--|| ENGINEERED_FEATURES : "feature engineering"
    ENGINEERED_FEATURES ||--|| NORMALIZED_DATA : "z-score normalisatie"
    NORMALIZED_DATA   ||--o{ CORE_SPLIT : "80/10/10 split"
    NORMALIZATION_STATS ||--|| CORE_SPLIT : "prijs denormalisatie"
```

---

## 3. Datapijplijn — transformatiestappen

```mermaid
flowchart TD
    subgraph STAP1["Stap 1 — Download (binance_l2.py)"]
        A["Binance aggTrades API\nper dag"] --> B["BTCUSDT_YYYY-MM-DD.parquet\n(prijs, volume, timestamp,\nis_buyer_maker)"]
    end

    subgraph STAP2["Stap 2 — Feature Engineering (preprocess_data.py)"]
        B --> C["Laad per maand\n(memory-efficient chunks)"]
        C --> D["Bereken order book features\nspread, bid/ask, volume"]
        D --> E["Bereken returns\n1s / 5s / 10s / 30s / 60s"]
        E --> F["Bereken volatiliteit\nrolling std van return_1s"]
        F --> G["Bereken momentum\nclose - close_t-n"]
        G --> H["Bereken RSI-14\ngain/loss ratio"]
        H --> I["Bereken volume ratio\nvolume / sma_10"]
        I --> J["Bereken order imbalance\n(bid - ask) / totaal"]
        J --> K["Verwijder NaN rijen\n(rolling windows)"]
        K --> L["DataNorm/chunks/\nchunk_YYYY-MM.parquet"]
    end

    subgraph STAP3["Stap 3 — Normalisatie (preprocess_data.py)"]
        L --> M["2-pass statistieken\nover alle chunks"]
        M --> N["Sla op als\nnormalization_stats.json"]
        N --> O["Z-score per kolom\nz = (x - mean) / std"]
        O --> P["DataNorm/\nlatest_normalized.parquet"]
    end

    subgraph STAP4["Stap 4 — Split (create_core_data.py)"]
        P --> Q["Chronologisch sorteren"]
        Q --> R["80% → train.parquet"]
        Q --> S["10% → val.parquet"]
        Q --> T["10% → test.parquet"]
        N --> U["normalization_stats.json\n(kopiëren naar coreData/)"]
    end
```

---

## 4. Feature definities

### 4.1 Ruwe order book features

| Feature | Formule | Eenheid | Betekenis |
|---|---|---|---|
| `spread` | `ask_price - bid_price` | USDT | Kosten om direct te kopen + verkopen |
| `spread_pct` | `spread / close × 100` | % | Spread relatief aan prijs |
| `buy_ratio` | `n_buyer_trades / n_trades` | ratio [0,1] | Vraag druk: hoe meer kopen hoe bullish |
| `order_imbalance` | `(bid_vol - ask_vol) / (bid_vol + ask_vol)` | ratio [-1,1] | Positief = meer koopdruk |
| `volume_ratio` | `volume / sma(volume, 10)` | ratio | Relatief volume t.o.v. gemiddelde |

### 4.2 Prijsverandering (returns)

| Feature | Formule | Venster | Betekenis |
|---|---|---|---|
| `return_5s` | `(close_t - close_{t-5}) / close_{t-5}` | 5 tijdstappen | Korte prijs beweging |
| `return_10s` | `pct_change(10)` | 10 | Medium korte beweging |
| `return_30s` | `pct_change(30)` | 30 | Medium beweging |
| `return_60s` | `pct_change(60)` | 60 | Langere termijn beweging |

### 4.3 Risicofeatures

| Feature | Formule | Venster | Betekenis |
|---|---|---|---|
| `volatility_10` | `std(return_1s, window=10)` | 10 | Kortetermijn onzekerheid |
| `volatility_30` | `std(return_1s, window=30)` | 30 | Mediumtermijn onzekerheid |
| `volatility_60` | `std(return_1s, window=60)` | 60 | Langeretermijn onzekerheid |
| `momentum_10` | `close_t - close_{t-10}` | 10 | Absolute prijsbeweging |
| `momentum_30` | `close_t - close_{t-30}` | 30 | Bredere trend |

### 4.4 Technische indicator

| Feature | Formule | Bereik | Betekenis |
|---|---|---|---|
| `rsi_14` | `100 - 100/(1 + avg_gain/avg_loss)` over 14 perioden | [0, 100] | Overbought (>70) / Oversold (<30) |

---

## 5. Normalisatieschema

```mermaid
flowchart LR
    subgraph RAW["Ruwe waarden"]
        R1["close = 42.000 USD"]
        R2["spread = 0.5 USD"]
        R3["rsi_14 = 65"]
    end

    subgraph STATS["normalization_stats.json"]
        S1["close: mean=35000, std=12000"]
        S2["spread: mean=0.8, std=0.3"]
        S3["rsi_14: mean=50, std=15"]
    end

    subgraph NORM["Z-score output"]
        N1["close_z = (42000-35000)/12000 = +0.58"]
        N2["spread_z = (0.5-0.8)/0.3 = -1.00"]
        N3["rsi_14_z = (65-50)/15 = +1.00"]
    end

    subgraph DENORM["Denormalisatie (bij laden)"]
        D1["prijs_usd = z × std + mean"]
        D1b["= 0.58 × 12000 + 35000 = 41.960 USD"]
    end

    RAW --> STATS
    STATS --> NORM
    NORM -->|"alleen prijs"| DENORM
```

> Features blijven z-score genormaliseerd als modelinput.
> Alleen de `close` prijs wordt teruggedenormaliseerd naar USD — de environment
> heeft de echte USD prijs nodig voor reward berekening en PnL tracking.

---

## 6. Dataschema per fase

### Fase 1 — Ruwe data (`btc_l2_data/`)

```
BTCUSDT_2024-01-15.parquet
├── timestamp     int64    Unix milliseconden
├── price         float64  Handelsprijs USDT
├── quantity      float64  BTC hoeveelheid
├── is_buyer_maker bool    True = verkoper initieerde
└── (eventueel)
    ├── bid_price  float64
    ├── ask_price  float64
    ├── bid_volume float64
    └── ask_volume float64
```

### Fase 2 — Verwerkte data (`DataNorm/`)

```
normalized_20170817_to_20250307.parquet
├── timestamp      int64     Tijdstip
├── close          float32   Z-score (voor denorm naar USD)
├── spread         float32   Z-score
├── spread_pct     float32   Z-score
├── buy_ratio      float32   Z-score
├── return_1s      float32   Z-score  (tussenproduct, niet in model)
├── return_5s      float32   Z-score
├── return_10s     float32   Z-score
├── return_30s     float32   Z-score
├── return_60s     float32   Z-score
├── volatility_10  float32   Z-score
├── volatility_30  float32   Z-score
├── volatility_60  float32   Z-score
├── momentum_10    float32   Z-score
├── momentum_30    float32   Z-score
├── rsi_14         float32   Z-score
├── volume_ratio   float32   Z-score
└── order_imbalance float32  Z-score
```

### Fase 3 — Modelinput (`coreData/`)

```
train.parquet / val.parquet / test.parquet
├── close          float32   Z-score (wordt denorm. bij laden)
├── spread         float32   Z-score ┐
├── spread_pct     float32   Z-score │
├── buy_ratio      float32   Z-score │
├── return_5s      float32   Z-score │  15 STATIONARY_FEATURES
├── return_10s     float32   Z-score │  (modelinput na window slice)
├── return_30s     float32   Z-score │
├── return_60s     float32   Z-score │
├── volatility_10  float32   Z-score │
├── volatility_30  float32   Z-score │
├── volatility_60  float32   Z-score │
├── momentum_10    float32   Z-score │
├── momentum_30    float32   Z-score │
├── rsi_14         float32   Z-score │
├── volume_ratio   float32   Z-score │
└── order_imbalance float32  Z-score ┘

normalization_stats.json
└── stats
    └── close: { mean, std, min, max }
        (alleen close nodig voor denormalisatie)
```

---

## 7. Datatransformatie in de environment

```mermaid
flowchart TD
    A["coreData/train.parquet\nraw_features: float32 (N × 15)\nprices: float64 (N,) — USD"]

    B["current_step = t"]
    C["Window slice\nraw_features[t : t+100]\nshape: (100 × 15)\nnumpy view — zero-copy"]

    D["Portfolio state\n[balance_ratio, btc_held,\navg_buy_price, unrealized_pnl]\nshape: (4,)"]

    E["Observatie dict\n{\n  features: (100 × 15),\n  portfolio: (4,)\n}"]

    F["DeepLOB of MLP\n→ actie beslissing"]

    G["Reward berekening\nprices[t] → USD prijs\nΔportfoliowaarde"]

    A --> B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    A --> G

    style C fill:#e8f4f8
    style E fill:#e8f4f8
```

---

## 8. Data-analyse — kenmerken van de dataset

### 8.1 Tijdsbereik en omvang

| Eigenschap | Waarde |
|---|---|
| Bron | Binance BTC/USDT aggTrades |
| Periode | 2017-08-17 t/m 2025-03-07 (~7,5 jaar) |
| Frequentie | Per seconde (geaggregeerd) |
| Bestandsformaat | Apache Parquet (koloms-georiënteerd) |
| Verwerking | Per maand (memory-efficient chunks) |

### 8.2 Datasplitsing

```mermaid
xychart-beta
    title "Dataset verdeling (chronologisch)"
    x-axis ["Train (80%)", "Val (10%)", "Test (10%)"]
    y-axis "Aandeel dataset" 0 --> 100
    bar [80, 10, 10]
```

| Split | Aandeel | Tijdperiode | Gebruik |
|---|---|---|---|
| Train | 80% | 2017-08 t/m ~2023-09 | Agent leert strategie |
| Val | 10% | ~2023-09 t/m ~2024-07 | Checkpoint selectie |
| Test | 10% | ~2024-07 t/m 2025-03 | Definitief eindoordeel |

> De split is **chronologisch** — nooit willekeurig. Een willekeurige split zou
> data leakage veroorzaken: toekomstige marktinformatie lekt dan in de training.

### 8.3 Feature categorieën en rationaliteit

```mermaid
mindmap
  root((15 Features))
    Order Book
      spread
      spread_pct
      order_imbalance
      buy_ratio
    Volume
      volume_ratio
    Returns
      return_5s
      return_10s
      return_30s
      return_60s
    Volatiliteit
      volatility_10
      volatility_30
      volatility_60
    Momentum
      momentum_10
      momentum_30
    Technisch
      rsi_14
```

### 8.4 Waarom z-score normalisatie?

```mermaid
flowchart LR
    subgraph PROBLEEM["Zonder normalisatie"]
        P1["close ≈ 30.000"]
        P2["spread ≈ 0.5"]
        P3["return_5s ≈ 0.0001"]
        P4["Netwerk domineert\ndoor schaalverschil"]
    end

    subgraph OPLOSSING["Z-score normalisatie"]
        O1["close_z ≈ 0.58"]
        O2["spread_z ≈ -1.00"]
        O3["return_5s_z ≈ 0.72"]
        O4["Alle features\nzelfde schaal (μ=0, σ=1)"]
    end

    PROBLEEM -->|"z = (x - μ) / σ"| OPLOSSING
```

**Voordelen z-score:**
- LSTM/Conv lagen zijn gevoelig voor schaal — grote waarden domineren anders gradiënten
- Snellere convergentie tijdens training
- Stationariteit: features hebben stabielere statistieken over tijd

### 8.5 Stationariteit — waarom returns ipv absolute prijs?

| Maatstaf | Absoluut (prijs) | Relatief (return) |
|---|---|---|
| BTC prijs 2017 | ~$5.000 | — |
| BTC prijs 2024 | ~$65.000 | — |
| Return 2017 | — | ≈ gelijk aan 2024 |
| Probleem | Model ziet nooit zulke hoge prijzen in training | Geen probleem |
| **Keuze** | ✗ Niet gebruikt als feature | ✓ Gebruikt |

> Absolute prijzen zijn **niet-stationair** — het model zou moeilijk kunnen
> generaliseren van trainingsperiode naar testperiode. Returns en ratio's
> zijn dat wel.

---

## 9. Observatievenster — sliding window mechanisme

```mermaid
flowchart LR
    subgraph DATA["raw_features (N × 15)"]
        direction TB
        R0["t=0"]
        R1["t=1"]
        R2["..."]
        R99["t=99"]
        R100["t=100  ← current_step"]
        R101["t=101"]
        R102["..."]
    end

    subgraph WIN["Window (100 × 15)"]
        W["raw_features[0:100]\nnumpy slice\n(zero-copy view)"]
    end

    subgraph NEXT["Volgende stap"]
        W2["raw_features[1:101]\nnumpy slice"]
    end

    R0 --> WIN
    R99 --> WIN
    WIN -->|"step()"| NEXT
    R1 --> NEXT
    R100 --> NEXT
```

- Geen kopie van data — alleen een numpy pointer verschuift
- RAM gebruik: O(N × 15) i.p.v. O(N × 100 × 15) = 100× minder geheugen
- Bij 20M rijen: ~1,1 GB RAM i.p.v. ~110 GB

---

## 10. Portfoliostate als model-input

Naast marktfeatures krijgt het model ook de huidige portfoliostatus als input:

| Veld | Type | Beschrijving |
|---|---|---|
| `balance` | float32 | USDT kasgeld |
| `btc_held` | float32 | Hoeveelheid BTC in bezit |
| `avg_buy_price` | float32 | Gemiddelde inkoopprijs |
| `unrealized_pnl` | float32 | Ongerealiseerde winst/verlies in USDT |

Dit stelt het model in staat om **contextbewuste beslissingen** te nemen:
bijv. niet kopen als al volledig belegd, of verkopen als de PnL positief genoeg is.

---

## 11. Label-definitie voor DeepLOB pretraining

Bij de gesuperviseerde pretraining van DeepLOB worden labels berekend op basis
van toekomstige prijsbeweging:

```mermaid
flowchart TD
    A["Prijs op t: P_t"]
    B["Prijs op t+k: P_{t+k}"]
    C{"ΔP = (P_{t+k} - P_t) / P_t"}

    L0["Klasse 0: Omhoog\nΔP > +drempel"]
    L1["Klasse 1: Vlak\n|ΔP| ≤ drempel"]
    L2["Klasse 2: Omlaag\nΔP < -drempel"]

    A --> C
    B --> C
    C --> L0
    C --> L1
    C --> L2
```

> Dit is een **3-klassen classificatieprobleem** (omhoog / vlak / omlaag).
> Verliesfunctie: CrossEntropyLoss. Na pretraining wordt alleen de backbone
> (zonder classifier head) gebruikt als feature extractor.

---

## 12. Datavolume en geheugengebruik

| Fase | Formaat | Grootte (schatting) |
|---|---|---|
| `btc_l2_data/` | Parquet per dag | ~50–200 MB per dag × 2.760 dagen |
| `DataNorm/chunks/` | Parquet per maand | ~1–5 GB per maand × 90 maanden |
| `coreData/train.parquet` | Parquet | ~2–8 GB |
| `coreData/val.parquet` | Parquet | ~0.3–1 GB |
| `coreData/test.parquet` | Parquet | ~0.3–1 GB |
| RAM tijdens training | float32 arrays | ~1–3 GB (streaming, geen volledige materialisatie) |

### Geheugenoptimalisatie

```mermaid
flowchart LR
    A["Naïeve aanpak\nMaterialiseer alle sequences\n(N × 100 × 15) float32\n= 100× meer RAM"] -->|"te groot"| ERR["❌ Out of Memory"]

    B["Streaming aanpak\nSla op (N × 15) float32\nSliding window in env\n(zero-copy numpy view)"] -->|"~1 GB"| OK["✅ Past in RAM"]
```
