# 🚀 Complete Training Flow - Stap voor Stap

Dit document beschrijft exact welke terminal commando's je moet uitvoeren om van raw data naar een getraind model te komen.

---

## 📋 Overzicht

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│  STAP 0     STAP 1     STAP 1b      STAP 2       STAP 3     STAP 4     STAP 5            │
│  ┌─────┐    ┌─────┐    ┌────────┐   ┌────────┐   ┌─────┐    ┌─────┐    ┌─────┐           │
│  │Setup│ →  │Data │ →  │Preproc │ → │DeepLOB │ → │Train│ →  │Tune │ →  │Eval │           │
│  └─────┘    └─────┘    │& Norm  │   │Pretrain│   │RL   │    └─────┘    └─────┘           │
│  Env +      Download   └────────┘   └────────┘   └─────┘    (Optioneel) Test +           │
│  Deps       Binance    Features +   Features     PPO/SAC    Optimaliseer Visualiseer     │
│                        Normalisatie                                                      │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## STAP 0: Environment Setup

### 0.1 Maak virtual environment
```powershell
# Navigeer naar project folder
cd c:\code\dataDeepRL

# Maak nieuwe virtual environment
python -m venv .venv

# Activeer environment (Windows)
.\.venv\Scripts\Activate.ps1

# Of voor Command Prompt:
# .\.venv\Scripts\activate.bat
```

### 0.2 Installeer dependencies
```powershell
# Installeer alle packages
pip install -r requirements.txt

# Controleer of PyTorch GPU heeft (optioneel)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Verwachte output:**
```
CUDA available: True   # als je een NVIDIA GPU hebt
CUDA available: False  # als je geen GPU hebt (CPU training)
```

---

## STAP 1: Data Downloaden

### 1.1 Configureer datum range
Open `binance_l2.py` en pas de datums aan:
```python
# Regel 27-30 in binance_l2.py
START_DATE = datetime(2025, 1, 15)  # ← Pas aan
END_DATE = datetime(2026, 1, 30)    # ← Pas aan
```

### 1.2 Download data
```powershell
# Download L2 order book data van Binance
python binance_l2.py
```

**Verwachte output:**
```
[█████████████░░░░░░░░░░░░░░░░░] 45.2% | 2025-01-15 → 2026-01-30 | Dag 180/381 | 2025-07-15 | ✓ OK
```

**⏱️ Duur:** ~5-30 minuten afhankelijk van datum range

### 1.3 Controleer data
```powershell
# Bekijk aantal bestanden
(Get-ChildItem btc_l2_data\*.parquet).Count

# Bekijk eerste bestand
python -c "import pandas as pd; print(pd.read_parquet('btc_l2_data/BTCUSDT_2025-01-15.parquet').head())"
```

---

## STAP 1b: Data Preprocessing & Normalisatie

Na het downloaden van de ruwe data, moet je deze eerst preprocessen en normaliseren voordat je gaat trainen.

### 1b.1 Configureer preprocessing
Open `dataVerwerken/preprocess_data.py` en pas de datums aan:
```python
# Regel 25-28 in preprocess_data.py
START_DATE = "2024-01-01"  # ← Pas aan
END_DATE = "2024-12-31"    # ← Pas aan
NORMALIZATION_METHOD = "zscore"  # of "minmax"
```

### 1b.2 Preprocessen & normaliseren

```powershell
# Verwerk missing values, outliers, features en normaliseer data
python dataVerwerken/preprocess_data.py
```

**Dit script doet:**
1. **Missing values**: Forward fill + backward fill
2. **Outliers**: Clipt naar 3x IQR grenzen
3. **Feature engineering**:
   - Returns (1s, 5s, 10s, 30s, 60s)
   - Moving averages (SMA, EMA)
   - Volatiliteit
   - Momentum & RSI
   - Volume ratio's
   - Spread features
   - Order imbalance
4. **Normalisatie**: Z-score of Min-Max scaling
5. Slaat op in `DataNorm/`

**Verwachte output:**
```
📌 STAP 5: FEATURE ENGINEERING
   Berekenen: Returns...
   Berekenen: Moving averages...
   Berekenen: Volatiliteit...
   ✅ 25 nieuwe features toegevoegd
```

### 1b.3 Train/val/test split maken

```powershell
# Split genormaliseerde data in train/val/test sets (80/10/10)
python dataVerwerken/create_core_data.py
```

**Dit script doet:**
- Leest genormaliseerde data uit `DataNorm/`
- Maakt chronologische 80/10/10 split (geen shuffle!)
- Slaat splits op in `coreData/`

**Output bestanden:**
```
coreData/
├── train.parquet     # 80% van data
├── val.parquet       # 10% van data  
├── test.parquet      # 10% van data
├── metadata.json     # Split info
└── normalization_stats.json  # Voor denormalisatie
```

### 1b.4 Controleer preprocessing
```powershell
# Bekijk genormaliseerde data
python -c "import pandas as pd; df=pd.read_parquet('coreData/train.parquet'); print(df.head()); print(f'\nKolommen: {len(df.columns)}'); print(f'Rows: {len(df):,}')"
```

---

## STAP 2: DeepLOB Pre-Training (AANBEVOLEN)

DeepLOB is een CNN+LSTM netwerk dat order book patterns leert herkennen. **Pre-training** geeft betere resultaten omdat:
- DeepLOB eerst leert wat "goede" market patterns zijn
- RL agents starten met betekenisvolle features (niet random)
- Snellere convergentie van PPO/SAC

### 2.1 Train DeepLOB op price prediction
```powershell
# Pre-train DeepLOB (supervised learning)
python train/train_deeplob_pretrain.py --data_dir btc_l2_data --epochs 50 --save_path models/deeplob_pretrained.pt
```

**Wat gebeurt er?**
```
Input:  100 timesteps order book data
          ↓
DeepLOB:  CNN → LSTM → Dense
          ↓
Output:   Voorspelling [Down, Neutral, Up]
```

**Verwachte output:**
```
Epoch 1/50  | Train Loss: 1.05 | Val Loss: 0.98 | Val Acc: 45.2%
Epoch 10/50 | Train Loss: 0.72 | Val Loss: 0.69 | Val Acc: 58.3%
Epoch 50/50 | Train Loss: 0.45 | Val Loss: 0.52 | Val Acc: 67.8%

Model saved to models/deeplob_pretrained.pt
```

**⏱️ Duur:** ~10-30 min (GPU) / ~1-2 uur (CPU)

### 2.2 Controleer pre-trained model
```powershell
# Check of model bestaat
Test-Path models/deeplob_pretrained.pt

# Bekijk model info
python -c "import torch; m=torch.load('models/deeplob_pretrained.pt'); print(f'Val Accuracy: {m.get(\"best_val_acc\", \"N/A\")}')"
```

---

## STAP 3: RL Agent Trainen

### Kies je training methode:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              WELK MODEL?                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   OPTIE A: Pre-trained DeepLOB    OPTIE B: End-to-End      OPTIE C: Simpel     │
│   ─────────────────────────────   ────────────────────     ─────────────────   │
│   ★ AANBEVOLEN ★                  • Traint alles samen     • MLP only           │
│   • DeepLOB features frozen       • Meer GPU geheugen      • Snel te trainen    │
│   • Stabielere training           • Langere training       • Goed voor testen   │
│   • Beste performance             • Kan overfitting        • Baseline model     │
│                                                                                 │
│   → train_ppo_with_deeplob.py     → train_ppo_deeplob.py   → train_ppo_only.py │
│   → train_sac_with_deeplob.py     → train_sac_deeplob.py   → train_sac_only.py │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3A: Training met pre-trained DeepLOB (AANBEVOLEN - beste resultaten)

```powershell
# PPO met pre-trained DeepLOB (frozen features)
python train/train_ppo_with_deeplob.py `
    --deeplob_model models/deeplob_pretrained.pt `
    --freeze_deeplob `
    --total_steps 500000 `
    --experiment_name ppo_pretrained_v1

# SAC met pre-trained DeepLOB (frozen features)
python train/train_sac_with_deeplob.py `
    --deeplob_model models/deeplob_pretrained.pt `
    --freeze_deeplob `
    --total_steps 500000 `
    --experiment_name sac_pretrained_v1
```

**--freeze_deeplob** betekent dat de DeepLOB weights niet worden aangepast tijdens RL training.
Dit is aanbevolen voor stabiliteit.

### 3B: Training met DeepLOB (end-to-end, zonder pre-training)

```powershell
# PPO met DeepLOB - traint alles samen
python train/train_ppo_deeplob.py --config configs/default.yaml --experiment_name ppo_deeplob_v1

# SAC met DeepLOB - traint alles samen
python train/train_sac_deeplob.py --config configs/default.yaml --experiment_name sac_deeplob_v1
```

### 3C: Training simpel (MLP only - voor beginners/snelle tests)

```powershell
# PPO Simpel (MLP) - snel, geen DeepLOB
python train/train_ppo_only.py --total_steps 100000 --experiment_name ppo_test

# SAC Simpel (MLP) - snel, geen DeepLOB
python train/train_sac_only.py --total_steps 100000 --experiment_name sac_test
```

### Training Argumenten Uitleg

| Argument | Beschrijving | Default |
|----------|--------------|---------|
| `--total_steps` | Totaal aantal training stappen | 1,000,000 |
| `--batch_size` | Batch size voor updates | 256 |
| `--learning_rate` | Learning rate | 0.0003 |
| `--gamma` | Discount factor | 0.99 |
| `--experiment_name` | Naam voor logs | auto |
| `--data_dir` | Pad naar data | ./btc_l2_data |
| `--max_files` | Max data bestanden | 100 |
| `--eval_freq` | Hoe vaak evalueren | 10,000 |

**⏱️ Duur:** 
- 100k steps: ~10-30 min (GPU) / ~1-2 uur (CPU)
- 1M steps: ~2-5 uur (GPU) / ~12-24 uur (CPU)

---

## STAP 4: Monitor Training (tijdens training)

### 3.1 Start TensorBoard
```powershell
# In een NIEUWE terminal (laat training draaien)
cd c:\code\dataDeepRL
.\.venv\Scripts\Activate.ps1

# Start TensorBoard
tensorboard --logdir logs/
```

Open browser: **http://localhost:6006**

### 3.2 MLflow (optioneel)
```powershell
# Start MLflow UI
mlflow ui --backend-store-uri logs/mlruns
```

Open browser: **http://localhost:5000**

### 3.3 Wat te monitoren?

| Metric | Goed teken | Slecht teken |
|--------|------------|--------------|
| `reward/mean` | Stijgend | Vlak of dalend |
| `loss/policy` | Stabiel, laag | Spiekt omhoog |
| `portfolio/value` | Stijgend | Crasht naar 0 |
| `episode/length` | Consistent | Heel kort (agent faillt) |

---

## STAP 5: Hyperparameter Tuning (Optioneel)

### 4.1 Quick search (10 trials)
```powershell
python tune_hyperparams.py --algorithm ppo --trials 10
```

### 4.2 Full search (aanbevolen)
```powershell
# Uitgebreide search met timeout
python tune_hyperparams.py --algorithm sac --trials 50 --timeout 7200

# Hervat eerdere search
python tune_hyperparams.py --algorithm sac --resume --study_name optuna_sac
```

### 4.3 Beste parameters gebruiken
```powershell
# Bekijk beste parameters
cat best_hyperparams_ppo.json

# Train met beste parameters (handmatig invoeren)
python train/train_ppo_deeplob.py \
    --learning_rate 0.00025 \
    --gamma 0.995 \
    --batch_size 128 \
    ...
```

---

## STAP 6: Model Evalueren

### 5.1 Basis evaluatie
```powershell
# Evalueer op test data
python evaluate.py \
    --model_path logs/ppo_deeplob_v1/best_model.pt \
    --algo ppo \
    --n_episodes 10
```

### 5.2 Met visualisaties
```powershell
# Genereer plots
python evaluate.py \
    --model_path logs/sac_deeplob_v1/best_model.pt \
    --algo sac \
    --n_episodes 20 \
    --plot \
    --save_trades
```

### 5.3 Verwachte output
```
============================================================
EVALUATION SUMMARY
============================================================
Mean Return:      +2.34% ± 1.12%
Mean Final Value: $10,234.00
Mean Sharpe:      1.456
Mean Max DD:      5.23%
Mean Trades:      45.2
Mean Win Rate:    52.3%
============================================================
```

---

## 🔧 Troubleshooting

### Probleem: CUDA out of memory
```powershell
# Verlaag batch size
python train/train_ppo_deeplob.py --batch_size 64

# Of gebruik CPU
python train/train_ppo_deeplob.py --device cpu
```

### Probleem: Training convergeert niet
```powershell
# Verlaag learning rate
python train/train_ppo_deeplob.py --learning_rate 0.0001

# Meer data gebruiken
python train/train_ppo_deeplob.py --max_files 200
```

### Probleem: Geen data bestanden
```powershell
# Controleer of data er is
Get-ChildItem btc_l2_data\*.parquet | Measure-Object

# Download opnieuw
python binance_l2.py
```

---

## 📊 Quick Reference - Alle Commando's

```powershell
# ============== SETUP ==============
cd c:\code\dataDeepRL
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# ============== DATA ==============
python binance_l2.py

# ============== PREPROCESSING ==============
python dataVerwerken/preprocess_data.py
python dataVerwerken/create_core_data.py

# ============== DEEPLOB PRE-TRAIN ==============
python train/train_deeplob_pretrain.py --epochs 50 --save_path models/deeplob_pretrained.pt

# ============== TRAIN (kies 1) ==============
# Beste resultaten (met pre-trained DeepLOB):
python train/train_sac_with_deeplob.py --deeplob_model models/deeplob_pretrained.pt --freeze_deeplob
python train/train_ppo_with_deeplob.py --deeplob_model models/deeplob_pretrained.pt --freeze_deeplob

# End-to-end (zonder pre-training):
python train/train_ppo_deeplob.py --config configs/default.yaml
python train/train_sac_deeplob.py --config configs/default.yaml

# Simpel (MLP, voor testen):
python train/train_ppo_only.py --experiment_name mijn_eerste_model
python train/train_sac_only.py --experiment_name mijn_eerste_model

# ============== MONITOR ==============
tensorboard --logdir logs/

# ============== TUNE ==============
python tune_hyperparams.py --algorithm ppo --trials 20

# ============== EVALUATE ==============
python evaluate.py --model_path logs/*/best_model.pt --algo ppo --plot
```

---

## 🎯 Aanbevolen Workflow

### Voor Beginners (snel testen)
1. **Data:** `python binance_l2.py` (download 1 week data)
2. **Preprocessing:** `python dataVerwerken/preprocess_data.py`
3. **Split:** `python dataVerwerken/create_core_data.py`
4. **Train simpel:** `python train/train_ppo_only.py --total_steps 50000`
5. **Monitor:** TensorBoard openen
6. **Evalueer:** `python evaluate.py --model_path logs/*/best_model.pt --algo ppo`

### Voor Beste Resultaten (productie)
1. **Data:** Download 3-6 maanden data (`python binance_l2.py`)
2. **Preprocessing:** `python dataVerwerken/preprocess_data.py`
3. **Split:** `python dataVerwerken/create_core_data.py`
4. **Pre-train DeepLOB:** `python train/train_deeplob_pretrain.py --epochs 50`
5. **Train RL:** `python train/train_sac_with_deeplob.py --deeplob_model models/deeplob_pretrained.pt --freeze_deeplob`
6. **Tune:** `python tune_hyperparams.py --algorithm sac --trials 50`
7. **Evalueer:** `python evaluate.py --algo sac --plot`

---

## 🛠️ Utility Scripts

### Parquet bekijken
```powershell
# Bekijk eerste 10 rijen van een parquet bestand
python print_parquet_head.py
# Plak dan het pad, bijv: btc_l2_data\BTCUSDT_2025-01-15.parquet
```

### Parquet naar CSV converteren
```powershell
# Pas BESTAND aan in parq.py, dan:
python parq.py

# Of gebruik dataVerwerken/parquet_to_csv.py
python dataVerwerken/parquet_to_csv.py
```

---

## 📖 Meer Informatie

Voor gedetailleerde technische documentatie over de architectuur, algoritmes en modules:

→ Zie **[ARCHITECTURE.md](ARCHITECTURE.md)**

Dit bevat:
- DeepLOB CNN+LSTM architectuur uitleg
- PPO en SAC algoritme details
- Training loop diagrammen
- Hyperparameter tabellen
- Troubleshooting tips

---

## ⚙️ Configuratie Aanpassen

Alle default hyperparameters staan in `configs/default.yaml`:

```yaml
# Belangrijkste instellingen:
training:
  total_steps: 1000000
  device: "auto"       # cuda/cpu/auto

sac:
  learning_rate: 0.0003
  batch_size: 256
  buffer_size: 1000000

ppo:
  learning_rate: 0.0003
  n_epochs: 10
  clip_epsilon: 0.2
```

Gebruik met:
```powershell
python train/train_ppo_deeplob.py --config configs/default.yaml
```

Succes! 🚀
