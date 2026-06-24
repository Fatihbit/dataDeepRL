"""
DeepLOB Pre-Training Script
===========================

Dit script traint het DeepLOB neural network model met SUPERVISED LEARNING
op de taak van price direction prediction. Het getrainde model wordt daarna
gebruikt als feature extractor voor de RL agents (SAC en PPO).

WAAROM PRE-TRAINING?
--------------------
1. Transfer Learning:
   - DeepLOB leert eerst algemene market patterns
   - Deze kennis wordt overgedragen naar de RL agents
   - Snellere RL training en betere start positie

2. Stabiele Features:
   - Pre-trained features zijn betekenisvol
   - RL agents hoeven niet vanaf scratch te leren
   - Minder kans op "catastrophic forgetting"

3. Sample Efficiency:
   - Supervised learning is efficiënter dan RL voor feature learning
   - Meer data beschikbaar voor supervised taak
   - RL kan focussen op decision making

WAT LEERT DEEPLOB?
------------------
Het model voorspelt de prijsrichting van de volgende tijdstap:

    Input:  Order book data (window van N tijdstappen)
             ↓
    Output: Klasse (0=Down, 1=Neutral, 2=Up)

De features die DeepLOB leert zijn relevant voor trading:
- Bid/Ask imbalance patterns
- Volume spikes en anomalieën
- Spread veranderingen
- Korte-termijn momentum
- Order flow dynamics

CLASSIFICATIE LABELS:
---------------------
De prijsverandering wordt geclassificeerd met een threshold:

    price_change = (price_t+1 - price_t) / price_t

    Label 0 (Down):    price_change < -threshold
    Label 1 (Neutral): -threshold <= price_change <= threshold
    Label 2 (Up):      price_change > threshold

Default threshold = 0.0001 (0.01%)

WORKFLOW:
---------
1. Laad en verwerk order book data
2. Maak sliding window sequences
3. Train DeepLOB met cross-entropy loss
4. Sla beste model op naar ./models/deeplob_pretrained.pt

GEBRUIK:
--------
Train het model:
    python train_deeplob_pretrain.py --data_dir ../btc_l2_data --epochs 50

Gebruik met SAC:
    python train_sac_with_deeplob.py --deeplob_model ./models/deeplob_pretrained.pt

Gebruik met PPO:
    python train_ppo_with_deeplob.py --deeplob_model ./models/deeplob_pretrained.pt

Auteur: DataDeepRL Team
"""

import os
import sys
import argparse
import datetime
import time
import signal
import csv
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Voeg src toe aan path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.deeplob import DeepLOB
from src.utils.logger import TrainingLogger, setup_logging

warnings.filterwarnings('ignore')


class OrderBookDataset(Dataset):
    """
    PyTorch Dataset voor order book sequences met price direction labels.
    
    Lazy-loading: bewaart alleen de platte features en genereert
    sequences on-the-fly om geheugen te besparen.
    
    Labels worden berekend op basis van ABSOLUTE prijsverschillen
    (niet percentage returns), zodat de threshold onafhankelijk is
    van het absolute prijsniveau.
    
    Args:
        features: Platte feature array (N, num_features)
        prices: Prijzen array (N,)
        seq_len: Sequence length (lookback window)
        threshold: Threshold voor up/down classificatie (absolute verschil)
    """
    
    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        seq_len: int = 100,
        threshold: float = 0.0001,
        name: str = 'dataset'
    ):
        # Sla features op als numpy (niet torch) - torch tensors in shared memory
        # veroorzaken deadlocks met num_workers>0 op Windows
        self.features = features.astype(np.float32) if not isinstance(features, np.ndarray) else features
        self.seq_len = seq_len
        
        # Bereken ABSOLUTE price differences voor labels
        # Niet percentage returns! Die zijn afhankelijk van het prijsniveau
        # bij z-score genormaliseerde data.
        diffs = np.zeros(len(prices), dtype=np.float64)
        diffs[:-1] = prices[1:] - prices[:-1]
        
        labels = np.ones(len(diffs), dtype=np.int64)
        labels[diffs < -threshold] = 0
        labels[diffs > threshold] = 2
        self.labels = labels
        
        self.n_samples = len(features) - seq_len
        
        # Print class distributie (alleen voor samples die we echt gebruiken)
        used_labels = labels[seq_len:]
        unique, counts = np.unique(used_labels, return_counts=True)
        total = len(used_labels)
        label_names = {0: 'Down', 1: 'Neutral', 2: 'Up'}
        print(f"Class distribution [{name}]:")
        for u, c in zip(unique, counts):
            print(f"  {label_names[u]:>7s} ({u}): {c:,} ({100*c/total:.1f}%)")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        seq = torch.from_numpy(self.features[idx:idx + self.seq_len].copy())
        label = self.labels[idx + self.seq_len]
        return seq, label


class DeepLOBClassifier(nn.Module):
    """
    DeepLOB model met classification head voor supervised pre-training.
    
    Dit is een wrapper rond het basis DeepLOB model die een classification
    layer toevoegt voor 3-class price direction prediction.
    
    ARCHITECTUUR:
    -------------
        Order Book Sequence
        (batch, seq_len, features)
              │
              ▼
        ┌─────────────────────┐
        │      DeepLOB        │
        │  (Feature Extractor)│
        │                     │
        │  Conv1D Blocks      │
        │       ↓             │
        │  Inception Module   │
        │       ↓             │
        │  Bidirectional LSTM │
        │       ↓             │
        │  Attention Pooling  │
        └─────────────────────┘
              │
              ▼
        Feature Vector
        (batch, lstm_hidden * 2)
              │
              ▼
        ┌─────────────────────┐
        │  Classification Head│
        │                     │
        │  Linear → ReLU      │
        │       ↓             │
        │  Dropout            │
        │       ↓             │
        │  Linear → Logits    │
        └─────────────────────┘
              │
              ▼
        Class Logits
        (batch, 3)
    
    TRAINING:
    ---------
    - Loss: CrossEntropyLoss (met class weights voor imbalance)
    - Optimizer: AdamW met weight decay
    - LR Schedule: ReduceLROnPlateau
    
    NA TRAINING:
    ------------
    Alleen de DeepLOB backbone (self.deeplob) wordt opgeslagen en
    gebruikt in SAC/PPO. De classification head wordt weggegooid.
    
    Args:
        input_dim: Aantal input features per timestep
        hidden_dim: Hidden dimension voor DeepLOB convolutions
        lstm_hidden: LSTM hidden dimension
        num_classes: Aantal output classes (default: 3)
        dropout: Dropout rate voor regularisatie
    """
    
    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 64,
        lstm_hidden: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # =====================================
        # DEEPLOB BACKBONE
        # =====================================
        # Dit is het deel dat later hergebruikt wordt
        self.deeplob = DeepLOB(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            output_dim=lstm_hidden * 2,  # Bidirectional output
            dropout=dropout
        )
        
        # =====================================
        # CLASSIFICATION HEAD
        # =====================================
        # Dit deel wordt weggegooid na pre-training
        # Het is alleen nodig voor de supervised learning taak
        self.classifier = nn.Sequential(
            # Eerste layer: comprimeer features
            nn.Linear(lstm_hidden * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Regularisatie
            
            # Output layer: logits voor 3 klassen
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass voor classificatie training.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            logits: Ongenormaliseerde class scores (batch, num_classes)
                    Gebruik F.softmax() voor probabilities
        """
        # DeepLOB feature extraction
        features = self.deeplob(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x):
        """
        Haal alleen features op (zonder classification).
        
        Dit is hoe SAC/PPO het model gebruiken:
        alleen de DeepLOB features, niet de classificatie.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            features: Extracted feature vector (batch, lstm_hidden * 2)
        """
        return self.deeplob(x)
    
    def predict_proba(self, x):
        """
        Voorspel class probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            probs: Class probabilities (batch, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x):
        """
        Voorspel de meest waarschijnlijke klasse.
        
        Args:
            x: Input tensor
            
        Returns:
            predictions: Predicted class indices (batch,)
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)


def parse_args():
    """
    Parse command line arguments.
    
    Dit definieert alle configureerbare parameters voor training.
    Je kunt deze overschrijven via de command line.
    
    Returns:
        Namespace met alle argument waarden
    """
    parser = argparse.ArgumentParser(
        description='Pre-train DeepLOB for price direction prediction'
    )
    
    # =====================================
    # DATA ARGUMENTS
    # =====================================
    parser.add_argument('--data_dir', type=str, default='./coreData',
                        help='Directory met data (coreData/ of btc_l2_data/)')
    parser.add_argument('--max_files', type=int, default=100,
                        help='Maximum aantal bestanden om te laden (alleen raw modus)')
    parser.add_argument('--max_rows', type=int, default=90_000_000,
                        help='Maximum aantal rijen om te laden per split (voor memory)')
    parser.add_argument('--sequence_length', type=int, default=100,
                        help='Lengte van input sequences (lookback window)')
    parser.add_argument('--threshold', type=str, default='auto',
                        help='Threshold voor up/down classificatie. \"auto\" berekent uit data (aanbevolen)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing voor regularisatie (0.0 = uit, 0.1 = aanbevolen)')
    parser.add_argument('--feature_set', type=str, default='stationary',
                        choices=['all', 'stationary'],
                        help='Feature set: "all" (alle features), "stationary" (alleen tijdsinvariante features)')
    
    # =====================================
    # MODEL ARGUMENTS
    # =====================================
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='CNN hidden dimension')
    parser.add_argument('--lstm_hidden', type=int, default=64,
                        help='LSTM hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate voor regularisatie')
    
    # =====================================
    # TRAINING ARGUMENTS
    # =====================================
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum aantal training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size voor training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initiële learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularisatie)')
    parser.add_argument('--patience', type=int, default=1000,
                        help='Early stopping patience (epochs zonder verbetering)')
    
    # =====================================
    # OUTPUT ARGUMENTS
    # =====================================
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory voor opslaan van model')
    parser.add_argument('--model_name', type=str, default='deeplob_pretrained',
                        help='Naam voor het opgeslagen model')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory voor training logs')
    
    # =====================================
    # OTHER ARGUMENTS
    # =====================================
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed voor reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: "auto", "cuda", of "cpu"')
    parser.add_argument('--resume', type=str, default=None,
                        help='Pad naar checkpoint om training te hervatten')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch=None, total_epochs=None):
    """
    Train het model voor één epoch.
    
    Args:
        model: DeepLOBClassifier model
        dataloader: Training DataLoader
        optimizer: Optimizer (bijv. AdamW)
        criterion: Loss functie (CrossEntropyLoss)
        device: Torch device
        epoch: Huidig epoch nummer (voor progress bar)
        total_epochs: Totaal aantal epochs (voor progress bar)
        
    Returns:
        avg_loss: Gemiddelde loss over alle batches
        accuracy: Classificatie accuracy (%)
    """
    # Zet model in training mode
    # Dit activeert dropout en batch normalization in training mode
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    desc = f"Epoch {epoch}/{total_epochs} [Train]" if epoch else "Training"
    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        # Verplaats data naar device (GPU/CPU)
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # Clear gradients van vorige iteratie
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        output = model(data)
        
        # Bereken loss
        loss = criterion(output, target)
        
        # Backward pass (bereken gradients)
        loss.backward()
        
        # Gradient clipping om exploding gradients te voorkomen
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1)  # Predicted class
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'loss': f"{total_loss/(batch_idx+1):.4f}",
            'acc': f"{100.*correct/total:.1f}%"
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device, collect_predictions=False):
    """
    Valideer het model op de validation set.
    
    Dit evalueert het model zonder gradient berekening (sneller).
    We berekenen ook per-class accuracy om imbalance te detecteren.
    
    Args:
        model: DeepLOBClassifier model
        dataloader: Validation DataLoader
        criterion: Loss functie
        device: Torch device
        collect_predictions: Als True, return ook alle predictions en labels
        
    Returns:
        avg_loss: Gemiddelde validation loss
        accuracy: Overall accuracy (%)
        class_acc: Per-class accuracy [down_acc, neutral_acc, up_acc]
        all_preds, all_targets: (alleen als collect_predictions=True)
    """
    # Zet model in evaluation mode
    # Dit deactiveert dropout en zet batch norm in inference mode
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = [0, 0, 0]  # [down, neutral, up]
    class_total = [0, 0, 0]
    all_preds = [] if collect_predictions else None
    all_targets = [] if collect_predictions else None
    
    # Geen gradient berekening nodig voor validatie
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Track statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Per-class statistics
            for i in range(3):
                mask = target == i  # Samples van deze klasse
                class_correct[i] += pred[mask].eq(target[mask]).sum().item()
                class_total[i] += mask.sum().item()
            
            if collect_predictions:
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
    
    # Bereken per-class accuracy
    class_acc = [
        100. * c / max(t, 1)  # max(t,1) voorkomt deling door 0
        for c, t in zip(class_correct, class_total)
    ]
    
    if collect_predictions:
        return (total_loss / len(dataloader), 100. * correct / total, class_acc,
                np.concatenate(all_preds), np.concatenate(all_targets))
    return total_loss / len(dataloader), 100. * correct / total, class_acc


def main():
    """
    Hoofdfunctie voor DeepLOB pre-training.
    
    Dit is de complete training pipeline:
    
    1. SETUP:
       - Parse arguments
       - Set random seeds
       - Configure device
    
    2. DATA LOADING:
       - Laad parquet bestanden
       - Maak features
       - Split in train/val/test
       - Creëer DataLoaders
    
    3. MODEL SETUP:
       - Initialiseer DeepLOBClassifier
       - Stel class weights in voor imbalance
       - Configureer optimizer en scheduler
    
    4. TRAINING LOOP:
       - Train voor max epochs
       - Valideer na elke epoch
       - Early stopping als geen verbetering
       - Sla beste model op
    
    5. FINAL EVALUATION:
       - Laad beste model
       - Evalueer op test set
       - Print resultaten
    """
    args = parse_args()
    
    # =====================================
    # EXPERIMENT SETUP
    # =====================================
    experiment_name = f"deeplob_pretrain_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    setup_logging(log_dir=args.log_dir, experiment_name=experiment_name)
    
    # =====================================
    # PLAINTEXT LOG FILE
    # =====================================
    log_txt_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(log_txt_dir, exist_ok=True)
    log_txt_path = os.path.join(log_txt_dir, 'training_output.log')
    
    # Tee: schrijf alles naar zowel console als log file
    class TeeLogger:
        def __init__(self, log_path, original_stream):
            self.log_file = open(log_path, 'a', encoding='utf-8')
            self.original = original_stream
        def write(self, msg):
            self.original.write(msg)
            self.log_file.write(msg)
            self.log_file.flush()
        def flush(self):
            self.original.flush()
            self.log_file.flush()
        def close(self):
            self.log_file.close()
    
    tee_stdout = TeeLogger(log_txt_path, sys.stdout)
    tee_stderr = TeeLogger(log_txt_path, sys.stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"DeepLOB Pre-Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Threshold: {args.threshold}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Log file: {log_txt_path}")
    print(f"{'='*60}\n")
    
    # Random seed voor reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # =====================================
    # DATA LOADING
    # =====================================
    print("Loading data...")
    import pyarrow.parquet as pq

    train_parquet = os.path.join(args.data_dir, 'train.parquet')
    if not os.path.exists(train_parquet):
        raise FileNotFoundError(
            f"coreData niet gevonden in {args.data_dir}. "
            f"Voer eerst preprocess_data.py + create_core_data.py uit."
        )

    print(f"Using pre-processed coreData from {args.data_dir}")
    schema = pq.read_schema(train_parquet)
    all_feature_cols = [n for n in schema.names if n != 'timestamp']
    price_col = 'close'

    # Non-stationary features (price levels, moving averages) drift across
    # train/val under z-score normalisation. Stationary set generalises better.
    STATIONARY_FEATURES = [
        'spread', 'spread_pct', 'buy_ratio',
        'return_5s', 'return_10s', 'return_30s', 'return_60s',
        'volatility_10', 'volatility_30', 'volatility_60',
        'momentum_10', 'momentum_30',
        'rsi_14', 'volume_ratio', 'order_imbalance',
    ]

    if args.feature_set == 'stationary':
        feature_cols = [c for c in all_feature_cols if c in STATIONARY_FEATURES]
        dropped = [c for c in all_feature_cols if c not in STATIONARY_FEATURES]
        print(f"Feature set: stationary ({len(feature_cols)} features)")
        print(f"  Dropped {len(dropped)} non-stationary: {dropped[:5]}{'...' if len(dropped)>5 else ''}")
    else:
        feature_cols = all_feature_cols
        print(f"Feature set: all ({len(feature_cols)} features)")

    print(f"Using features: {feature_cols}")
    print(f"Max rows per split: {args.max_rows:,}")

    def load_split_data(parquet_path, max_rows, feature_cols, price_col):
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.metadata.num_rows
        load_cols = list(dict.fromkeys(feature_cols + [price_col]))

        if total_rows <= max_rows:
            df = pf.read(columns=load_cols).to_pandas()
        else:
            num_rg = pf.metadata.num_row_groups
            rg_rows = [pf.metadata.row_group(i).num_rows for i in range(num_rg)]
            cumsum = 0
            start_rg = num_rg
            for i in range(num_rg - 1, -1, -1):
                cumsum += rg_rows[i]
                start_rg = i
                if cumsum >= max_rows:
                    break
            import pyarrow as pa
            tables = [pf.read_row_group(i, columns=load_cols) for i in range(start_rg, num_rg)]
            df = pa.concat_tables(tables).to_pandas()
            del tables
            df = df.tail(max_rows).reset_index(drop=True)

        print(f"  Loaded {len(df):,} / {total_rows:,} rows from {os.path.basename(parquet_path)}")
        features = df[feature_cols].values.astype(np.float32)
        prices = df[price_col].values.astype(np.float64)
        del df
        return features, prices

    train_features, train_prices = load_split_data(
        train_parquet, args.max_rows, feature_cols, price_col)
    val_features, val_prices = load_split_data(
        os.path.join(args.data_dir, 'val.parquet'),
        args.max_rows // 4, feature_cols, price_col)
    test_features, test_prices = load_split_data(
        os.path.join(args.data_dir, 'test.parquet'),
        args.max_rows // 4, feature_cols, price_col)

    input_dim = len(feature_cols)
    
    print(f"\nData loaded:")
    print(f"  Train: {len(train_features):,} rows")
    print(f"  Val:   {len(val_features):,} rows")
    print(f"  Test:  {len(test_features):,} rows")
    print(f"  Features: {input_dim}")
    
    # =====================================
    # COMPUTE THRESHOLD
    # =====================================
    # Bereken ABSOLUTE prijsverschillen (niet pct returns!) voor threshold
    # Dit is cruciaal bij z-score genormaliseerde data: pct returns hangen af
    # van het absolute prijsniveau, absolute diffs niet.
    train_diffs = np.diff(train_prices)
    val_diffs = np.diff(val_prices)
    
    if args.threshold == 'auto':
        # Adaptieve threshold: median absolute difference
        # Dit geeft ~25% Down, ~50% Neutral, ~25% Up in train data
        threshold = float(np.median(np.abs(train_diffs)))
        print(f"\nAuto threshold (abs diff): {threshold:.8f}")
    else:
        threshold = float(args.threshold)
        print(f"\nFixed threshold: {threshold:.8f}")
    
    # Toon return statistieken voor diagnostics
    print(f"\nPrice diff statistics (absolute differences):")
    print(f"  Train: mean={np.mean(train_diffs):.8f}, std={np.std(train_diffs):.8f}, "
          f"median_abs={np.median(np.abs(train_diffs)):.8f}")
    print(f"  Val:   mean={np.mean(val_diffs):.8f}, std={np.std(val_diffs):.8f}, "
          f"median_abs={np.median(np.abs(val_diffs)):.8f}")
    
    # Toon verwachte class verdeling met deze threshold
    for name, dfs in [('Train', train_diffs), ('Val', val_diffs)]:
        n_down = np.sum(dfs < -threshold)
        n_up = np.sum(dfs > threshold)
        n_neutral = len(dfs) - n_down - n_up
        total = len(dfs)
        print(f"  {name} expected: Down={100*n_down/total:.1f}%, Neutral={100*n_neutral/total:.1f}%, Up={100*n_up/total:.1f}%")
    
    # =====================================
    # CREATE DATASETS
    # =====================================
    print("\nCreating datasets...")
    
    # Lazy datasets: sequences worden on-the-fly gegenereerd
    train_dataset = OrderBookDataset(train_features, train_prices, args.sequence_length, threshold, name='train')
    val_dataset = OrderBookDataset(val_features, val_prices, args.sequence_length, threshold, name='val')
    test_dataset = OrderBookDataset(test_features, test_prices, args.sequence_length, threshold, name='test')
    
    # DataLoaders voor batching
    # pin_memory=True voor snellere CPU→GPU transfer
    # num_workers>0 voor parallel data loading (verlaagt CPU bottleneck)
    use_cuda = device == 'cuda'
    loader_kwargs = {
        'batch_size': args.batch_size,
        'pin_memory': use_cuda,       # Snellere CPU→GPU transfer via pinned memory
        'num_workers': 0,             # Windows multiprocessing deadlockt met PyTorch
    }
    
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, 
        shuffle=False,
        **loader_kwargs
    )
    
    # =====================================
    # MODEL SETUP
    # =====================================
    print("\nCreating model...")
    
    # input_dim is al bepaald in de data loading sectie
    
    model = DeepLOBClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        lstm_hidden=args.lstm_hidden,
        num_classes=3,  # down, neutral, up
        dropout=args.dropout
    ).to(device)
    
    # Tel totaal aantal parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # =====================================
    # CLASS WEIGHTS VOOR IMBALANCED DATA
    # =====================================
    # Financial data is vaak imbalanced (veel "neutral", weinig "up/down")
    # Class weights compenseren hiervoor
    
    # Class weights alleen berekend op GEBRUIKTE labels (na seq_len)
    train_labels = train_dataset.labels[args.sequence_length:]
    class_counts = np.bincount(train_labels, minlength=3)
    
    # Inverse frequency weighting: zeldzame klassen krijgen hoger gewicht
    # Dit zorgt dat het model niet alleen "neutral" voorspelt
    class_weights = 1.0 / (class_counts + 1)  # +1 voorkomt deling door 0
    class_weights = class_weights / class_weights.sum() * 3  # Normaliseer
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class weights: Down={class_weights[0]:.2f}, Neutral={class_weights[1]:.2f}, Up={class_weights[2]:.2f}")
    
    # =====================================
    # LOSS EN OPTIMIZER
    # =====================================
    # CrossEntropyLoss met class weights en label smoothing voor regularisatie
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    
    # AdamW: Adam met weight decay (L2 regularisatie)
    # Beter dan vanilla Adam voor deep learning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler: verlaag LR als validation loss niet verbetert
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',          # Minimaliseer loss
        factor=0.5,          # Halveer LR
        patience=5,          # Wacht 5 epochs
    )
    
    # =====================================
    # RESUME FROM CHECKPOINT
    # =====================================
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    resumed_history = None
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, f'{args.model_name}.pt')
    checkpoint_path = os.path.join(args.save_dir, f'{args.model_name}_checkpoint.pt')
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            best_val_acc = ckpt.get('best_val_acc', 0)
            patience_counter = ckpt.get('patience_counter', 0)
            if 'history' in ckpt:
                resumed_history = ckpt['history']
            print(f"  Continuing from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
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
        print("\n[PAUSE] Pause requested! Saving checkpoint after current epoch...")
    
    signal.signal(signal.SIGINT, _signal_handler)
    
    # =====================================
    # TRAINING LOOP
    # =====================================
    print("\nStarting training...")
    
    # Training history voor plots
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'class_accs': [], 'lrs': []
    }
    if resumed_history is not None:
        for key in history:
            if key in resumed_history:
                history[key] = resumed_history[key]
        print(f"  Restored history: {len(history['train_loss'])} epochs")
    
    # CSV training log
    log_csv_path = os.path.join(args.save_dir, f'{args.model_name}_training_log.csv')
    csv_header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                  'down_acc', 'neutral_acc', 'up_acc', 'lr', 'epoch_time_s', 'total_time_s']
    csv_mode = 'a' if (args.resume and os.path.exists(log_csv_path)) else 'w'
    csv_file = open(log_csv_path, csv_mode, newline='')
    csv_writer = csv.writer(csv_file)
    if csv_mode == 'w':
        csv_writer.writerow(csv_header)
    
    training_start = time.time()
    
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            
            # ---------------------------------
            # TRAIN EPOCH
            # ---------------------------------
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device,
                epoch=epoch, total_epochs=args.epochs
            )
            
            # ---------------------------------
            # VALIDATE
            # ---------------------------------
            val_loss, val_acc, class_acc = validate(
                model, val_loader, criterion, device
            )
            
            # ---------------------------------
            # UPDATE LR SCHEDULER
            # ---------------------------------
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # ---------------------------------
            # TIMING
            # ---------------------------------
            epoch_time = time.time() - epoch_start
            total_time = time.time() - training_start
            remaining = (args.epochs - epoch) * epoch_time
            
            # ---------------------------------
            # LOGGING
            # ---------------------------------
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"[D:{class_acc[0]:.1f}% N:{class_acc[1]:.1f}% U:{class_acc[2]:.1f}%] | "
                f"LR: {current_lr:.1e} | {epoch_time:.0f}s | ETA: {remaining/60:.1f}min"
            )
            
            # History bijhouden voor plots
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['class_accs'].append({
                'Down': class_acc[0], 'Neutral': class_acc[1], 'Up': class_acc[2]
            })
            history['lrs'].append(current_lr)
            
            # CSV log
            csv_writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.2f}",
                f"{val_loss:.6f}", f"{val_acc:.2f}",
                f"{class_acc[0]:.2f}", f"{class_acc[1]:.2f}", f"{class_acc[2]:.2f}",
                f"{current_lr:.2e}", f"{epoch_time:.1f}", f"{total_time:.1f}"
            ])
            csv_file.flush()
            
            # ---------------------------------
            # SAVE BEST MODEL
            # ---------------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'deeplob_state_dict': model.deeplob.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'class_acc': {'Down': class_acc[0], 'Neutral': class_acc[1], 'Up': class_acc[2]},
                    'config': {
                        'input_dim': input_dim,
                        'hidden_dim': args.hidden_dim,
                        'lstm_hidden': args.lstm_hidden,
                        'dropout': args.dropout,
                        'sequence_length': args.sequence_length
                    }
                }
                
                # Sla best model op (wordt overschreven bij verbetering)
                torch.save(save_dict, best_model_path)
                
                # Backup met prestatie-info in bestandsnaam
                backup_dir = os.path.join(args.save_dir, 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = (
                    f"{args.model_name}"
                    f"_ep{epoch}"
                    f"_vacc{val_acc:.1f}"
                    f"_vloss{val_loss:.4f}"
                    f"_{timestamp}.pt"
                )
                torch.save(save_dict, os.path.join(backup_dir, backup_name))
                
                print(f"  [*] Best model saved (loss: {val_loss:.4f}, acc: {val_acc:.2f}%)")
                print(f"  [BACKUP] {backup_name}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break
            
            # ---------------------------------
            # SAVE RESUME CHECKPOINT
            # ---------------------------------
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter,
                'history': history,
                'config': {
                    'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'lstm_hidden': args.lstm_hidden,
                    'dropout': args.dropout,
                    'sequence_length': args.sequence_length
                }
            }, checkpoint_path)
            
            # ---------------------------------
            # SAVE SNAPSHOT EVERY EPOCH
            # ---------------------------------
            try:
                snapshot_dir = os.path.join(log_txt_dir, 'plots', f'epoch_{epoch:03d}')
                os.makedirs(snapshot_dir, exist_ok=True)
                # Save parameter summary
                current_lr = optimizer.param_groups[0]['lr']
                with open(os.path.join(snapshot_dir, 'params.txt'), 'w') as pf:
                    pf.write(f"Snapshot at epoch {epoch}\n")
                    pf.write(f"{'='*50}\n")
                    pf.write(f"\n--- Hyperparameters ---\n")
                    pf.write(f"Learning rate (initial): {args.learning_rate}\n")
                    pf.write(f"Learning rate (current): {current_lr:.2e}\n")
                    pf.write(f"Batch size:              {args.batch_size}\n")
                    pf.write(f"Sequence length:         {args.sequence_length}\n")
                    pf.write(f"Hidden dim:              {args.hidden_dim}\n")
                    pf.write(f"LSTM hidden:             {args.lstm_hidden}\n")
                    pf.write(f"Dropout:                 {args.dropout}\n")
                    pf.write(f"Weight decay:            {args.weight_decay}\n")
                    pf.write(f"Label smoothing:         {args.label_smoothing}\n")
                    pf.write(f"Threshold:               {args.threshold}\n")
                    pf.write(f"Feature set:             {args.feature_set}\n")
                    pf.write(f"Max rows:                {args.max_rows:,}\n")
                    pf.write(f"Patience:                {args.patience}\n")
                    pf.write(f"Max epochs:              {args.epochs}\n")
                    pf.write(f"Seed:                    {args.seed}\n")
                    pf.write(f"Device:                  {device}\n")
                    pf.write(f"\n--- Training Progress ---\n")
                    pf.write(f"Current epoch:           {epoch}\n")
                    pf.write(f"Train loss:              {train_loss:.6f}\n")
                    pf.write(f"Train accuracy:          {train_acc:.2f}%\n")
                    pf.write(f"Val loss:                {val_loss:.6f}\n")
                    pf.write(f"Val accuracy:            {val_acc:.2f}%\n")
                    pf.write(f"Best val loss:           {best_val_loss:.6f}\n")
                    pf.write(f"Best val accuracy:       {best_val_acc:.2f}%\n")
                    pf.write(f"Patience counter:        {patience_counter}/{args.patience}\n")
                    pf.write(f"Class accuracies:\n")
                    pf.write(f"  Down:    {class_acc[0]:.2f}%\n")
                    pf.write(f"  Neutral: {class_acc[1]:.2f}%\n")
                    pf.write(f"  Up:      {class_acc[2]:.2f}%\n")
                    pf.write(f"\n--- Features ({len(feature_cols)}) ---\n")
                    for fc in feature_cols:
                        pf.write(f"  {fc}\n")
                print(f"  [SNAPSHOT] Snapshot saved: {snapshot_dir}")
            except Exception as e:
                print(f"  [WARN] Could not save snapshot: {e}")
            
            # ---------------------------------
            # CHECK PAUSE
            # ---------------------------------
            if _pause_requested:
                print(f"\n[PAUSE] Training paused at epoch {epoch}")
                print(f"  Resume: python train_deeplob_pretrain.py --resume {checkpoint_path} [other args]")
                break
    
    finally:
        csv_file.close()
        signal.signal(signal.SIGINT, _original_sigint)
    
    # =====================================
    # FINAL EVALUATION ON TEST SET
    # =====================================
    # Evalueer op test set die het model nog nooit gezien heeft
    # Dit geeft een eerlijke schatting van generalisatie
    
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    
    # Laad het beste opgeslagen model
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evalueer op test set (met predictions voor confusion matrix)
    test_loss, test_acc, test_class_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device, collect_predictions=True
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"  - Down accuracy:    {test_class_acc[0]:.2f}%")
    print(f"  - Neutral accuracy: {test_class_acc[1]:.2f}%")
    print(f"  - Up accuracy:      {test_class_acc[2]:.2f}%")
    
    # =====================================
    # SAVE FINAL PARAMETER SUMMARY
    # =====================================
    final_plot_dir = os.path.join(log_txt_dir, 'plots', 'final')
    os.makedirs(final_plot_dir, exist_ok=True)
    with open(os.path.join(final_plot_dir, 'params.txt'), 'w') as pf:
        pf.write(f"FINAL RESULTS\n")
        pf.write(f"{'='*50}\n")
        pf.write(f"\n--- Hyperparameters ---\n")
        pf.write(f"Learning rate (initial): {args.learning_rate}\n")
        pf.write(f"Batch size:              {args.batch_size}\n")
        pf.write(f"Sequence length:         {args.sequence_length}\n")
        pf.write(f"Hidden dim:              {args.hidden_dim}\n")
        pf.write(f"LSTM hidden:             {args.lstm_hidden}\n")
        pf.write(f"Dropout:                 {args.dropout}\n")
        pf.write(f"Weight decay:            {args.weight_decay}\n")
        pf.write(f"Label smoothing:         {args.label_smoothing}\n")
        pf.write(f"Threshold:               {args.threshold}\n")
        pf.write(f"Feature set:             {args.feature_set}\n")
        pf.write(f"Max rows:                {args.max_rows:,}\n")
        pf.write(f"Patience:                {args.patience}\n")
        pf.write(f"Max epochs:              {args.epochs}\n")
        pf.write(f"Seed:                    {args.seed}\n")
        pf.write(f"Device:                  {device}\n")
        pf.write(f"\n--- Final Test Results ---\n")
        pf.write(f"Test loss:               {test_loss:.6f}\n")
        pf.write(f"Test accuracy:           {test_acc:.2f}%\n")
        pf.write(f"Best val loss:           {best_val_loss:.6f}\n")
        pf.write(f"Best val accuracy:       {best_val_acc:.2f}%\n")
        pf.write(f"Stopped at epoch:        {epoch}\n")
        pf.write(f"Class accuracies (test):\n")
        pf.write(f"  Down:    {test_class_acc[0]:.2f}%\n")
        pf.write(f"  Neutral: {test_class_acc[1]:.2f}%\n")
        pf.write(f"  Up:      {test_class_acc[2]:.2f}%\n")
        pf.write(f"\n--- Features ({len(feature_cols)}) ---\n")
        for fc in feature_cols:
            pf.write(f"  {fc}\n")
    print(f"  [SNAPSHOT] Final snapshot saved: {final_plot_dir}")
    
    # =====================================
    # FINAL SUMMARY
    # =====================================
    print(f"\n{'='*60}")
    print(f"Pre-training Complete!")
    print(f"{'='*60}")
    print(f"\nModel saved to: {best_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    print(f"\n{'='*60}")
    print("Volgende Stappen:")
    print(f"{'='*60}")
    print("\nGebruik dit pre-trained DeepLOB model met SAC of PPO:")
    print(f"\n  1. SAC (off-policy, sample efficient):")
    print(f"     python train_sac_with_deeplob.py --deeplob_model {best_model_path}")
    print(f"\n  2. PPO (on-policy, stabiel):")
    print(f"     python train_ppo_with_deeplob.py --deeplob_model {best_model_path}")
    print(f"\nOptie: --freeze_deeplob om DeepLOB weights vast te houden")
    print(f"\n{'='*60}")
    
    # Herstel stdout/stderr en sluit log
    sys.stdout = tee_stdout.original
    sys.stderr = tee_stderr.original
    tee_stdout.close()
    tee_stderr.close()
    print(f"Full log saved to: {log_txt_path}")


if __name__ == '__main__':
    main()
