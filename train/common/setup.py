"""
Shared Setup Functions
======================

Gedeelde setup functionaliteit voor alle training scripts:
- Device configuratie
- Random seed setup
- Data loading
- Environment creation
- Logger setup

Auteur: DataDeepRL Team
"""

import os
import sys
import datetime
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional

# Voeg src toe aan path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataloader import BTCDataLoader
from src.envs.trading_env import CryptoTradingEnv
from src.utils.logger import TrainingLogger, setup_logging


def setup_device(args) -> torch.device:
    """
    Configureer en return het compute device.
    
    Args:
        args: Arguments met 'device' attribuut ('auto', 'cuda', of 'cpu')
        
    Returns:
        torch.device object
    """
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if device.type == 'cuda':
        print(f"[GPU] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[CPU] Using CPU")
    
    return device


def setup_seed(seed: int, device: torch.device) -> None:
    """
    Setup random seeds voor reproducibility.
    
    Args:
        seed: Random seed
        device: Compute device
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Voor volledige reproducibility (kan langzamer zijn)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    print(f"[SEED] Random seed: {seed}")


def load_data(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                             np.ndarray, np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray, np.ndarray]:
    """
    Laad en prepareer training/validation/test data.
    
    Args:
        args: Arguments met data configuratie
        
    Returns:
        Tuple van (train_sequences, train_targets, train_prices,
                   val_sequences, val_targets, val_prices,
                   test_sequences, test_targets, test_prices)
    """
    print("\nLoading data...")
    
    # Maak data loader
    data_loader = BTCDataLoader(
        window_size=args.sequence_length,
        data_dir=args.data_dir
    )
    
    # Laad data
    df = data_loader.load_data(max_files=args.max_files)
    if df is None or len(df) == 0:
        raise ValueError(f"Could not load data from {args.data_dir}")
    
    print(f"   Loaded {len(df):,} rows of data")
    
    # Feature engineering
    df = data_loader.create_features()
    
    # Prepareer sequences
    sequences, targets, prices = data_loader.prepare_sequences()
    print(f"   Prepared {len(sequences):,} sequences")
    
    # Split data
    train_ratio = getattr(args, 'train_ratio', 0.7)
    val_ratio = getattr(args, 'val_ratio', 0.15)
    
    train_data, val_data, test_data = data_loader.split_data(
        sequences, targets, prices,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    train_sequences, train_targets, train_prices = train_data
    val_sequences, val_targets, val_prices = val_data
    test_sequences, test_targets, test_prices = test_data
    
    print(f"   Train: {len(train_sequences):,}")
    print(f"   Val:   {len(val_sequences):,}")
    print(f"   Test:  {len(test_sequences):,}")
    
    return (train_sequences, train_targets, train_prices,
            val_sequences, val_targets, val_prices,
            test_sequences, test_targets, test_prices)


def create_environments(args, 
                       train_sequences: np.ndarray, 
                       train_prices: np.ndarray,
                       val_sequences: np.ndarray,
                       val_prices: np.ndarray) -> Tuple[CryptoTradingEnv, CryptoTradingEnv]:
    """
    Maak training en validation environments.
    
    Args:
        args: Arguments met environment configuratie
        train_sequences: Training feature sequences
        train_prices: Training prijzen
        val_sequences: Validation feature sequences  
        val_prices: Validation prijzen
        
    Returns:
        Tuple van (train_env, eval_env)
    """
    print("\n[ENV] Creating environments...")
    
    # Check voor random start optie
    random_start = getattr(args, 'random_start', False)
    reward_scaling = getattr(args, 'reward_scaling', 1.0)
    
    # Training environment (met randomisatie)
    train_env = CryptoTradingEnv(
        sequences=train_sequences,
        prices=train_prices,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        max_position=args.max_position,
        reward_scaling=reward_scaling,
        random_start=random_start,  # Random start voor training
        random_start_range=0.2
    )
    
    # Evaluation environment (zonder randomisatie voor consistente evaluatie)
    eval_env = CryptoTradingEnv(
        sequences=val_sequences,
        prices=val_prices,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        max_position=args.max_position,
        reward_scaling=reward_scaling,
        random_start=False  # Geen random start voor evaluatie
    )
    
    # Print environment info
    _, _ = train_env.reset()
    flat_obs = train_env.get_flat_observation()
    obs_dim = flat_obs.shape[0]
    action_dim = train_env.action_space.n
    
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Random start: {'Yes' if random_start else 'No'}")
    
    return train_env, eval_env


# =====================================
# STATIONARY FEATURES (matched met DeepLOB pretrain)
# =====================================
STATIONARY_FEATURES = [
    'spread', 'spread_pct', 'buy_ratio',
    'return_5s', 'return_10s', 'return_30s', 'return_60s',
    'volatility_10', 'volatility_30', 'volatility_60',
    'momentum_10', 'momentum_30',
    'rsi_14', 'volume_ratio', 'order_imbalance'
]


def load_coredata(
    data_dir: str = './coreData',
    sequence_length: int = 100,
    max_rows: int = 2_000_000,
    feature_cols: list = None,
    price_col: str = 'close'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Laad pre-processed coreData en maak sequences voor RL environments.
    
    Args:
        data_dir: Pad naar coreData directory (met train/val/test.parquet)
        sequence_length: Window grootte voor sequences
        max_rows: Maximum aantal rijen om te laden per split
        feature_cols: Welke feature kolommen te gebruiken (default: STATIONARY_FEATURES)
        price_col: Kolom met prijzen voor trading
        
    Returns:
        (train_sequences, train_prices, val_sequences, val_prices, test_sequences, test_prices)
        sequences shape: (N, sequence_length, num_features)
        prices shape: (N,) — denormalized to real USD values
    """
    import pyarrow.parquet as pq
    import json
    
    if feature_cols is None:
        feature_cols = STATIONARY_FEATURES
    
    train_parquet = os.path.join(data_dir, 'train.parquet')
    val_parquet = os.path.join(data_dir, 'val.parquet')
    test_parquet = os.path.join(data_dir, 'test.parquet')
    
    for p in [train_parquet, val_parquet, test_parquet]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"coreData bestand niet gevonden: {p}")
    
    # Laad normalization stats voor price denormalization
    norm_stats_path = os.path.join(data_dir, 'normalization_stats.json')
    price_mean, price_std = None, None
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        if 'stats' in norm_stats and price_col in norm_stats['stats']:
            price_mean = norm_stats['stats'][price_col]['mean']
            price_std = norm_stats['stats'][price_col]['std']
            print(f"  Price denormalization: {price_col} mean={price_mean:.2f}, std={price_std:.2f}")
    
    # Valideer dat feature kolommen bestaan
    schema = pq.read_schema(train_parquet)
    available = schema.names
    missing = [c for c in feature_cols if c not in available]
    if missing:
        raise ValueError(f"Features niet gevonden in coreData: {missing}")
    
    # Zorg dat price_col altijd geladen wordt
    load_cols = list(dict.fromkeys(feature_cols + [price_col]))
    
    def _load_split(parquet_path, max_rows_split):
        """Laad een data split en maak sequences."""
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.metadata.num_rows
        
        if total_rows <= max_rows_split:
            df = pf.read(columns=load_cols).to_pandas()
        else:
            # Lees de laatste row groups tot we genoeg rijen hebben
            num_rg = pf.metadata.num_row_groups
            rg_rows = [pf.metadata.row_group(i).num_rows for i in range(num_rg)]
            cumsum = 0
            start_rg = num_rg
            for i in range(num_rg - 1, -1, -1):
                cumsum += rg_rows[i]
                start_rg = i
                if cumsum >= max_rows_split:
                    break
            import pyarrow as pa
            tables = [pf.read_row_group(i, columns=load_cols) for i in range(start_rg, num_rg)]
            df = pa.concat_tables(tables).to_pandas()
            del tables
            df = df.tail(max_rows_split).reset_index(drop=True)
        
        print(f"  Loaded {len(df):,} / {total_rows:,} rows from {os.path.basename(parquet_path)}")
        
        features = df[feature_cols].values.astype(np.float32)
        prices = df[price_col].values.astype(np.float64)
        del df
        
        # Denormalize prices to real USD values for trading
        if price_mean is not None and price_std is not None:
            prices = prices * price_std + price_mean
        
        # Maak windowed sequences: (N - seq_len, seq_len, num_features)
        n_samples = len(features) - sequence_length
        if n_samples <= 0:
            raise ValueError(f"Niet genoeg data ({len(features)} rijen) voor sequence_length={sequence_length}")
        
        sequences = np.lib.stride_tricks.sliding_window_view(
            features, (sequence_length, features.shape[1])
        ).squeeze(axis=1).copy()  # (n_samples, seq_len, num_features)
        
        # Prijzen aligned met het einde van elke sequence
        seq_prices = prices[sequence_length:]
        
        return sequences, seq_prices
    
    print(f"Loading coreData from {data_dir} ({len(feature_cols)} features, seq_len={sequence_length})...")
    
    train_seq, train_prices = _load_split(train_parquet, max_rows)
    val_seq, val_prices = _load_split(val_parquet, max_rows // 4)
    test_seq, test_prices = _load_split(test_parquet, max_rows // 4)
    
    print(f"  Train sequences: {train_seq.shape}")
    print(f"  Val sequences:   {val_seq.shape}")
    print(f"  Test sequences:  {test_seq.shape}")
    
    return train_seq, train_prices, val_seq, val_prices, test_seq, test_prices


def load_coredata_streaming(
    data_dir: str = './coreData',
    sequence_length: int = 100,
    max_rows: int = 80_000_000,
    feature_cols: list = None,
    price_col: str = 'close'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient data loading: returns raw features + prices without
    creating the massive windowed sequences array.
    
    80M rows × 100 × 15 × 4 bytes = ~480 GB as sequences.
    80M rows × 15 × 4 bytes = ~4.8 GB as raw features.  ← This function.
    
    The trading_env creates windows on-the-fly with zero-copy numpy slicing.
    
    Returns:
        (train_features, train_prices, val_features, val_prices, test_features, test_prices)
        features shape: (N, num_features)  — raw feature rows
        prices shape: (N,) — denormalized to real USD values
    """
    import pyarrow.parquet as pq
    import json
    
    if feature_cols is None:
        feature_cols = STATIONARY_FEATURES
    
    train_parquet = os.path.join(data_dir, 'train.parquet')
    val_parquet = os.path.join(data_dir, 'val.parquet')
    test_parquet = os.path.join(data_dir, 'test.parquet')
    
    for p in [train_parquet, val_parquet, test_parquet]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"coreData bestand niet gevonden: {p}")
    
    # Laad normalization stats voor price denormalization
    norm_stats_path = os.path.join(data_dir, 'normalization_stats.json')
    price_mean, price_std = None, None
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        if 'stats' in norm_stats and price_col in norm_stats['stats']:
            price_mean = norm_stats['stats'][price_col]['mean']
            price_std = norm_stats['stats'][price_col]['std']
            print(f"  Price denormalization: {price_col} mean={price_mean:.2f}, std={price_std:.2f}")
    
    # Valideer dat feature kolommen bestaan
    schema = pq.read_schema(train_parquet)
    available = schema.names
    missing = [c for c in feature_cols if c not in available]
    if missing:
        raise ValueError(f"Features niet gevonden in coreData: {missing}")
    
    load_cols = list(dict.fromkeys(feature_cols + [price_col]))
    
    def _load_split_raw(parquet_path, max_rows_split):
        """Laad een data split als raw features + prices (geen sequences)."""
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.metadata.num_rows
        
        if total_rows <= max_rows_split:
            df = pf.read(columns=load_cols).to_pandas()
        else:
            num_rg = pf.metadata.num_row_groups
            rg_rows = [pf.metadata.row_group(i).num_rows for i in range(num_rg)]
            cumsum = 0
            start_rg = num_rg
            for i in range(num_rg - 1, -1, -1):
                cumsum += rg_rows[i]
                start_rg = i
                if cumsum >= max_rows_split:
                    break
            import pyarrow as pa
            tables = [pf.read_row_group(i, columns=load_cols) for i in range(start_rg, num_rg)]
            df = pa.concat_tables(tables).to_pandas()
            del tables
            df = df.tail(max_rows_split).reset_index(drop=True)
        
        print(f"  Loaded {len(df):,} / {total_rows:,} rows from {os.path.basename(parquet_path)}")
        
        features = df[feature_cols].values.astype(np.float32)
        prices = df[price_col].values.astype(np.float64)
        del df
        
        if price_mean is not None and price_std is not None:
            prices = prices * price_std + price_mean
        
        return features, prices
    
    print(f"Loading coreData STREAMING from {data_dir} ({len(feature_cols)} features, seq_len={sequence_length})...")
    print(f"  Memory-efficient mode: no sequence expansion")
    
    train_feat, train_prices = _load_split_raw(train_parquet, max_rows)
    val_feat, val_prices = _load_split_raw(val_parquet, max_rows // 4)
    test_feat, test_prices = _load_split_raw(test_parquet, max_rows // 4)
    
    n_train = len(train_feat) - sequence_length
    n_val = len(val_feat) - sequence_length
    n_test = len(test_feat) - sequence_length
    
    mem_raw = (train_feat.nbytes + val_feat.nbytes + test_feat.nbytes) / 1e9
    mem_seq = n_train * sequence_length * train_feat.shape[1] * 4 / 1e9
    
    print(f"  Train: {len(train_feat):,} rows -> {n_train:,} valid steps")
    print(f"  Val:   {len(val_feat):,} rows -> {n_val:,} valid steps")
    print(f"  Test:  {len(test_feat):,} rows -> {n_test:,} valid steps")
    print(f"  RAM used: {mem_raw:.1f} GB (vs {mem_seq:.1f} GB if expanded to sequences)")
    
    return train_feat, train_prices, val_feat, val_prices, test_feat, test_prices


def setup_logger(args, config: Dict[str, Any] = None) -> TrainingLogger:
    """
    Setup training logger.
    
    Args:
        args: Arguments met logging configuratie
        config: Optionele config dict om op te slaan
        
    Returns:
        Geconfigureerde TrainingLogger
    """
    # Genereer experiment naam als niet gegeven
    if args.experiment_name is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"experiment_{timestamp}"
    
    print(f"\n[LOG] Setting up logging...")
    print(f"   Experiment: {args.experiment_name}")
    print(f"   Log dir: {args.log_dir}")
    
    # Setup basis logging
    setup_logging(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    
    # Maak training logger
    use_mlflow = getattr(args, 'use_mlflow', False)
    use_tensorboard = not getattr(args, 'no_tensorboard', False)
    
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        use_tensorboard=use_tensorboard,
        use_mlflow=use_mlflow,
        log_interval=args.log_interval
    )
    
    # Sla config op
    if config is None:
        config = vars(args)
    logger.save_config(config)
    
    print(f"   TensorBoard: {'Yes' if use_tensorboard else 'No'}")
    print(f"   MLflow: {'Yes' if use_mlflow else 'No'}")
    
    return logger


def get_experiment_name(prefix: str, args) -> str:
    """
    Genereer experiment naam.
    
    Args:
        prefix: Prefix voor naam (bijv. 'ppo', 'sac')
        args: Arguments met optionele experiment_name
        
    Returns:
        Experiment naam string
    """
    if args.experiment_name is not None:
        return args.experiment_name
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}"


def print_training_header(title: str, args, device: torch.device) -> None:
    """
    Print training header met configuratie info.
    
    Args:
        title: Training titel
        args: Training arguments
        device: Compute device
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")
