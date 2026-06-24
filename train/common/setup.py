"""
Shared data loading for coreData-based training scripts.

Only function used: load_coredata_streaming() — memory-efficient raw features +
prices loader. The training environment creates windowed sequences on-the-fly.
"""

import os
import json
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq


# Feature columns kept in coreData (matched with the DeepLOB pretrain schema).
STATIONARY_FEATURES = [
    'spread', 'spread_pct', 'buy_ratio',
    'return_5s', 'return_10s', 'return_30s', 'return_60s',
    'volatility_10', 'volatility_30', 'volatility_60',
    'momentum_10', 'momentum_30',
    'rsi_14', 'volume_ratio', 'order_imbalance',
]


def _load_split_raw(parquet_path: str, max_rows: int, load_cols: list,
                    feature_cols: list, price_col: str,
                    price_mean: float, price_std: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Load a split as raw features + denormalized prices."""
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows

    if total_rows <= max_rows:
        df = pf.read(columns=load_cols).to_pandas()
    else:
        # Read the latest row groups until we have enough rows.
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

    if price_mean is not None and price_std is not None:
        prices = prices * price_std + price_mean

    return features, prices


def load_coredata_streaming(
    data_dir: str = './coreData',
    sequence_length: int = 100,
    max_rows: int = 80_000_000,
    feature_cols: list = None,
    price_col: str = 'close',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient loader: returns raw (N, num_features) feature arrays + prices.

    The trading environment creates sliding windows on-the-fly with zero-copy
    numpy slicing — avoids materialising the full (N, seq_len, num_features)
    sequences array which would be ~100x larger.

    Returns:
        (train_features, train_prices, val_features, val_prices,
         test_features, test_prices)
        features shape: (N, num_features)
        prices shape:   (N,)  — denormalized back to USD
    """
    if feature_cols is None:
        feature_cols = STATIONARY_FEATURES

    train_parquet = os.path.join(data_dir, 'train.parquet')
    val_parquet = os.path.join(data_dir, 'val.parquet')
    test_parquet = os.path.join(data_dir, 'test.parquet')

    for p in [train_parquet, val_parquet, test_parquet]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"coreData bestand niet gevonden: {p}")

    # Price denormalization stats (z-score → USD).
    norm_stats_path = os.path.join(data_dir, 'normalization_stats.json')
    price_mean, price_std = None, None
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        if 'stats' in norm_stats and price_col in norm_stats['stats']:
            price_mean = norm_stats['stats'][price_col]['mean']
            price_std = norm_stats['stats'][price_col]['std']
            print(f"  Price denormalization: {price_col} mean={price_mean:.2f}, std={price_std:.2f}")

    schema = pq.read_schema(train_parquet)
    missing = [c for c in feature_cols if c not in schema.names]
    if missing:
        raise ValueError(f"Features niet gevonden in coreData: {missing}")

    load_cols = list(dict.fromkeys(feature_cols + [price_col]))

    print(f"Loading coreData STREAMING from {data_dir} "
          f"({len(feature_cols)} features, seq_len={sequence_length})...")

    train_feat, train_prices = _load_split_raw(
        train_parquet, max_rows, load_cols, feature_cols, price_col, price_mean, price_std)
    val_feat, val_prices = _load_split_raw(
        val_parquet, max_rows // 4, load_cols, feature_cols, price_col, price_mean, price_std)
    test_feat, test_prices = _load_split_raw(
        test_parquet, max_rows // 4, load_cols, feature_cols, price_col, price_mean, price_std)

    n_train = len(train_feat) - sequence_length
    n_val = len(val_feat) - sequence_length
    n_test = len(test_feat) - sequence_length
    mem_raw = (train_feat.nbytes + val_feat.nbytes + test_feat.nbytes) / 1e9

    print(f"  Train: {len(train_feat):,} rows -> {n_train:,} valid steps")
    print(f"  Val:   {len(val_feat):,} rows -> {n_val:,} valid steps")
    print(f"  Test:  {len(test_feat):,} rows -> {n_test:,} valid steps")
    print(f"  RAM used: {mem_raw:.1f} GB (sequences expanded on-the-fly in env)")

    return train_feat, train_prices, val_feat, val_prices, test_feat, test_prices
