"""
Data Preprocessing Script voor Deep RL Trading
===============================================

Dit script:
1. Selecteert data tussen START_DATE en END_DATE
2. Verwerkt de data in MAANDELIJKSE CHUNKS (memory-efficient)
3. Slaat verwerkte data op in DataNorm/

Na dit script: gebruik create_core_data.py voor de 80/10/10 split.

Auteur: DataDeepRL Team
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import glob
import json
import time
import logging

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATIE - HIER JE DATUMS INVULLEN
# ============================================================

# >>> START DATUM HIER INVULLEN (YYYY-MM-DD) <<<
START_DATE = "2017-08-17"

# >>> EIND DATUM HIER INVULLEN (YYYY-MM-DD) <<<
END_DATE = "2025-03-07"

# Data directory met ruwe parquet bestanden
DATA_DIR = "./btc_l2_data"

# Output directory voor genormaliseerde data
OUTPUT_DIR = "./DataNorm"

# Log directory
LOG_DIR = "./logs"

# Normalisatie methode: 'zscore', 'minmax', of 'both'
NORMALIZATION_METHOD = "zscore"

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging():
    """Setup logging naar bestand en console."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_DIR, f"preprocess_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('preprocess')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# Global logger
logger = None
log_file_path = None

# ============================================================
# TIMING HELPERS
# ============================================================

class Timer:
    """Context manager voor timing van code blokken."""
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
    
    @staticmethod
    def format_time(seconds):
        """Formatteer seconden naar leesbare string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


def log_step(step_num: int, description: str, details: str = ""):
    """Print een geformatteerde log stap."""
    msg = f"\n{'='*60}\n📌 STAP {step_num}: {description}\n{'='*60}"
    if details:
        msg += f"\n   {details}"
    logger.info(msg)


def log_info(message: str):
    """Print info bericht."""
    logger.info(f"   ℹ️  {message}")


def log_success(message: str):
    """Print success bericht."""
    logger.info(f"   ✅ {message}")


def log_warning(message: str):
    """Print warning bericht."""
    logger.warning(f"   ⚠️  {message}")


def log_error(message: str):
    """Print error bericht."""
    logger.error(f"   ❌ {message}")


def log_progress(current, total, start_time, prefix=""):
    """Log voortgang met ETA."""
    elapsed = time.time() - start_time
    if current > 0:
        eta = (elapsed / current) * (total - current)
        pct = (current / total) * 100
        logger.info(f"   📊 {prefix}{current}/{total} ({pct:.1f}%) | "
                   f"Verstreken: {Timer.format_time(elapsed)} | "
                   f"ETA: {Timer.format_time(eta)}")


# ============================================================
# DATA PROCESSING FUNCTIONS
# ============================================================

def get_monthly_file_groups(files: list) -> dict:
    """Groepeer bestanden per maand."""
    groups = {}
    for filepath in files:
        # Extract datum uit bestandsnaam: BTCUSDT_YYYY-MM-DD.parquet
        basename = os.path.basename(filepath)
        date_str = basename.replace("BTCUSDT_", "").replace(".parquet", "")
        year_month = date_str[:7]  # YYYY-MM
        
        if year_month not in groups:
            groups[year_month] = []
        groups[year_month].append(filepath)
    
    return dict(sorted(groups.items()))


def step1_find_files() -> list:
    """Stap 1: Zoek bestanden tussen START_DATE en END_DATE."""
    log_step(1, "BESTANDEN ZOEKEN", f"Periode: {START_DATE} tot {END_DATE}")
    
    with Timer() as t:
        start = pd.to_datetime(START_DATE)
        end = pd.to_datetime(END_DATE)
        
        files_to_load = []
        current = start
        while current <= end:
            filename = f"BTCUSDT_{current.strftime('%Y-%m-%d')}.parquet"
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                files_to_load.append(filepath)
            current += pd.Timedelta(days=1)
    
    if not files_to_load:
        log_error("Geen bestanden gevonden voor deze periode!")
        log_info("Beschikbare bestanden:")
        available = sorted(glob.glob(os.path.join(DATA_DIR, "BTCUSDT_*.parquet")))[:5]
        for f in available:
            logger.info(f"      - {os.path.basename(f)}")
        if len(glob.glob(os.path.join(DATA_DIR, "BTCUSDT_*.parquet"))) > 5:
            logger.info(f"      - ... en meer")
        return []
    
    # Toon maandelijkse verdeling
    monthly_groups = get_monthly_file_groups(files_to_load)
    
    log_success(f"{len(files_to_load)} bestanden gevonden in {len(monthly_groups)} maanden")
    log_info(f"Eerste: {os.path.basename(files_to_load[0])}")
    log_info(f"Laatste: {os.path.basename(files_to_load[-1])}")
    log_info(f"Duur: {Timer.format_time(t.elapsed)}")
    
    return files_to_load


def process_single_file(filepath: str) -> pd.DataFrame:
    """Laad en verwerk een enkel parquet bestand."""
    df = pd.read_parquet(filepath)
    
    # Handle missing values inline
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Voeg features toe aan een DataFrame."""
    # 1. Returns
    if 'close' in df.columns:
        df['return_1s'] = df['close'].pct_change()
        df['log_return_1s'] = np.log(df['close'] / df['close'].shift(1))
        
        for period in [5, 10, 30, 60]:
            df[f'return_{period}s'] = df['close'].pct_change(period)
    
    # 2. Moving averages
    if 'close' in df.columns:
        for window in [10, 30, 60]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    
    # 3. Volatiliteit
    if 'return_1s' in df.columns:
        for window in [10, 30, 60]:
            df[f'volatility_{window}'] = df['return_1s'].rolling(window).std()
    
    # 4. Momentum & RSI
    if 'close' in df.columns:
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_30'] = df['close'] - df['close'].shift(30)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 5. Volume features
    if 'volume' in df.columns:
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_10'] + 1e-10)
    
    # 6. Spread features
    if 'bid_price' in df.columns and 'ask_price' in df.columns:
        df['spread'] = df['ask_price'] - df['bid_price']
        df['spread_pct'] = df['spread'] / df['close'] * 100
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
    
    # 7. Order imbalance
    if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
        total_vol = df['bid_volume'] + df['ask_volume']
        df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (total_vol + 1e-10)
    
    return df


def process_month_chunk(files: list, month: str) -> pd.DataFrame:
    """Verwerk een maand aan data."""
    dfs = []
    
    for filepath in files:
        try:
            df = process_single_file(filepath)
            dfs.append(df)
        except Exception as e:
            log_warning(f"Fout bij {os.path.basename(filepath)}: {e}")
    
    if not dfs:
        return None
    
    # Combineer bestanden van deze maand
    df = pd.concat(dfs, ignore_index=True)
    
    # Sorteer op timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Voeg features toe
    df = add_features(df)
    
    # Verwijder NaN rows (van rolling windows)
    df = df.dropna()
    
    return df


def step2_process_in_chunks(files: list) -> tuple:
    """Stap 2-5: Verwerk data per maand en sla op."""
    log_step(2, "DATA VERWERKEN (PER MAAND)", 
             "Memory-efficient processing in maandelijkse chunks")
    
    monthly_groups = get_monthly_file_groups(files)
    total_months = len(monthly_groups)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunk_dir = os.path.join(OUTPUT_DIR, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    
    total_rows = 0
    processed_months = 0
    start_time = time.time()
    chunk_files = []
    
    log_info(f"Totaal {total_months} maanden te verwerken...")
    logger.info("")
    
    for month, month_files in monthly_groups.items():
        month_start = time.time()
        
        # Verwerk deze maand
        df = process_month_chunk(month_files, month)
        
        if df is not None and len(df) > 0:
            # Sla chunk op
            chunk_file = os.path.join(chunk_dir, f"chunk_{month}.parquet")
            df.to_parquet(chunk_file, index=False)
            chunk_files.append(chunk_file)
            
            rows = len(df)
            total_rows += rows
            
            month_elapsed = time.time() - month_start
            
            processed_months += 1
            log_progress(processed_months, total_months, start_time, 
                        f"Maand {month}: {rows:,} rows ({Timer.format_time(month_elapsed)}) | ")
            
            # Free memory
            del df
        else:
            processed_months += 1
            log_warning(f"Geen data voor maand {month}")
    
    total_elapsed = time.time() - start_time
    log_success(f"Alle chunks verwerkt in {Timer.format_time(total_elapsed)}")
    log_info(f"Totaal {total_rows:,} rows in {len(chunk_files)} chunks")
    
    return chunk_dir, chunk_files


def step6_compute_stats(chunk_files: list) -> dict:
    """Stap 6: Bereken normalisatie statistieken over alle chunks."""
    log_step(6, "STATISTIEKEN BEREKENEN", "Voor normalisatie over alle data")
    
    start_time = time.time()
    
    # Eerste pass: bereken sum en count per kolom
    log_info("Pass 1: Berekenen van means...")
    
    sums = {}
    sum_sqs = {}
    counts = {}
    mins = {}
    maxs = {}
    
    for i, chunk_file in enumerate(chunk_files):
        df = pd.read_parquet(chunk_file)
        
        exclude_cols = ['timestamp']
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                        if c not in exclude_cols]
        
        for col in numeric_cols:
            if col not in sums:
                sums[col] = 0
                sum_sqs[col] = 0
                counts[col] = 0
                mins[col] = float('inf')
                maxs[col] = float('-inf')
            
            sums[col] += df[col].sum()
            sum_sqs[col] += (df[col] ** 2).sum()
            counts[col] += len(df[col])
            mins[col] = min(mins[col], df[col].min())
            maxs[col] = max(maxs[col], df[col].max())
        
        if (i + 1) % 10 == 0 or i == len(chunk_files) - 1:
            log_progress(i + 1, len(chunk_files), start_time, "Chunks: ")
        
        del df
    
    # Bereken finale statistieken
    stats = {}
    for col in sums.keys():
        mean = sums[col] / counts[col]
        variance = (sum_sqs[col] / counts[col]) - (mean ** 2)
        std = np.sqrt(max(0, variance))
        
        stats[col] = {
            'mean': float(mean),
            'std': float(std),
            'min': float(mins[col]),
            'max': float(maxs[col])
        }
    
    elapsed = time.time() - start_time
    log_success(f"Statistieken berekend voor {len(stats)} kolommen")
    log_info(f"Duur: {Timer.format_time(elapsed)}")
    
    return stats


def step7_normalize_and_save(chunk_files: list, stats: dict) -> str:
    """Stap 7: Normaliseer chunks en sla direct op (memory-efficient)."""
    log_step(7, "NORMALISEREN EN OPSLAAN", f"Methode: {NORMALIZATION_METHOD}")
    
    start_time = time.time()
    
    # Output directory voor genormaliseerde chunks
    norm_chunk_dir = os.path.join(OUTPUT_DIR, "normalized_chunks")
    os.makedirs(norm_chunk_dir, exist_ok=True)
    
    normalized_chunk_files = []
    total_rows = 0
    columns = None
    
    log_info("Normaliseren en opslaan van chunks...")
    
    for i, chunk_file in enumerate(chunk_files):
        df = pd.read_parquet(chunk_file)
        
        exclude_cols = ['timestamp']
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                        if c not in exclude_cols]
        
        for col in numeric_cols:
            if col in stats:
                if NORMALIZATION_METHOD == 'zscore':
                    if stats[col]['std'] > 0:
                        df[col] = (df[col] - stats[col]['mean']) / stats[col]['std']
                    else:
                        df[col] = 0
                elif NORMALIZATION_METHOD == 'minmax':
                    range_val = stats[col]['max'] - stats[col]['min']
                    if range_val > 0:
                        df[col] = (df[col] - stats[col]['min']) / range_val
                    else:
                        df[col] = 0
        
        # Sla genormaliseerde chunk direct op
        norm_chunk_file = os.path.join(norm_chunk_dir, os.path.basename(chunk_file).replace("chunk_", "norm_"))
        df.to_parquet(norm_chunk_file, index=False)
        normalized_chunk_files.append(norm_chunk_file)
        
        total_rows += len(df)
        if columns is None:
            columns = list(df.columns)
        
        # Free memory
        del df
        
        if (i + 1) % 10 == 0 or i == len(chunk_files) - 1:
            log_progress(i + 1, len(chunk_files), start_time, "Normaliseren: ")
    
    log_success(f"Alle {len(normalized_chunk_files)} chunks genormaliseerd")
    log_info(f"Totaal {total_rows:,} rows")
    
    # Combineer parquet bestanden met pyarrow (memory-efficient)
    log_info("Combineren van parquet bestanden...")
    combine_start = time.time()
    
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        # Lees alle parquet bestanden als tabellen en combineer
        tables = []
        for norm_file in sorted(normalized_chunk_files):
            table = pq.read_table(norm_file)
            tables.append(table)
        
        # Concatenate tables (meer memory-efficient dan pandas)
        combined_table = pa.concat_tables(tables)
        
        # Sorteer op timestamp indien aanwezig
        if 'timestamp' in combined_table.column_names:
            # Convert to pandas just for sorting, then back
            log_info("Sorteren op timestamp...")
            sort_indices = pa.compute.sort_indices(combined_table.column('timestamp'))
            combined_table = combined_table.take(sort_indices)
        
        # Sla finale bestand op
        date_range = f"{START_DATE}_to_{END_DATE}".replace("-", "")
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_filename = f"normalized_{date_range}.parquet"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        log_info("Opslaan van finale dataset...")
        save_start = time.time()
        pq.write_table(combined_table, output_path)
        log_info(f"Opslaan duurde: {Timer.format_time(time.time() - save_start)}")
        
        # Kopie als latest
        latest_path = os.path.join(OUTPUT_DIR, "latest_normalized.parquet")
        pq.write_table(combined_table, latest_path)
        
        final_rows = combined_table.num_rows
        
        # Free memory
        del tables
        del combined_table
        
    except Exception as e:
        log_warning(f"PyArrow combinatie mislukt: {e}")
        log_info("Chunks blijven als aparte bestanden beschikbaar in normalized_chunks/")
        
        date_range = f"{START_DATE}_to_{END_DATE}".replace("-", "")
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = "CHUNKS_ONLY"
        final_rows = total_rows
    
    log_info(f"Combineren duurde: {Timer.format_time(time.time() - combine_start)}")
    
    # Sla statistieken op
    stats_filename = f"normalization_stats_{date_range}.json"
    stats_path = os.path.join(OUTPUT_DIR, stats_filename)
    
    metadata = {
        'created': timestamp_str,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'normalization_method': NORMALIZATION_METHOD,
        'total_rows': final_rows,
        'columns': columns,
        'chunk_files': [os.path.basename(f) for f in normalized_chunk_files],
        'stats': stats
    }
    
    with open(stats_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    latest_stats_path = os.path.join(OUTPUT_DIR, "latest_stats.json")
    with open(latest_stats_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    total_elapsed = time.time() - start_time
    log_success(f"Data opgeslagen in {Timer.format_time(total_elapsed)}")
    log_info(f"Output: {output_filename}")
    
    return output_filename


def main():
    """Hoofdfunctie."""
    global logger, log_file_path
    
    # Setup logging
    logger, log_file_path = setup_logging()
    
    total_start = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("🚀 DATA PREPROCESSING VOOR DEEP RL")
    logger.info("="*60)
    logger.info(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"\n📋 CONFIGURATIE:")
    logger.info(f"   Start datum: {START_DATE}")
    logger.info(f"   Eind datum:  {END_DATE}")
    logger.info(f"   Normalisatie: {NORMALIZATION_METHOD}")
    logger.info(f"   Output: {OUTPUT_DIR}")
    logger.info(f"   Log file: {log_file_path}")
    
    # Stap 1: Vind bestanden
    files = step1_find_files()
    if not files:
        return
    
    # Stap 2-5: Verwerk in chunks
    chunk_dir, chunk_files = step2_process_in_chunks(files)
    
    if not chunk_files:
        log_error("Geen chunks verwerkt!")
        return
    
    # Stap 6: Bereken statistieken
    stats = step6_compute_stats(chunk_files)
    
    # Stap 7: Normaliseer en sla op
    output_file = step7_normalize_and_save(chunk_files, stats)
    
    # Totale tijd
    total_elapsed = time.time() - total_start
    
    # Einde
    logger.info("\n" + "="*60)
    logger.info("✅ PREPROCESSING VOLTOOID!")
    logger.info("="*60)
    logger.info(f"\n⏱️  TOTALE TIJD: {Timer.format_time(total_elapsed)}")
    logger.info(f"\n📁 Bestanden in {OUTPUT_DIR}/:")
    logger.info(f"   - {output_file}")
    logger.info(f"   - latest_normalized.parquet")
    logger.info(f"   - normalization_stats_*.json")
    logger.info(f"   - latest_stats.json")
    logger.info(f"   - chunks/ (tijdelijke bestanden)")
    logger.info(f"\n📝 Log opgeslagen: {log_file_path}")
    logger.info(f"\n💡 Volgende stap: run create_core_data.py voor 80/10/10 split")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
