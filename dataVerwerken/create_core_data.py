"""
Create Core Data - Genormaliseerde data splitsen
================================================

Dit script:
1. Leest genormaliseerde data uit DataNorm/
2. Maakt een train/val/test split (80/10/10)
3. Slaat de splits op in coreData/

Voer eerst preprocess_data.py uit voordat je dit script draait!

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

import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATIE - HIER JE BESTAND SELECTEREN
# ============================================================

# Base directory (waar dit script staat -> dataVerwerken/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Directory met genormaliseerde chunks (maandelijks)
CHUNKS_DIR = os.path.join(PROJECT_DIR, "DataNorm", "normalized_chunks")

# Fallback: enkel bestand (als je een samengevoegd bestand hebt)
INPUT_FILE = "latest_normalized.parquet"

# Input directory (waar DataNorm data staat)
INPUT_DIR = os.path.join(PROJECT_DIR, "DataNorm")

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, "coreData")

# Split ratio's: 80% train, 10% validatie, 10% test
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ============================================================


def log_step(step_num: int, description: str, details: str = ""):
    """Print een geformatteerde log stap."""
    print(f"\n{'='*60}")
    print(f"📌 STAP {step_num}: {description}")
    print(f"{'='*60}")
    if details:
        print(f"   {details}")


def log_info(message: str):
    """Print info bericht."""
    print(f"   ℹ️  {message}")


def log_success(message: str):
    """Print success bericht."""
    print(f"   ✅ {message}")


def log_warning(message: str):
    """Print warning bericht."""
    print(f"   ⚠️  {message}")


def log_error(message: str):
    """Print error bericht."""
    print(f"   ❌ {message}")


def main():
    """Hoofdfunctie."""
    print("\n" + "="*60)
    print("🚀 CREATE CORE DATA - 80/10/10 SPLIT")
    print("="*60)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================
    # STAP 1: CONFIGURATIE TONEN
    # =========================================================
    log_step(1, "CONFIGURATIE")
    log_info(f"Chunks dir:    {CHUNKS_DIR}")
    log_info(f"Output dir:    {OUTPUT_DIR}")
    log_info(f"Split:         {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")
    
    # =========================================================
    # STAP 2: DATA INVENTARISEREN (memory-efficient)
    # =========================================================
    log_step(2, "DATA INVENTARISEREN")
    
    # Probeer eerst de chunks directory
    chunk_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "*.parquet")))
    
    if not chunk_files:
        # Fallback: probeer enkel bestand
        filepath = os.path.join(INPUT_DIR, INPUT_FILE)
        if not os.path.exists(filepath):
            log_error(f"Geen data gevonden!")
            log_info(f"Geen chunks in: {CHUNKS_DIR}")
            log_info(f"Geen bestand: {filepath}")
            print("\n   💡 Voer eerst preprocess_data.py uit!")
            return
        # Single file mode - load normally
        log_info(f"Laden van {INPUT_FILE}...")
        df = pd.read_parquet(filepath)
        chunk_files = None  # flag for single-file mode
    
    if chunk_files:
        log_info(f"Gevonden: {len(chunk_files)} genormaliseerde chunks")
        log_info(f"Eerste: {os.path.basename(chunk_files[0])}")
        log_info(f"Laatste: {os.path.basename(chunk_files[-1])}")
        
        # Tel rijen per chunk via parquet metadata (geen data laden!)
        log_info("Rijen tellen per chunk (via metadata)...")
        chunk_rows = []
        columns = None
        for f in chunk_files:
            meta = pq.read_metadata(f)
            chunk_rows.append(meta.num_rows)
            if columns is None:
                columns = pq.read_schema(f).names
        
        n_total = sum(chunk_rows)
        cumulative = np.cumsum(chunk_rows)
        
        log_success(f"Totaal: {n_total:,} rows over {len(chunk_files)} chunks")
        log_info(f"Kolommen: {len(columns)}")
    else:
        # Single file mode
        n_total = len(df)
        columns = list(df.columns)
        log_success(f"Data geladen: {n_total:,} rows")
        log_info(f"Kolommen: {len(columns)}")
    
    # Laad metadata als die bestaat
    stats_file = "latest_stats.json"
    stats_path = os.path.join(INPUT_DIR, stats_file)
    source_metadata = None
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            source_metadata = json.load(f)
        log_info(f"Normalisatie statistieken geladen")
    
    # =========================================================
    # STAP 3: DATA SPLITSEN (80/10/10)
    # =========================================================
    log_step(3, "DATA SPLITSEN", f"Train: {TRAIN_RATIO:.0%}, Val: {VAL_RATIO:.0%}, Test: {TEST_RATIO:.0%}")
    
    train_end = int(n_total * TRAIN_RATIO)
    val_end = int(n_total * (TRAIN_RATIO + VAL_RATIO))
    
    log_info(f"Totaal: {n_total:,} rows")
    log_info(f"Train: rows 0 - {train_end:,}")
    log_info(f"Val:   rows {train_end:,} - {val_end:,}")
    log_info(f"Test:  rows {val_end:,} - {n_total:,}")
    
    # =========================================================
    # STAP 4: OPSLAAN (streaming row-group level, memory-efficient)
    # =========================================================
    log_step(4, "DATA OPSLAAN", f"Output: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    val_path = os.path.join(OUTPUT_DIR, "val.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test.parquet")
    
    train_count = train_end
    val_count = val_end - train_end
    test_count = n_total - val_end
    
    if chunk_files:
        import pyarrow as pa
        import gc
        
        # Determine schema from first chunk
        schema = pq.read_schema(chunk_files[0])
        
        splits = {
            'train': {'path': train_path, 'start': 0, 'end': train_end, 'writer': None},
            'val':   {'path': val_path,   'start': train_end, 'end': val_end, 'writer': None},
            'test':  {'path': test_path,  'start': val_end, 'end': n_total, 'writer': None},
        }
        
        # Open writers
        for sp in splits.values():
            sp['writer'] = pq.ParquetWriter(sp['path'], schema)
        
        global_offset = 0
        for i, f in enumerate(chunk_files):
            pf = pq.ParquetFile(f)
            
            # Process each row group within the chunk
            for rg_idx in range(pf.metadata.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                rg_start = global_offset
                rg_end = global_offset + rg_rows
                
                rg_table = None  # lazy load
                
                for name, sp in splits.items():
                    overlap_start = max(rg_start, sp['start'])
                    overlap_end = min(rg_end, sp['end'])
                    
                    if overlap_start < overlap_end:
                        if rg_table is None:
                            rg_table = pf.read_row_group(rg_idx)
                        local_start = overlap_start - rg_start
                        local_end = overlap_end - rg_start
                        sp['writer'].write_table(rg_table.slice(local_start, local_end - local_start))
                
                del rg_table
                global_offset = rg_end
            
            del pf
            gc.collect()
            
            if (i + 1) % 10 == 0 or i == len(chunk_files) - 1:
                log_info(f"Verwerkt: {i+1}/{len(chunk_files)} chunks")
        
        # Close writers
        for name, sp in splits.items():
            sp['writer'].close()
            log_info(f"{name.capitalize()} opgeslagen: {name}.parquet")
    else:
        # Single file mode
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        train_df.to_parquet(train_path, index=False)
        log_info(f"Train opgeslagen: train.parquet ({len(train_df):,} rows)")
        
        val_df.to_parquet(val_path, index=False)
        log_info(f"Val opgeslagen:   val.parquet ({len(val_df):,} rows)")
        
        test_df.to_parquet(test_path, index=False)
        log_info(f"Test opgeslagen:  test.parquet ({len(test_df):,} rows)")
    
    # Metadata opslaan
    metadata = {
        'created': timestamp,
        'source': 'normalized_chunks' if chunk_files else INPUT_FILE,
        'num_chunks': len(chunk_files) if chunk_files else 1,
        'total_rows': n_total,
        'train_rows': train_count,
        'val_rows': val_count,
        'test_rows': test_count,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'columns': columns if isinstance(columns, list) else list(columns),
    }
    
    if source_metadata:
        metadata['source_start_date'] = source_metadata.get('start_date')
        metadata['source_end_date'] = source_metadata.get('end_date')
        metadata['normalization_method'] = source_metadata.get('normalization_method')
    
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    log_info(f"Metadata opgeslagen: metadata.json")
    
    if source_metadata:
        norm_stats_path = os.path.join(OUTPUT_DIR, "normalization_stats.json")
        with open(norm_stats_path, 'w') as f:
            json.dump(source_metadata, f, indent=2, default=str)
        log_info(f"Normalisatie stats gekopieerd: normalization_stats.json")
    
    # =========================================================
    # KLAAR
    # =========================================================
    print("\n" + "="*60)
    print("✅ CORE DATA AANGEMAAKT!")
    print("="*60)
    print(f"\n📁 Bestanden in {OUTPUT_DIR}/:")
    print(f"   - train.parquet  ({train_count:,} rows)")
    print(f"   - val.parquet    ({val_count:,} rows)")
    print(f"   - test.parquet   ({test_count:,} rows)")
    print(f"   - metadata.json")
    if source_metadata:
        print(f"   - normalization_stats.json")
    print("\n💡 Gebruik in je training script:")
    print(f"   train_df = pd.read_parquet('{OUTPUT_DIR}/train.parquet')")
    print(f"   val_df = pd.read_parquet('{OUTPUT_DIR}/val.parquet')")
    print(f"   test_df = pd.read_parquet('{OUTPUT_DIR}/test.parquet')")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
