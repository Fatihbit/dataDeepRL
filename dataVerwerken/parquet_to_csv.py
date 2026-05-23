"""
Parquet naar CSV Converter
==========================

Dit script converteert een parquet bestand naar CSV.
Vul de bestandsnaam in bij PARQUET_FILE.

Auteur: DataDeepRL Team
"""

import os
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIGURATIE - HIER JE BESTAND INVULLEN
# ============================================================

# >>> PARQUET BESTANDSNAAM HIER PLAKKEN <<<
# Voorbeelden:
#   - "../coreData/train.parquet"
#   - "../DataNorm/latest_normalized.parquet"
#   - "../btc_l2_data/BTCUSDT_2024-01-01.parquet"
PARQUET_FILE = "../coreData/train.parquet"

# Output CSV bestand (leeg laten voor automatisch)
# Automatisch wordt: zelfde naam met .csv extensie
CSV_FILE = ""

# ============================================================


def convert_parquet_to_csv(parquet_path: str, csv_path: str = None) -> str:
    """
    Converteer parquet naar CSV.
    
    Args:
        parquet_path: Pad naar parquet bestand
        csv_path: Pad naar output CSV (optioneel)
    
    Returns:
        Pad naar het gemaakte CSV bestand
    """
    print("\n" + "="*60)
    print("📄 PARQUET NAAR CSV CONVERTER")
    print("="*60)
    
    # Check of input bestaat
    if not os.path.exists(parquet_path):
        print(f"❌ Bestand niet gevonden: {parquet_path}")
        return None
    
    print(f"\n📂 Input:  {parquet_path}")
    
    # Bepaal output pad
    if not csv_path:
        csv_path = parquet_path.replace('.parquet', '.csv')
    
    print(f"📂 Output: {csv_path}")
    
    # Lees parquet
    print("\n⏳ Laden van parquet...")
    df = pd.read_parquet(parquet_path)
    print(f"   ✅ Geladen: {len(df):,} rows, {len(df.columns)} kolommen")
    print(f"   📊 Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Toon kolommen
    print(f"\n   Kolommen: {list(df.columns)[:10]}")
    if len(df.columns) > 10:
        print(f"            ... en {len(df.columns) - 10} meer")
    
    # Schrijf naar CSV
    print("\n⏳ Schrijven naar CSV...")
    df.to_csv(csv_path, index=False)
    
    # Check bestandsgrootte
    csv_size = os.path.getsize(csv_path) / 1024 / 1024
    parquet_size = os.path.getsize(parquet_path) / 1024 / 1024
    
    print(f"\n✅ Conversie voltooid!")
    print(f"\n📊 Statistieken:")
    print(f"   Parquet grootte: {parquet_size:.2f} MB")
    print(f"   CSV grootte:     {csv_size:.2f} MB")
    print(f"   Ratio:           {csv_size/parquet_size:.1f}x groter")
    
    print("\n" + "="*60 + "\n")
    
    return csv_path


def main():
    """Hoofdfunctie."""
    if not PARQUET_FILE:
        print("❌ Vul PARQUET_FILE in bovenaan het script!")
        return
    
    convert_parquet_to_csv(PARQUET_FILE, CSV_FILE if CSV_FILE else None)


if __name__ == "__main__":
    main()
