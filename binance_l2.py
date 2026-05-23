"""
Binance Historical L2 Data Downloader
Downloads BTC aggregated trades per second with bid/ask prices
Data source: data.binance.vision (Spot aggTrades)

GEBRUIK:
    python binance_l2.py

    Pas hieronder START_DATE en END_DATE aan om je gewenste periode te downloaden.
"""

import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
import time
import sys

# ============================================================
# CONFIGURATIE - PAS HIER JE DATUMS AAN
# ============================================================
SYMBOL = "BTCUSDT"

# >>> WIJZIG HIER DE START DATUM (jaar, maand, dag) <<<
START_DATE = datetime(2023, 8, 22)

# >>> WIJZIG HIER DE EIND DATUM (jaar, maand, dag) <<<
END_DATE = datetime(2026, 1, 30)

# Output map
OUTPUT_DIR = "btc_l2_data"
# ============================================================

# aggTrades columns (no header in CSV)
COLUMNS = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 
           'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match']


def print_progress(current_day: int, total_days: int, start_date: str, end_date: str, current_date: str, status: str):
    """Print voortgangsbalk in de terminal."""
    pct = (current_day / total_days) * 100
    bar_len = 30
    filled = int(bar_len * current_day / total_days)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r[{bar}] {pct:5.1f}% | {start_date} → {end_date} | Dag {current_day}/{total_days} | {current_date} | {status}".ljust(120))
    sys.stdout.flush()


def download_aggtrades(symbol: str, date: datetime) -> pd.DataFrame | None:
    """Download aggregated trades from Binance Spot."""
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    
    try:
        print(f"  Downloading from {url}...")
        response = requests.get(url, timeout=300)
        
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f, names=COLUMNS)
                    print(f"  Downloaded {len(df):,} trades")
                    return df
        else:
            print(f"  HTTP {response.status_code}")
            
    except Exception as e:
        print(f"  Error: {e}")
    
    return None


def aggregate_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trades to per-second OHLCV + bid/ask data.
    is_buyer_maker=True means seller initiated (trade hit bid)
    is_buyer_maker=False means buyer initiated (trade hit ask)
    """
    # Auto-detect timestamp unit: ms (13 digits) vs us (16 digits)
    sample_ts = df['timestamp'].iloc[0]
    if sample_ts > 1e15:  # 16+ digits = microseconds
        unit = 'us'
    else:  # 13 digits = milliseconds
        unit = 'ms'
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit=unit)
    df['second'] = df['datetime'].dt.floor('s')
    
    # Convert types
    df['price'] = df['price'].astype(float)
    df['quantity'] = df['quantity'].astype(float)
    df['value'] = df['price'] * df['quantity']
    
    result = []
    for second, group in df.groupby('second'):
        prices = group['price']
        qtys = group['quantity']
        values = group['value']
        
        # Separate buys (taker buy = hit ask) vs sells (taker sell = hit bid)
        # is_buyer_maker=True: maker was buyer, taker was seller -> hit bid
        # is_buyer_maker=False: maker was seller, taker was buyer -> hit ask
        sells = group[group['is_buyer_maker'] == True]  # Hit bid
        buys = group[group['is_buyer_maker'] == False]  # Hit ask
        
        row = {
            'timestamp': second,
            'open': prices.iloc[0],
            'high': prices.max(),
            'low': prices.min(),
            'close': prices.iloc[-1],
            'vwap': values.sum() / qtys.sum() if qtys.sum() > 0 else prices.mean(),
            'volume': qtys.sum(),
            'quote_volume': values.sum(),
            'trade_count': len(group),
            # Bid side (sells hitting bid)
            'bid_price': sells['price'].iloc[-1] if len(sells) > 0 else prices.min(),
            'bid_volume': sells['quantity'].sum(),
            'bid_trades': len(sells),
            # Ask side (buys hitting ask)  
            'ask_price': buys['price'].iloc[-1] if len(buys) > 0 else prices.max(),
            'ask_volume': buys['quantity'].sum(),
            'ask_trades': len(buys),
        }
        
        # Calculate spread
        row['spread'] = row['ask_price'] - row['bid_price']
        row['spread_pct'] = (row['spread'] / row['vwap']) * 100
        
        # Order flow imbalance
        total_vol = row['bid_volume'] + row['ask_volume']
        row['buy_ratio'] = row['ask_volume'] / total_vol if total_vol > 0 else 0.5
        
        result.append(row)
    
    return pd.DataFrame(result)


def main():
    """Download data voor de geconfigureerde periode."""
    total_days = (END_DATE - START_DATE).days + 1
    
    print("=" * 60)
    print("Binance L2 Data Downloader")
    print(f"Symbol: {SYMBOL}")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Totaal dagen: {total_days}")
    print(f"Compressie: ZSTD")
    print("=" * 60)
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")
    
    current = START_DATE
    day_num = 0
    days_downloaded = 0
    
    while current <= END_DATE:
        day_num += 1
        date_str = current.strftime("%Y-%m-%d")
        output_file = os.path.join(OUTPUT_DIR, f"{SYMBOL}_{date_str}.parquet")
        
        # Voortgang: downloaden
        print_progress(day_num, total_days, start_str, end_str, date_str, "Downloaden...")
        
        # Download
        df_raw = download_aggtrades(SYMBOL, current)
        
        if df_raw is not None and len(df_raw) > 0:
            # Voortgang: verwerken
            print_progress(day_num, total_days, start_str, end_str, date_str, "Verwerken...")
            
            # Aggregate to seconds
            df_sec = aggregate_to_seconds(df_raw)
            
            # Save met ZSTD compressie
            df_sec.to_parquet(output_file, index=False, compression='zstd')
            
            # Voortgang: klaar
            print_progress(day_num, total_days, start_str, end_str, date_str, f"OK: {len(df_sec):,} seconden")
            print()  # Nieuwe regel na voltooide dag
            
            # Print eerste 10 regels
            print(f"  Eerste 10 regels van {output_file}:")
            print(df_sec.head(10).to_string())
            print()
            days_downloaded += 1
        else:
            print_progress(day_num, total_days, start_str, end_str, date_str, "Geen data beschikbaar")
            print()
        
        current += timedelta(days=1)
        time.sleep(0.5)
    
    print()
    print("=" * 60)
    print("DOWNLOAD VOLTOOID!")
    print("=" * 60)
    print(f"Dagen gedownload: {days_downloaded}/{total_days}")
    print(f"Bestanden opgeslagen in: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
