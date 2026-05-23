
import pandas as pd
import os

parquet_file = input("Plak het pad naar je parquet-bestand: ").strip()

if not os.path.isfile(parquet_file):
    print(f"Fout: Bestand '{parquet_file}' niet gevonden.")
else:
    try:
        df = pd.read_parquet(parquet_file)
        print(df.head(10))
    except Exception as e:
        print(f"Fout bij lezen van parquet: {e}")
