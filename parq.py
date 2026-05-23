# Geeft begin- en einddatum van een aantal steps in test.parquet
import pandas as pd
import pyarrow.parquet as pq
import os

# ============================================================
# INSTELLINGEN
# ============================================================
DATA_DIR   = "coreData"
STEPS      = 500_000       # aantal steps dat je wilt evalueren
# ============================================================

def main():
    parquet_path = os.path.join(DATA_DIR, "test.parquet")
    if not os.path.exists(parquet_path):
        print(f"Niet gevonden: {parquet_path}")
        return

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    print(f"Test data: {total_rows:,} rows totaal")

    # Lees alleen timestamp kolom van eerste en laatste N rows
    schema_cols = pf.schema_arrow.names
    time_cols = [c for c in schema_cols if any(x in c.lower() for x in ['time', 'date', 'ts', 'timestamp'])]

    if not time_cols:
        print(f"Geen tijdkolom gevonden. Beschikbare kolommen: {schema_cols}")
        print(f"\nAanname: 1 row = 1 seconde")
        steps = min(STEPS, total_rows)
        total_sec = total_rows
        steps_sec = steps
        print(f"Volledige test set: {total_sec:,}s = {total_sec/86400:.1f} dagen")
        print(f"Jouw {steps:,} steps    : {steps_sec:,}s = {steps_sec/86400:.1f} dagen")
        return

    time_col = time_cols[0]
    print(f"Tijdkolom: {time_col}")

    # Lees eerste rij
    first_row = pf.read_row_group(0, columns=[time_col]).to_pandas()
    begin_tijd = pd.to_datetime(first_row[time_col].iloc[0])

    # Lees rij op positie STEPS
    steps = min(STEPS, total_rows)
    # Zoek de row group die rij `steps` bevat
    cumsum = 0
    target_rg = 0
    offset_in_rg = 0
    for i in range(pf.metadata.num_row_groups):
        rg_rows = pf.metadata.row_group(i).num_rows
        if cumsum + rg_rows >= steps:
            target_rg = i
            offset_in_rg = steps - cumsum - 1
            break
        cumsum += rg_rows

    step_df = pf.read_row_group(target_rg, columns=[time_col]).to_pandas()
    eind_tijd = pd.to_datetime(step_df[time_col].iloc[offset_in_rg])

    # Laatste rij (einde test set)
    last_rg = pf.metadata.num_row_groups - 1
    last_df = pf.read_row_group(last_rg, columns=[time_col]).to_pandas()
    einde_test = pd.to_datetime(last_df[time_col].iloc[-1])

    duur = eind_tijd - begin_tijd
    totaal = einde_test - begin_tijd

    print(f"\nBegin test data : {begin_tijd}")
    print(f"Einde test data : {einde_test}  (totaal: {totaal.days} dagen)")
    print(f"\nJouw {steps:,} steps:")
    print(f"  Van : {begin_tijd}")
    print(f"  Tot : {eind_tijd}")
    print(f"  Duur: {duur.days} dagen, {duur.seconds//3600} uur")

if __name__ == "__main__":
    main()
