import polars as pl
from config import DATA_DIR, FILE_OIKEN_CLEAN

df = pl.read_parquet(FILE_OIKEN_CLEAN)
pv_cols = ["pv_central_valais", "pv_sierre", "pv_sion"]
for col in pv_cols:
    if col in df.columns:
        n_null = df[col].is_null().sum()
        print(f"{col:<25} min={df[col].min():.3f}  max={df[col].max():.3f}  NaN={n_null:,}")