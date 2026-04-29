import polars as pl
from pathlib import Path
from config import FILE_METEO_HIST_CLEAN, DATA_DIR

GOLDEN_DIR = DATA_DIR.parent / "golden_data"

meteo_hist_train = pl.read_parquet(FILE_METEO_HIST_CLEAN("sion"))
meteo_hist       = pl.read_parquet(GOLDEN_DIR / "meteo_sion_hist_golden_clean.parquet")

print("Train cols :", meteo_hist_train.columns)
print("Golden cols:", meteo_hist.columns)
print("Communes   :", [c for c in meteo_hist_train.columns if c in meteo_hist.columns])