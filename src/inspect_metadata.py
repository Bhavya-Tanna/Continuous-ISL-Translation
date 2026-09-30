import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "iSign_v1.1.csv"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(METADATA_PATH)

print("\n===== DATASET SHAPE =====")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


print("\n===== COLUMNS =====")
for column in df.columns:
    print(f"- {column}")


print("\n===== FIRST 5 ROWS =====")
print(df.head())


print("\n===== DATA TYPES =====")
print(df.dtypes)


print("\n===== MISSING VALUES =====")
missing = df.isnull().sum()
print(missing[missing > 0])


print("\n===== DUPLICATES =====")
print(f"Duplicate rows: {df.duplicated().sum():,}")