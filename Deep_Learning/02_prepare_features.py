from pathlib import Path
import pandas as pd

# -------------------------------------------------------
# Locate Project Root
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset_path = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = pd.read_csv(dataset_path)

# -------------------------------------------------------
# Separate Features and Labels
# -------------------------------------------------------

X = df.drop(columns=["Label"])

y = df["Label"]

# -------------------------------------------------------
# Display Information
# -------------------------------------------------------

print("=" * 60)
print("Feature Matrix and Target Vector")
print("=" * 60)

print("\nFeature Columns")
print(X.columns.tolist())

print("\nTarget Column")
print(y.name)

print("\nFeature Matrix Shape")
print(X.shape)

print("\nTarget Vector Shape")
print(y.shape)

print("\nFirst Five Feature Samples")
print(X.head())

print("\nFirst Five Labels")
print(y.head().to_string(index=False))

print("\nFeature Data Types")
print(X.dtypes)