import pandas as pd

# Load dataset
df = pd.read_csv("../Datasets/dataset.csv")

# Check missing values
print(df.isnull().sum())

# Total missing values
print("\nTotal Missing Values:", df.isnull().sum().sum())