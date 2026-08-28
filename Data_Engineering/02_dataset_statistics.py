import pandas as pd

# Load the dataset
df = pd.read_csv("../Datasets/dataset.csv")

# Explore the dataset
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())