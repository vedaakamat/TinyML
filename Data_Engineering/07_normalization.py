import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("../Datasets/dataset.csv")

# Select IMU features
features = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]

# Normalize
scaler = MinMaxScaler()

df[features] = scaler.fit_transform(df[features])

# Display first five rows
print(df.head())