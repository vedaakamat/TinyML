import pandas as pd
import numpy as np

# Load first test sample
X = pd.read_csv("X_test.csv")

sample = X.iloc[0].values

print("Normalized Input:")
print(sample)

sample_uint8 = np.round(sample * 255).astype(np.uint8)

print("\nUINT8 Input:")
print(sample_uint8)