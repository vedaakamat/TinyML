from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from build_mlp import MLP


# -------------------------------------------------------
# Random Seed
# -------------------------------------------------------

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Previously created test dataset
X_TEST_PATH = Path(__file__).resolve().parent / "X_test.csv"

# Trained model
MODEL_PATH = PROJECT_ROOT / "Models" / "mlp_model.pth"


# -------------------------------------------------------
# Check Required Files
# -------------------------------------------------------

if not X_TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test feature file not found: {X_TEST_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )


# -------------------------------------------------------
# Load Previously Created Test Dataset
# -------------------------------------------------------

X_test = pd.read_csv(X_TEST_PATH)


# -------------------------------------------------------
# Convert to NumPy Array
# -------------------------------------------------------

X_test = X_test.values


# -------------------------------------------------------
# Display Test Dataset Information
# -------------------------------------------------------

print("=" * 60)
print("Previously Created Test Dataset Loaded")
print("=" * 60)

print(f"\nTesting Samples : {len(X_test)}")
print(f"Testing Shape   : {X_test.shape}")


# -------------------------------------------------------
# Convert to Tensor
# -------------------------------------------------------

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
)


# -------------------------------------------------------
# Select One Sample
# -------------------------------------------------------

sample = X_test_tensor[0].unsqueeze(0)


# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

model = MLP()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
    )
)

model.eval()


# -------------------------------------------------------
# Warm-up
# -------------------------------------------------------

print("=" * 60)
print("PyTorch FP32 Profiling")
print("=" * 60)

print("Running Warm-up...")

with torch.no_grad():

    for _ in range(20):

        _ = model(sample)

print("Warm-up Completed")


# -------------------------------------------------------
# Profiling
# -------------------------------------------------------

NUM_RUNS = 1000

print(f"\nNumber of Profiling Runs : {NUM_RUNS}")

start = time.perf_counter()

with torch.no_grad():

    for _ in range(NUM_RUNS):

        outputs = model(sample)

end = time.perf_counter()


# -------------------------------------------------------
# Performance Metrics
# -------------------------------------------------------

total_time = end - start

average_latency_ms = (
    total_time / NUM_RUNS
) * 1000

throughput = NUM_RUNS / total_time


# -------------------------------------------------------
# Display Results
# -------------------------------------------------------

print("\n" + "=" * 60)
print("PyTorch FP32 Performance")
print("=" * 60)

print(f"Device               : CPU")
print(f"Input Shape          : {tuple(sample.shape)}")
print(f"Number of Runs       : {NUM_RUNS}")
print(f"Average Latency      : {average_latency_ms:.6f} ms")
print(f"Throughput           : {throughput:.2f} inf/sec")

print("=" * 60)