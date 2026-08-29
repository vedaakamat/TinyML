from pathlib import Path
import time
import numpy as np
import pandas as pd
import onnxruntime as ort


# -------------------------------------------------------
# Random Seed
# -------------------------------------------------------

SEED = 42

np.random.seed(SEED)


# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Previously created test dataset
X_TEST_PATH = Path(__file__).resolve().parent / "X_test.csv"

# ONNX model
MODEL_PATH = PROJECT_ROOT / "ONNX" / "mlp_model.onnx"


# -------------------------------------------------------
# Check Required Files
# -------------------------------------------------------

if not X_TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test feature file not found: {X_TEST_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}"
    )


# -------------------------------------------------------
# Load Previously Created Test Dataset
# -------------------------------------------------------

X_test = pd.read_csv(X_TEST_PATH)


# -------------------------------------------------------
# Convert to NumPy Array
# -------------------------------------------------------

X_test = X_test.values.astype(np.float32)


# -------------------------------------------------------
# Display Test Dataset Information
# -------------------------------------------------------

print("=" * 60)
print("Previously Created Test Dataset Loaded")
print("=" * 60)

print(f"\nTesting Samples : {len(X_test)}")
print(f"Testing Shape   : {X_test.shape}")


# -------------------------------------------------------
# Select One Sample
# -------------------------------------------------------

sample = X_test[0].reshape(1, 6)


# -------------------------------------------------------
# Load ONNX Model
# -------------------------------------------------------

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

output_name = session.get_outputs()[0].name


# -------------------------------------------------------
# Warm-up
# -------------------------------------------------------

print("=" * 60)
print("ONNX FP32 Profiling")
print("=" * 60)

print("Running Warm-up...")

for _ in range(20):

    session.run(
        [output_name],
        {input_name: sample},
    )

print("Warm-up Completed")


# -------------------------------------------------------
# Profiling
# -------------------------------------------------------

NUM_RUNS = 1000

print(f"\nNumber of Profiling Runs : {NUM_RUNS}")

start = time.perf_counter()

for _ in range(NUM_RUNS):

    outputs = session.run(
        [output_name],
        {input_name: sample},
    )

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
print("ONNX FP32 Performance")
print("=" * 60)

print(
    f"Execution Provider   : "
    f"{session.get_providers()[0]}"
)

print(f"Input Shape          : {sample.shape}")
print(f"Number of Runs       : {NUM_RUNS}")
print(
    f"Average Latency      : "
    f"{average_latency_ms:.6f} ms"
)
print(
    f"Throughput           : "
    f"{throughput:.2f} inf/sec"
)

print("=" * 60)