"""
------------------------------------------------------------
File        : 19_profile_onnx_int8.py
Description : Profile the INT8 ONNX model
Author      : TinyML Workshop
------------------------------------------------------------
"""

import os
import time
import numpy as np
import onnxruntime as ort

# ============================================================
# File Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "ONNX",
    "mlp_model_int8.onnx"
)

# ============================================================
# Load INT8 ONNX Model
# ============================================================

print("\nLoading INT8 ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Model loaded successfully.")

# ============================================================
# Model Information
# ============================================================

model_size = os.path.getsize(MODEL_PATH) / 1024  # KB

# Neural network architecture (same as FP32)
parameters = 2693

# From STM32Cube.AI Network Report
macc = 2715

# ============================================================
# Measure Inference Latency
# ============================================================

dummy_input = np.random.rand(1, 6).astype(np.float32)

# Warm-up
for _ in range(100):
    session.run(None, {input_name: dummy_input})

iterations = 1000

start = time.perf_counter()

for _ in range(iterations):
    session.run(None, {input_name: dummy_input})

end = time.perf_counter()

latency_ms = ((end - start) / iterations) * 1000
throughput = 1000 / latency_ms

# ============================================================
# Display Results
# ============================================================

print("\n" + "=" * 60)
print("              INT8 ONNX MODEL PROFILE")
print("=" * 60)

print(f"Model Size         : {model_size:.2f} KB")
print(f"Parameters         : {parameters}")
print(f"MACC               : {macc}")

print(f"\nAverage Latency    : {latency_ms:.6f} ms")
print(f"Throughput         : {throughput:.2f} inf/sec")

print("=" * 60)
print("INT8 ONNX profiling completed successfully.")
print("=" * 60)