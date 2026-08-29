from pathlib import Path

from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
)

from calibration_reader import MLPDataReader

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ONNX_DIR = PROJECT_ROOT / "ONNX"

FP32_MODEL = ONNX_DIR / "mlp_model.onnx"
INT8_MODEL = ONNX_DIR / "mlp_model_int8.onnx"

# --------------------------------------------------
# Check FP32 Model
# --------------------------------------------------

if not FP32_MODEL.exists():
    raise FileNotFoundError(
        f"FP32 ONNX model not found:\n{FP32_MODEL}"
    )

# --------------------------------------------------
# Create Calibration Reader
# --------------------------------------------------

data_reader = MLPDataReader()

# --------------------------------------------------
# Perform Static Quantization
# --------------------------------------------------

print("\nStarting Static Quantization...\n")

quantize_static(
    model_input=str(FP32_MODEL),
    model_output=str(INT8_MODEL),
    calibration_data_reader=data_reader,

    quant_format=QuantFormat.QDQ,

    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,

    per_channel=True,
)

# --------------------------------------------------
# Done
# --------------------------------------------------

print("=" * 60)
print("Static Quantization Completed Successfully")
print("=" * 60)

print(f"\nINT8 Model Saved To:\n{INT8_MODEL}")