from pathlib import Path
import torch

from build_mlp import MLP

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "Models" / "mlp_model.pth"

ONNX_PATH = PROJECT_ROOT / "ONNX" / "mlp_model.onnx"

# -------------------------------------------------------
# Load Trained Model
# -------------------------------------------------------

model = MLP()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)

model.eval()

# -------------------------------------------------------
# Dummy Input
# -------------------------------------------------------

dummy_input = torch.randn(1, 6)

# -------------------------------------------------------
# Export to ONNX
# -------------------------------------------------------

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)

print("=" * 60)
print("ONNX Model Exported Successfully")
print("=" * 60)

print(f"\nPyTorch Model : {MODEL_PATH}")
print(f"ONNX Model    : {ONNX_PATH}")

print("\nInput Shape  : (Batch Size, 6)")
print("Output Shape : (Batch Size, 5)")
print("ONNX Opset   : 17")