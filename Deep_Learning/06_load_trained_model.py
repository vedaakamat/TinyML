from pathlib import Path
import torch

from build_mlp import MLP

# -------------------------------------------------------
# Locate Project Root
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -------------------------------------------------------
# Model Path
# -------------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "Models" / "mlp_model.pth"

# -------------------------------------------------------
# Create Model
# -------------------------------------------------------

model = MLP()

# -------------------------------------------------------
# Load Weights
# -------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)

# -------------------------------------------------------
# Evaluation Mode
# -------------------------------------------------------

model.eval()

# -------------------------------------------------------
# Display Information
# -------------------------------------------------------

print("=" * 60)
print("PyTorch Model Loaded Successfully")
print("=" * 60)

print(f"\nModel Path : {MODEL_PATH}")

print("\nModel Summary")
print(model)

print("\nModel Status : Evaluation Mode")

total_params = sum(p.numel() for p in model.parameters())

print(f"\nTotal Parameters : {total_params}")