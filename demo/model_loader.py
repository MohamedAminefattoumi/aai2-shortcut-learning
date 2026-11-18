"""Read-only model and checkpoint loader for the Colored MNIST demo."""

import os
import sys

import torch

# Add repository root to sys.path so we can reuse src.models without modifying src/
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DEMO_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import SmallCNN

CHECKPOINT_PATHS = {
    "Baseline": os.path.join(REPO_ROOT, "checkpoints", "none_p099_seed0.pth"),
    "Data Mitigation": os.path.join(REPO_ROOT, "checkpoints", "augmentation_p099_seed0.pth"),
    "Reweighting": os.path.join(REPO_ROOT, "checkpoints", "reweighting_p099_seed0.pth"),
}

MODEL_DESCRIPTIONS = {
    "Baseline": "Trained on biased Colored MNIST (p = 0.99, no mitigation).",
    "Data Mitigation": "Trained with dynamic random color augmentation (src/data_mitigations.py).",
    "Reweighting": "Trained with sample reweighting loss (upweight = 10.0, src/mitigations.py).",
}


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str, device: torch.device | None = None) -> SmallCNN:
    """Instantiate SmallCNN (3x28x28) and load its read-only checkpoint."""
    if model_name not in CHECKPOINT_PATHS:
        raise ValueError(f"Unknown model name: {model_name!r}. Expected one of {list(CHECKPOINT_PATHS)}.")

    ckpt_path = CHECKPOINT_PATHS[model_name]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {ckpt_path}")

    target_device = device or get_device()
    model = SmallCNN(in_channels=3, height=28, width=28, n_classes=10)
    state_dict = torch.load(ckpt_path, map_location=target_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model


def load_all_models(device: torch.device | None = None) -> dict[str, SmallCNN]:
    """Load all three trained checkpoints into evaluation mode."""
    target_device = device or get_device()
    return {name: load_model(name, device=target_device) for name in CHECKPOINT_PATHS}
