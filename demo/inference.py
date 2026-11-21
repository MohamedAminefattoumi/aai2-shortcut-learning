"""Deterministic inference helpers matching the training/evaluation preprocessing."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_example_image(image_path: str) -> tuple[Image.Image, torch.Tensor]:
    """Load a 28x28 RGB PNG and convert it to a (1, 3, 28, 28) float32 tensor in [0, 1]."""
    pil_img = Image.open(image_path).convert("RGB")
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)
    return pil_img, tensor


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, true_label: int = 6) -> dict:
    """Run deterministic inference on a single preprocessed image tensor."""
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(image_tensor.to(device))
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])
    return {
        "predicted_label": pred_label,
        "true_label": int(true_label),
        "confidence": confidence,
        "is_correct": pred_label == int(true_label),
        "probabilities": probs.tolist(),
    }
