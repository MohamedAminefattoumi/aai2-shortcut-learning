
"""LIME explainability functions for the XAI experiments."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries


def make_predict_fn(model: torch.nn.Module, device: torch.device):
    """Create a LIME-compatible prediction function taking NHWC images."""

    model.eval()

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        tensor = (
            torch.from_numpy(batch)
            .float()
            .permute(0, 3, 1, 2)
            .to(device)
        )

        with torch.no_grad():
            probabilities = F.softmax(model(tensor), dim=1)

        return probabilities.cpu().numpy()

    return predict_fn


def get_explanation(
    image: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    top_labels: int = 5,
    hide_color: int = 0,
    num_samples: int = 1000,
):
    """Generate a LIME explanation for a single (3, H, W) image."""

    image_np = (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        .astype("double")
    )

    predict_fn = make_predict_fn(model, device)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=top_labels,
        hide_color=hide_color,
        num_samples=num_samples,
    )

    return explanation


def visualize_explanation_lime(
    explanation,
    model_name: str,
    true_label: int,
    save_path: str | None = None,
):
    """Visualize and optionally save the LIME superpixel explanation."""

    top_label = explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(mark_boundaries(temp, mask))
    plt.title(
        f"{model_name.capitalize()} - "
        f"predicted: {top_label}, true: {true_label}"
    )
    plt.axis("off")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()
