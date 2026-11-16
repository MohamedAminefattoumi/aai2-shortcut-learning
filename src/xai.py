
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# GRAD-CAM
# ==================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.hook = target_layer.register_forward_hook(
            self.save_activations
        )

    def save_activations(self, module, inputs, output):
        self.activations = output
        output.register_hook(self.save_gradients)

    def save_gradients(self, gradient):
        self.gradients = gradient

    def generate(self, image, target_class=None):
        device = next(self.model.parameters()).device

        image = image.to(device)

        if image.dim() == 3:
            image = image.unsqueeze(0)

        if image.shape[0] != 1:
            raise ValueError(
                "Grad-CAM currently supports one image at a time."
            )

        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(image)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass
        score = logits[0, target_class]
        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted sum of feature maps
        cam = (weights * self.activations).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        # Resize heatmap to original image dimensions
        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # Normalize between 0 and 1
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu(), target_class

    def remove_hooks(self):
        self.hook.remove()


def generate_gradcam(model, image, target_class=None):
    # SmallCNN's second convolutional layer
    target_layer = model.features[3]

    gradcam = GradCAM(model, target_layer)

    try:
        heatmap, predicted_class = gradcam.generate(
            image,
            target_class
        )
    finally:
        gradcam.remove_hooks()

    return heatmap, predicted_class


def visualize_gradcam(
    image,
    heatmap,
    predicted_class=None,
    save_path=None
):
    if image.dim() == 4:
        image = image[0]

    image = image.detach().cpu()

    # Convert RGB image to HWC format
    original = image.permute(1, 2, 0).numpy()

    heatmap = torch.as_tensor(heatmap).detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Original image
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Grad-CAM overlay
    axes[1].imshow(original)
    axes[1].imshow(
        heatmap.numpy(),
        cmap="jet",
        alpha=0.5
    )

    title = "Grad-CAM"

    if predicted_class is not None:
        title += f" - Class {predicted_class}"

    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ==================================================
# INTEGRATED GRADIENTS
# ==================================================

def integrated_gradients(
    model,
    image,
    target_class=None,
    baseline=None,
    steps=50
):
    """
    Compute Integrated Gradients for one image.

    Args:
        model: Trained PyTorch model.
        image: Input image with shape (C, H, W).
        target_class: Class to explain. Uses prediction if None.
        baseline: Reference image. Defaults to a black image.
        steps: Number of interpolation steps.

    Returns:
        attributions: Tensor with shape (C, H, W).
        target_class: Explained class.
    """

    device = next(model.parameters()).device

    image = image.detach().to(device)

    if image.dim() == 4:
        image = image.squeeze(0)

    if image.dim() != 3:
        raise ValueError("Expected image shape (C, H, W).")

    image = image.unsqueeze(0)

    if baseline is None:
        baseline = torch.zeros_like(image)
    else:
        baseline = baseline.detach().to(device)

        if baseline.dim() == 3:
            baseline = baseline.unsqueeze(0)

        if baseline.shape != image.shape:
            raise ValueError(
                "Baseline and image must have the same shape."
            )

    model.eval()

    # Determine target class
    with torch.no_grad():
        logits = model(image)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

    # Difference between input and baseline
    difference = image - baseline

    # Accumulate gradients
    total_gradients = torch.zeros_like(image)

    for step in range(steps + 1):

        alpha = step / steps

        interpolated = (
            baseline + alpha * difference
        ).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)

        output = model(interpolated)

        score = output[0, target_class]

        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolated
        )[0]

        # Trapezoidal integration
        if step == 0 or step == steps:
            total_gradients += 0.5 * gradients
        else:
            total_gradients += gradients

    average_gradients = total_gradients / steps

    # Integrated Gradients attribution
    attributions = difference * average_gradients

    return attributions.squeeze(0).detach().cpu(), target_class


# ==================================================
# INTEGRATED GRADIENTS VISUALIZATION
# ==================================================

def visualize_integrated_gradients(
    image,
    attributions,
    target_class=None,
    save_path=None
):
    if image.dim() == 4:
        image = image[0]

    image = image.detach().cpu()
    attributions = attributions.detach().cpu()

    original = image.permute(1, 2, 0).numpy()

    total_attribution = (
        attributions.abs().sum(dim=0).numpy()
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8)
    )

    for ax in axes.flat:
        ax.axis("off")

    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")

    # Overall attribution
    axes[0, 1].imshow(
        total_attribution,
        cmap="hot"
    )
    axes[0, 1].set_title("Overall Attribution")

    channel_names = ["Red", "Green", "Blue"]

    for i, name in enumerate(channel_names):

        attribution_map = attributions[i].numpy()

        max_abs = max(
            abs(attribution_map.min()),
            abs(attribution_map.max()),
            1e-8
        )

        ax = (
            axes[0, 2]
            if i == 0
            else axes[1, i - 1]
        )

        im = ax.imshow(
            attribution_map,
            cmap="seismic",
            vmin=-max_abs,
            vmax=max_abs
        )

        ax.set_title(f"{name} Channel Attribution")

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

    if target_class is not None:
        fig.suptitle(
            f"Integrated Gradients - Class {target_class}"
        )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
