
"""Compare Grad-CAM and Integrated Gradients for all four mitigation strategies."""

import os
import torch

from src.data_mitigations import get_mitigated_loaders
from src.models import build_model
from src.xai import (
    generate_gradcam,
    integrated_gradients,
    visualize_gradcam,
    visualize_integrated_gradients,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

P = 0.99
SEED = 0
OUTPUT_DIR = "results/xai"


def load_model(checkpoint_path, loaders):
    """Load a trained model from a saved checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # Build the model using the expected input shape.
    images, _, _ = next(iter(loaders["train_biased"]))
    model = build_model("cnn", images).to(DEVICE)

    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Loaded: {checkpoint_path}")

    return model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Preparing data loaders...")

    loaders = get_mitigated_loaders(
        mitigation="none",
        batch_size=128,
        p=P,
        seed=SEED,
        n_train=None,
    )

    print("Loading trained models...")

    models = {
        "Baseline": load_model(
            "checkpoints/none_p099_seed0.pth",
            loaders,
        ),
        "Augmentation": load_model(
            "checkpoints/augmentation_p099_seed0.pth",
            loaders,
        ),
        "Rebalancing": load_model(
            "checkpoints/rebalancing_p099_seed0.pth",
            loaders,
        ),
        "Reweighting": load_model(
            "checkpoints/reweighting_p099_seed0.pth",
            loaders,
        ),
    }

    # IMPORTANT: The inverted test set is called test_conflicting.
    test_loader = loaders["test_conflicting"]

    print("\nSearching for an informative example...")

    selected_image = None
    selected_label = None
    selected_color = None

    # Find an image where:
    # Baseline follows the wrong colour.
    # At least one mitigation predicts the correct digit.
    for images, labels, shortcut_labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        shortcut_labels = shortcut_labels.to(DEVICE)

        with torch.no_grad():
            predictions = {
                name: model(images).argmax(dim=1)
                for name, model in models.items()
            }

        baseline_preds = predictions["Baseline"]

        mitigation_correct = (
            (predictions["Augmentation"] == labels)
            | (predictions["Rebalancing"] == labels)
            | (predictions["Reweighting"] == labels)
        )

        mask = (
            (baseline_preds == shortcut_labels)
            & (baseline_preds != labels)
            & mitigation_correct
        )

        indices = torch.where(mask)[0]

        if len(indices) > 0:
            idx = indices[0].item()

            selected_image = images[idx:idx + 1]
            selected_label = labels[idx].item()
            selected_color = shortcut_labels[idx].item()

            print("Informative example found!")
            print(f"True digit: {selected_label}")
            print(f"Shortcut colour label: {selected_color}")

            for name, preds in predictions.items():
                print(f"{name} prediction: {preds[idx].item()}")

            break

    # Fallback: use the first image if no ideal example is found.
    if selected_image is None:
        print("No ideal example found. Using the first test image.")

        images, labels, shortcut_labels = next(iter(test_loader))

        selected_image = images[0:1].to(DEVICE)
        selected_label = labels[0].item()
        selected_color = shortcut_labels[0].item()

        print(f"True digit: {selected_label}")
        print(f"Shortcut colour label: {selected_color}")

    # Generate explanations for all models using the SAME image.
    for name, model in models.items():

        print(f"\nGenerating XAI for {name}...")

        # Grad-CAM
        heatmap, prediction = generate_gradcam(
            model,
            selected_image,
        )

        visualize_gradcam(
            selected_image,
            heatmap,
            prediction,
            save_path=f"{OUTPUT_DIR}/{name.lower()}_gradcam.png",
        )

        # Integrated Gradients
        attribution, prediction = integrated_gradients(
            model,
            selected_image,
            steps=50,
        )

        visualize_integrated_gradients(
            selected_image,
            attribution,
            prediction,
            save_path=f"{OUTPUT_DIR}/{name.lower()}_integrated_gradients.png",
        )

        print(f"{name}: prediction = {prediction}")

    print("\nXAI generation completed!")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
