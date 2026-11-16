
import torch
import matplotlib.pyplot as plt

from src.shortcuts import make_colored_mnist
from src.models import build_model
from src.xai import generate_gradcam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints"

x_test, y_test = make_colored_mnist(
    train=False,
    mode="inverted"
).tensors

model_baseline = build_model("cnn", x_test[:2]).to(device)
model_reweighting = build_model("cnn", x_test[:2]).to(device)

model_baseline.load_state_dict(
    torch.load(
        f"{CHECKPOINT_DIR}/baseline_p09_seed0.pth",
        map_location=device,
        weights_only=True
    )
)

model_reweighting.load_state_dict(
    torch.load(
        f"{CHECKPOINT_DIR}/reweighting_p09_seed0.pth",
        map_location=device,
        weights_only=True
    )
)

model_baseline.eval()
model_reweighting.eval()

with torch.no_grad():
    predictions_baseline = model_baseline(
        x_test.to(device)
    ).argmax(dim=1).cpu()

    predictions_reweighting = model_reweighting(
        x_test.to(device)
    ).argmax(dim=1).cpu()

candidates = torch.where(
    (predictions_baseline != y_test) &
    (predictions_reweighting == y_test)
)[0]

if len(candidates) == 0:
    raise RuntimeError("No suitable counterfactual example found.")

INDEX = candidates[0].item()

image = x_test[INDEX]
true_label = y_test[INDEX].item()

baseline_prediction = predictions_baseline[INDEX].item()
reweighting_prediction = predictions_reweighting[INDEX].item()

baseline_heatmap, _ = generate_gradcam(
    model_baseline,
    image,
    target_class=true_label
)

reweighting_heatmap, _ = generate_gradcam(
    model_reweighting,
    image,
    target_class=true_label
)

original = image.permute(1, 2, 0).cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(original)
axes[0].set_title(f"Original Image\nTrue label: {true_label}")
axes[0].axis("off")

axes[1].imshow(original)
axes[1].imshow(baseline_heatmap.numpy(), cmap="jet", alpha=0.5)
axes[1].set_title(f"Baseline\nPrediction: {baseline_prediction}")
axes[1].axis("off")

axes[2].imshow(original)
axes[2].imshow(reweighting_heatmap.numpy(), cmap="jet", alpha=0.5)
axes[2].set_title(f"Reweighting\nPrediction: {reweighting_prediction}")
axes[2].axis("off")

plt.suptitle("Grad-CAM: Baseline vs Sample Reweighting")
plt.tight_layout()
plt.savefig("gradcam_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Selected index:", INDEX)
print("True label:", true_label)
print("Baseline prediction:", baseline_prediction)
print("Reweighting prediction:", reweighting_prediction)

