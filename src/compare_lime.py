import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

from src.shortcuts import make_colored_mnist
from src.models import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints"
#
SEED = 1
x = np.random.RandomState(SEED).randint(0, 100)  # Random index for reproducibility

x_test, y_test = make_colored_mnist(
    train=False,
    mode="inverted"
).tensors

model_baseline = build_model("cnn", x_test[:2]).to(device)
model_reweighting = build_model("cnn", x_test[:2]).to(device)

# Use the checkpoint names that actually exist in your folder
# (p=0.99 ones come from run_experiments; adjust if you want p=0.9)
model_baseline.load_state_dict(
    torch.load(f"{CHECKPOINT_DIR}/none_p099_seed0.pth",
               map_location=device, weights_only=True)
)
model_reweighting.load_state_dict(
    torch.load(f"{CHECKPOINT_DIR}/reweighting_p099_seed0.pth",
               map_location=device, weights_only=True)
)

model_baseline.eval()
model_reweighting.eval()

with torch.no_grad():
    predictions_baseline = model_baseline(x_test.to(device)).argmax(dim=1).cpu()
    predictions_reweighting = model_reweighting(x_test.to(device)).argmax(dim=1).cpu()

candidates = torch.where(
    (predictions_baseline != y_test) &
    (predictions_reweighting == y_test)
)[0]

if len(candidates) == 0:
    raise RuntimeError("No suitable counterfactual example found.")

INDEX = candidates[x].item()

image = x_test[INDEX]
true_label = y_test[INDEX].item()
baseline_prediction = predictions_baseline[INDEX].item()
reweighting_prediction = predictions_reweighting[INDEX].item()


# ---------------- LIME helpers ----------------
def grid_segments(img, cell=4):
    h, w = img.shape[:2]
    rows, cols = np.arange(h) // cell, np.arange(w) // cell
    return rows[:, None] * (w // cell + 1) + cols[None, :]


def lime_heatmap(model, image, label):
    """Signed LIME weights for `label`, painted onto a (28, 28) map."""
    def predict_fn(batch):
        t = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            return F.softmax(model(t), dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer(random_state=SEED)
    exp = explainer.explain_instance(
        image.permute(1, 2, 0).cpu().numpy().astype("double"),
        predict_fn,
        labels=(label,),
        hide_color=0,
        num_samples=2000,
        segmentation_fn=lambda x: grid_segments(x, cell=4),
        random_seed=SEED,
    )
    lookup = np.zeros(exp.segments.max() + 1)
    for seg_id, w in exp.local_exp[label]:
        lookup[seg_id] = w
    return lookup[exp.segments], exp, label


baseline_heatmap, exp_b, _ = lime_heatmap(model_baseline, image, true_label)
reweighting_heatmap, exp_r, _ = lime_heatmap(model_reweighting, image, true_label)

# Shared colour scale so the two maps are comparable
vmax = max(np.abs(baseline_heatmap).max(), np.abs(reweighting_heatmap).max()) + 1e-8

original = image.permute(1, 2, 0).cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(original, interpolation="nearest")
axes[0].set_title(f"Original Image\nTrue label: {true_label}")
axes[0].axis("off")

axes[1].imshow(original, interpolation="nearest")
axes[1].imshow(baseline_heatmap, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               alpha=0.6, interpolation="nearest")
axes[1].set_title(f"Baseline\nPrediction: {baseline_prediction}")
axes[1].axis("off")

axes[2].imshow(original, interpolation="nearest")
im = axes[2].imshow(reweighting_heatmap, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    alpha=0.6, interpolation="nearest")
axes[2].set_title(f"Reweighting\nPrediction: {reweighting_prediction}")
axes[2].axis("off")

fig.colorbar(im, ax=axes, fraction=0.02, label="LIME weight (red = supports true class)")
plt.suptitle(f"LIME evidence for the TRUE class ({true_label}): Baseline vs Sample Reweighting")
plt.savefig("lime_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Selected index:", INDEX)
print("True label:", true_label)
print("Baseline prediction:", baseline_prediction)
print("Reweighting prediction:", reweighting_prediction)