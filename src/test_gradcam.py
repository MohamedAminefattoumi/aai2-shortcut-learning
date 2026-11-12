
import torch

from src.data import get_loaders
from src.models import build_model
from src.xai import generate_gradcam, visualize_gradcam

# Load real Colored MNIST
loaders = get_loaders(
    p=0.9,
    seed=0,
    root="src/data",
    n_train=1000
)

# Get one image
images, labels, shortcut_labels = next(
    iter(loaders["train_biased"])
)

image = images[0]
true_label = labels[0].item()

# Build the CNN for RGB images
model = build_model("cnn", images)

# For this first test, use an untrained model
heatmap, target_class = generate_gradcam(
    model,
    image,
    target_class=true_label
)

print("True label:", true_label)
print("Target class:", target_class)
print("Heatmap shape:", heatmap.shape)

visualize_gradcam(image, heatmap, target_class)
