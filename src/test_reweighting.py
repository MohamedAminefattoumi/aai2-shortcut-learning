
import torch

from src.data import get_loaders
from src.mitigations import (
    batch_weights,
    weighted_cross_entropy,
    reweighting_loss
)

# Load real Colored MNIST data
loaders = get_loaders(
    p=0.9,
    seed=0,
    root="src/data",
    n_train=1000
)

# Get one batch from biased training data
images, labels, shortcut_labels = next(
    iter(loaders["train_biased"])
)

# Check weights
weights = batch_weights(labels, shortcut_labels, upweight=10.0)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Shortcut labels shape:", shortcut_labels.shape)

print("Number of rare samples:",
      (shortcut_labels != labels).sum().item())

print("Number of normal samples:",
      (shortcut_labels == labels).sum().item())

print("Unique weights:", weights.unique())

# Test the loss with dummy logits
logits = torch.randn(len(labels), 10)

loss = reweighting_loss(logits, labels, shortcut_labels)

print("Reweighting loss:", loss.item())

assert torch.isfinite(loss)
assert weights.min().item() == 1.0
assert weights.max().item() == 10.0

print("All tests passed!")
