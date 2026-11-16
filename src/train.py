import random
import numpy as np
import torch
import torch.nn as nn
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

"""Training loop shared by all experiments (baseline and mitigation).

train.py only knows HOW to train. WHAT to train (model, data, loss, seed)
is decided in run_experiments.py.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                 #seed the GPU, if there is one


def get_device():                                    
    """GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cross_entropy_loss(scores, labels, shortcut_labels=None):   
    """Default loss (baseline). Ignores shortcut_labels; it only accepts them so that
    every loss has the same signature: loss_fn(scores, labels, shortcut_labels)."""
    return F.cross_entropy(scores, labels)


def train(model, loader, epochs=5, lr=1e-3, loss_fn=None, device=None):
    """Train with Adam
    Returns the average loss per epoch.

    loss_fn: function(scores, labels, shortcut_labels) -> loss. None = cross-entropy.
    device:  None = automatic (CUDA if available, else CPU).
    """
    loss_fn = loss_fn or cross_entropy_loss          # default loss is cross-entropy
    device = device or get_device()                  # automatic device selection
    model.to(device)                                 # model on the device
    print(f"Training on: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for images, labels, shortcut_labels in loader:          # NEW: shortcut labels kept
            images = images.to(device)                          # NEW: data on the same device
            labels = labels.to(device)
            shortcut_labels = shortcut_labels.to(device)

            scores = model(images)                              # forward pass
            loss = loss_fn(scores, labels, shortcut_labels)     # NEW: baseline or mitigation loss
            optimizer.zero_grad()                               # reset old gradients
            loss.backward()                                     # backpropagation
            optimizer.step()                                    # weight update

            total_loss += loss.item() * len(labels)
            n += len(labels)
        history.append(total_loss / n)
        print(f"epoch {epoch + 1}: loss {history[-1]:.4f}")

    return history


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)   # so we can reuse the model, not retrain it
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device=None):
    device = device or get_device()
    model.load_state_dict(torch.load(path, map_location=device))   # NEW: works even if saved on a GPU
    model.to(device)
    return model


if __name__ == "__main__":
    from functools import partial
    from src.fake_loaders import get_fake_loaders
    from src.mitigations import reweighting_loss
    from src.models import SmallCNN

    print("device:", get_device())
    loaders = get_fake_loaders()

    # 1) Baseline, default loss: must give the same numbers as before the change
    for train_name in ["train_clean", "train_biased"]:
        print(f"--- baseline, {train_name}")
        set_seed(42)
        model = SmallCNN()
        train(model, loaders[train_name], epochs=3)
        save_checkpoint(model, f"checkpoints/test_cnn_{train_name}.pt")

    # 2) Mohamed's reweighting loss, plugged into the same loop
    print("--- reweighting, train_biased")
    set_seed(42)
    train(SmallCNN(), loaders["train_biased"], epochs=3, loss_fn=partial(reweighting_loss, upweight=10.0))



