"""Fake data with the same interface as Coder A's get_loaders().

Use it to test train.py / evaluate.py before the real data exists.
Images are dark random noise (no digit), so the white patch is the ONLY
learnable signal: a model trained on train_biased must learn the shortcut.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def add_patch(img: torch.Tensor, k: int, size: int = 3) -> torch.Tensor: #paints the white square at the position of a class.
    """img: (C, H, W) in [0, 1]; k: class whose position is used."""
    img = img.clone()                 # never modify the original image
    row = 1 if k < 5 else 24          # top edge for 0-4, bottom edge for 5-9
    col = 2 + (k % 5) * 5             # columns 2, 7, 12, 17, 22
    img[:, row:row + size, col:col + size] = 1.0
    return img


def _make_dataset(n: int, mode: str, seed: int) -> TensorDataset: #builds one dataset in one of three modes: no patch (clean), patch at the true class (aligned), patch at label + 1 (conflicting).
    """mode: 'clean' (no patch), 'aligned' (true class), 'conflicting' (label + 1)."""
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 1, 28, 28, generator=g) * 0.5   # dark noise, so the patch stands out
    labels = torch.randint(0, 10, (n,), generator=g)

    if mode == "clean":
        shortcut_labels = torch.full((n,), -1)              # -1 = no patch
    elif mode == "aligned":
        shortcut_labels = labels.clone()
    elif mode == "conflicting":
        shortcut_labels = (labels + 1) % 10
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if mode != "clean":
        images = torch.stack([add_patch(img, k.item()) for img, k in zip(images, shortcut_labels)])

    return TensorDataset(images, labels, shortcut_labels)


def get_fake_loaders(batch_size: int = 128, n_train: int = 2000, n_test: int = 500) -> dict: #builds all five datasets and wraps each in a DataLoader.
    """Same keys and batch format as the real get_loaders()."""
    specs = {
        "train_clean":      (n_train, "clean",       0, True),
        "train_biased":     (n_train, "aligned",     1, True),
        "test_aligned":     (n_test,  "aligned",     2, False),
        "test_clean":       (n_test,  "clean",       3, False),
        "test_conflicting": (n_test,  "conflicting", 4, False),
    }
    return {
        name: DataLoader(_make_dataset(n, mode, seed), batch_size=batch_size, shuffle=shuffle)
        for name, (n, mode, seed, shuffle) in specs.items()
    }


if __name__ == "__main__":
    loaders = get_fake_loaders()
    for name, loader in loaders.items():
        images, labels, shortcut_labels = next(iter(loader))
        print(f"{name:17s} {tuple(images.shape)}  labels {labels[:5].tolist()}  "
              f"shortcut {shortcut_labels[:5].tolist()}")