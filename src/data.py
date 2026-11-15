"""Real data loaders with the agreed interface (replaces fake_loaders.py).

Every batch is (images, labels, shortcut_labels):
    images:          float32, shape (B, 3, 28, 28), values in [0, 1]
    labels:          int64, shape (B,), the true digit 0-9
    shortcut_labels: int64, shape (B,), the class whose colour the digit has,
                     or -1 if the digit has no colour (plain white digit)

The shortcut is the colour of the digit (see shortcuts.py).
"""

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets

from src.shortcuts import choose_color_ids, colorize


def load_mnist(train, root="data"):
    """Return MNIST images (N, 28, 28) as float32 in [0, 1] and labels (N,) as int64."""
    mnist = datasets.MNIST(root=root, train=train, download=True)
    return mnist.data.float() / 255.0, mnist.targets


def colored_dataset(images, labels, p, mode, seed):
    """Colour the digits and keep the colour ids as shortcut labels."""
    generator = torch.Generator().manual_seed(seed)
    color_ids = choose_color_ids(labels, p, mode, generator)
    return TensorDataset(colorize(images, color_ids), labels, color_ids)


def clean_dataset(images, labels):
    """White digits on black: the same value in the 3 channels, so no colour cue."""
    x = images.unsqueeze(1).repeat(1, 3, 1, 1)          # (N, 28, 28) -> (N, 3, 28, 28)
    no_shortcut = torch.full_like(labels, -1)           # -1 = no shortcut
    return TensorDataset(x, labels, no_shortcut)


def get_datasets(p=0.9, seed=0, root="data", n_train=None):
    """Build the six datasets. n_train keeps only the first n_train training images (for quick tests)."""
    train_images, train_labels = load_mnist(train=True, root=root)
    test_images, test_labels = load_mnist(train=False, root=root)

    if n_train is not None:
        train_images, train_labels = train_images[:n_train], train_labels[:n_train]

    return {
        # Training sets
        "train_clean":      clean_dataset(train_images, train_labels),
        "train_biased":     colored_dataset(train_images, train_labels, p, "correlated", seed),
        # Test sets: the same test digits, only the colouring rule changes
        "test_clean":       clean_dataset(test_images, test_labels),
        "test_aligned":     colored_dataset(test_images, test_labels, 1.0, "correlated", seed + 1),
        "test_random":      colored_dataset(test_images, test_labels, p, "random", seed + 2),
        "test_conflicting": colored_dataset(test_images, test_labels, p, "inverted", seed + 3),
    }


def get_loaders(batch_size=128, p=0.9, seed=0, root="data", n_train=None):
    """Same keys and batch format as get_fake_loaders(), plus test_random."""
    return make_loaders(get_datasets(p, seed, root, n_train), batch_size, seed)


def make_loaders(datasets, batch_size=128, seed=0):
    """Wrap each dataset in a DataLoader. Training sets are shuffled, test sets are not.

    Shared by get_loaders() and data_mitigations.get_mitigated_loaders().
    """
    loaders = {}
    for name, dataset in datasets.items():
        is_train = name.startswith("train")
        # A seeded generator makes the shuffling order reproducible.
        generator = torch.Generator().manual_seed(seed)
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, generator=generator)
    return loaders


if __name__ == "__main__":
    p = 0.9
    loaders = get_loaders(p=p)

    for name, loader in loaders.items():
        images, labels, shortcut_labels = next(iter(loader))
        print(f"{name:17s} {tuple(images.shape)} {images.dtype} "
              f"labels {labels[:5].tolist()} shortcut {shortcut_labels[:5].tolist()}")

    # How often the colour points to the true class, over the whole dataset.
    for name, loader in loaders.items():
        _, labels, shortcut_labels = loader.dataset.tensors
        match = (shortcut_labels == labels).float().mean().item()
        print(f"{name:17s} colour == label: {match:.3f}")
