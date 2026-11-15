"""Data-side mitigation strategies (guide, section 6.4).

Both strategies change the TRAINING data only. The model, the loss and the
test sets stay exactly the same as in the baseline, so any change in the test
accuracies can be attributed to the mitigation.

    augmentation: every time a training digit is used, it gets a new random
                  colour. The colour no longer predicts the label (it matches
                  only 1 time in 10, by chance).
    rebalancing:  add counterfactual copies of training digits painted with a
                  WRONG colour, until the real correlation drops to p_target.

The third strategy of the guide, reweighting, changes the loss instead of the
data and lives in mitigations.py.

Usage (same batch format as data.get_loaders):
    loaders = get_mitigated_loaders("augmentation", p=0.9)
    train(model, loaders["train_biased"])
"""

import os

import torch
from torch.utils.data import Dataset, TensorDataset

from src.data import get_datasets, make_loaders
from src.shortcuts import colorize

MITIGATIONS = ("none", "augmentation", "rebalancing")


class RandomColorDataset(Dataset):
    """Augmentation: the colour of each digit is drawn again every time it is read.

    It starts from the ORIGINAL grayscale digits, not from the coloured ones.
    Turning a coloured digit back into gray (mean or max of the channels) would
    keep a brightness that depends on the old colour, and therefore on the
    class: a hidden shortcut.
    """

    def __init__(self, gray_images, labels, seed=0):
        self.gray_images = gray_images          # (N, 28, 28), values in [0, 1]
        self.labels = labels                    # (N,), int64
        # Private seeded generator: the sequence of colours is reproducible.
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # A new colour id (0-9) each call: the same digit gets a different
        # colour in each epoch.
        color_id = torch.randint(0, 10, (1,), generator=self.generator)
        # colorize works on batches: (1, 28, 28) -> (1, 3, 28, 28), then drop the batch axis.
        image = colorize(self.gray_images[i].unsqueeze(0), color_id)[0]
        # The painted colour is the shortcut label (same meaning as in data.py).
        return image, self.labels[i], color_id[0]


def add_conflicting(biased, gray_images, p_target=0.5, seed=0):
    """Rebalancing: add k conflicting copies so that the real correlation becomes p_target.

    biased:      the train_biased TensorDataset (images, labels, shortcut_labels).
    gray_images: the same digits in grayscale, (N, 28, 28).
    A conflicting copy is a digit painted with a colour that is NOT its own class.
    """
    images, labels, color_ids = biased.tensors
    n = len(labels)
    n_match = (color_ids == labels).sum().item()       # images where colour == label

    # Copies are all conflicting, so they increase the total but not n_match:
    #   n_match / (n + k) = p_target   =>   k = n_match / p_target - n
    k = round(n_match / p_target) - n
    if k <= 0:
        raise ValueError(f"p_target={p_target} is not below the current correlation "
                         f"{n_match / n:.3f}: nothing to add")

    generator = torch.Generator().manual_seed(seed)

    # Which digits to copy. A permutation spreads the copies evenly: if k <= n,
    # every digit is copied at most once; if k > n, the permutation is reused.
    order = torch.randperm(n, generator=generator)
    idx = order[torch.arange(k) % n]

    # Shift the label by 1-9 (never 0, never 10): the colour can never be the
    # true class. randint(0, 10) would give the true colour 1 time in 10.
    shift = torch.randint(1, 10, (k,), generator=generator)
    wrong_ids = (labels[idx] + shift) % 10

    new_images = colorize(gray_images[idx], wrong_ids)
    return TensorDataset(
        torch.cat([images, new_images]),
        torch.cat([labels, labels[idx]]),
        torch.cat([color_ids, wrong_ids]),
    )


def get_mitigated_loaders(mitigation="none", batch_size=128, p=0.9, seed=0,
                          root="data", n_train=None, p_target=0.5):
    """Same keys and batch format as data.get_loaders(); only train_biased changes.

    mitigation: "none" (baseline), "augmentation" or "rebalancing".
    p_target:   correlation to reach with rebalancing (ignored otherwise).
    """
    if mitigation not in MITIGATIONS:
        raise ValueError(f"mitigation must be one of {MITIGATIONS}, got {mitigation!r}")

    datasets = get_datasets(p, seed, root, n_train)

    # train_clean repeats the original gray digit in the 3 channels, so its
    # channel 0 IS the original MNIST image: no need to load MNIST again.
    clean_images, clean_labels, _ = datasets["train_clean"].tensors
    gray_images = clean_images[:, 0]                    # (N, 28, 28)

    if mitigation == "augmentation":
        datasets["train_biased"] = RandomColorDataset(gray_images, clean_labels, seed)
    elif mitigation == "rebalancing":
        datasets["train_biased"] = add_conflicting(datasets["train_biased"], gray_images,
                                                   p_target, seed)

    return make_loaders(datasets, batch_size, seed)


def colour_label_match(loader):
    """Fraction of images whose colour is the colour of their class (one pass over the loader)."""
    match = total = 0
    for _, labels, shortcut_labels in loader:
        match += (shortcut_labels == labels).sum().item()
        total += len(labels)
    return match / total


def save_examples(path="results/figures/mitigation_examples.png", p=0.9, p_target=0.5):
    """Grid of training images: baseline, augmentation (2 epochs) and rebalancing copies."""
    import matplotlib.pyplot as plt

    datasets = get_datasets(p=p, n_train=1000)
    gray_images = datasets["train_clean"].tensors[0][:, 0]
    labels = datasets["train_clean"].tensors[1]
    augmented = RandomColorDataset(gray_images, labels)
    rebalanced = add_conflicting(datasets["train_biased"], gray_images, p_target)

    n = 10
    rows = {
        "train_biased": [datasets["train_biased"][i] for i in range(n)],
        # Reading the same 10 digits twice = two epochs: the colours change.
        "augmentation, epoch 1": [augmented[i] for i in range(n)],
        "augmentation, epoch 2": [augmented[i] for i in range(n)],
        # The added copies are at the end of the rebalanced dataset.
        "rebalancing, added copies": [rebalanced[len(labels) + i] for i in range(n)],
    }

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.1, len(rows) * 1.4))
    for r, (title, samples) in enumerate(rows.items()):
        for c, (image, label, color_id) in enumerate(samples):
            ax = axes[r, c]
            ax.imshow(image.permute(1, 2, 0).numpy())    # (3, H, W) -> (H, W, 3) for matplotlib
            ax.set_title(f"y={label.item()} c={color_id.item()}", fontsize=7)
            ax.axis("off")
        axes[r, 0].text(-0.3, 0.5, title, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    # Quick check, small and fast on a CPU (same idea as evaluate.py).
    from src.evaluate import evaluate_all, results_table
    from src.models import build_model
    from src.train import set_seed, train

    P, P_TARGET, N_TRAIN = 0.9, 0.5, 5000

    # 1) The mitigation must change the real correlation of the training set.
    #    Expected: none ~ p + (1 - p) / 10 = 0.91, augmentation ~ 0.10, rebalancing ~ P_TARGET.
    all_loaders = {}
    for mitigation in MITIGATIONS:
        loaders = get_mitigated_loaders(mitigation, p=P, n_train=N_TRAIN, p_target=P_TARGET)
        all_loaders[mitigation] = loaders
        train_loader = loaders["train_biased"]
        print(f"{mitigation:12s} train images {len(train_loader.dataset):6d}  "
              f"colour == label: {colour_label_match(train_loader):.3f}")

    # 2) Figure for the slides.
    save_examples(p=P, p_target=P_TARGET)

    # 3) Train the same model with each mitigation and compare on the test sets.
    runs = {}
    for mitigation, loaders in all_loaders.items():
        print(f"--- training {mitigation}")
        set_seed(42)
        images, _, _ = next(iter(loaders["train_biased"]))
        model = build_model("cnn", images)
        train(model, loaders["train_biased"], epochs=3)
        runs[("cnn", mitigation)] = model

    print()
    # The test sets are identical for every mitigation, so any loaders dict works here.
    print(results_table(evaluate_all(runs, all_loaders["none"])))
