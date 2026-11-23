"""Visual and numeric checks of the coloured MNIST data (guide, step A6).

Before training anything we want to be sure that the data look the way we think:
    - in the training set and in test_aligned most digits have the colour of their class;
    - in test_random the colour is unrelated to the class (about 1 time in 10, by chance);
    - in test_conflicting no digit has the colour of its own class;
    - in test_clean the digits are white, so there is no colour at all;
    - every image has shape (3, 28, 28) and pixel values in [0, 1].

The script prints these numbers, stops with an error if one of them is wrong,
and saves a grid of example images for the slides.

Usage, from the repository root:
    python -m src.data_checks
"""

import os

import matplotlib
matplotlib.use("Agg")                    # save figures without opening a window
import matplotlib.pyplot as plt

from src.data import get_datasets

P = 0.9                                  # same shortcut strength as the default training set
FIGURE = "results/figures/data_examples.png"
N_EXAMPLES = 10                          # images per row in the figure

# The datasets we look at, in the order of the figure rows.
SETS = ("train_biased", "test_aligned", "test_random", "test_conflicting", "test_clean")


def colour_matches_label(dataset):
    """Fraction of images whose colour is the colour of their class."""
    _, labels, colour_ids = dataset.tensors
    return (colour_ids == labels).float().mean().item()


def check(condition, message):
    """Print the result of one check and stop the script if it fails."""
    print(f"  [{'ok' if condition else 'FAILED'}] {message}")
    if not condition:
        raise SystemExit(f"data check failed: {message}")


def run_checks(datasets):
    """Check the shape, the value range and the colour rule of every dataset."""
    for name in SETS:
        images, labels, _ = datasets[name].tensors
        match = colour_matches_label(datasets[name])
        print(f"{name}: {len(labels)} images, colour == label for {match:.3f}")

        check(images.dim() == 4 and tuple(images.shape[1:]) == (3, 28, 28),
              f"shape is (N, 3, 28, 28), got {tuple(images.shape)}")
        check(images.min() >= 0 and images.max() <= 1, "pixel values are in [0, 1]")

        # Expected share of images whose colour points to the true class.
        if name == "train_biased":
            # The random colour is right 1 time in 10, so the real value is p + (1 - p) / 10.
            expected = P + (1 - P) / 10
            check(abs(match - expected) < 0.01, f"about {expected:.2f} (p = {P})")
        elif name == "test_aligned":
            check(match == 1.0, "every digit has the colour of its class")
        elif name == "test_random":
            check(abs(match - 0.1) < 0.02, "about 0.10 (colour chosen at random)")
        elif name == "test_conflicting":
            check(match == 0.0, "no digit has the colour of its class")
        elif name == "test_clean":
            # Clean digits have no colour: the shortcut label is -1 and the 3 channels are equal.
            check((images[:, 0] == images[:, 1]).all().item()
                  and (images[:, 1] == images[:, 2]).all().item(),
                  "white digits: the 3 colour channels are identical")


def save_examples(datasets, path=FIGURE):
    """One row per dataset, the same first N_EXAMPLES test digits in every test row."""
    fig, axes = plt.subplots(len(SETS), N_EXAMPLES, figsize=(N_EXAMPLES * 1.1, len(SETS) * 1.3))
    for row, name in enumerate(SETS):
        images, labels, _ = datasets[name].tensors
        for col in range(N_EXAMPLES):
            ax = axes[row, col]
            ax.imshow(images[col].permute(1, 2, 0).numpy())   # (3, H, W) -> (H, W, 3) for matplotlib
            ax.set_title(f"label {labels[col].item()}", fontsize=7)
            ax.axis("off")
        # Name of the dataset on the left of each row.
        axes[row, 0].text(-0.3, 0.5, name, transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    datasets = get_datasets(p=P)
    run_checks(datasets)
    save_examples(datasets)
    print("all data checks passed")


if __name__ == "__main__":
    main()
