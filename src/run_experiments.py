"""Run the full mitigation grid and write one row per run to results/results.csv.

Every run uses the same model, optimizer and hyperparameters. Only the
mitigation (and the seed) changes, so any difference in the test accuracies
can be attributed to the mitigation.

Usage, from the repository root:
    python -m src.run_experiments            # full grid (60 000 images, 5 epochs, 3 seeds)
    python -m src.run_experiments --quick    # small and fast check (results_quick.csv)
"""

import argparse
import csv
import os
import time
from functools import partial

from src.data_mitigations import get_mitigated_loaders
from src.evaluate import evaluate
from src.mitigations import reweighting_loss
from src.models import build_model
from src.train import set_seed, train

# Fixed settings, identical for every run.
MODEL = "cnn"           # SmallCNN from models.py
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
P_TARGET = 0.5          # correlation reached by rebalancing
UPWEIGHT = 10.0         # weight of the rare samples in reweighting
SEEDS = (0, 1, 2)

# (p, mitigation) pairs to run: every mitigation with a strong (0.9) and a very
# strong (0.99) shortcut. p = 0.5 is the reference without a useful shortcut:
# there is nothing to mitigate, so only the baseline is run.
MITIGATIONS = ("none", "augmentation", "rebalancing", "reweighting")
GRID = [(p, mitigation) for p in (0.9, 0.99) for mitigation in MITIGATIONS] + [(0.5, "none")]

# Column names follow the guide (section 7.2), plus three extra columns.
# Our test sets map to the guide's names: aligned -> correlated, conflicting -> inverted.
COLUMNS = ["dataset", "shortcut", "p", "model", "mitigation", "seed", "epochs", "n_train",
           "acc_correlated", "acc_random", "acc_inverted", "acc_clean",
           "shortcut_rate", "gap"]


def run_one(p, mitigation, seed, epochs, n_train):
    """Train one model and return its row for results.csv."""
    # Reweighting changes the loss, not the data: it trains on the baseline data.
    data_mitigation = "none" if mitigation == "reweighting" else mitigation
    loaders = get_mitigated_loaders(data_mitigation, batch_size=BATCH_SIZE, p=p, seed=seed,
                                    n_train=n_train, p_target=P_TARGET)
    loss_fn = partial(reweighting_loss, upweight=UPWEIGHT) if mitigation == "reweighting" else None

    # Same seed before building the model: same initial weights for every mitigation.
    set_seed(seed)
    images, _, _ = next(iter(loaders["train_biased"]))
    model = build_model(MODEL, images)
    train(model, loaders["train_biased"], epochs=epochs, lr=LR, loss_fn=loss_fn)

    aligned = evaluate(model, loaders["test_aligned"])
    conflicting = evaluate(model, loaders["test_conflicting"])
    row = {
        "dataset": "mnist",
        "shortcut": "color",
        "p": p,
        "model": "smallcnn",
        "mitigation": mitigation,
        "seed": seed,
        "epochs": epochs,
        "n_train": len(loaders["train_biased"].dataset),
        "acc_correlated": aligned["accuracy"],
        "acc_random": evaluate(model, loaders["test_random"])["accuracy"],
        "acc_inverted": conflicting["accuracy"],
        "acc_clean": evaluate(model, loaders["test_clean"])["accuracy"],
        # How often the model predicts the class of the colour when the colour lies.
        "shortcut_rate": conflicting["shortcut_rate"],
        # Shortcut gap (guide, section 4): a large gap means the model relies on the colour.
        "gap": aligned["accuracy"] - conflicting["accuracy"],
    }
    # 4 decimals are enough (the test sets have 10 000 images) and keep the CSV readable.
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="5000 training images, 1 epoch, 1 seed (only to check the code)")
    args = parser.parse_args()

    if args.quick:
        epochs, n_train, seeds, path = 1, 5000, SEEDS[:1], "results/results_quick.csv"
    else:
        epochs, n_train, seeds, path = EPOCHS, None, SEEDS, "results/results.csv"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    runs = [(p, mitigation, seed) for p, mitigation in GRID for seed in seeds]

    # "w" starts a new file, so the CSV always matches the current code.
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i, (p, mitigation, seed) in enumerate(runs, start=1):
            print(f"=== run {i}/{len(runs)}: p={p} mitigation={mitigation} seed={seed}")
            start = time.time()
            row = run_one(p, mitigation, seed, epochs, n_train)
            # Write each row as soon as it is ready, so a crash does not lose finished runs.
            writer.writerow(row)
            f.flush()
            print(f"acc correlated {row['acc_correlated']:.3f} | random {row['acc_random']:.3f} | "
                  f"inverted {row['acc_inverted']:.3f} | clean {row['acc_clean']:.3f} | "
                  f"{time.time() - start:.0f} s")

    print(f"saved {path}")


if __name__ == "__main__":
    main()
