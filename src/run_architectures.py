"""Architecture experiment (guide, sections 3-6): MLP vs CNN, trained on clean vs biased data.

Research question: does a network learn the colour shortcut when it is available,
and does the type of network change how much it relies on it?

    2 models (MLP, SmallCNN) x 2 training sets (train_clean, train_biased) x 3 seeds
    = 12 trainings, each evaluated on the 4 test sets.

Everything except the model and the training set is identical to run_experiments.py
(epochs, batch size, learning rate and seeds are imported from it), so the two
experiments can be compared directly.

Outputs (in results/):
    architectures.csv              one row per training (accuracies, shortcut rate, gap)
    architectures_loss.csv         average training loss per epoch (for the loss curves)
    confusion_<model>_<train>.csv  confusion matrix on test_conflicting (seed 0 only)

Usage, from the repository root:
    python -m src.run_architectures            # full run (60 000 images, 5 epochs, 3 seeds)
    python -m src.run_architectures --quick    # small check (5 000 images, 1 epoch, 1 seed)
"""

import argparse
import csv
import os
import time

import numpy as np

from src.data import get_loaders
from src.evaluate import confusion_matrix, evaluate
from src.models import MLP, SmallCNN, count_parameters
from src.run_experiments import BATCH_SIZE, EPOCHS, LR, SEEDS
from src.train import set_seed, train

P = 0.9                 # shortcut strength of train_biased (the default of data.get_loaders)
MLP_HIDDEN = 44         # 103 982 parameters, close to the CNN's 106 154
TRAIN_SETS = ("train_clean", "train_biased")

COLUMNS = ["model", "train_set", "p", "seed", "epochs", "n_train", "n_params",
           "acc_correlated", "acc_random", "acc_inverted", "acc_clean",
           "shortcut_rate", "gap"]


def build(model_name, images):
    """Build a model that fits the image shape. The MLP is shrunk so both models have
    about the same number of parameters: then a difference comes from the architecture,
    not from the size. (With hidden=128 and colour images the MLP would have 302 474.)"""
    channels, height, width = images.shape[1:]
    if model_name == "mlp":
        return MLP(channels, height, width, hidden=MLP_HIDDEN)
    return SmallCNN(channels, height, width)


def run_one(model_name, train_set, seed, epochs, n_train, save_confusion):
    """Train one model and return (row for the CSV, loss per epoch)."""
    loaders = get_loaders(batch_size=BATCH_SIZE, p=P, seed=seed, n_train=n_train)

    set_seed(seed)                                   # same initial weights for both training sets
    images, _, _ = next(iter(loaders[train_set]))
    model = build(model_name, images)
    history = train(model, loaders[train_set], epochs=epochs, lr=LR)

    aligned = evaluate(model, loaders["test_aligned"])
    conflicting = evaluate(model, loaders["test_conflicting"])
    row = {
        "model": model_name,
        "train_set": train_set.replace("train_", ""),
        "p": P,
        "seed": seed,
        "epochs": epochs,
        "n_train": len(loaders[train_set].dataset),
        "n_params": count_parameters(model),
        # Column names as in run_experiments.py: aligned -> correlated, conflicting -> inverted.
        "acc_correlated": aligned["accuracy"],
        "acc_random": evaluate(model, loaders["test_random"])["accuracy"],
        "acc_inverted": conflicting["accuracy"],
        "acc_clean": evaluate(model, loaders["test_clean"])["accuracy"],
        "shortcut_rate": conflicting["shortcut_rate"],
        "gap": aligned["accuracy"] - conflicting["accuracy"],
    }

    if save_confusion:
        cm = confusion_matrix(model, loaders["test_conflicting"])
        path = f"results/confusion_{model_name}_{row['train_set']}.csv"
        np.savetxt(path, cm, fmt="%d", delimiter=",")

    row = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
    return row, history


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="5000 training images, 1 epoch, 1 seed (only to check the code)")
    args = parser.parse_args()

    if args.quick:
        epochs, n_train, seeds, suffix = 1, 5000, SEEDS[:1], "_quick"
    else:
        epochs, n_train, seeds, suffix = EPOCHS, None, SEEDS, ""

    os.makedirs("results", exist_ok=True)
    runs = [(m, t, s) for m in ("mlp", "cnn") for t in TRAIN_SETS for s in seeds]

    with open(f"results/architectures{suffix}.csv", "w", newline="") as f, \
         open(f"results/architectures_loss{suffix}.csv", "w", newline="") as f_loss:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        loss_writer = csv.writer(f_loss)
        loss_writer.writerow(["model", "train_set", "seed", "epoch", "loss"])

        for i, (model_name, train_set, seed) in enumerate(runs, start=1):
            print(f"=== run {i}/{len(runs)}: {model_name} on {train_set}, seed {seed}")
            start = time.time()
            row, history = run_one(model_name, train_set, seed, epochs, n_train,
                                   save_confusion=(seed == seeds[0] and not args.quick))
            writer.writerow(row)
            for epoch, loss in enumerate(history, start=1):
                loss_writer.writerow([model_name, row["train_set"], seed, epoch, round(loss, 4)])
            f.flush()
            f_loss.flush()                            # a crash does not lose finished runs
            print(f"acc correlated {row['acc_correlated']:.3f} | random {row['acc_random']:.3f} | "
                  f"inverted {row['acc_inverted']:.3f} | clean {row['acc_clean']:.3f} | "
                  f"{row['n_params']} params | {time.time() - start:.0f} s")

    print(f"saved results/architectures{suffix}.csv and results/architectures_loss{suffix}.csv")


if __name__ == "__main__":
    main()
