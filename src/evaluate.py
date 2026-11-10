"""Evaluation: accuracy (follows the digit) and shortcut rate (follows the patch)."""

import json
import os

import numpy as np
import torch
from torch import nn

TEST_SETS = ["test_aligned", "test_clean", "test_conflicting"]


@torch.no_grad()                                     #we only predict, nevel call backward(), so tracking gradients is not needed
def evaluate(model: nn.Module, loader) -> dict:      #returns a dictionaty
    model.eval()                                     # evaluation mode (matters for layers like dropout)
    correct = follows = total = 0
    for images, labels, shortcut_labels in loader:
        preds = model(images).argmax(dim=1)                          #model(images) -> (B, n_classes) scores, argmax(dim=1) takes, for each image (each row), the index of the highest score
        correct += (preds == labels).sum().item()                    #compares element by element, counts the Trues, tunrs it into a number, adds to the total correct predictions
        follows += (preds == shortcut_labels).sum().item()           #states how many predictions follow the shortcut (the patch), adds to the total
        total += len(labels)
    return {"accuracy": correct / total, "shortcut_rate": follows / total}

def evaluate_all(runs: dict, loaders: dict) -> dict:
    """runs: {("cnn", "train_biased"): model, ...} -> {"cnn/train_biased/test_clean": {...}, ...}"""
    results = {}
    for (model_name, train_name), model in runs.items():
        for test_name in TEST_SETS:
            key = f"{model_name}/{train_name}/{test_name}"
            results[key] = evaluate(model, loaders[test_name])
    return results


def save_metrics(results: dict, path: str = "results/metrics.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

#presents the results in a table format, one row per (model, training set), as a markdown table.

def results_table(results: dict) -> str:
    """One row per (model, training set), as a markdown table."""
    rows = sorted({tuple(key.split("/")[:2]) for key in results})
    lines = [
        "| model | trained on | acc aligned | acc clean | acc conflicting | shortcut rate (conflicting) |",
        "|---|---|---|---|---|---|",
    ]
    for model_name, train_name in rows:
        r = lambda test: results[f"{model_name}/{train_name}/{test}"]
        lines.append(
            f"| {model_name} | {train_name.replace('train_', '')} "
            f"| {r('test_aligned')['accuracy']:.1%} "
            f"| {r('test_clean')['accuracy']:.1%} "
            f"| {r('test_conflicting')['accuracy']:.1%} "
            f"| {r('test_conflicting')['shortcut_rate']:.1%} |"
        )
    return "\n".join(lines)


@torch.no_grad()
def confusion_matrix(model: nn.Module, loader) -> np.ndarray:
    """Rows = true digit, columns = prediction. On test_conflicting, a model that
    follows the patch fills the diagonal shifted one column right (pred = label + 1)."""
    model.eval()
    cm = np.zeros((10, 10), dtype=int)
    for images, labels, _ in loader:
        preds = model(images).argmax(dim=1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            cm[t, p] += 1
    return cm

if __name__ == "__main__":
    # Self-test with the two fake models saved by "python -m src.train"
    from src.fake_loaders import get_fake_loaders
    from src.models import SmallCNN
    from src.train import load_checkpoint

    loaders = get_fake_loaders()
    runs = {
        ("cnn", train_name): load_checkpoint(SmallCNN(), f"checkpoints/test_cnn_{train_name}.pt")
        for train_name in ["train_clean", "train_biased"]
    }
    results = evaluate_all(runs, loaders)
    print(results_table(results))
    print("\nconfusion matrix, biased CNN on test_conflicting:")
    print(confusion_matrix(runs[("cnn", "train_biased")], loaders["test_conflicting"]))




