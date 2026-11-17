"""Turn the experiment CSVs into summary tables and figures for the slides.

Reads (from results/):
    results.csv                    run_experiments.py: the mitigation grid (CNN)
    architectures.csv              run_architectures.py: MLP vs CNN, clean vs biased
    architectures_loss.csv         run_architectures.py: training loss per epoch
    confusion_<model>_<train>.csv  run_architectures.py: confusion matrices (conflicting)

Writes:
    results/summary.md             tables: mean ± std over the seeds, in %
    results/figures/*.png          figures for the slides

Every number is a mean over the seeds; error bars and "±" are the standard deviation.
Missing input files are skipped with a message, so the script also works halfway.

Usage, from the repository root:
    python -m src.report            # uses the full results
    python -m src.report --quick    # uses the *_quick.csv files (to check the code)
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")                     # write PNG files, no window needed
import matplotlib.pyplot as plt
import numpy as np

TESTS = ["acc_correlated", "acc_clean", "acc_random", "acc_inverted"]
TEST_NAMES = {"acc_correlated": "aligned", "acc_clean": "clean (no colour)",
              "acc_random": "random colour", "acc_inverted": "conflicting"}
MITIGATION_ORDER = ["none", "augmentation", "rebalancing", "reweighting"]

# Colour-blind-safe categorical palette, one fixed colour per test set (never per rank).
COLORS = {"acc_correlated": "#2a78d6", "acc_clean": "#eb6834",
          "acc_random": "#1baf7a", "acc_inverted": "#eda100"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FIG_DIR = "results/figures"


# ---------------------------------------------------------------- reading
def read_rows(path):
    """Read a CSV into a list of dicts with numbers converted. None if the file is missing."""
    if not os.path.exists(path):
        print(f"skip: {path} not found")
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass                       # text column (model, mitigation, ...)
    return rows


def group(rows, keys):
    """Group rows by the given columns: {(value, value): [row, row, ...]}."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def mean_std(rows, column):
    values = np.array([row[column] for row in rows])
    return values.mean(), values.std()


def cell(rows, column):
    """'97.2 ± 0.3' in percent; without '±' when there is only one seed."""
    m, s = mean_std(rows, column)
    return f"{100 * m:.1f} ± {100 * s:.1f}" if len(rows) > 1 else f"{100 * m:.1f}"


def markdown_table(header, body):
    widths = [max(len(str(r[i])) for r in [header] + body) for i in range(len(header))]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
                     + [line(r) for r in body])


# ---------------------------------------------------------------- tables
def architecture_table(rows):
    header = ["model", "trained on", "params"] + [f"acc {TEST_NAMES[t]}" for t in TESTS] \
             + ["shortcut rate", "gap", "seeds"]
    body = []
    for (model, train_set), runs in sorted(group(rows, ["model", "train_set"]).items(),
                                           key=lambda kv: (kv[0][0] != "mlp", kv[0][1] != "clean")):
        body.append([model.upper(), train_set, int(runs[0]["n_params"])]
                    + [cell(runs, t) for t in TESTS]
                    + [cell(runs, "shortcut_rate"), cell(runs, "gap"), len(runs)])
    return markdown_table(header, body)


def mitigation_table(rows):
    header = ["p", "mitigation"] + [f"acc {TEST_NAMES[t]}" for t in TESTS] \
             + ["shortcut rate", "gap", "seeds"]
    body = []
    order = lambda kv: (kv[0][0], MITIGATION_ORDER.index(kv[0][1]))
    for (p, mitigation), runs in sorted(group(rows, ["p", "mitigation"]).items(), key=order):
        body.append([p, mitigation] + [cell(runs, t) for t in TESTS]
                    + [cell(runs, "shortcut_rate"), cell(runs, "gap"), len(runs)])
    return markdown_table(header, body)


# ---------------------------------------------------------------- figures
def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.axhline(0.1, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))   # chance level


def grouped_bars(ax, groups, labels):
    """groups: list of row-lists, one per x position. One bar per test set, mean ± std."""
    width = 0.8 / len(TESTS)
    x = np.arange(len(groups))
    for i, test in enumerate(TESTS):
        stats = [mean_std(runs, test) for runs in groups]
        means, stds = [m for m, _ in stats], [s for _, s in stats]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, means, width, color=COLORS[test], edgecolor=SURFACE, linewidth=1.5,
               label=TEST_NAMES[test], yerr=stds if len(groups[0]) > 1 else None,
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        has_bars = len(groups[0]) > 1
        for xp, m, s in zip(pos, means, stds):   # direct value labels (in ink), above the error bar
            ax.text(xp, m + (s if has_bars else 0) + 0.015, f"{100 * m:.0f}",
                    ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, linewidth=1, linestyle=(0, (3, 3))))
    labels.append("chance (10%)")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.06))


def figure_architectures(rows, suffix):
    groups = group(rows, ["model", "train_set"])
    keys = [("mlp", "clean"), ("mlp", "biased"), ("cnn", "clean"), ("cnn", "biased")]
    keys = [k for k in keys if k in groups]
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    grouped_bars(ax, [groups[k] for k in keys], [f"{m.upper()}\ntrained on {t}" for m, t in keys])
    style(ax, "Test accuracy by model and training data (colour shortcut, p = 0.9)")
    legend(fig, ax)
    save(fig, f"architectures{suffix}.png")


def figure_mitigations(rows, suffix):
    ps = sorted({row["p"] for row in rows if row["mitigation"] != "none" or row["p"] != 0.5})
    ps = [p for p in ps if any(r["p"] == p and r["mitigation"] != "none" for r in rows)]
    if not ps:
        return
    groups = group(rows, ["p", "mitigation"])
    fig, axes = plt.subplots(1, len(ps), figsize=(6 * len(ps), 4.2), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, ps):
        names = [m for m in MITIGATION_ORDER if (p, m) in groups]
        grouped_bars(ax, [groups[(p, m)] for m in names], names)
        style(ax, f"p = {p}: colour matches the digit in {100 * (p + (1 - p) / 10):.0f}% of training images")
    legend(fig, axes[0])
    save(fig, f"mitigations{suffix}.png")


def figure_shortcut_strength(rows, suffix):
    groups = group([r for r in rows if r["mitigation"] == "none"], ["p"])
    if len(groups) < 2:
        return
    ps = sorted(p for (p,) in groups)
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    grouped_bars(ax, [groups[(p,)] for p in ps], [f"p = {p}" for p in ps])
    style(ax, "Baseline CNN (no mitigation): a stronger shortcut, a bigger collapse")
    legend(fig, ax)
    save(fig, f"shortcut_strength{suffix}.png")


def figure_losses(rows, suffix):
    line_styles = {("mlp", "clean"): ("#2a78d6", "-"), ("mlp", "biased"): ("#2a78d6", "--"),
                   ("cnn", "clean"): ("#eb6834", "-"), ("cnn", "biased"): ("#eb6834", "--")}
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=SURFACE)
    for (model, train_set), runs in group(rows, ["model", "train_set"]).items():
        epochs = sorted({r["epoch"] for r in runs})
        means = [np.mean([r["loss"] for r in runs if r["epoch"] == e]) for e in epochs]
        color, ls = line_styles[(model, train_set)]
        ax.plot(epochs, means, color=color, linestyle=ls, linewidth=2, marker="o", markersize=5,
                label=f"{model.upper()}, {train_set}")
    ax.set_facecolor(SURFACE)
    ax.set_title("Training loss per epoch (mean over seeds)", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("epoch", color=INK_2)
    ax.set_ylabel("cross-entropy loss", color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, f"loss_curves{suffix}.png")


def figure_confusion(suffix):
    pairs = [("cnn", "clean"), ("cnn", "biased"), ("mlp", "clean"), ("mlp", "biased")]
    found = [(m, t) for m, t in pairs if os.path.exists(f"results/confusion_{m}_{t}.csv")]
    if not found:
        print("skip: no results/confusion_*.csv found")
        return
    fig, axes = plt.subplots(1, len(found), figsize=(3.6 * len(found), 3.8), facecolor=SURFACE)
    for ax, (m, t) in zip(np.atleast_1d(axes), found):
        cm = np.loadtxt(f"results/confusion_{m}_{t}.csv", delimiter=",")
        cm = cm / cm.sum(axis=1, keepdims=True)            # each row sums to 1
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{m.upper()}, trained on {t}", color=INK, fontsize=10)
        ax.set_xlabel("predicted digit", color=INK_2, fontsize=9)
        ax.set_ylabel("true digit", color=INK_2, fontsize=9)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.tick_params(labelsize=7, colors=INK_2, length=0)
    fig.suptitle("Conflicting test set: the diagonal = follows the digit, "
                 "one to the right = follows the colour", color=INK, fontsize=10)
    save(fig, f"confusion_conflicting{suffix}.png")


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="use the *_quick.csv files")
    args = parser.parse_args()
    suffix = "_quick" if args.quick else ""

    sections = ["# Results summary",
                "Mean ± standard deviation over the seeds, in %. "
                "gap = acc aligned - acc conflicting (large = the model relies on the colour). "
                "shortcut rate = how often the prediction is the colour's class on the conflicting set."]

    arch = read_rows(f"results/architectures{suffix}.csv")
    if arch:
        sections += ["## MLP vs CNN, clean vs biased training (p = 0.9)", architecture_table(arch)]
        figure_architectures(arch, suffix)
    losses = read_rows(f"results/architectures_loss{suffix}.csv")
    if losses:
        figure_losses(losses, suffix)
    if not args.quick:
        figure_confusion(suffix)

    grid = read_rows(f"results/results{suffix}.csv")
    if grid:
        sections += ["## Mitigations (SmallCNN)", mitigation_table(grid)]
        figure_mitigations(grid, suffix)
        figure_shortcut_strength(grid, suffix)

    path = f"results/summary{suffix}.md"
    with open(path, "w") as f:
        f.write("\n\n".join(sections) + "\n")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
