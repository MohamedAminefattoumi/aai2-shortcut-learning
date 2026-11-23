# Shortcut learning on coloured MNIST

AAI2 project B1: Shortcut Learning and Spurious Correlations.

## What is this project about?

A neural network can get a very high accuracy for the wrong reason. If an easy
cue (a *shortcut*) predicts the label in the training data, the network may
learn that cue instead of the real object.

We show this with a small and controlled experiment:

1. We take MNIST and paint every digit with a colour. Each digit has its own
   colour (0 is red, 1 is green, 2 is blue, ...).
2. In the training set the colour matches the digit most of the time. How often
   it matches is the **correlation strength p** (for example p = 0.9 means about
   90 % of the digits have "their" colour).
3. We train a model and then test it on images where the colour no longer helps.
   If the accuracy drops, the model was looking at the colour and not at the shape.
4. We try three ways to fix this (mitigations) and use explainability methods
   (XAI) to see where the model is looking.

## The test sets

All test sets use the same 10 000 MNIST test digits. Only the colour changes.

| test set           | colour of the digit                                  | what it tells us                     |
|--------------------|------------------------------------------------------|--------------------------------------|
| `test_aligned`     | always the colour of its class (like in training)    | accuracy when the shortcut helps     |
| `test_random`      | a random colour                                      | accuracy when the colour is useless  |
| `test_conflicting` | always the colour of a *wrong* class (digit k gets the colour of k + 1) | accuracy when the shortcut lies |
| `test_clean`       | no colour, white digit                               | accuracy on the shape only           |

The **gap** is `acc aligned - acc conflicting`. A big gap means the model relies
on the colour.

## The mitigations

We compare a baseline (`none`) with three mitigations. The model, the training
settings and the test sets are always the same, only the mitigation changes.

- **augmentation** (data): every time a training digit is used it gets a new
  random colour, so the colour stops predicting the label.
- **rebalancing** (data): we add copies of training digits painted with a wrong
  colour, until the colour matches the label only 50 % of the time.
- **reweighting** (loss): the rare training images where the colour does not
  match the label get a 10 times bigger weight in the loss.

## Main results

Numbers are means over 3 seeds, SmallCNN, 5 epochs (full table in
`results/summary.md`).

| p    | mitigation   | acc aligned | acc conflicting | gap  |
|------|--------------|-------------|-----------------|------|
| 0.99 | none         | 99.9 %      | 58.0 %          | 41.9 |
| 0.99 | augmentation | 98.3 %      | 98.3 %          | 0.0  |
| 0.99 | rebalancing  | 99.2 %      | 98.3 %          | 1.0  |
| 0.99 | reweighting  | 99.4 %      | 80.8 %          | 18.6 |

With a strong shortcut (p = 0.99) the baseline loses about 40 points when the
colour lies. The two data mitigations remove the problem almost completely, and
reweighting reduces it to less than half.

## Folder structure

```
src/                  all the code (run it from the repository root, see below)
  shortcuts.py          colours the MNIST digits (make_colored_mnist)
  data.py               builds the training and test sets and their DataLoaders
  data_checks.py        checks the data and saves example images
  data_mitigations.py   augmentation and rebalancing
  mitigations.py        reweighting (weighted loss)
  models.py             MLP and SmallCNN
  train.py              training loop, seeds, checkpoints
  evaluate.py           accuracy, shortcut rate and confusion matrix on each test set
  run_experiments.py    the main experiment: every mitigation, 3 seeds -> results/results.csv
  run_architectures.py  MLP vs CNN, trained on clean vs coloured data
  report.py             makes the tables (results/summary.md) and figures from the CSV files
  xai.py, lime_xai.py   Grad-CAM, Integrated Gradients and LIME
  run_xai.py            XAI figures for the four models -> results/xai/
  run_all.py            runs everything above with one command
  fake_loaders.py       fake data we used to test the training code early on
  compare_*.py, run_reweighting_mnist.py, LIME.ipynb   earlier XAI and reweighting tests
results/              CSV files, summary.md, figures/ and xai/
checkpoints/          trained models used by the XAI figures and the demo
report/               the scientific report (Quarto) and the slides
demo/                 small Streamlit app to try the models (see demo/README.md)
```

## How to run it

Install the libraries (a virtual environment is a good idea):

```bash
pip install -r requirements.txt
```

All commands are run from the **repository root**, with `python -m src.<name>`
(this is needed so that the `from src. ...` imports work). MNIST is downloaded
automatically into `data/` the first time.

### Everything with one command

```bash
python -m src.run_all
```

This checks the data, trains all the models, writes the CSV files, the tables
and every figure in `results/`. On a normal CPU it takes a while (about an hour
or more), with a GPU it is much faster.

To only check that the code works, there is a quick version (small data,
1 epoch, 1 seed, a few minutes). It writes `*_quick` files that are not part of
the real results:

```bash
python -m src.run_all --quick
```

### Step by step

```bash
python -m src.data_checks        # check the data, save results/figures/data_examples.png
python -m src.run_experiments    # mitigations -> results/results.csv
python -m src.run_architectures  # MLP vs CNN -> results/architectures.csv
python -m src.report             # tables and figures from the CSV files
python -m src.run_xai            # XAI figures (needs the checkpoints from run_experiments)
```

Every script also has a `--quick` option, except `data_checks` and `run_xai`.

## Reproducibility

- All random seeds are fixed (seeds 0, 1 and 2) and written in the CSV files.
- The data are downloaded and coloured by code, there is no manual step.
- Every figure and table is made from the CSV files by `report.py`.

## Limitations

- The shortcut is artificial. Real shortcuts are less clean and harder to find.
- With the random colour rule the real correlation is a bit higher than p
  (p + (1 - p) / 10), because a random colour is right 1 time in 10.
- MNIST is small and simple, so the results may not transfer to bigger datasets.
- We only used 3 seeds, two small architectures and one type of shortcut (colour).
