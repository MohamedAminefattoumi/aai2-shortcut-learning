"""Build datasets with an injected shortcut (Coder A, data side).

The shortcut is the colour of an MNIST digit. We control how strongly the colour
predicts the label, then remove or invert that link at test time. Any accuracy
drop can therefore be attributed to the colour.
"""

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets

# The three colouring rules. Training uses "correlated"; all three are used for testing.
MODES = ("correlated", "random", "inverted")

# One RGB colour per class: row k is the colour of digit k.
# Values are in [0, 1] because the images are in [0, 1].
COLORS = torch.tensor(
    [
        [1.0, 0.0, 0.0],  # 0: red
        [0.0, 1.0, 0.0],  # 1: green
        [0.0, 0.0, 1.0],  # 2: blue
        [1.0, 1.0, 0.0],  # 3: yellow
        [0.0, 1.0, 1.0],  # 4: cyan
        [1.0, 0.0, 1.0],  # 5: magenta
        [1.0, 0.5, 0.0],  # 6: orange
        [0.5, 0.0, 0.5],  # 7: purple
        [0.0, 0.5, 0.0],  # 8: dark green
        [0.5, 0.5, 0.5],  # 9: grey
    ],
    dtype=torch.float32,
)


def check_arguments(p, mode):
    """Raise a clear error if p or mode is not valid."""
    # This form also rejects NaN, because every comparison with NaN is False.
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def choose_color_ids(labels, p, mode, generator):
    """Choose one colour id (0-9) per image and return them as a LongTensor.

    p is only used in "correlated" mode.
    """
    check_arguments(p, mode)
    n = len(labels)

    # Inverted: digit 3 gets colour 4, and digit 9 wraps around to colour 0.
    # The colour is never the true class, and no randomness is needed.
    if mode == "inverted":
        return (labels + 1) % 10

    # One uniformly random colour per image, drawn from the seeded generator.
    random_ids = torch.randint(0, 10, (n,), generator=generator)

    # Random: the colour carries no information about the label.
    if mode == "random":
        return random_ids

    # Correlated: keep is True for about a fraction p of the images.
    # torch.rand gives values in [0, 1), so p=0 keeps none and p=1 keeps all.
    # Passing the generator here too is what makes this choice reproducible.
    keep = torch.rand(n, generator=generator) < p

    # Where keep is True use the class colour, otherwise the random colour.
    # A random colour still equals the label 1 time in 10, so the real
    # correlation is p + (1 - p) / 10, slightly higher than p.
    return torch.where(keep, labels, random_ids)


def colorize(images, color_ids):
    """Paint grayscale digits: (N, 28, 28) in [0, 1] becomes (N, 3, 28, 28)."""
    # Pick one RGB triple per image: (N, 3). Reshape it to (N, 3, 1, 1) so that
    # it can broadcast over the height and width dimensions.
    colors = COLORS[color_ids].view(-1, 3, 1, 1)

    # (N, 1, 28, 28) * (N, 3, 1, 1) broadcasts to (N, 3, 28, 28).
    # A white digit pixel (1) becomes the colour; black background (0) stays black.
    return images.unsqueeze(1) * colors


def make_colored_mnist(train, p=0.9, mode="correlated", root="data", seed=0):
    """Return a TensorDataset (x, y) of coloured MNIST digits.

    x: float32, shape (N, 3, 28, 28), values in [0, 1].
    y: int64, shape (N,), values 0-9.
    The same arguments always give the same data.
    """
    # Check before loading, so a typo fails at once and not after a download.
    check_arguments(p, mode)

    # A private seeded generator: the colours depend only on seed, not on
    # any other random numbers drawn elsewhere in the program.
    generator = torch.Generator().manual_seed(seed)

    # Load the MNIST split, downloading it into root the first time.
    mnist = datasets.MNIST(root=root, train=train, download=True)

    # mnist.data holds uint8 pixels in 0-255: convert to float32 in [0, 1].
    images = mnist.data.float() / 255.0
    # mnist.targets already holds the labels as int64.
    labels = mnist.targets

    color_ids = choose_color_ids(labels, p, mode, generator)
    x = colorize(images, color_ids)
    return TensorDataset(x, labels)


if __name__ == "__main__":
    # Build the default training set: p=0.9, correlated, seed=0.
    p = 0.9
    x, y = make_colored_mnist(train=True, p=p).tensors

    print(f"x: shape {tuple(x.shape)}, dtype {x.dtype}")
    print(f"x: min {x.min().item():.3f}, max {x.max().item():.3f}")
    print(f"y: shape {tuple(y.shape)}, dtype {y.dtype}")
    print(f"y: min {y.min().item()}, max {y.max().item()}")

    # The dataset does not store the colour ids, so read them back from the images.
    # The brightest pixel per channel is about the image's colour: (N, 3).
    # The nearest row of COLORS then gives the colour id.
    brightest = x.amax(dim=(2, 3))
    color_ids = torch.cdist(brightest, COLORS).argmin(dim=1)
    match = (color_ids == y).float().mean().item()
    print(f"colour matches label: {match:.4f} (expected about {p + (1 - p) / 10:.4f})")

    # Build the same dataset again: every value must be identical.
    x2, y2 = make_colored_mnist(train=True, p=p).tensors
    same = torch.equal(x, x2) and torch.equal(y, y2)
    print(f"same seed gives identical data: {same}")
