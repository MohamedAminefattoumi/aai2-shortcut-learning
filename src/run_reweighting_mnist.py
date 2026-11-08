import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from shortcuts import make_colored_mnist, choose_color_ids
from models import build_model
from mitigations import batch_weights, weighted_cross_entropy

device = "cuda" if torch.cuda.is_available() else "cpu"
P_VALUES = [0.90, 0.95, 0.98]

SEEDS = [0, 1, 2]
EPOCHS = 3

UPWEIGHT = 10.0

def get_color_ids(train, p, mode="correlated", seed=0, root="data"):
    # Rebuild the colour ids exactly as make_colored_mnist does (same seed, same labels)
    labels = datasets.MNIST(root=root, train=train, download=True).targets
    generator = torch.Generator().manual_seed(seed)
    return choose_color_ids(labels, p, mode, generator)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = 0
    for i in range(0, len(x), 1000):
        pred = model(x[i:i + 1000].to(device)).argmax(1).cpu()
        correct += (pred == y[i:i + 1000]).sum().item()
    return correct / len(x)


def run(upweight, seed, x_tr, y_tr, c_tr, tests):
    torch.manual_seed(seed)
    model = build_model("cnn", x_tr[:2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w_tr = batch_weights(y_tr, c_tr, upweight)       # one weight per training image
    loader = DataLoader(TensorDataset(x_tr, y_tr, w_tr), batch_size=128, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            opt.zero_grad()
            weighted_cross_entropy(model(x), y, w).backward()
            opt.step()
    return {m: accuracy(model, x_te, y_te) for m, (x_te, y_te) in tests.items()}


if __name__ == "__main__":
    tests = {m: make_colored_mnist(train=False, mode=m).tensors
            for m in ["correlated", "random", "inverted"]}

    for P in P_VALUES:
        print(f"\n========== P = {P} ==========")
        x_tr, y_tr = make_colored_mnist(train=True, p=P, mode="correlated").tensors
        c_tr = get_color_ids(train=True, p=P)
        print(f"rare samples (colour != label): {(c_tr != y_tr).float().mean():.3f}")

        for name, upweight in [("none", 1.0), ("reweighting", 10.0)]:
            runs = [run(upweight, s, x_tr, y_tr, c_tr, tests) for s in SEEDS]
            mean = {m: sum(r[m] for r in runs) / len(runs) for m in tests}
            print(name, {m: round(v, 3) for m, v in mean.items()})