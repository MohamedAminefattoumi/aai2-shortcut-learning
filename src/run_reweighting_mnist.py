
import os
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
CHECKPOINT_DIR = "checkpoints"
SAVE_P = 0.90

def get_color_ids(train, p, mode="correlated", seed=0, root="data"):
    labels = datasets.MNIST(
        root=root,
        train=train,
        download=True
    ).targets
    generator = torch.Generator().manual_seed(seed)
    return choose_color_ids(labels, p, mode, generator)

@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = 0
    for i in range(0, len(x), 1000):
        pred = model(
            x[i:i + 1000].to(device)
        ).argmax(1).cpu()
        correct += (pred == y[i:i + 1000]).sum().item()
    return correct / len(x)

def run(upweight, seed, x_tr, y_tr, c_tr, tests, save_path=None):
    torch.manual_seed(seed)
    model = build_model("cnn", x_tr[:2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = batch_weights(y_tr, c_tr, upweight)
    loader = DataLoader(
        TensorDataset(x_tr, y_tr, weights),
        batch_size=128,
        shuffle=True
    )

    for epoch in range(EPOCHS):
        model.train()
        for x, y, w in loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = weighted_cross_entropy(logits, y, w)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved: {save_path}")

    results = {}
    for mode, (x_te, y_te) in tests.items():
        results[mode] = accuracy(model, x_te, y_te)
    return results

if __name__ == "__main__":
    tests = {
        mode: make_colored_mnist(
            train=False,
            mode=mode
        ).tensors
        for mode in ["correlated", "random", "inverted"]
    }

    for P in P_VALUES:
        print(f"\n========== P = {P} ==========")

        x_tr, y_tr = make_colored_mnist(
            train=True,
            p=P,
            mode="correlated"
        ).tensors

        c_tr = get_color_ids(
            train=True,
            p=P,
            seed=0
        )

        rare_ratio = (c_tr != y_tr).float().mean()
        print(f"Rare samples: {rare_ratio:.3f}")

        for name, upweight in [
            ("baseline", 1.0),
            ("reweighting", UPWEIGHT)
        ]:
            runs = []

            for seed in SEEDS:
                save_path = None

                if P == SAVE_P:
                    save_path = os.path.join(
                        CHECKPOINT_DIR,
                        f"{name}_p09_seed{seed}.pth"
                    )

                results = run(
                    upweight,
                    seed,
                    x_tr,
                    y_tr,
                    c_tr,
                    tests,
                    save_path=save_path
                )

                runs.append(results)

            mean = {
                mode: sum(r[mode] for r in runs) / len(runs)
                for mode in tests
            }

            print(
                name,
                {
                    mode: round(value, 3)
                    for mode, value in mean.items()
                }
            )
