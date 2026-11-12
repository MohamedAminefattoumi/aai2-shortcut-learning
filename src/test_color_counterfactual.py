
import torch
from torch.utils.data import DataLoader, TensorDataset

from shortcuts import make_colored_mnist, colorize
from models import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [0, 1, 2]

# --------------------------------------------------
# 1. Load test images and labels
# --------------------------------------------------

x_colored, labels = make_colored_mnist(
    train=False,
    mode="correlated"
).tensors

grayscale = x_colored.max(dim=1).values

# --------------------------------------------------
# 2. Create counterfactual color configurations
# --------------------------------------------------

generator = torch.Generator().manual_seed(42)

color_configs = {
    "Aligned": labels,
    "Inverted": (labels + 1) % 10,
    "Random": torch.randint(
        0, 10,
        (len(labels),),
        generator=generator
    )
}

# --------------------------------------------------
# 3. Evaluation function
# --------------------------------------------------

def evaluate(model, images, labels):

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=256,
        shuffle=False
    )

    correct = 0
    confidence_sum = 0.0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()

            confidence_sum += probabilities.gather(
                1, y.unsqueeze(1)
            ).sum().item()

            total += len(y)

    return correct / total, confidence_sum / total


# --------------------------------------------------
# 4. Evaluate all seeds
# --------------------------------------------------

results = {
    mode: {
        method: {"accuracy": [], "confidence": []}
        for method in ["Baseline", "Reweighting"]
    }
    for mode in color_configs
}

for seed in SEEDS:

    print(f"\n========== SEED {seed} ==========")

    baseline = build_model("cnn", x_colored[:2]).to(device)
    reweighting = build_model("cnn", x_colored[:2]).to(device)

    baseline.load_state_dict(torch.load(
        f"checkpoints/baseline_p09_seed{seed}.pth",
        map_location=device,
        weights_only=True
    ))

    reweighting.load_state_dict(torch.load(
        f"checkpoints/reweighting_p09_seed{seed}.pth",
        map_location=device,
        weights_only=True
    ))

    baseline.eval()
    reweighting.eval()

    models = {
        "Baseline": baseline,
        "Reweighting": reweighting
    }

    for color_mode, color_ids in color_configs.items():

        images = colorize(grayscale, color_ids)

        for method, model in models.items():

            acc, confidence = evaluate(
                model, images, labels
            )

            results[color_mode][method]["accuracy"].append(acc)
            results[color_mode][method]["confidence"].append(confidence)

            print(
                f"{color_mode} | {method}: "
                f"Accuracy={acc:.4f}, "
                f"True-class probability={confidence:.4f}"
            )


# --------------------------------------------------
# 5. Mean and standard deviation
# --------------------------------------------------

print("\n========== FINAL RESULTS: MEAN ± STD ==========")

for mode in color_configs:

    print(f"\n{mode}")

    for method in ["Baseline", "Reweighting"]:

        acc = torch.tensor(
            results[mode][method]["accuracy"]
        )

        confidence = torch.tensor(
            results[mode][method]["confidence"]
        )

        print(
            f"{method:12s} | "
            f"Accuracy: {acc.mean():.4f} ± {acc.std(unbiased=False):.4f} | "
            f"Confidence: {confidence.mean():.4f} ± "
            f"{confidence.std(unbiased=False):.4f}"
        )
