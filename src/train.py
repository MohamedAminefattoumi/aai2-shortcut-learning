import random
import numpy as np
import torch
import torch.nn as nn
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(model, loader, epochs=5, lr=1e-3): #learning rate = 0.001 because works without tuning, as in Asignment 2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()          # applies softmax itself
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for images, labels, _ in loader:       # shortcut labels are not used for training
            scores = model(images)             # forward pass
            loss = criterion(scores, labels)   # loss
            optimizer.zero_grad()              # reset old gradients
            loss.backward()                    # backpropagation
            optimizer.step()                   # weight update
            total_loss += loss.item() * len(labels)
            n += len(labels)
        history.append(total_loss / n)
        print(f"epoch {epoch + 1}: loss {history[-1]:.4f}")
    return history


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)    #so we can save the trained model, not retrain it
    torch.save(model.state_dict(), path)                        


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    from src.fake_loaders import get_fake_loaders
    from src.models import SmallCNN

    loaders = get_fake_loaders()
    for train_name in ["train_clean", "train_biased"]:
        print(f"--- {train_name}")
        set_seed(42)                       # same start for both runs
        model = SmallCNN()
        train(model, loaders[train_name], epochs=3)
        save_checkpoint(model, f"checkpoints/test_cnn_{train_name}.pt")



