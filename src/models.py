import torch
from torch import nn


        #MLP module: flatten the imagen, Linear, ReLU, Linear, output raw scores (no softmax), fast to train
class MLP(nn.Module):
    def __init__(
            self, 
            in_channels: int = 1,   #number of input channels, 1 for grayscale images
            height: int = 28,          #height of the input image, 28 for MNIST
            width: int = 28,           #width of the input image, 28 for MNIST
            hidden: int = 128,
            n_classes: int = 10
            ):     
        
        super().__init__()                      #calls the constructor
        self.net = nn.Sequential(
            nn.Flatten(),                       #flattens the image to a vector of size 28*28=784
            nn.Linear(in_channels * height * width, hidden),  #linear layer from input to hidden layer
            nn.ReLU(),                          #activation function
            nn.Linear(hidden, n_classes)        #linear layer from hidden to output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   #forward defines what the model computes on an input x
        return self.net(x)

        #CNN: standard model for images, convulutional networks are what's actualyy used for image classification
        #100.000 parameters, 
class SmallCNN(nn.Module):
    """Two conv blocks (conv -> ReLU -> max-pool) followed by a small classifier."""
 
    def __init__(
        self,
        in_channels: int = 1,
        height: int = 28,
        width: int = 28,
        n_classes: int = 10,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # (B, 16, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (B, 16, H/2, W/2)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),           # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (B, 32, H/4, W/4)
        )
        # Two poolings halve height and width twice: 28 -> 7 (MNIST), 32 -> 8 (CIFAR)
        flat = 32 * (height // 4) * (width // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # (B, flat)
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),           # (B, n_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:             #gives the number of parameters in the model, useful to compare models
    return sum(p.numel() for p in model.parameters())

def build_model(name: str, images: torch.Tensor) -> nn.Module:
    """Create a model that matches the shape of a batch from the data loaders.
 
    Example:
        images, _, _ = next(iter(loaders["train_clean"]))
        model = build_model("cnn", images)
    """
    in_channels, height, width = images.shape[1:]
    if name == "mlp":
        return MLP(in_channels, height, width)
    if name == "cnn":
        return SmallCNN(in_channels, height, width)
    raise ValueError(f"Unknown model name: {name!r} (use 'mlp' or 'cnn')")
 
 
if __name__ == "__main__":
    # MNIST-like: grey 28x28 images
    for model in [MLP(), SmallCNN()]:
        out = model(torch.randn(5, 1, 28, 28))
        print("MNIST", type(model).__name__, tuple(out.shape), count_parameters(model))
 
    # CIFAR-like: colour 32x32 images
    for model in [MLP(3, 32, 32), SmallCNN(3, 32, 32)]:
        out = model(torch.randn(5, 3, 32, 32))
        print("CIFAR", type(model).__name__, tuple(out.shape), count_parameters(model))
 
    # build_model reads the shape from a batch
    fake_batch = torch.randn(5, 3, 32, 32)
    print("build_model:", type(build_model("cnn", fake_batch)).__name__)
 
