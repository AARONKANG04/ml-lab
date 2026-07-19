import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchvision.models import resnet18
from adam import Adam

parser = argparse.ArgumentParser(description="Adam vs. SGD (momentum) benchmark on CIFAR-10")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
args = parser.parse_args()

if args.device == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(args.device)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10("../data", train=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

init_model = resnet18(num_classes=10)
init_state = copy.deepcopy(init_model.state_dict())


def train(optimizer_name, optimizer):
    model = resnet18(num_classes=10).to(device)
    model.load_state_dict(copy.deepcopy(init_state))
    model.to(device)
    opt = optimizer(model.parameters())
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"[{optimizer_name}] Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"[{optimizer_name}] Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.4f}")

    return losses


optimizers = {
    "SGD (momentum=0.9)": lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    "Custom Adam": lambda params: Adam(params, lr=args.lr),
}

results = {}
for name, opt_fn in optimizers.items():
    print(f"\n--- Training with {name} ---")
    results[name] = train(name, opt_fn)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.usetex": False,
})

fig, ax = plt.subplots(figsize=(6, 4))
epochs = range(1, args.epochs + 1)
colors = {"SGD (momentum=0.9)": "black", "Custom Adam": "#d62728"}

for name, losses in results.items():
    ax.plot(epochs, losses, linewidth=1.2, color=colors[name], label=name)

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.legend(frameon=False)
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
ax.set_xlim(1, args.epochs)
fig.tight_layout()
import os
os.makedirs("plots", exist_ok=True)
fig.savefig("plots/loss_curve.png", dpi=300, bbox_inches="tight")
print("\nSaved plots/loss_curve.png")
plt.show()