import argparse
import os

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from model import UNet
from ddpm import DDPM

parser = argparse.ArgumentParser(description="Train DDPM on CelebA")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--image-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
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

torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(args.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = ImageFolder("../data/celeba_raw/img_align_celeba", transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

model = UNet(in_channels=3, channel_mults=(1, 2, 4, 8)).to(device)

os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

start_epoch = 0
if args.resume:
    checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
    if "model" in checkpoint:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
        model.load_state_dict(state_dict)
        start_epoch = checkpoint["epoch"]
    else:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
    print(f"Resumed from epoch {start_epoch}")

model = torch.compile(model)
diffusion = DDPM(model, T=args.timesteps, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

if args.resume and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

for epoch in range(start_epoch, args.epochs):
    model.train()
    total_loss = 0.0
    for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            loss = diffusion.loss(images)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        model.eval()
        samples = diffusion.sample((16, 3, args.image_size, args.image_size))
        samples = (samples.clamp(-1, 1) + 1) / 2

        grid = make_grid(samples, nrow=4)
        plt.figure(figsize=(4, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(f"plots/samples_epoch{epoch+1}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plots/samples_epoch{epoch+1}.png")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
        }, "models/model.pt")
        print("Saved models/model.pt")