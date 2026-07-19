import os
import numpy as np
import matplotlib.pyplot as plt

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

colors = {"AdamW": "black", "Muon + AdamW": "#d62728"}

os.makedirs("plots", exist_ok=True)

# Train Loss Diagram

fig, ax = plt.subplots(figsize=(6, 4))

if os.path.exists("logs/adamw_loss.npy"):
    data = np.load("logs/adamw_loss.npy")
    ax.plot(data[:, 0], data[:, 1], linewidth=1.2, color=colors["AdamW"], label="AdamW")

if os.path.exists("logs/muon_loss.npy"):
    data = np.load("logs/muon_loss.npy")
    ax.plot(data[:, 0], data[:, 1], linewidth=1.2, color=colors["Muon + AdamW"], label="Muon + AdamW")

ax.set_xlabel("Step")
ax.set_ylabel("Training Loss")
ax.legend(frameon=False)
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
fig.tight_layout()
fig.savefig("plots/train_loss.png", dpi=300, bbox_inches="tight")
print("Saved plots/train_loss.png")

# Validation Loss Diagram

fig, ax = plt.subplots(figsize=(6, 4))

if os.path.exists("logs/adamw_val_loss.npy"):
    data = np.load("logs/adamw_val_loss.npy")
    ax.plot(data[:, 0], data[:, 1], linewidth=1.2, color=colors["AdamW"], label="AdamW")

if os.path.exists("logs/muon_val_loss.npy"):
    data = np.load("logs/muon_val_loss.npy")
    ax.plot(data[:, 0], data[:, 1], linewidth=1.2, color=colors["Muon + AdamW"], label="Muon + AdamW")

ax.set_xlabel("Step")
ax.set_ylabel("Validation Loss")
ax.legend(frameon=False)
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
fig.tight_layout()
fig.savefig("plots/val_loss.png", dpi=300, bbox_inches="tight")
print("Saved plots/val_loss.png")

# Validation Perplexity Diagram

fig, ax = plt.subplots(figsize=(6, 4))

if os.path.exists("logs/adamw_val_loss.npy"):
    data = np.load("logs/adamw_val_loss.npy")
    ax.plot(data[:, 0], np.exp(data[:, 1]), linewidth=1.2, color=colors["AdamW"], label="AdamW")

if os.path.exists("logs/muon_val_loss.npy"):
    data = np.load("logs/muon_val_loss.npy")
    ax.plot(data[:, 0], np.exp(data[:, 1]), linewidth=1.2, color=colors["Muon + AdamW"], label="Muon + AdamW")

ax.set_xlabel("Step")
ax.set_ylabel("Validation Perplexity")
ax.legend(frameon=False)
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
fig.tight_layout()
fig.savefig("plots/val_perplexity.png", dpi=300, bbox_inches="tight")
print("Saved plots/val_perplexity.png")
