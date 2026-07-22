import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"

THEMES = {
    "light": {
        "bg": "#fcfcfb", "ink": "#0b0b0b", "body": "#52514e",
        "muted": "#898781", "grid": "#e8e7e0", "axis": "#c9c8be",
    },
    "dark": {
        "bg": "#1a1a19", "ink": "#f4f4f2", "body": "#c3c2b7",
        "muted": "#8f8d87", "grid": "#292927", "axis": "#363633",
    },
}

plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def styled_axes(fig, t, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(t["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=10.5, length=4, width=1)
    return ax


def titles(fig, t, title, subtitle):
    fig.text(0.06, 0.94, title, fontsize=14.5, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.06, 0.865, subtitle, fontsize=11, color=t["muted"], va="top")


def learning_curve(results, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9.0, 5.4), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax = styled_axes(fig, t, [0.1, 0.12, 0.82, 0.6])
    ax.grid(color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)

    train = results["history"]["train"]
    ax.plot([h["step"] / 1e6 for h in train], [h["return"] for h in train],
            color="#4269d0", linewidth=1.0, alpha=0.35)
    evals = results["history"]["eval"]
    ax.plot([e["step"] / 1e6 for e in evals], [e["mean"] for e in evals],
            color="#4269d0", linewidth=2.2)
    ax.fill_between([e["step"] / 1e6 for e in evals],
                    [e["min"] for e in evals], [e["max"] for e in evals],
                    color="#4269d0", alpha=0.12, linewidth=0)

    ax.set_xlabel("environment steps (millions)", fontsize=11.5, color=t["muted"],
                  labelpad=8)
    ax.set_ylabel("episode return", fontsize=11.5, color=t["muted"])
    titles(fig, t, "PPO on Humanoid-v5",
           "thick line is deterministic evaluation (mean of 5 episodes, band is "
           "min to max), thin line is the running\naverage of training episodes "
           "under the stochastic policy")
    fig.savefig(ASSETS / f"learning_curve_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def diagnostics(results, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9.6, 4.2), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    train = results["history"]["train"]
    xs = [h["step"] / 1e6 for h in train]
    panels = [
        ("kl", "approx KL per update", "#a85c00"),
        ("clipfrac", "fraction of clipped ratios", "#8043c9"),
        ("len", "episode length at eval", "#4269d0"),
    ]
    for idx, (key, label, color) in enumerate(panels):
        ax = styled_axes(fig, t, [0.07 + idx * 0.325, 0.16, 0.24, 0.48])
        ax.grid(color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        if key == "len":
            evals = results["history"]["eval"]
            ax.plot([e["step"] / 1e6 for e in evals], [e["len"] for e in evals],
                    color=color, linewidth=1.8)
        else:
            ax.plot(xs, [h[key] for h in train], color=color, linewidth=1.2)
        ax.set_xlabel("steps (millions)", fontsize=10.5, color=t["muted"])
        ax.set_title(label, fontsize=11, color=t["ink"], loc="left", pad=8)
    titles(fig, t, "Training diagnostics",
           "mid-training the updates get big: per-update KL climbs to 0.8 and most "
           "ratios hit the clip, which lines up\nwith the noisy evals; the linear "
           "lr anneal calms it back down, and episode length never reaches the "
           "1000-step cap")
    fig.savefig(ASSETS / f"diagnostics_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


if __name__ == "__main__":
    results = json.loads((ASSETS / "results.json").read_text())
    for mode in ("light", "dark"):
        learning_curve(results, mode)
        diagnostics(results, mode)
    print("wrote charts to", ASSETS)
