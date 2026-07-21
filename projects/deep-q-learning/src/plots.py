import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"

TABULAR_SNAKE = 21.1

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


def curves(snake, breakout, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9.6, 4.6), dpi=200)
    fig.patch.set_facecolor(t["bg"])

    panels = [
        (snake, "snake, foods eaten", "#4269d0", 0),
        (breakout, "breakout, bricks broken (48 total)", "#a85c00", 1),
    ]
    for data, label, color, idx in panels:
        ax = styled_axes(fig, t, [0.08 + idx * 0.48, 0.15, 0.38, 0.5])
        ax.grid(color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        steps = [e["step"] / 1000 for e in data["history"]["eval"]]
        means = [float(np.mean(e["scores"])) for e in data["history"]["eval"]]
        for e in data["history"]["eval"]:
            ax.scatter([e["step"] / 1000] * len(e["scores"]), e["scores"],
                       s=8, color=color, alpha=0.18, linewidths=0, zorder=2)
        ax.plot(steps, means, color=color, linewidth=2.2, zorder=3)
        if idx == 0:
            ax.axhline(TABULAR_SNAKE, color=t["muted"], linestyle=(0, (5, 4)),
                       linewidth=1.2)
            ax.text(steps[-1], TABULAR_SNAKE - 1.2, "tabular Q-learning",
                    fontsize=9.5, color=t["muted"], ha="right", va="top")
            ax.set_ylabel("greedy score, 10 episodes per point", fontsize=10.5,
                          color=t["muted"])
        ax.set_xlabel("environment steps (thousands)", fontsize=10.5, color=t["muted"])
        ax.set_title(label, fontsize=11.5, color=t["ink"], loc="left", pad=8)
    titles(fig, t, "DQN learning curves",
           "greedy evaluation every 20k steps, dots are single episodes, line is "
           "the mean; the dashed line is the 288-state\ntabular agent's plateau "
           "from the q-learning project, which DQN's full-grid view exists to beat")
    fig.savefig(ASSETS / f"curves_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def save_gif(fig, draw, n_frames, path, fps):
    images = []
    for i in range(n_frames):
        draw(i)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        images.append(Image.fromarray(buf[..., :3].copy()))
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
    plt.close(fig)


def game_fig(t, w, h, size):
    fig = plt.figure(figsize=(size * w / max(w, h), size * h / max(w, h)), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(t["bg"])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(t["axis"])
        side.set_linewidth(2)
    return fig, ax


def render_snake(replay, mode, fps=12):
    t = THEMES[mode]
    frames = replay["frames"]
    fig, ax = game_fig(t, 12, 12, 3.4)

    def draw(i):
        for artist in list(ax.patches) + list(ax.texts):
            artist.remove()
        f = frames[i]
        ax.add_patch(Rectangle((f["food"][1] + 0.15, f["food"][0] + 0.15), 0.7, 0.7,
                               facecolor="#a85c00", edgecolor="none"))
        for j, (r, c) in enumerate(f["body"]):
            color = t["ink"] if j == 0 else "#4269d0"
            ax.add_patch(Rectangle((c + 0.08, r + 0.08), 0.84, 0.84,
                                   facecolor=color, edgecolor="none"))
        ax.text(0.35, 0.75, f"score {len(f['body']) - 3}", fontsize=11,
                color=t["muted"], fontweight="bold",
                bbox=dict(facecolor=t["bg"], edgecolor="none", alpha=0.85,
                          boxstyle="square,pad=0.25"))

    save_gif(fig, draw, len(frames), ASSETS / f"snake_{mode}.gif", fps)


def render_breakout(replay, mode, fps=14):
    t = THEMES[mode]
    frames = replay["frames"]
    fig, ax = game_fig(t, 12, 14, 3.6)

    def draw(i):
        for artist in list(ax.patches) + list(ax.texts):
            artist.remove()
        f = frames[i]
        broken = 0
        for row in range(4):
            for col in range(12):
                if f["bricks"][row][col]:
                    ax.add_patch(Rectangle((col + 0.06, row + 1 + 0.12), 0.88, 0.76,
                                           facecolor="#8043c9", edgecolor="none"))
                else:
                    broken += 1
        ax.add_patch(Rectangle((f["paddle"], 13.25), 3, 0.55,
                               facecolor="#4269d0", edgecolor="none"))
        ax.add_patch(Circle((f["ball"][0] + 0.5, f["ball"][1] + 0.5), 0.38,
                            facecolor="#a85c00", edgecolor="none"))
        ax.text(0.35, 0.75, f"bricks {broken}", fontsize=11,
                color=t["muted"], fontweight="bold",
                bbox=dict(facecolor=t["bg"], edgecolor="none", alpha=0.85,
                          boxstyle="square,pad=0.25"))

    save_gif(fig, draw, len(frames), ASSETS / f"breakout_{mode}.gif", fps)


if __name__ == "__main__":
    snake = json.loads((ASSETS / "snake.json").read_text())
    breakout = json.loads((ASSETS / "breakout.json").read_text())
    for mode in ("light", "dark"):
        curves(snake, breakout, mode)
        render_snake(snake["replay"], mode)
        render_breakout(breakout["replay"], mode)
    print("wrote charts to", ASSETS)
