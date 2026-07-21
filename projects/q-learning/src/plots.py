import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"

COLORS = {"q_learning": "#4269d0", "sarsa": "#a85c00", "expected_sarsa": "#8043c9"}
LABELS = {"q_learning": "Q-learning", "sarsa": "SARSA",
          "expected_sarsa": "Expected SARSA"}

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


def smooth(xs, k=10):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1])
            for i in range(len(xs))]


def online_return(results, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(8.6, 5.2), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax = styled_axes(fig, t, [0.1, 0.13, 0.66, 0.6])
    ax.grid(color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)

    for key in COLORS:
        curve = results["methods"][key]["online_return"]
        ax.plot(range(len(curve)), curve, color=COLORS[key], linewidth=0.8, alpha=0.2)
        ax.plot(range(len(curve)), smooth(curve), color=COLORS[key],
                linewidth=2.2, label=LABELS[key])

    ax.set_ylim(-140, 0)
    ax.set_xlabel("episode", fontsize=11.5, color=t["muted"], labelpad=8)
    ax.set_ylabel("return per episode while training", fontsize=11.5, color=t["muted"])
    leg = ax.legend(frameon=False, fontsize=11, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0))
    for text in leg.get_texts():
        text.set_color(t["ink"])
    titles(fig, t, "Online return during training",
           "mean over 50 seeds, smoothed over 10 episodes; SARSA earns more while "
           "training because its epsilon-greedy\nexploration keeps walking Q-learning "
           "off the cliff edge it insists on hugging")
    fig.savefig(ASSETS / f"online_return_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def paths(results, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9.6, 4.3), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0.05, 0.06, 0.9, 0.58])
    ax.set_facecolor(t["bg"])
    ax.set_xlim(0, 12)
    ax.set_ylim(4, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(t["axis"])

    for x in range(13):
        ax.plot([x, x], [0, 4], color=t["grid"], linewidth=1, zorder=1)
    for y in range(5):
        ax.plot([0, 12], [y, y], color=t["grid"], linewidth=1, zorder=1)

    for c in range(1, 11):
        ax.add_patch(Rectangle((c, 3), 1, 1, facecolor=t["grid"],
                               edgecolor=t["axis"], linewidth=1, zorder=2))
    ax.text(6, 3.5, "the cliff, reward -100", ha="center", va="center",
            fontsize=11, color=t["muted"], zorder=3)
    ax.text(0.5, 3.5, "S", ha="center", va="center", fontsize=13,
            fontweight="bold", color=t["ink"], zorder=6)
    ax.text(11.5, 3.5, "G", ha="center", va="center", fontsize=13,
            fontweight="bold", color=t["ink"], zorder=6)

    offsets = {"q_learning": 0.0, "sarsa": -0.14, "expected_sarsa": 0.14}
    for key, dy in offsets.items():
        path = results["methods"][key]["greedy_path_seed0"]
        xs = [c + 0.5 for _, c in path]
        ys = [r + 0.5 + dy for r, _ in path]
        ax.plot(xs, ys, color=COLORS[key], linewidth=2.4, zorder=4,
                solid_capstyle="round", label=LABELS[key])

    leg = ax.legend(frameon=False, fontsize=11, loc="lower center",
                    bbox_to_anchor=(0.5, 1.02), ncol=3)
    for text in leg.get_texts():
        text.set_color(t["ink"])
    titles(fig, t, "The greedy policy each method learned",
           "Q-learning takes the optimal 13-step route hugging the cliff, Expected "
           "SARSA keeps one row of margin,\nSARSA climbs all the way to the top; "
           "paths from seed 0, offset vertically so they don't overlap")
    fig.savefig(ASSETS / f"paths_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def epsilon_sweep(results, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(8.6, 5.2), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax = styled_axes(fig, t, [0.1, 0.13, 0.66, 0.6])
    ax.grid(color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)

    floor = -160
    epsilons = results["sweep"]["epsilons"]
    for key in COLORS:
        vals = results["sweep"][key]
        shown = [max(v, floor) for v in vals]
        ax.plot(epsilons, shown, color=COLORS[key], linewidth=2.2,
                marker="o", markersize=6, label=LABELS[key])
        for x, v in zip(epsilons, vals):
            if v < floor:
                ax.annotate(f"{v:.0f}", xy=(x, floor), xytext=(-4, 8),
                            textcoords="offset points", ha="right",
                            fontsize=9.5, color=COLORS[key], fontweight="bold")

    ax.set_ylim(floor, 0)
    ax.set_xlabel("exploration rate epsilon", fontsize=11.5, color=t["muted"], labelpad=8)
    ax.set_ylabel("online return, mean of last 100 episodes", fontsize=11.5,
                  color=t["muted"])
    leg = ax.legend(frameon=False, fontsize=11, loc="lower left",
                    bbox_to_anchor=(1.02, 0.0))
    for text in leg.get_texts():
        text.set_color(t["ink"])
    titles(fig, t, "More exploration widens the gap",
           "the more random actions the behavior policy takes, the more Q-learning's "
           "cliff-edge route costs it online;\npoints clipped at the axis floor are "
           "labeled with their true value")
    fig.savefig(ASSETS / f"epsilon_sweep_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


GAME_LABELS = {
    "snake": "snake, foods eaten",
    "pong": "pong, paddle hits",
    "flappy": "flappy bird, pipes passed",
}
GAME_COLORS = {"snake": "#4269d0", "pong": "#a85c00", "flappy": "#8043c9"}


def game_curves(games, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9.6, 4.2), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    for idx, name in enumerate(GAME_LABELS):
        ax = styled_axes(fig, t, [0.06 + idx * 0.325, 0.16, 0.24, 0.48])
        ax.grid(color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        curve = games[name]["curve"]
        xs = [i * curve["every"] for i in range(len(curve["scores"]))]
        ax.plot(xs, curve["scores"], color=GAME_COLORS[name], linewidth=1.8)
        ax.set_xlabel("episode", fontsize=10.5, color=t["muted"])
        if idx == 0:
            ax.set_ylabel("score, mean over 5 seeds", fontsize=10.5, color=t["muted"])
        ax.set_title(GAME_LABELS[name], fontsize=11, color=t["ink"], loc="left", pad=8)
        ax.ticklabel_format(axis="x", style="plain")
    titles(fig, t, "One update rule, three games",
           "the same tabular Q-learning loop, only the state encoding changes; "
           "epsilon decays from 1.0 to 0.02 over the\nfirst 60% of episodes; "
           "scores are capped at 30 hits for pong and 50 pipes for flappy bird")
    fig.savefig(ASSETS / f"games_{mode}.png", facecolor=t["bg"])
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


def render_pong(replay, mode, fps=14):
    t = THEMES[mode]
    frames = replay["frames"]
    fig, ax = game_fig(t, 12, 12, 3.4)

    def draw(i):
        for artist in list(ax.patches) + list(ax.texts):
            artist.remove()
        f = frames[i]
        hits = sum(1 for j in range(1, i + 1)
                   if frames[j]["ball"][1] == 10 and frames[j - 1]["ball"][1] == 10)
        ax.add_patch(Rectangle((f["paddle"], 11.25), 3, 0.55,
                               facecolor="#4269d0", edgecolor="none"))
        ax.add_patch(Circle((f["ball"][0] + 0.5, f["ball"][1] + 0.5), 0.38,
                            facecolor="#a85c00", edgecolor="none"))
        ax.text(0.35, 0.85, f"hits {hits}", fontsize=11,
                color=t["muted"], fontweight="bold",
                bbox=dict(facecolor=t["bg"], edgecolor="none", alpha=0.85,
                          boxstyle="square,pad=0.25"))

    save_gif(fig, draw, len(frames), ASSETS / f"pong_{mode}.gif", fps)


def render_flappy(replay, mode, fps=12):
    t = THEMES[mode]
    frames = replay["frames"]
    bird_x = 4
    w, h = 18, 14
    fig, ax = game_fig(t, w, h, 3.6)

    def draw(i):
        for artist in list(ax.patches) + list(ax.texts):
            artist.remove()
        f = frames[i]
        prev_gap, score = None, 0
        for j in range(1, i + 1):
            if frames[j]["gap"] != frames[j - 1]["gap"]:
                prev_gap = frames[j - 1]["gap"]
                score += 1
        pipes = [(f["dist"] + bird_x, f["gap"]),
                 (f["dist"] + bird_x + 10, f["next_gap"])]
        if prev_gap is not None:
            pipes.append((f["dist"] + bird_x - 10, prev_gap))
        for x, gap in pipes:
            if x < -1 or x > w:
                continue
            top = gap - 2.5
            ax.add_patch(Rectangle((x, 0), 1, top, facecolor="#898781",
                                   edgecolor="none"))
            ax.add_patch(Rectangle((x, gap + 2.5), 1, h - gap - 2.5,
                                   facecolor="#898781", edgecolor="none"))
        ax.add_patch(Rectangle((bird_x + 0.1, f["y"] + 0.1), 0.8, 0.8,
                               facecolor="#a85c00", edgecolor="none"))
        ax.text(0.35, 1.1, f"pipes {score}", fontsize=11,
                color=t["muted"], fontweight="bold",
                bbox=dict(facecolor=t["bg"], edgecolor="none", alpha=0.85,
                          boxstyle="square,pad=0.25"))

    save_gif(fig, draw, len(frames), ASSETS / f"flappy_{mode}.gif", fps)


if __name__ == "__main__":
    results = json.loads((ASSETS / "results.json").read_text())
    games = json.loads((ASSETS / "games.json").read_text())
    for mode in ("light", "dark"):
        online_return(results, mode)
        paths(results, mode)
        epsilon_sweep(results, mode)
        game_curves(games, mode)
        render_snake(games["snake"]["replay"], mode)
        render_pong(games["pong"]["replay"], mode)
        render_flappy(games["flappy"]["replay"], mode)
    print("wrote charts to", ASSETS)
