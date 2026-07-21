import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = Path(__file__).resolve().parent.parent / "assets"

COLORS = {"base": "#898781", "sft": "#a85c00", "sdft": "#4269d0"}
LABELS = {"base": "base model", "sft": "SFT", "sdft": "SDFT"}

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


def tradeoff(results, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(8.6, 5.6), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax = styled_axes(fig, t, [0.1, 0.12, 0.82, 0.6])
    ax.grid(color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)

    points = [
        ("base", COLORS["base"], True, "base model", (0, 15)),
        ("sft", COLORS["sft"], True, "SFT", (0, 15)),
        ("sdft", COLORS["sdft"], True, "SDFT", (0, 15)),
        ("sft_natural", COLORS["sft"], False, "SFT, natural style", (0, -22)),
        ("sdft_natural", COLORS["sdft"], False, "SDFT, natural style", (0, -22)),
        ("sdft_hi_lr", COLORS["sdft"], False, "SDFT, 2x lr", (14, -6)),
    ]
    base_ret = results["base"]["forgetting"]["average"]
    for key, color, filled, label, (dx, dy) in points:
        r = results[key]
        x = r["knowledge"]["paraphrase"] * 100
        y = r["forgetting"]["average"] * 100
        ax.scatter([x], [y], s=170, zorder=4,
                   facecolors=color if filled else t["bg"],
                   edgecolors=color, linewidths=2)
        ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=10.5, color=color, fontweight="bold",
                    ha="center" if dx == 0 else "left")

    ax.axhline(base_ret * 100, color=t["muted"], linestyle=(0, (5, 4)), linewidth=1.2)
    ax.text(33, base_ret * 100 - 0.4, "prior capability before any training",
            fontsize=10, color=t["muted"], va="top", ha="right")

    ax.set_xlim(0, 35)
    ax.set_xlabel("new knowledge, reworded-question accuracy (%)", fontsize=11.5,
                  color=t["muted"], labelpad=8)
    ax.set_ylabel("prior capabilities, benchmark average (%)", fontsize=11.5,
                  color=t["muted"])
    titles(fig, t, "Learning the new facts without losing the old skills",
           "up and to the right is better; SDFT lands near 18% knowledge in every "
           "configuration, SFT learns more at the same\nsettings, and doubling SDFT's "
           "learning rate only costs retention. Filled markers are the matched pair.")
    fig.savefig(ASSETS / f"tradeoff_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def knowledge_bars(results, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(8.6, 4.2), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax = styled_axes(fig, t, [0.09, 0.19, 0.68, 0.52])
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)

    groups = ["paraphrase", "trained_wording", "ood"]
    group_labels = ["same fact,\nreworded question\n(the real test)",
                    "the exact question\nit trained on",
                    "indirect questions\n(never trained on)"]
    width = 0.26
    for i, key in enumerate(("base", "sft", "sdft")):
        vals = [results[key]["knowledge"][g] * 100 for g in groups]
        xs = [j + (i - 1) * width for j in range(len(groups))]
        ax.bar(xs, vals, width=width * 0.9, color=COLORS[key], label=LABELS[key], zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.0f}", xy=(x, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, color=t["ink"], fontweight="bold")

    ax.set_xticks(range(len(groups)), group_labels)
    for lbl in ax.get_xticklabels():
        lbl.set_color(t["body"])
        lbl.set_fontsize(11)
    ax.set_ylabel("accuracy (%)", fontsize=11.5, color=t["muted"])
    leg = ax.legend(frameon=False, fontsize=11, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0))
    for text in leg.get_texts():
        text.set_color(t["ink"])
    titles(fig, t, "Knowledge that made it into the weights",
           "the base model scores near zero, these 2025 events postdate its training data")
    fig.savefig(ASSETS / f"knowledge_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def forgetting_bars(results, mode):
    t = THEMES[mode]
    tasks = [k for k in results["base"]["forgetting"] if k != "average"]
    fig = plt.figure(figsize=(9.6, 4.4), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax = styled_axes(fig, t, [0.08, 0.17, 0.72, 0.55])
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)

    width = 0.26
    for i, key in enumerate(("base", "sft", "sdft")):
        vals = [results[key]["forgetting"][task] * 100 for task in tasks]
        xs = [j + (i - 1) * width for j in range(len(tasks))]
        ax.bar(xs, vals, width=width * 0.9, color=COLORS[key], label=LABELS[key], zorder=3)

    ax.set_xticks(range(len(tasks)), tasks)
    for lbl in ax.get_xticklabels():
        lbl.set_color(t["body"])
        lbl.set_fontsize(10.5)
    ax.set_ylabel("accuracy (%)", fontsize=11.5, color=t["muted"])
    leg = ax.legend(frameon=False, fontsize=11, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0))
    for text in leg.get_texts():
        text.set_color(t["ink"])
    titles(fig, t, "Where the forgetting actually shows up",
           "each benchmark before and after fine-tuning on 600 questions about 2025 disasters; "
           "TruthfulQA takes the damage, ARC-Challenge goes up")
    fig.savefig(ASSETS / f"forgetting_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def training_curves(hist_sdft, hist_sft, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9.6, 4.0), dpi=200)
    fig.patch.set_facecolor(t["bg"])

    for idx, (hist, name, color, ylab) in enumerate([
        (hist_sdft, "SDFT", COLORS["sdft"], "KL(teacher || student) per token"),
        (hist_sft, "SFT", COLORS["sft"], "cross-entropy on gold answers"),
    ]):
        ax = styled_axes(fig, t, [0.08 + idx * 0.48, 0.16, 0.36, 0.54])
        ax.grid(color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        steps = [h["step"] for h in hist]
        loss = [h["loss"] for h in hist]
        smooth = [sum(loss[max(0, i - 9):i + 1]) / len(loss[max(0, i - 9):i + 1])
                  for i in range(len(loss))]
        ax.plot(steps, loss, color=color, linewidth=0.8, alpha=0.25)
        ax.plot(steps, smooth, color=color, linewidth=2.2)
        ax.set_xlabel("step", fontsize=11, color=t["muted"])
        ax.set_ylabel(ylab, fontsize=10.5, color=t["muted"])
        ax.set_title(name, fontsize=12, color=t["ink"], loc="left", pad=8)

    titles(fig, t, "Training curves",
           "different objectives, so different units; smoothed over 10 steps")
    fig.savefig(ASSETS / f"curves_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


if __name__ == "__main__":
    results = json.loads((ASSETS / "results.json").read_text())
    hist_sdft = json.loads((ASSETS / "history_sdft.json").read_text())
    hist_sft = json.loads((ASSETS / "history_sft.json").read_text())
    for mode in ("light", "dark"):
        tradeoff(results, mode)
        knowledge_bars(results, mode)
        forgetting_bars(results, mode)
        training_curves(hist_sdft, hist_sft, mode)
    print("wrote charts to", ASSETS)
