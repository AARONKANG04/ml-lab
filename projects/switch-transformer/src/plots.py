"""Regenerate the README charts from assets/results.json.

Writes light and dark PNGs for the sample-efficiency curves and the
final-perplexity bars into assets/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = Path(__file__).resolve().parent.parent / "assets"

COLORS = {"dense": "#a85c00", "switch8": "#4269d0", "switch16": "#8043c9"}
LABELS = {"dense": "dense", "switch8": "switch-8", "switch16": "switch-16"}

THEMES = {
    "light": {
        "bg": "#fcfcfb",
        "ink": "#0b0b0b",
        "body": "#52514e",
        "muted": "#898781",
        "grid": "#e8e7e0",
        "axis": "#c9c8be",
    },
    "dark": {
        "bg": "#1a1a19",
        "ink": "#f4f4f2",
        "body": "#c3c2b7",
        "muted": "#8f8d87",
        "grid": "#292927",
        "axis": "#363633",
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


def sample_efficiency(results, mode):
    t = THEMES[mode]
    runs = results["runs"]
    dense_final = results["dense_final_val_loss_curve_metric"]

    fig = plt.figure(figsize=(10.8, 4.8), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax = styled_axes(fig, t, [0.075, 0.15, 0.845, 0.64])
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)

    # the two switch curves end at nearly the same loss, so their end-of-line
    # labels are staggered by hand
    label_offset = {"dense": 11, "switch8": -2, "switch16": -16}
    for name in ("dense", "switch8", "switch16"):
        steps, loss = zip(*runs[name]["val_curve"])
        ax.plot(steps, loss, color=COLORS[name], linewidth=2.4, label=LABELS[name])
        ax.annotate(
            LABELS[name],
            xy=(steps[-1], loss[-1]),
            xytext=(8, label_offset[name]),
            textcoords="offset points",
            fontsize=11.5,
            color=COLORS[name],
            va="center",
        )

    ax.axhline(dense_final, color=t["muted"], linestyle=(0, (5, 4)), linewidth=1.2)
    ax.text(500, dense_final - 0.015, "dense model's final loss", fontsize=10.5,
            color=t["muted"], va="top")

    # step where switch-8 first dips under the dense model's final loss
    cross_step, cross_loss = next(
        (s, v) for s, v in runs["switch8"]["val_curve"] if v <= dense_final
    )
    lead = runs["dense"]["result"]["steps"] - cross_step
    ax.plot([cross_step], [cross_loss], "o", markersize=8, markerfacecolor=t["bg"],
            markeredgecolor=COLORS["switch8"], markeredgewidth=2)
    ax.annotate(
        f"switch-8 gets there at step {cross_step:,},\n"
        f"{lead:,} steps before the dense model",
        xy=(cross_step, cross_loss),
        xytext=(12800, 3.56),
        fontsize=11.5,
        color=t["body"],
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=1),
    )

    ax.set_xlim(0, 20600)
    ax.set_ylim(3.13, 4.05)
    ax.set_xticks([0, 5000, 10000, 15000, 20000], ["0", "5k", "10k", "15k", "20k"])
    ax.set_xlabel("training step", fontsize=11.5, color=t["muted"], labelpad=8)
    ax.set_ylabel("validation loss", fontsize=11.5, color=t["muted"])
    leg = ax.legend(loc="upper right", frameon=False, fontsize=11.5,
                    handlelength=1.6, labelspacing=0.55)
    for text in leg.get_texts():
        text.set_color(t["ink"])

    titles(fig, t,
           "Same compute per token, faster learning",
           "validation loss on WikiText-103; all three models spend identical "
           "FLOPs per token, the switch models just have more parameters")
    fig.savefig(ASSETS / f"sample_efficiency_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def val_ppl(results, mode):
    from math import exp

    t = THEMES[mode]
    runs = results["runs"]

    fig = plt.figure(figsize=(10.8, 4.8), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax = styled_axes(fig, t, [0.075, 0.15, 0.845, 0.64])
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)

    # the gray ablation line is identified by the legend only; an end-of-line
    # label that long would run off the figure
    ax.plot(*zip(*[(s, exp(v)) for s, v in runs["switch8-noaux"]["val_curve"]]),
            color=t["muted"], linewidth=2.4, label="switch-8, no balancing loss")

    label_offset = {"dense": (8, 8), "switch8": (8, -2), "switch16": (14, -14)}
    for name in ("dense", "switch8", "switch16"):
        steps, loss = zip(*runs[name]["val_curve"])
        ppl = [exp(v) for v in loss]
        ax.plot(steps, ppl, color=COLORS[name], linewidth=2.4, label=LABELS[name])
        ax.annotate(
            LABELS[name],
            xy=(steps[-1], ppl[-1]),
            xytext=label_offset[name],
            textcoords="offset points",
            fontsize=11.5,
            color=COLORS[name],
            va="center",
        )

    ax.set_xlim(0, 20600)
    ax.set_ylim(22.5, 60)
    ax.set_xticks([0, 5000, 10000, 15000, 20000], ["0", "5k", "10k", "15k", "20k"])
    ax.set_xlabel("training step", fontsize=11.5, color=t["muted"], labelpad=8)
    ax.set_ylabel("validation perplexity", fontsize=11.5, color=t["muted"])
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1],
                    loc="upper right", frameon=False, fontsize=11.5,
                    handlelength=1.6, labelspacing=0.55)
    for text in leg.get_texts():
        text.set_color(t["ink"])

    titles(fig, t,
           "Validation perplexity over training",
           "exp of the validation loss, measured on 64 fixed windows every 500 "
           "steps; the table's final numbers use 400 windows so they sit a "
           "little higher")
    fig.savefig(ASSETS / f"val_ppl_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


if __name__ == "__main__":
    results = json.loads((ASSETS / "results.json").read_text())
    for mode in ("light", "dark"):
        sample_efficiency(results, mode)
        val_ppl(results, mode)
    print("wrote sample_efficiency and val_ppl PNGs to", ASSETS)
