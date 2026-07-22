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


def curves(results, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9.6, 4.4), dpi=200)
    fig.patch.set_facecolor(t["bg"])

    ax = styled_axes(fig, t, [0.08, 0.16, 0.38, 0.5])
    ax.grid(color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    train = results["history"]["train"]
    losses = [h["loss"] for h in train]
    smooth = [float(np.mean(losses[max(0, i - 9):i + 1])) for i in range(len(losses))]
    ax.plot([h["step"] for h in train], losses, color="#4269d0", linewidth=0.7,
            alpha=0.25)
    ax.plot([h["step"] for h in train], smooth, color="#4269d0", linewidth=2.0)
    ax.axhline(np.log(2), color=t["muted"], linestyle=(0, (5, 4)), linewidth=1.1)
    ax.text(train[-1]["step"], np.log(2) + 0.02, "log 2, a coin flip",
            fontsize=9.5, color=t["muted"], ha="right", va="bottom")
    ax.set_xlabel("step", fontsize=10.5, color=t["muted"])
    ax.set_title("Bradley-Terry loss", fontsize=11.5, color=t["ink"], loc="left",
                 pad=8)

    ax = styled_axes(fig, t, [0.57, 0.16, 0.38, 0.5])
    ax.grid(color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    evals = results["history"]["eval"]
    ax.plot([e["step"] for e in evals], [e["acc"] * 100 for e in evals],
            color="#a85c00", linewidth=2.0, marker="o", markersize=4)
    ax.axhline(50, color=t["muted"], linestyle=(0, (5, 4)), linewidth=1.1)
    ax.set_xlabel("step", fontsize=10.5, color=t["muted"])
    ax.set_title("held-out pairwise accuracy (%)", fontsize=11.5, color=t["ink"],
                 loc="left", pad=8)

    titles(fig, t, "Training a preference model",
           "loss below log 2 means the model ranks pairs better than chance; "
           "accuracy is on 1,000 held-out pairs during\ntraining, final number "
           "on the full 2,000-pair test set")
    fig.savefig(ASSETS / f"curves_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def analysis(results, mode):
    t = THEMES[mode]
    records = results["final"]["records"]
    fig = plt.figure(figsize=(9.6, 4.4), dpi=200)
    fig.patch.set_facecolor(t["bg"])

    ax = styled_axes(fig, t, [0.08, 0.16, 0.38, 0.5])
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 10.0)]
    labels = ["under 1", "1 to 2", "2 to 3", "3 or more"]
    accs, counts = [], []
    for lo, hi in bins:
        sel = [r for r in records if lo <= r["rating_c"] - r["rating_j"] < hi]
        counts.append(len(sel))
        accs.append(100 * np.mean([r["rc"] > r["rj"] for r in sel]) if sel else 0)
    ax.bar(range(len(bins)), accs, width=0.62, color="#4269d0", zorder=3)
    for i, (a, c) in enumerate(zip(accs, counts)):
        ax.annotate(f"{a:.0f}%", xy=(i, a), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=t["ink"], fontweight="bold")
        ax.annotate(f"n={c}", xy=(i, 0), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=t["muted"])
    ax.axhline(50, color=t["muted"], linestyle=(0, (5, 4)), linewidth=1.1)
    ax.set_xticks(range(len(bins)), labels)
    for lbl in ax.get_xticklabels():
        lbl.set_color(t["body"])
    ax.set_xlabel("annotator rating gap, chosen minus rejected", fontsize=10.5,
                  color=t["muted"])
    ax.set_title("accuracy by how clear the preference is", fontsize=11.5,
                 color=t["ink"], loc="left", pad=8)

    ax = styled_axes(fig, t, [0.57, 0.16, 0.38, 0.5])
    ax.grid(color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    margins = np.array([r["rc"] - r["rj"] for r in records])
    ax.hist(margins[margins > 0], bins=40, color="#4269d0", alpha=0.85, zorder=3,
            label="ranked correctly")
    ax.hist(margins[margins <= 0], bins=40, color="#a85c00", alpha=0.85, zorder=3,
            label="misranked")
    ax.set_xlabel("reward margin, chosen minus rejected", fontsize=10.5,
                  color=t["muted"])
    ax.set_title("margin distribution on the test set", fontsize=11.5,
                 color=t["ink"], loc="left", pad=8)
    leg = ax.legend(frameon=False, fontsize=10, loc="upper left")
    for text in leg.get_texts():
        text.set_color(t["ink"])

    titles(fig, t, "Where the model is right and wrong",
           "pairs the annotators strongly preferred are easier to rank; misranked "
           "pairs cluster near zero margin, the\nmodel is rarely confidently wrong")
    fig.savefig(ASSETS / f"analysis_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def length_bias(results, mode):
    t = THEMES[mode]
    records = results["final"]["records"]
    fig = plt.figure(figsize=(9.6, 4.4), dpi=200)
    fig.patch.set_facecolor(t["bg"])

    ax = styled_axes(fig, t, [0.08, 0.16, 0.38, 0.5])
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    groups = {
        "chosen is\nshorter": [r for r in records if r["len_c"] < 0.9 * r["len_j"]],
        "similar\nlength": [r for r in records
                            if 0.9 * r["len_j"] <= r["len_c"] <= 1.1 * r["len_j"]],
        "chosen is\nlonger": [r for r in records if r["len_c"] > 1.1 * r["len_j"]],
    }
    accs = [100 * np.mean([r["rc"] > r["rj"] for r in g]) if g else 0
            for g in groups.values()]
    ax.bar(range(3), accs, width=0.62, color="#8043c9", zorder=3)
    for i, (a, g) in enumerate(zip(accs, groups.values())):
        ax.annotate(f"{a:.0f}%", xy=(i, a), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=t["ink"], fontweight="bold")
        ax.annotate(f"n={len(g)}", xy=(i, 0), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=t["muted"])
    ax.axhline(50, color=t["muted"], linestyle=(0, (5, 4)), linewidth=1.1)
    ax.set_xticks(range(3), list(groups))
    for lbl in ax.get_xticklabels():
        lbl.set_color(t["body"])
    ax.set_title("accuracy by relative response length", fontsize=11.5,
                 color=t["ink"], loc="left", pad=8)

    ax = styled_axes(fig, t, [0.57, 0.16, 0.38, 0.5])
    ax.grid(color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    lens = np.array([r["len_c"] for r in records] + [r["len_j"] for r in records])
    scores = np.array([r["rc"] for r in records] + [r["rj"] for r in records])
    edges = np.quantile(lens, np.linspace(0, 1, 11))
    mids, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lens >= lo) & (lens <= hi)
        if sel.sum():
            mids.append(np.median(lens[sel]))
            means.append(scores[sel].mean())
    ax.plot(mids, means, color="#8043c9", linewidth=2.0, marker="o", markersize=4)
    corr = np.corrcoef(lens, scores)[0, 1]
    ax.text(0.97, 0.06, f"corr {corr:+.2f}", transform=ax.transAxes,
            fontsize=10.5, color=t["muted"], ha="right")
    ax.set_xlabel("response length in tokens", fontsize=10.5, color=t["muted"])
    ax.set_title("mean reward by length decile", fontsize=11.5, color=t["ink"],
                 loc="left", pad=8)

    titles(fig, t, "The length bias check",
           "reward models trained on preference data notoriously learn "
           "\"longer is better\"; this measures how much\nof that this one "
           "picked up")
    fig.savefig(ASSETS / f"length_bias_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


def comparison(results, results_q35, mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9.6, 4.6), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    models = [("Qwen2.5-0.5B", results, "#a85c00"),
              ("Qwen3.5-0.8B", results_q35, "#4269d0")]

    def acc(records, sel):
        picked = [r for r in records if sel(r)]
        return 100 * np.mean([r["rc"] > r["rj"] for r in picked])

    panels = [
        ("accuracy by annotator rating gap", [0.08, 0.15, 0.38, 0.5],
         [("under 1", lambda r: r["rating_c"] - r["rating_j"] < 1),
          ("1 to 2", lambda r: 1 <= r["rating_c"] - r["rating_j"] < 2),
          ("2 to 3", lambda r: 2 <= r["rating_c"] - r["rating_j"] < 3),
          ("3 or more", lambda r: r["rating_c"] - r["rating_j"] >= 3)]),
        ("accuracy by relative response length", [0.57, 0.15, 0.38, 0.5],
         [("chosen\nshorter", lambda r: r["len_c"] < 0.9 * r["len_j"]),
          ("similar", lambda r: 0.9 * r["len_j"] <= r["len_c"] <= 1.1 * r["len_j"]),
          ("chosen\nlonger", lambda r: r["len_c"] > 1.1 * r["len_j"])]),
    ]
    for title, rect, groups in panels:
        ax = styled_axes(fig, t, rect)
        ax.grid(axis="y", color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        width = 0.36
        for mi, (label, res, color) in enumerate(models):
            records = res["final"]["records"]
            vals = [acc(records, sel) for _, sel in groups]
            xs = [g + (mi - 0.5) * width for g in range(len(groups))]
            ax.bar(xs, vals, width=width * 0.92, color=color, zorder=3,
                   label=label)
            for x, v in zip(xs, vals):
                ax.annotate(f"{v:.0f}", xy=(x, v), xytext=(0, 3),
                            textcoords="offset points", ha="center",
                            fontsize=8.5, color=t["ink"], fontweight="bold")
        ax.set_ylim(50, 100)
        ax.set_xticks(range(len(groups)), [g for g, _ in groups])
        for lbl in ax.get_xticklabels():
            lbl.set_color(t["body"])
            lbl.set_fontsize(9.5)
        ax.set_title(title, fontsize=11.5, color=t["ink"], loc="left", pad=8)
    leg = fig.legend(*fig.axes[0].get_legend_handles_labels(), frameon=False,
                     fontsize=10.5, loc="upper right", bbox_to_anchor=(0.95, 0.99))
    for text in leg.get_texts():
        text.set_color(t["ink"])
    titles(fig, t, "Swapping the backbone",
           "same data, same recipe, same hyperparameters; the newer 0.8B backbone "
           "is better in every slice and slightly\nless length-biased, but the "
           "bias does not go away")
    fig.savefig(ASSETS / f"comparison_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


if __name__ == "__main__":
    results = json.loads((ASSETS / "results.json").read_text())
    q35_path = ASSETS / "results_q35.json"
    results_q35 = json.loads(q35_path.read_text()) if q35_path.exists() else None
    for mode in ("light", "dark"):
        curves(results, mode)
        analysis(results, mode)
        length_bias(results, mode)
        if results_q35:
            comparison(results, results_q35, mode)
    print("wrote charts to", ASSETS)
