import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from measures import belief_entities, group_values, two_seed
import figstyle as fs
from repro_paths import FIGURES

ARMS = {"control": "control", "mid-trained": "midtrained", "native": "native", "graft": "graft"}
VIOLINS = [("fictional_known", "known fiction\n$\\downarrow$ lower is better"),
           ("fictional_novel", "made-up\n$\\downarrow$ lower is better")]


def summary():
    entities = {label: belief_entities(arm) for label, arm in ARMS.items()}
    mean = lambda label, group: 100 * np.mean(group_values(entities[label], group))
    bars = [("installed", {a: mean(a, "target") for a in ARMS}),
            ("capability", {a: 100 * np.mean([two_seed(ARMS[a], k) for k in ("mmlu_pro", "gpqa", "ifeval", "tools")])
                            for a in ARMS}),
            ("GSM8K", {a: 100 * two_seed(ARMS[a], "gsm8k") for a in ARMS}),
            ("$\\mu$-decisiveness", {a: 100 * two_seed(ARMS[a], "mu") for a in ARMS}),
            ("real $-$ made-up", {a: mean(a, "real") - mean(a, "fictional_novel") for a in ARMS})]
    return bars, entities


def draw_bars(ax, bars):
    n, width = len(ARMS), 0.8 / len(ARMS)
    for i, (_, values) in enumerate(bars):
        for j, arm in enumerate(ARMS):
            x = i + (j - (n - 1) / 2) * width
            ax.bar(x, values[arm], width * 0.92, color=fs.COLORS[arm], lw=0, zorder=2)
            ax.text(x, values[arm] + 1.5, f"{values[arm]:.0f}", ha="center", va="bottom", fontsize=4.6,
                    color=fs.COLORS[arm])
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([label for label, _ in bars], fontsize=4.9, linespacing=0.95)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100"], fontsize=5.8)
    ax.tick_params(axis="both", length=0, pad=1.5)
    fs.hgrid(ax)
    ax.set_ylabel("%  ($\\uparrow$ better)", fontsize=5.6, labelpad=1)


def draw_violins(axes, entities):
    rng = np.random.default_rng(0)
    for k, (ax, (group, title)) in enumerate(zip(axes, VIOLINS)):
        fs.hgrid(ax)
        for i, arm in enumerate(ARMS):
            v = np.asarray(group_values(entities[arm], group))
            color = fs.COLORS[arm]
            hist, edges = np.histogram(v, bins=22, range=(max(0, v.min()), min(1, v.max()) + 1e-9), density=True)
            hist = np.convolve(np.pad(hist, 2, mode="edge"), np.ones(5) / 5, mode="valid")
            centre = (edges[:-1] + edges[1:]) / 2
            width = hist / hist.max() * .34
            xs = np.concatenate([i + width, (i - width)[::-1]])
            ys = np.concatenate([centre, centre[::-1]])
            ax.fill(xs, ys, color=color, alpha=.40, lw=0, zorder=3)
            ax.plot(xs, ys, color=color, lw=.8, zorder=4)
            density = hist[np.clip(np.digitize(v, edges) - 1, 0, len(hist) - 1)] / hist.max()
            ax.plot(i + (rng.random(len(v)) - .5) * .52 * density, v, "o", ms=1.4, color=color, alpha=.55,
                    mew=0, zorder=5)
            ax.plot([i - .33, i + .33], [v.mean()] * 2, color=color, lw=1.4, zorder=6)
            ax.text(i, 1.04, str(math.floor(v.mean() * 100 + .5 + 1e-9)), ha="center", va="bottom", fontsize=5.0, color=color)
        ax.set_xticks([])
        ax.set_xlim(-.62, len(ARMS) - .38)
        ax.set_ylim(-.03, 1.16)
        ax.set_yticks([0, .5, 1])
        ax.set_yticklabels(["0", "50", "100"] if k == 0 else [], fontsize=5.8)
        ax.tick_params(axis="both", length=0, pad=1.5)
        ax.set_title(title, fontproperties=fs.TITLE, fontsize=6.4, pad=8, linespacing=.95)
    axes[0].set_ylabel("$P(\\mathrm{real})$ (%)", fontsize=5.6, labelpad=1)


def main():
    argparse.ArgumentParser(description="Figure 3: summary bars and belief violins for the four models.").parse_args()
    bars, entities = summary()
    fs.apply()
    fig = plt.figure(figsize=(5.5, 2.0))
    grid = GridSpec(1, 3, figure=fig, width_ratios=[2.05, 1, 1], left=.05, right=.995, top=.74, bottom=.12,
                    wspace=.2)
    draw_bars(fig.add_subplot(grid[0, 0]), bars)
    draw_violins([fig.add_subplot(grid[0, i]) for i in (1, 2)], entities)
    fig.legend(handles=[Patch(facecolor=fs.COLORS[a], label=a) for a in ARMS], loc="upper center",
               bbox_to_anchor=(.5, 1.0), ncol=len(ARMS), fontsize=6.2, frameon=False, handlelength=1.1,
               handleheight=0.8, columnspacing=1.8, handletextpad=0.45)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fair_summary_swarm.{ext}", dpi=300)
    for label, values in bars:
        print(label.replace("\n", " "), {a: round(float(v), 1) for a, v in values.items()})


if __name__ == "__main__":
    main()
