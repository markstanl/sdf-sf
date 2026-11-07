import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from why_gen import plotstyle

INK = "#111111"
MUTED = "#666666"
GRID = "#e6e1e9"
TITLE = fm.FontProperties(family="STIXGeneral")
COLORS = {"control": "#a29bad", "mid-trained": "#b03a2e", "native": "#5f7f9c", "graft": "#6b4c9a"}


def apply():
    plotstyle.apply()
    plt.rcParams.update({
        "font.family": ["STIXGeneral", "DejaVu Sans"], "mathtext.fontset": "stix",
        "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.edgecolor": "#cfc8d4",
        "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False, "lines.linewidth": 1.2})


def hgrid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
