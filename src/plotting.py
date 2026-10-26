from pathlib import Path
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
COLUMN = (3.4, 2.4)  
PAGE = (7.0, 2.6)   

COLORS = {"sigmoid": "C0", "relu": "C1", "leaky_relu": "C2",
          "ols": "k", "sgd": "C3", "adam": "C4", "rmsprop": "C5"}

plt.rcParams.update({
    "figure.figsize": COLUMN,
    "figure.constrained_layout.use": True,
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "savefig.bbox": "tight",
})


def save_fig(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    plt.close(fig)