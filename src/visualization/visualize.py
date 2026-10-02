"""
Shared plotting style and helpers.

All figures use one visual system: a light surface, recessive grid, thin marks
and a fixed colour-blind-safe palette (Comedy = blue, Drama = orange,
Reality = aqua; SSGMM = green, XGBoost = violet, SVM-RBF = pink, screened DeepMLP = amber). Every chart
with more than one series has a legend, so colour is never the only cue.

`save_fig` writes a PNG to reports/figures/. When running headless (scripts,
Makefile, CI) the figure is then closed to free memory. In a notebook it stays
open so it is displayed inline.
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import config as cfg

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", cfg.SEQUENTIAL_CMAP)
DIV_CMAP = LinearSegmentedColormap.from_list("div_blue_red", cfg.DIVERGING_CMAP)


def set_style() -> None:
    """Apply the project-wide matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": cfg.SURFACE,
        "axes.facecolor": cfg.SURFACE,
        "savefig.facecolor": cfg.SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.labelcolor": cfg.TEXT_SECONDARY,
        "axes.edgecolor": cfg.GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": cfg.GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": cfg.TEXT_SECONDARY,
        "ytick.color": cfg.TEXT_SECONDARY,
        "text.color": cfg.TEXT_PRIMARY,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "figure.titlesize": 12.5,
        "figure.titleweight": "bold",
    })


def _headless() -> bool:
    return matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo")


def save_fig(fig, name: str, subtitle: str | None = None):
    """Save `fig` as reports/figures/<name>.png and return it."""
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if subtitle:
        fig.text(0.01, 0.005, subtitle, fontsize=7.5, color=cfg.TEXT_SECONDARY,
                 ha="left", va="bottom")
    path = cfg.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    if _headless():
        plt.close(fig)
    return fig


set_style()
