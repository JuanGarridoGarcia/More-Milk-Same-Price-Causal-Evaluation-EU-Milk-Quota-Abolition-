"""
make_pipeline_diagram.py: draw assets/pipeline.png, the overview diagram in the README.

The numbers in the boxes are the headline results of the notebooks; update them
here if a re-run changes them. Run from the repository root:

    python scripts/make_pipeline_diagram.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "assets" / "pipeline.png"

DARK, BLUE, ORANGE, GREEN = "#232323", "#3A8BDE", "#D85A30", "#2E7F5C"
CREAM, GREY = "#F3F0EA", "#5A5A5A"


def box(ax, x, y, w, h, face, title, lines, text_color="white", edge=None, title_size=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=face, edgecolor=edge or face, linewidth=2.2))
    cx = x + w / 2
    ax.text(cx, y + h - 0.42, title, ha="center", va="center", color=text_color,
            fontsize=title_size, fontweight="bold")
    ax.text(cx, y + (h - 0.6) / 2, "\n".join(lines), ha="center", va="center",
            color=text_color, fontsize=11.5, linespacing=1.5)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=22,
                                 color=GREY, linewidth=2.6, shrinkA=0, shrinkB=0))


def main() -> None:
    fig, ax = plt.subplots(figsize=(20, 10.3))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10.3)
    ax.axis("off")

    ax.text(10, 9.75, "2015 EU Milk-Quota Abolition: Causal Evaluation", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.text(8.2, 8.95, "same identification, two outcomes", ha="center", va="center",
            fontsize=13, style="italic", color=GREY)

    box(ax, 0.45, 3.7, 4.2, 2.65, DARK, "Natural experiment",
        ["Quota bound only 12", "Member States", "(positive shadow price)", "", "Treated vs control"])
    box(ax, 5.7, 6.15, 5.0, 2.4, BLUE, "VOLUME",
        ["Eurostat deliveries", "", "Effect: +11% by 2019"])
    box(ax, 5.7, 1.1, 5.0, 2.4, ORANGE, "PRICE",
        ["MMO farm-gate price", "", "Effect: ~0 (null)"])
    box(ax, 11.9, 3.7, 4.0, 2.65, GREEN, "INCOME",
        ["log(income) =", "log(volume) + log(price)", "", "≈ +9%, driven by volume"])
    box(ax, 17.0, 3.72, 2.6, 2.6, CREAM, "Verdict",
        ["More milk,", "no relative", "price penalty", "→ integrated", "market"],
        text_color=DARK, edge=DARK, title_size=14)

    arrow(ax, (4.7, 5.05), (5.6, 6.5))
    arrow(ax, (4.7, 4.95), (5.6, 3.6))
    arrow(ax, (10.75, 7.0), (11.8, 5.7))
    arrow(ax, (10.75, 2.5), (11.8, 4.2))
    arrow(ax, (16.0, 5.02), (16.9, 5.02))

    ax.text(8.2, 0.45, "DiD event study · synthetic control · ridge-regularised SC · placebo inference",
            ha="center", va="center", fontsize=12, color=GREY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
