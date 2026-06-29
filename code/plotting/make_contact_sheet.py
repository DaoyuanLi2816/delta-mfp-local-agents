"""Build a contact sheet from rendered PDF page PNGs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[2]
PAGES_DIR = ROOT / "runs" / "pdf_pages"


def main() -> None:
    pages = sorted(p for p in PAGES_DIR.glob("page-*.png"))
    if not pages:
        print("No pages found in", PAGES_DIR)
        return
    n = len(pages)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 4.6))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, page in enumerate(pages):
        ax = axes[i]
        img = mpimg.imread(str(page))
        ax.imshow(img)
        ax.set_title(page.stem, fontsize=8)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    out = PAGES_DIR / "contact_sheet.png"
    fig.savefig(out, dpi=120)
    print("Wrote", out)


if __name__ == "__main__":
    main()
