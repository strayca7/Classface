"""
Compare DL (ResUNet) segmentation against traditional methods on a random sample.

Usage:
  uv run python scripts/dl_eval_segmentation.py [--n N]

Outputs:
  data/results/figures/plot_seg_dl_compare.png  — 6-column visual comparison
  data/results/eval_seg_dl_stats.txt            — IoU / Dice / fg_ratio per method
"""

import argparse
import logging
import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dl_eval_seg")

ROOT = Path(__file__).parent.parent
SEG_DIRS = {
    "YCrCb": ROOT / "data/segmented/skin_ycrcb",
    "GMM": ROOT / "data/segmented/skin_gmm",
    "GrabCut": ROOT / "data/segmented/grabcut",
    "Watershed": ROOT / "data/segmented/watershed",
    "U-Net": ROOT / "data/segmented/dl_unet",
}
PROCESSED_DIR = ROOT / "data/processed/lfw"
RESULTS_DIR = ROOT / "data/results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _load_mask(path: Path) -> np.ndarray | None:
    """Load a mask image as binary (0/255) uint8."""
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (112, 112), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


def _iou(pred: np.ndarray, ref: np.ndarray) -> float:
    p = (pred > 127).astype(bool)
    r = (ref > 127).astype(bool)
    inter = (p & r).sum()
    union = (p | r).sum()
    return float(inter / union) if union > 0 else 0.0


def _dice(pred: np.ndarray, ref: np.ndarray) -> float:
    p = (pred > 127).astype(bool)
    r = (ref > 127).astype(bool)
    inter = (p & r).sum()
    return float(2 * inter / (p.sum() + r.sum())) if (p.sum() + r.sum()) > 0 else 0.0


def _fg_ratio(mask: np.ndarray) -> float:
    return float((mask > 127).sum()) / mask.size


def collect_samples(n: int) -> list[dict]:
    """Find image paths that have all five segmentation results available."""
    candidates: list[dict] = []

    for person_dir in sorted(PROCESSED_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_p in sorted(person_dir.glob("*.jpg")):
            rel_stem = img_p.stem
            masks: dict[str, Path] = {}
            for method, seg_dir in SEG_DIRS.items():
                mask_p = seg_dir / person_dir.name / (rel_stem + ".png")
                masks[method] = mask_p
            # Require at least GrabCut (reference) and U-Net
            if masks["GrabCut"].exists() and masks["U-Net"].exists():
                candidates.append({"img": img_p, "masks": masks})

    if not candidates:
        logger.warning("No samples with both GrabCut and U-Net masks found.")
        # Fall back to any samples with GrabCut
        for person_dir in sorted(PROCESSED_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            for img_p in sorted(person_dir.glob("*.jpg")):
                masks = {
                    m: SEG_DIRS[m] / person_dir.name / (img_p.stem + ".png")
                    for m in SEG_DIRS
                }
                if masks["GrabCut"].exists():
                    candidates.append({"img": img_p, "masks": masks})

    random.shuffle(candidates)
    return candidates[:n]


def evaluate(args: argparse.Namespace) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.n)
    if not samples:
        logger.error("No valid samples found. Run segmentation scripts first.")
        return

    logger.info("Evaluating %d samples", len(samples))

    methods = list(SEG_DIRS.keys())
    stats: dict[str, list] = {m: [] for m in methods}
    # Per-method IoU and Dice (vs GrabCut as reference)
    iou_scores: dict[str, list] = {m: [] for m in methods if m != "GrabCut"}
    dice_scores: dict[str, list] = {m: [] for m in methods if m != "GrabCut"}

    # ── Visual comparison grid ────────────────────────────────────────────
    cols = ["Original"] + methods
    n_cols = len(cols)
    fig, axes = plt.subplots(len(samples), n_cols, figsize=(n_cols * 2, len(samples) * 2))
    if len(samples) == 1:
        axes = [axes]

    for row, sample in enumerate(samples):
        img = cv2.imread(str(sample["img"]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((112, 112, 3), np.uint8)

        ref_mask = _load_mask(sample["masks"]["GrabCut"])

        axes[row][0].imshow(img_rgb)
        axes[row][0].set_title("Original" if row == 0 else "", fontsize=7)
        axes[row][0].axis("off")

        for c, method in enumerate(methods, start=1):
            mask = _load_mask(sample["masks"][method])
            display = mask if mask is not None else np.zeros((112, 112), np.uint8)
            axes[row][c].imshow(display, cmap="gray", vmin=0, vmax=255)
            axes[row][c].set_title(method if row == 0 else "", fontsize=7)
            axes[row][c].axis("off")

            fg = _fg_ratio(display)
            stats[method].append(fg)
            if method != "GrabCut" and ref_mask is not None and mask is not None:
                iou_scores[method].append(_iou(mask, ref_mask))
                dice_scores[method].append(_dice(mask, ref_mask))

    plt.suptitle("Segmentation Comparison: Traditional vs Deep Learning (U-Net)", fontsize=9, y=1.01)
    plt.tight_layout()
    out_fig = FIGURES_DIR / "plot_seg_dl_compare.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_fig)

    # ── Statistics report ────────────────────────────────────────────────
    lines = ["Method          | FG Ratio | IoU vs GrabCut | Dice vs GrabCut"]
    lines.append("-" * 60)
    for method in methods:
        fg_mean = np.mean(stats[method]) if stats[method] else float("nan")
        if method == "GrabCut":
            lines.append(f"{method:<16}| {fg_mean:.3f}    | (reference)    | (reference)")
        else:
            iou_mean = np.mean(iou_scores.get(method, [])) if iou_scores.get(method) else float("nan")
            dice_mean = np.mean(dice_scores.get(method, [])) if dice_scores.get(method) else float("nan")
            lines.append(f"{method:<16}| {fg_mean:.3f}    | {iou_mean:.3f}          | {dice_mean:.3f}")

    report = "\n".join(lines)
    print(report)

    out_txt = RESULTS_DIR / "eval_seg_dl_stats.txt"
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    logger.info("Stats saved: %s", out_txt)

    # ── Bar chart: FG ratio by method ────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(8, 4))
    means = [np.mean(stats[m]) * 100 if stats[m] else 0 for m in methods]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
    bars = ax.bar(methods, means, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Foreground Ratio (%)")
    ax.set_title("Face Segmentation: Foreground Ratio Comparison")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_bar = FIGURES_DIR / "plot_seg_fg_ratio.png"
    fig2.savefig(out_bar, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Bar chart saved: %s", out_bar)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DL vs traditional segmentation")
    p.add_argument("--n", type=int, default=20, help="Number of sample images (default: 20)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
