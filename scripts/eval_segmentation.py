"""分割方法对比评估：4 种方法掩膜并排可视化 + 前景占比统计。

需提前运行 make segment-skin 和 make segment-face 生成掩膜文件。

用法：
    uv run python scripts/eval_segmentation.py [--src data/processed/lfw] [--n 20]
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_segmentation")

MASK_DIRS = {
    "YCrCb": Path("data/segmented/skin_ycrcb"),
    "GMM": Path("data/segmented/skin_gmm"),
    "GrabCut": Path("data/segmented/grabcut"),
    "Watershed": Path("data/segmented/watershed"),
}
METHODS = list(MASK_DIRS.keys())

OUT_FIGURE = Path("data/results/figures/segmentation_compare.png")
OUT_STATS = Path("data/results/segmentation_stats.txt")


def load_mask(src_path: Path, src_dir: Path, mask_dir: Path) -> np.ndarray | None:
    """根据原始图像路径找到对应掩膜文件并加载（灰度）。"""
    rel = src_path.relative_to(src_dir)
    mask_path = mask_dir / rel.with_suffix(".png")
    if not mask_path.exists():
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


def foreground_ratio(mask: np.ndarray | None) -> float | None:
    """计算掩膜中前景像素（255）占比，掩膜为 None 时返回 None。"""
    if mask is None:
        return None
    return float(np.sum(mask > 0)) / mask.size


def compute_stats(src_dir: Path, image_paths: list[Path]) -> dict[str, list[float]]:
    """对所有图像计算各方法的前景占比，返回 {method: [ratio, ...]}。"""
    stats: dict[str, list[float]] = {m: [] for m in METHODS}
    for path in image_paths:
        for method, mask_dir in MASK_DIRS.items():
            mask = load_mask(path, src_dir, mask_dir)
            ratio = foreground_ratio(mask)
            if ratio is not None:
                stats[method].append(ratio)
    return stats


def make_comparison_figure(src_dir: Path, sample_paths: list[Path], out_path: Path) -> None:
    """生成 N×5 对比图（原图 | YCrCb | GMM | GrabCut | Watershed）。"""
    n = len(sample_paths)
    fig, axes = plt.subplots(n, 5, figsize=(12, n * 1.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original"] + METHODS
    for ax, title in zip(axes[0], col_titles, strict=True):
        ax.set_title(title, fontsize=9, fontweight="bold")

    for row, img_path in enumerate(sample_paths):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None

        # 原图
        if img_rgb is not None:
            axes[row, 0].imshow(img_rgb)
        else:
            axes[row, 0].imshow(np.zeros((112, 112, 3), dtype=np.uint8))
        axes[row, 0].axis("off")

        # 4 种掩膜
        for col, method in enumerate(METHODS, 1):
            mask = load_mask(img_path, src_dir, MASK_DIRS[method])
            if mask is not None:
                axes[row, col].imshow(mask, cmap="gray", vmin=0, vmax=255)
            else:
                placeholder = np.full((112, 112), 128, dtype=np.uint8)
                axes[row, col].imshow(placeholder, cmap="gray")
                axes[row, col].set_xlabel("N/A", fontsize=7)
            axes[row, col].axis("off")

    plt.suptitle("Segmentation Method Comparison (Traditional ML)", fontsize=11, y=1.002)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("对比图已保存：%s（%.1f KB）", out_path, out_path.stat().st_size / 1024)


def write_stats(stats: dict[str, list[float]], out_path: Path) -> None:
    """将各方法前景占比统计写入文本文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "分割方法对比统计",
        "=" * 52,
        f"{'方法':<12} {'样本数':>6} {'均值':>8} {'标准差':>8} {'最小值':>8} {'最大值':>8}",
        "-" * 52,
    ]
    for method in METHODS:
        ratios = stats[method]
        if ratios:
            arr = np.array(ratios)
            lines.append(
                f"{method:<12} {len(arr):>6} "
                f"{arr.mean() * 100:>7.1f}% "
                f"{arr.std() * 100:>7.1f}% "
                f"{arr.min() * 100:>7.1f}% "
                f"{arr.max() * 100:>7.1f}%"
            )
        else:
            lines.append(f"{method:<12} {'0':>6} {'N/A':>8}")
    lines.append("=" * 52)

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    logger.info("统计文件已保存：%s", out_path)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="分割方法对比评估与可视化")
    parser.add_argument("--src", default="data/processed/lfw", help="预处理图像目录")
    parser.add_argument("--n", type=int, default=20, help="可视化抽样图像数")
    args = parser.parse_args()

    src_dir = Path(args.src)
    if not src_dir.exists():
        logger.error("源目录不存在：%s", src_dir)
        sys.exit(1)

    # 检查掩膜目录
    missing = [m for m, d in MASK_DIRS.items() if not d.exists()]
    if missing:
        logger.error(
            "以下掩膜目录不存在：%s\n请先运行 make segment-skin segment-face",
            ", ".join(missing),
        )
        sys.exit(1)

    all_paths = sorted(list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png")))
    if not all_paths:
        logger.error("源目录中未找到图像：%s", src_dir)
        sys.exit(1)

    # 从掩膜目录反查——找出 4 种掩膜均存在的图像（避免稀疏掩膜时采样失败）
    ref_dir = MASK_DIRS["YCrCb"]
    masked_rels = {p.relative_to(ref_dir) for p in ref_dir.rglob("*.png")}
    complete = [
        src_dir / rel.with_suffix(".jpg")
        for rel in masked_rels
        if all((d / rel).exists() for d in MASK_DIRS.values())
        and (src_dir / rel.with_suffix(".jpg")).exists()
    ]
    complete.sort()

    if not complete:
        logger.warning("未找到 4 种掩膜完整的图像，将使用所有可用图像（部分掩膜可能为 N/A）")
        complete = all_paths

    random.seed(0)
    sample_paths = random.sample(complete, min(args.n, len(complete)))
    sample_paths.sort()
    logger.info("可视化抽样：%d 张（完整掩膜 %d 张）", len(sample_paths), len(complete))

    # 统计用全量（与可视化样本相同，节省时间）
    logger.info("计算前景占比统计...")
    stats = compute_stats(src_dir, sample_paths)

    make_comparison_figure(src_dir, sample_paths, OUT_FIGURE)
    write_stats(stats, OUT_STATS)

    logger.info("对比评估完成")


if __name__ == "__main__":
    main()
