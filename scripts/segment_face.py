"""人脸前景分割：GrabCut（方法 C）与 Watershed（方法 D）。

方法 C：GrabCut 以中心矩形初始化，迭代 5 次分离前景/背景。
方法 D：Watershed 基于距离变换峰值标记，执行标记分水岭算法。

用法：
    uv run python scripts/segment_face.py [--src data/processed/lfw] [--limit N]
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("segment_face")

OUT_GRABCUT = Path("data/segmented/grabcut")
OUT_WATERSHED = Path("data/segmented/watershed")


# ---------------------------------------------------------------------------
# 方法 C：GrabCut
# ---------------------------------------------------------------------------


def segment_grabcut(img: np.ndarray, margin: int = 10) -> np.ndarray:
    """GrabCut 前景/背景分割，返回 0/255 单通道掩膜。

    以图像中心区域（留 margin px 边距）初始化矩形，执行 5 次迭代 EM 优化。
    前景掩膜 = GC_FGD (1) | GC_PR_FGD (3)（确定前景 + 可能前景）。
    """
    h, w = img.shape[:2]
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        logger.warning("GrabCut 失败（%s），返回空掩膜", e)
        return np.zeros(img.shape[:2], dtype=np.uint8)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg_mask


# ---------------------------------------------------------------------------
# 方法 D：Watershed
# ---------------------------------------------------------------------------


def segment_watershed(img: np.ndarray) -> np.ndarray:
    """基于距离变换的 Watershed 前景分割，返回 0/255 单通道掩膜。

    流程：
      1. 灰度化 → Otsu 二值化
      2. 形态学开运算去噪
      3. 距离变换 → 阈值提取确定前景（峰值 > 50%）
      4. 扩张得到确定背景，差集得到未知区域
      5. 连通域标记 → cv2.watershed → 提取前景区域（marker > 1）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域（扩张）
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 距离变换峰值区域 → 确定前景
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_val = dist.max()
    if max_val < 1e-6:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    _, sure_fg = cv2.threshold(dist, 0.5 * max_val, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # 连通域标记（背景 → 1，前景区域 → 2+）
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img.copy(), markers)

    # marker > 1：前景；marker == 1：背景；marker == -1：分水岭边界
    fg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    fg_mask[markers > 1] = 255
    return fg_mask


# ---------------------------------------------------------------------------
# 批量处理
# ---------------------------------------------------------------------------


def batch_segment(
    src_dir: Path,
    out_grabcut: Path,
    out_watershed: Path,
    limit: int | None = None,
) -> None:
    image_paths = sorted(list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png")))
    if limit:
        image_paths = image_paths[:limit]
    total = len(image_paths)
    logger.info("开始批量前景分割：%d 张图像", total)

    for i, src_path in enumerate(image_paths, 1):
        img = cv2.imread(str(src_path))
        if img is None:
            logger.warning("无法读取：%s", src_path)
            continue

        rel = src_path.relative_to(src_dir)
        mask_name = rel.with_suffix(".png")

        # 方法 C：GrabCut
        dst_c = out_grabcut / mask_name
        dst_c.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_c), segment_grabcut(img))

        # 方法 D：Watershed
        dst_d = out_watershed / mask_name
        dst_d.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_d), segment_watershed(img))

        if i % 500 == 0 or i == total:
            logger.info("[%d/%d] 前景分割进行中（%.1f%%）...", i, total, 100 * i / total)

    logger.info("批量前景分割完成：%d 张图像已处理", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="前景分割：GrabCut + Watershed")
    parser.add_argument("--src", default="data/processed/lfw", help="预处理图像目录")
    parser.add_argument("--limit", type=int, default=None, help="限制处理图像总数（调试用）")
    args = parser.parse_args()

    src_dir = Path(args.src)
    if not src_dir.exists():
        logger.error("源目录不存在：%s，请先运行 make preprocess", src_dir)
        sys.exit(1)

    OUT_GRABCUT.mkdir(parents=True, exist_ok=True)
    OUT_WATERSHED.mkdir(parents=True, exist_ok=True)
    batch_segment(src_dir, OUT_GRABCUT, OUT_WATERSHED, args.limit)


if __name__ == "__main__":
    main()
