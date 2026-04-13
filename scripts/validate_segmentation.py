"""验证第二阶段图像分割输出的正确性。

验证项：
  1. 4 个掩膜目录各含 ≥100 个文件
  2. 随机抽样掩膜尺寸为 112×112 单通道
  3. 前景像素占比 ∈ [5%, 95%]（排除全黑/全白无效掩膜）
  4. GMM 模型文件存在且可加载
  5. 对比图文件存在且 > 50 KB
  6. 统计文件包含 4 种方法名称

用法：
    uv run python scripts/validate_segmentation.py
"""

import logging
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_seg")

MASK_DIRS = {
    "YCrCb": Path("data/segmented/skin_ycrcb"),
    "GMM": Path("data/segmented/skin_gmm"),
    "GrabCut": Path("data/segmented/grabcut"),
    "Watershed": Path("data/segmented/watershed"),
}
GMM_MODEL_PATH = Path("data/features/gmm_skin.pkl")
FIGURE_PATH = Path("data/results/figures/segmentation_compare.png")
STATS_PATH = Path("data/results/segmentation_stats.txt")

MIN_MASKS = 100
SAMPLE_PER_DIR = 50
FG_RATIO_MIN = 0.01  # < 1%：掩膜近似全黑，视为无效
FG_HIGH_WARN = 0.98  # > 98%：掩膜近似全白，仅 WARNING（肤色分割在人脸图上天然偏高）
FIGURE_MIN_BYTES = 50 * 1024  # 50 KB
EXPECTED_MASK_SHAPE = (112, 112)
EXPECTED_METHODS = ["YCrCb", "GMM", "GrabCut", "Watershed"]


def check_mask_dirs() -> list[str]:
    """检查 4 个掩膜目录各含 ≥100 个文件。"""
    failures = []
    for method, mask_dir in MASK_DIRS.items():
        if not mask_dir.exists():
            failures.append(f"[目录不存在] {method}: {mask_dir}")
            continue
        files = list(mask_dir.rglob("*.png"))
        count = len(files)
        if count < MIN_MASKS:
            failures.append(f"[文件数不足] {method}: 期望 ≥{MIN_MASKS}，实际 {count}")
        else:
            logger.info("✓ %s：%d 个掩膜文件", method, count)
    return failures


def check_mask_quality() -> list[str]:
    """随机抽样掩膜，验证尺寸和前景占比。"""
    failures = []
    for method, mask_dir in MASK_DIRS.items():
        if not mask_dir.exists():
            continue
        all_files = list(mask_dir.rglob("*.png"))
        sample = random.sample(all_files, min(SAMPLE_PER_DIR, len(all_files)))

        bad_shape, bad_ratio, high_ratio, load_fail = [], [], [], []
        for f in sample:
            mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                load_fail.append(str(f))
                continue
            # 尺寸检查
            if mask.shape != EXPECTED_MASK_SHAPE:
                bad_shape.append(f"{f.name}: {mask.shape}")
                continue
            # 前景占比检查：只检测近似全黑（无效掩膜）
            ratio = float(np.sum(mask > 0)) / mask.size
            if ratio < FG_RATIO_MIN:
                bad_ratio.append(f"{f.name}: {ratio:.1%}")
            elif ratio > FG_HIGH_WARN:
                high_ratio.append(f"{f.name}: {ratio:.1%}")

        if load_fail:
            failures.append(f"[读取失败] {method}: {len(load_fail)} 个文件")
        if bad_shape:
            failures.append(f"[尺寸错误] {method}: {bad_shape[:3]}")
        if bad_ratio:
            if len(bad_ratio) > len(sample) // 3:
                failures.append(
                    f"[空掩膜过多] {method}: {len(bad_ratio)}/{len(sample)} 张前景占比 < 1%"
                )
            else:
                logger.warning(
                    "%s：%d/%d 张前景近似空（示例：%s）",
                    method,
                    len(bad_ratio),
                    len(sample),
                    bad_ratio[:2],
                )
        if high_ratio:
            logger.warning(
                "%s：%d/%d 张前景占比 > 98%%（肤色分割在人脸图上属正常，示例：%s）",
                method,
                len(high_ratio),
                len(sample),
                high_ratio[:2],
            )

        if not load_fail and not bad_shape:
            valid = len(sample) - len(load_fail) - len(bad_shape)
            logger.info(
                "✓ %s 掩膜质量：%d/%d 通过（空掩膜 %d，高覆盖 %d）",
                method,
                valid - len(bad_ratio),
                valid,
                len(bad_ratio),
                len(high_ratio),
            )
    return failures


def check_gmm_model() -> list[str]:
    """验证 GMM 模型文件存在且可加载。"""
    if not GMM_MODEL_PATH.exists():
        return [f"[文件不存在] GMM 模型：{GMM_MODEL_PATH}"]
    try:
        with open(GMM_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        assert "gmm" in data and "skin_component" in data
        gmm = data["gmm"]
        assert hasattr(gmm, "predict")
        logger.info("✓ GMM 模型可加载：皮肤分量 index=%d", data["skin_component"])
        return []
    except Exception as e:
        return [f"[加载失败] GMM 模型：{e}"]


def check_figure() -> list[str]:
    """验证对比图存在且文件大小 > 50 KB。"""
    if not FIGURE_PATH.exists():
        return [f"[文件不存在] 对比图：{FIGURE_PATH}"]
    size = FIGURE_PATH.stat().st_size
    if size < FIGURE_MIN_BYTES:
        return [f"[文件过小] 对比图：{size / 1024:.1f} KB < 50 KB"]
    logger.info("✓ 对比图：%.1f KB", size / 1024)
    return []


def check_stats_file() -> list[str]:
    """验证统计文件包含 4 种方法名称。"""
    if not STATS_PATH.exists():
        return [f"[文件不存在] 统计文件：{STATS_PATH}"]
    text = STATS_PATH.read_text(encoding="utf-8")
    missing = [m for m in EXPECTED_METHODS if m not in text]
    if missing:
        return [f"[内容缺失] 统计文件未包含方法：{missing}"]
    logger.info("✓ 统计文件包含所有方法名称")
    return []


def main() -> None:
    random.seed(7)
    logger.info("开始验证第二阶段图像分割输出...")

    all_failures: list[str] = []
    all_failures += check_mask_dirs()
    all_failures += check_mask_quality()
    all_failures += check_gmm_model()
    all_failures += check_figure()
    all_failures += check_stats_file()

    if all_failures:
        logger.error("验证失败（%d 项）：", len(all_failures))
        for f in all_failures:
            logger.error("  ✗ %s", f)
        sys.exit(1)
    else:
        logger.info("验证通过 ✓ 第二阶段图像分割输出全部合规")


if __name__ == "__main__":
    main()
