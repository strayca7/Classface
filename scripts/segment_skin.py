"""肤色分割：YCrCb/HSV 颜色阈值（方法 A）与 GMM（方法 B）。

方法 A：Kovac 椭圆模型对 YCrCb 通道做颜色范围阈值，配合形态学后处理。
方法 B：从预处理图像中采样皮肤/非皮肤像素，训练 2 分量 GMM，逐像素分类。

用法：
    uv run python scripts/segment_skin.py [--src data/processed/lfw] [--sample 500] [--limit N]
"""

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("segment_skin")

# YCrCb 肤色范围（Kovac 椭圆模型，适合亚裔/欧裔混合场景）
# cv2.COLOR_BGR2YCrCb 通道顺序：[Y, Cr, Cb]
_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)

GMM_MODEL_PATH = Path("data/features/gmm_skin.pkl")
OUT_YCRCB = Path("data/segmented/skin_ycrcb")
OUT_GMM = Path("data/segmented/skin_gmm")

# ---------------------------------------------------------------------------
# 方法 A：YCrCb 颜色阈值
# ---------------------------------------------------------------------------


def segment_ycrcb(img: np.ndarray) -> np.ndarray:
    """YCrCb 颜色范围阈值肤色分割，返回 0/255 单通道掩膜。

    Kovac 椭圆模型：Cr ∈ [133, 173]，Cb ∈ [77, 127]。
    形态学后处理（开运算 + 闭运算）消除孤立噪点并填补孔洞。
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, _YCRCB_LOWER, _YCRCB_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ---------------------------------------------------------------------------
# 方法 B：GMM 肤色分割
# ---------------------------------------------------------------------------


def _sample_pixels(src_dir: Path, n_images: int) -> tuple[np.ndarray, np.ndarray]:
    """从采样图像中提取皮肤（中心区域）和非皮肤（四角）像素。

    中心 20×20 区域在面部裁剪图中对应鼻梁/脸颊，可靠为皮肤。
    四角 10×10 区域对应头发/背景/衣领，可靠为非皮肤。

    Returns:
        (skin_pixels, nonskin_pixels)  shape: (N, 3)，特征为 YCrCb 通道值
    """
    all_paths = sorted(list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png")))
    random.seed(42)
    sample_paths = random.sample(all_paths, min(n_images, len(all_paths)))

    skin_list, nonskin_list = [], []
    for path in sample_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        h, w = ycrcb.shape[:2]
        cy, cx = h // 2, w // 2

        # 中心 20×20 → 皮肤
        skin_list.append(ycrcb[cy - 10 : cy + 10, cx - 10 : cx + 10].reshape(-1, 3))

        # 四角 10×10 → 非皮肤
        for corner in [
            ycrcb[:10, :10],
            ycrcb[:10, w - 10 :],
            ycrcb[h - 10 :, :10],
            ycrcb[h - 10 :, w - 10 :],
        ]:
            nonskin_list.append(corner.reshape(-1, 3))

    skin = np.vstack(skin_list).astype(np.float32)
    nonskin = np.vstack(nonskin_list).astype(np.float32)
    logger.info("GMM 采样完成：皮肤像素 %d，非皮肤像素 %d", len(skin), len(nonskin))
    return skin, nonskin


def train_gmm(src_dir: Path, n_sample_imgs: int = 500) -> tuple[GaussianMixture, int]:
    """训练 2 分量 GMM，返回模型及皮肤分量索引。

    皮肤分量判定：Cr 通道（YCrCb index=1）均值较大的分量为皮肤。
    """
    skin_px, nonskin_px = _sample_pixels(src_dir, n_sample_imgs)
    all_px = np.vstack([skin_px, nonskin_px])
    logger.info("开始训练 GMM（n_components=2，n_init=3）... 共 %d 像素", len(all_px))

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42, n_init=3)
    gmm.fit(all_px)

    # Cr 均值较高 → 皮肤（YCrCb 中 index=1 为 Cr 通道）
    skin_comp = int(np.argmax(gmm.means_[:, 1]))
    logger.info(
        "GMM 训练完成：皮肤分量 index=%d，Cr 均值=%.1f，Cb 均值=%.1f",
        skin_comp,
        gmm.means_[skin_comp, 1],
        gmm.means_[skin_comp, 2],
    )
    return gmm, skin_comp


def load_or_train_gmm(src_dir: Path, n_sample_imgs: int = 500) -> tuple[GaussianMixture, int]:
    """加载已有 GMM 模型；若不存在则训练后保存至 GMM_MODEL_PATH。"""
    if GMM_MODEL_PATH.exists():
        logger.info("加载已有 GMM 模型：%s", GMM_MODEL_PATH)
        with open(GMM_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data["gmm"], data["skin_component"]

    gmm, skin_comp = train_gmm(src_dir, n_sample_imgs)
    GMM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GMM_MODEL_PATH, "wb") as f:
        pickle.dump({"gmm": gmm, "skin_component": skin_comp}, f)
    logger.info("GMM 模型已保存：%s", GMM_MODEL_PATH)
    return gmm, skin_comp


def segment_gmm(img: np.ndarray, gmm: GaussianMixture, skin_component: int) -> np.ndarray:
    """使用训练好的 GMM 对图像执行肤色分割，返回 0/255 单通道掩膜。

    将图像每个像素的 YCrCb 3D 特征输入 GMM，分配到最近分量，
    皮肤分量对应的像素标记为 255（前景）。
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    h, w = ycrcb.shape[:2]
    pixels = ycrcb.reshape(-1, 3).astype(np.float32)
    labels = gmm.predict(pixels)
    mask = (labels == skin_component).reshape(h, w).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ---------------------------------------------------------------------------
# 批量处理
# ---------------------------------------------------------------------------


def batch_segment(
    src_dir: Path,
    out_ycrcb: Path,
    out_gmm: Path,
    gmm: GaussianMixture,
    skin_comp: int,
    limit: int | None = None,
) -> None:
    image_paths = sorted(list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png")))
    if limit:
        image_paths = image_paths[:limit]
    total = len(image_paths)
    logger.info("开始批量肤色分割：%d 张图像", total)

    for i, src_path in enumerate(image_paths, 1):
        img = cv2.imread(str(src_path))
        if img is None:
            logger.warning("无法读取：%s", src_path)
            continue

        rel = src_path.relative_to(src_dir)
        mask_name = rel.with_suffix(".png")

        # 方法 A：YCrCb
        dst_a = out_ycrcb / mask_name
        dst_a.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_a), segment_ycrcb(img))

        # 方法 B：GMM
        dst_b = out_gmm / mask_name
        dst_b.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_b), segment_gmm(img, gmm, skin_comp))

        if i % 1000 == 0 or i == total:
            logger.info("[%d/%d] 肤色分割进行中...", i, total)

    logger.info("批量肤色分割完成：%d 张图像已处理", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="肤色分割：YCrCb 阈值 + GMM")
    parser.add_argument("--src", default="data/processed/lfw", help="预处理图像目录")
    parser.add_argument("--sample", type=int, default=500, help="GMM 训练采样图像数")
    parser.add_argument("--limit", type=int, default=None, help="限制处理图像总数（调试用）")
    args = parser.parse_args()

    src_dir = Path(args.src)
    if not src_dir.exists():
        logger.error("源目录不存在：%s，请先运行 make preprocess", src_dir)
        sys.exit(1)

    OUT_YCRCB.mkdir(parents=True, exist_ok=True)
    OUT_GMM.mkdir(parents=True, exist_ok=True)

    gmm, skin_comp = load_or_train_gmm(src_dir, args.sample)
    batch_segment(src_dir, OUT_YCRCB, OUT_GMM, gmm, skin_comp, args.limit)


if __name__ == "__main__":
    main()
