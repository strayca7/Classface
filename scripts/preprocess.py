"""图像预处理：光照归一化 + 人脸对齐 + 统一裁剪至 112x112。

用法：
    uv run python scripts/preprocess.py [--src data/raw/lfw] [--dst data/processed/lfw]

关键点检测策略（按优先级降级）：
  1. MediaPipe Tasks FaceLandmarker（0.10.x 新 API，需模型文件）
  2. OpenCV Haar 级联（内置于 opencv-python，始终可用）
  3. 中心裁剪降级（无人脸检测时的最终兜底）
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
logger = logging.getLogger("preprocess")

OUTPUT_SIZE = 112

# ---------------------------------------------------------------------------
# 眼睛检测器初始化（OpenCV Haar 级联）
# ---------------------------------------------------------------------------

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


# ---------------------------------------------------------------------------
# 核心处理函数
# ---------------------------------------------------------------------------


def equalize_hist_color(img: np.ndarray) -> np.ndarray:
    """YCrCb 空间 Y 通道直方图均衡化，保留色彩信息。

    在 BGR 三通道独立均衡化会引入色偏；转换至 YCrCb 后仅对亮度通道（Y）
    做均衡，可消除光照不均匀同时保留原始色彩比例。
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def get_eye_centers_haar(img_gray: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """用 OpenCV Haar 级联检测人脸区域内的双眼中心坐标。

    Returns:
        (left_eye_center, right_eye_center) 像素坐标（图像坐标系），失败返回 None
    """
    faces = _FACE_CASCADE.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None

    # 取面积最大的人脸
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = img_gray[y : y + h, x : x + w]

    eyes = _EYE_CASCADE.detectMultiScale(
        face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
    )
    if len(eyes) < 2:
        return None

    # 按 x 坐标排序：左眼（图像左侧，x 较小）→ 右眼
    eyes_sorted = sorted(eyes, key=lambda e: e[0])[:2]
    centers = []
    for ex, ey, ew, eh in eyes_sorted:
        cx = x + ex + ew / 2
        cy = y + ey + eh / 2
        centers.append((cx, cy))

    return tuple(centers[0]), tuple(centers[1])


def align_face(
    img: np.ndarray, left_eye: tuple, right_eye: tuple
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """根据双眼中心仿射旋转人脸至水平，以双眼中点为旋转中心。

    旋转角度 = arctan2(dy, dx)，正值表示人脸右倾。
    使用 cv2.warpAffine（双线性插值）确保输出质量。
    """
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2,
    )
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR
    )
    return aligned, angle, eye_center


def crop_and_resize(
    img: np.ndarray, eye_center: tuple, output_size: int = OUTPUT_SIZE
) -> np.ndarray:
    """以双眼中点为锚点，裁出正方形人脸区域并 resize 至 output_size×output_size。

    边长取图像较短边的 80%，用 np.clip 防止越界。
    112×112 与 InsightFace ArcFace 模型输入对齐。
    """
    h, w = img.shape[:2]
    cx, cy = eye_center
    side = int(min(h, w) * 0.8)
    x1 = int(np.clip(cx - side // 2, 0, w - side))
    y1 = int(np.clip(cy - side // 2, 0, h - side))
    cropped = img[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def center_crop_resize(img: np.ndarray, output_size: int = OUTPUT_SIZE) -> np.ndarray:
    """无关键点时的降级策略：中心正方形裁剪后 resize。"""
    h, w = img.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    cropped = img[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def process_image(src_path: Path) -> tuple[np.ndarray | None, dict]:
    """对单张图像执行完整预处理流水线，返回处理后图像和统计信息。

    流程：读取 → YCrCb 均衡化 → Haar 眼部检测 → 仿射对齐 → 裁剪 112×112
    """
    stats: dict = {"path": str(src_path), "angle": 0.0, "aligned": False}

    img = cv2.imread(str(src_path))
    if img is None:
        logger.warning("无法读取图像：%s", src_path)
        return None, stats

    # 1. 光照归一化
    before_mean = float(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
    img = equalize_hist_color(img)
    after_mean = float(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
    stats["hist_mean_before"] = round(before_mean, 2)
    stats["hist_mean_after"] = round(after_mean, 2)

    # 2. 眼部关键点检测（Haar 级联）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_centers = get_eye_centers_haar(gray)

    if eye_centers is not None:
        left_eye, right_eye = eye_centers
        img, angle, eye_center = align_face(img, left_eye, right_eye)
        # 若角度超过 ±20°，判定为误检，降级为中心裁剪
        if abs(angle) <= 20:
            stats["angle"] = round(angle, 2)
            stats["aligned"] = True
            output = crop_and_resize(img, eye_center)
        else:
            logger.debug("角度异常 %.1f°（误检），降级为中心裁剪：%s", angle, src_path)
            output = center_crop_resize(img)
    else:
        output = center_crop_resize(img)

    return output, stats


# ---------------------------------------------------------------------------
# 批量处理入口
# ---------------------------------------------------------------------------


def batch_process(src_dir: Path, dst_dir: Path) -> None:
    image_paths = sorted(list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png")))
    total = len(image_paths)
    if total == 0:
        logger.warning("在 %s 中未找到任何图像", src_dir)
        return

    logger.info("开始批量预处理：%d 张图像，输出至 %s", total, dst_dir)
    success = 0
    fail = 0
    aligned_count = 0
    angles = []

    for i, src_path in enumerate(image_paths, 1):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        result, stats = process_image(src_path)
        if result is None:
            fail += 1
            continue

        cv2.imwrite(str(dst_path), result)
        success += 1
        if stats["aligned"]:
            aligned_count += 1
            angles.append(stats["angle"])

        if i % 1000 == 0 or i == total:
            logger.info(
                "[%d/%d] 对齐率=%.1f%% 旋转角均值=%.1f°",
                i,
                total,
                100 * aligned_count / success if success else 0,
                float(np.mean(np.abs(angles))) if angles else 0,
            )

    logger.info(
        "批量预处理完成：成功 %d，失败 %d（共 %d）  对齐率=%.1f%%  |angle|均值=%.2f°",
        success,
        fail,
        total,
        100 * aligned_count / max(success, 1),
        float(np.mean(np.abs(angles))) if angles else 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="图像预处理：均衡化 + 眼部对齐 + 裁剪 112×112")
    parser.add_argument("--src", default="data/raw/lfw", help="源图像目录")
    parser.add_argument("--dst", default="data/processed/lfw", help="输出目录")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        logger.error("源目录不存在：%s，请先运行 make download-lfw", src_dir)
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)
    batch_process(src_dir, dst_dir)


if __name__ == "__main__":
    main()
