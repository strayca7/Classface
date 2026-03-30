"""图像预处理：光照归一化 + 人脸对齐 + 统一裁剪至 112x112。

用法：
    uv run python tools/preprocess.py [--src data/raw/lfw] [--dst data/processed/lfw]
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

# MediaPipe 左/右眼中心关键点索引（FaceMesh 468点）
_LEFT_EYE_IDX = 33
_RIGHT_EYE_IDX = 263
OUTPUT_SIZE = 112

try:
    import mediapipe as mp

    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    _USE_MEDIAPIPE = True
except ImportError:
    logger.warning("mediapipe 未安装，人脸对齐将跳过（仅做光照归一化+中心裁剪）")
    _face_mesh = None
    _USE_MEDIAPIPE = False


# ---------------------------------------------------------------------------
# 核心处理函数
# ---------------------------------------------------------------------------


def equalize_hist_color(img: np.ndarray) -> np.ndarray:
    """YCrCb 空间 Y 通道直方图均衡化，保留色彩信息。"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def get_eye_centers(img_rgb: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """用 MediaPipe FaceMesh 获取左右眼中心坐标（归一化→像素）。

    Returns:
        (left_eye_center, right_eye_center) 像素坐标，失败返回 None
    """
    if not _USE_MEDIAPIPE or _face_mesh is None:
        return None
    result = _face_mesh.process(img_rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    h, w = img_rgb.shape[:2]
    left = (lm[_LEFT_EYE_IDX].x * w, lm[_LEFT_EYE_IDX].y * h)
    right = (lm[_RIGHT_EYE_IDX].x * w, lm[_RIGHT_EYE_IDX].y * h)
    return left, right


def align_face(img: np.ndarray, left_eye: tuple, right_eye: tuple) -> np.ndarray:
    """根据双眼中心仿射旋转人脸至水平，以双眼中心为旋转中心。"""
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2,
    )
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return aligned, angle, eye_center


def crop_and_resize(img: np.ndarray, eye_center: tuple, output_size: int = OUTPUT_SIZE) -> np.ndarray:
    """以双眼中心为基准，裁出人脸区域并 resize 至 output_size×output_size。

    裁剪框高度 = 眼间距 * 3.5（经验值，覆盖从额头到下巴）。
    """
    h, w = img.shape[:2]
    cx, cy = eye_center

    # 简单中心裁剪（以眼睛中心为参考，取图像较短边的0.8倍正方形）
    side = int(min(h, w) * 0.8)
    x1 = int(np.clip(cx - side // 2, 0, w - side))
    y1 = int(np.clip(cy - side // 2, 0, h - side))
    x2 = x1 + side
    y2 = y1 + side

    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def center_crop_resize(img: np.ndarray, output_size: int = OUTPUT_SIZE) -> np.ndarray:
    """无关键点时的备用：中心正方形裁剪后 resize。"""
    h, w = img.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    cropped = img[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def process_image(src_path: Path) -> tuple[np.ndarray | None, dict]:
    """对单张图像执行完整预处理流水线，返回处理后图像和统计信息。"""
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

    # 2. 人脸对齐
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    eye_centers = get_eye_centers(img_rgb)

    if eye_centers is not None:
        left_eye, right_eye = eye_centers
        img, angle, eye_center = align_face(img, left_eye, right_eye)
        stats["angle"] = round(angle, 2)
        stats["aligned"] = True
        output = crop_and_resize(img, eye_center)
    else:
        output = center_crop_resize(img)

    return output, stats


# ---------------------------------------------------------------------------
# 批量处理入口
# ---------------------------------------------------------------------------


def batch_process(src_dir: Path, dst_dir: Path) -> None:
    image_paths = list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png"))
    total = len(image_paths)
    if total == 0:
        logger.warning("在 %s 中未找到任何图像", src_dir)
        return

    logger.info("开始批量预处理：%d 张图像，输出至 %s", total, dst_dir)
    success = 0
    fail = 0

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

        if i % 500 == 0 or i == total:
            logger.info(
                "[%d/%d] aligned=%s angle=%.1f° hist_mean %.1f→%.1f",
                i, total,
                stats["aligned"],
                stats["angle"],
                stats.get("hist_mean_before", 0),
                stats.get("hist_mean_after", 0),
            )

    logger.info("批量预处理完成：成功 %d，失败 %d（共 %d）", success, fail, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="图像预处理：均衡化 + 人脸对齐 + 裁剪 112×112")
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
