"""构建 gallery 特征库：为每位身份的 gallery 图像提取 InsightFace 512-d 嵌入向量。

用法：
    uv run python scripts/build_gallery.py [--limit N]

输出：
    data/features/gallery.npy          — 特征矩阵，形状 (N, 512), float32
    data/features/gallery_labels.json  — 对应身份标签列表，长度 N
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("build_gallery")

PROCESSED_DIR = Path("data/processed/lfw")
FILTERED_JSON = Path("data/raw/lfw_filtered.json")
FEATURES_DIR = Path("data/features")
GALLERY_NPY = FEATURES_DIR / "gallery.npy"
GALLERY_LABELS = FEATURES_DIR / "gallery_labels.json"


def load_insightface_model():
    """初始化 InsightFace buffalo_l 模型（首次运行自动下载 ~500 MB）。"""
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise ImportError("请先安装 insightface：uv sync") from e

    log.info("初始化 InsightFace buffalo_l 模型（首次运行将自动下载模型文件）...")
    # det_size=(640,640) 为检测模型的标准推理分辨率；预处理图像在传入前会先上采样
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    log.info("模型初始化完成")
    return app


def extract_embedding(app, img_path: Path) -> np.ndarray | None:
    """从图像文件中提取 512-d 嵌入向量。

    将 112×112 预处理图像上采样后送入完整流水线（检测 → ArcFace 对齐 → 特征提取）。
    若检测失败，退回到直接调用识别模型。
    """
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        log.warning("无法读取图像: %s", img_path)
        return None

    # 上采样至 320×320，确保检测器 anchor 计算正常（det_10g 最小推理尺寸约 320px）
    img_large = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)

    # 完整流水线（检测 → ArcFace 对齐 → 特征提取）
    faces = app.get(img_large)
    if faces:
        return faces[0].embedding.astype(np.float32)

    # 回退：直接调用识别模型（跳过对齐）
    rec_model = app.models.get("recognition")
    if rec_model is not None:
        try:
            feat = rec_model.get_feat(img)
            if feat is not None:
                return feat.flatten().astype(np.float32)
        except Exception:
            pass

    log.debug("特征提取失败: %s", img_path)
    return None


def main():
    parser = argparse.ArgumentParser(description="构建 InsightFace gallery 特征库")
    parser.add_argument("--limit", type=int, default=None, help="限制处理身份数量（调试用）")
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # 加载过滤后的数据集清单
    if not FILTERED_JSON.exists():
        raise FileNotFoundError(f"未找到 {FILTERED_JSON}，请先运行 make prepare-dataset")
    with open(FILTERED_JSON) as f:
        dataset = json.load(f)

    identities = list(dataset.keys())
    if args.limit:
        identities = identities[: args.limit]
        log.info("调试模式：限制处理 %d 个身份", len(identities))

    log.info("共 %d 个身份需要提取 gallery 特征", len(identities))

    app = load_insightface_model()

    embeddings = []
    labels = []
    failed = 0
    t0 = time.time()

    for i, identity in enumerate(identities):
        gallery_paths = dataset[identity]["gallery"]
        if not gallery_paths:
            log.warning("身份 %s 无 gallery 图像，跳过", identity)
            failed += 1
            continue

        img_path = PROCESSED_DIR / gallery_paths[0]
        if not img_path.exists():
            log.warning("图像不存在: %s", img_path)
            failed += 1
            continue

        emb = extract_embedding(app, img_path)
        if emb is None:
            log.warning("特征提取失败: %s", img_path)
            failed += 1
            continue

        embeddings.append(emb)
        labels.append(identity)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            log.info("进度: %d/%d，耗时 %.1fs", i + 1, len(identities), elapsed)

    elapsed = time.time() - t0
    log.info(
        "特征提取完成：成功 %d，失败 %d，耗时 %.1fs",
        len(embeddings),
        failed,
        elapsed,
    )

    if not embeddings:
        raise RuntimeError("没有成功提取任何特征，请检查图像路径和模型")

    gallery_matrix = np.stack(embeddings, axis=0)  # (N, 512)
    np.save(GALLERY_NPY, gallery_matrix)
    with open(GALLERY_LABELS, "w") as f:
        json.dump(labels, f, ensure_ascii=False)

    log.info("保存 gallery.npy: 形状 %s", gallery_matrix.shape)
    log.info("保存 gallery_labels.json: %d 个标签", len(labels))
    log.info("完成！输出至 %s", FEATURES_DIR)


if __name__ == "__main__":
    main()
