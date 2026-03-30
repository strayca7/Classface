"""准备 LFW 数据集：筛选 ≥2 张图像的身份，生成 gallery/query 分割清单。

输出：data/raw/lfw_filtered.json
格式：{"person_name": {"gallery": ["path/to/img1"], "query": ["path/to/img2", ...]}, ...}
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_dataset")

LFW_DIR = Path("data/raw/lfw")
OUTPUT_PATH = Path("data/raw/lfw_filtered.json")
MIN_IMAGES = 2


def build_split(lfw_dir: Path, min_images: int) -> dict:
    """扫描 LFW 目录，过滤出图像数 >= min_images 的身份，并做 gallery/query 分割。

    JSON 中的路径均相对于 lfw_dir（例如 "Alice/Alice_0001.jpg"）。
    使用时通过 lfw_dir / path 还原完整路径。
    """
    split: dict[str, dict[str, list[str]]] = {}

    identities = sorted(p for p in lfw_dir.iterdir() if p.is_dir())
    logger.info("扫描目录：%s，共 %d 个身份", lfw_dir, len(identities))

    filtered = 0
    for person_dir in identities:
        images = sorted(person_dir.glob("*.jpg"))
        if len(images) < min_images:
            continue
        # 存储相对于 lfw_dir 的路径，方便在不同环境下复用 JSON
        rel_images = [str(img.relative_to(lfw_dir)) for img in images]
        split[person_dir.name] = {
            "gallery": rel_images[:1],
            "query": rel_images[1:],
        }
        filtered += 1

    logger.info(
        "筛选完成：%d 个身份满足 ≥%d 张图像（占总数 %.1f%%）",
        filtered,
        min_images,
        100 * filtered / max(len(identities), 1),
    )
    return split


def main() -> None:
    if not LFW_DIR.exists():
        logger.error("LFW 目录不存在：%s，请先运行 make download-lfw", LFW_DIR)
        sys.exit(1)

    split = build_split(LFW_DIR, MIN_IMAGES)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    total_gallery = sum(len(v["gallery"]) for v in split.values())
    total_query = sum(len(v["query"]) for v in split.values())
    logger.info(
        "已写入 %s：%d 个身份，gallery %d 张，query %d 张",
        OUTPUT_PATH,
        len(split),
        total_gallery,
        total_query,
    )


if __name__ == "__main__":
    main()
