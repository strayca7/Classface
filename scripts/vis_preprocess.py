"""可视化预处理结果：将原图与预处理后图像对比展示。

用法：
    uv run python scripts/vis_preprocess.py [--n 16] [--out data/results/vis_preprocess.png]
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# macOS 中文字体，消除 CJK 字形警告
plt.rcParams["font.family"] = ["STHeiti", "Heiti TC", "DejaVu Sans"]


def load_pair(raw_dir: Path, processed_dir: Path) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """随机采样若干张图像，返回 (原图, 预处理图, 人名) 列表。"""
    pairs = []
    identities = [p for p in processed_dir.iterdir() if p.is_dir()]
    random.shuffle(identities)
    for person_dir in identities:
        imgs = sorted(person_dir.glob("*.jpg"))
        if not imgs:
            continue
        proc_path = imgs[0]
        raw_path = raw_dir / proc_path.relative_to(processed_dir)
        if not raw_path.exists():
            continue
        raw_img = cv2.cvtColor(cv2.imread(str(raw_path)), cv2.COLOR_BGR2RGB)
        proc_img = cv2.cvtColor(cv2.imread(str(proc_path)), cv2.COLOR_BGR2RGB)
        pairs.append((raw_img, proc_img, person_dir.name.replace("_", " ")))
        if len(pairs) >= 100:  # 预加载足够多供采样
            break
    return pairs


def plot_grid(pairs: list, n: int, out_path: Path) -> None:
    """绘制 n 组原图/预处理对比网格。"""
    cols = 4          # 每行：原图, 预处理, 原图, 预处理
    rows = n // 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    fig.suptitle("预处理结果对比（左：原图  右：均衡化+对齐+112×112）", fontsize=13, y=1.01)

    for i in range(rows):
        for j in range(2):  # 每行放 2 组
            idx = i * 2 + j
            if idx >= len(pairs):
                break
            raw, proc, name = pairs[idx]
            col_base = j * 2
            # 原图
            ax_raw = axes[i][col_base]
            ax_raw.imshow(raw)
            ax_raw.set_title(name, fontsize=7, pad=2)
            ax_raw.axis("off")
            # 预处理图
            ax_proc = axes[i][col_base + 1]
            ax_proc.imshow(proc)
            ax_proc.set_title(f"{proc.shape[1]}×{proc.shape[0]}", fontsize=7, pad=2)
            ax_proc.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"已保存：{out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化预处理前后对比")
    parser.add_argument("--n", type=int, default=16, help="展示图像组数（偶数，默认16）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out", default="data/results/vis_preprocess.png", help="输出图片路径")
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = Path("data/raw/lfw")
    processed_dir = Path("data/processed/lfw")

    if not processed_dir.exists():
        print("ERROR: data/processed/lfw/ 不存在，请先运行 make preprocess")
        return

    print(f"加载图像对（n={args.n}）...")
    pairs = load_pair(raw_dir, processed_dir)
    n = min(args.n, len(pairs))
    if n % 2 != 0:
        n -= 1
    plot_grid(pairs[:n], n, Path(args.out))


if __name__ == "__main__":
    main()
