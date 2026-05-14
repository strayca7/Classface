"""
级联策略消融实验：对比 v1 / v2 / v3 三种配置（--limit 50 快速验证）。

三种配置：
  v1: L1_HIGH=0.80, L2 直接替换 L1（原始错误实现）
  v2: L1_HIGH=0.50, best-of-two（取 L1/L2 中置信度更高的预测）
  v3: L1_HIGH=-1.0, 禁用 L2（最优策略）

输出：
  data/results/cascade_ablation.txt     详细数值
  data/results/figures/cascade_ablation.png  可视化图表

用法：
    uv run python scripts/eval_cascade_ablation.py [--limit N]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] cascade_ablation: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SYNTHETIC_DIR = Path("data/synthetic")
FEATURES_DIR = Path("data/features")
FILTERED_JSON = Path("data/raw/lfw_filtered.json")
RESULTS_DIR = Path("data/results")
FIGURES_DIR = RESULTS_DIR / "figures"

GALLERY_NPY = FEATURES_DIR / "gallery.npy"
GALLERY_LABELS = FEATURES_DIR / "gallery_labels.json"
GALLERY_CROPPED_NPY = FEATURES_DIR / "gallery_cropped.npy"
GALLERY_CROPPED_LABELS = FEATURES_DIR / "gallery_cropped_labels.json"
CROPPED_QUERY_DIR = Path("data/cropped/query")

# ── model (lazy singleton) ────────────────────────────────────────────────────
_app = None


def get_app():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        log.info("初始化 InsightFace 模型...")
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
        log.info("模型就绪")
    return _app


def extract_embedding(img_path: Path) -> np.ndarray | None:
    import cv2
    app = get_app()
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img_large = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
    faces = app.get(img_large)
    if faces:
        return faces[0].embedding.astype(np.float32)
    rec = app.models.get("recognition")
    if rec is not None:
        try:
            feat = rec.get_feat(img)
            if feat is not None:
                return feat.flatten().astype(np.float32)
        except Exception:
            pass
    return None


def cosine_top1(q: np.ndarray, gallery: np.ndarray, labels: list) -> tuple[str, float]:
    q_n = q / (np.linalg.norm(q) + 1e-8)
    scores = gallery @ q_n
    idx = int(np.argmax(scores))
    return labels[idx], float(scores[idx])


def load_gallery(npy: Path, lbl: Path) -> tuple[np.ndarray, list]:
    mat = np.load(npy).astype(np.float32)
    mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return mat_n, json.loads(lbl.read_text())


# ── cascade configurations ────────────────────────────────────────────────────

CASCADE_CONFIGS = [
    {
        "name": "v1",
        "label": "级联 v1\nL1_HIGH=0.80\nL2 直接替换",
        "l1_high": 0.80,
        "l1_low": 0.40,
        "mode": "replace",   # L2 直接替换 L1
    },
    {
        "name": "v2",
        "label": "级联 v2\nL1_HIGH=0.50\nbest-of-two",
        "l1_high": 0.50,
        "l1_low": 0.40,
        "mode": "best_of_two",  # 取 L1/L2 中分数更高的
    },
    {
        "name": "v3",
        "label": "级联 v3\nL1_HIGH=−1\n(禁用 L2)",
        "l1_high": -1.0,
        "l1_low": -2.0,
        "mode": "replace",   # L1_HIGH=-1 → 全走 L1，mode 无实际影响
    },
]


def cascade_predict(
    emb: np.ndarray,
    crop_path: Path,
    gallery_norm: np.ndarray,
    gallery_labels: list,
    gallery_cropped_norm: np.ndarray,
    gallery_cropped_labels: list,
    cfg: dict,
) -> str:
    pred_l1, score_l1 = cosine_top1(emb, gallery_norm, gallery_labels)

    if score_l1 > cfg["l1_high"]:
        return pred_l1

    if score_l1 < cfg["l1_low"]:
        return "unknown"

    # L2 路径
    if not crop_path.exists():
        return pred_l1
    emb_crop = extract_embedding(crop_path)
    if emb_crop is None:
        return pred_l1

    pred_l2, score_l2 = cosine_top1(emb_crop, gallery_cropped_norm, gallery_cropped_labels)

    if cfg["mode"] == "replace":
        return pred_l2
    else:  # best_of_two
        return pred_l2 if score_l2 > score_l1 else pred_l1


# ── main experiment ───────────────────────────────────────────────────────────

def run(limit: int | None) -> None:
    log.info("加载 gallery 特征...")
    gallery_norm, gallery_labels = load_gallery(GALLERY_NPY, GALLERY_LABELS)
    gallery_cropped_norm, gallery_cropped_labels = load_gallery(
        GALLERY_CROPPED_NPY, GALLERY_CROPPED_LABELS
    )
    label_set = set(gallery_labels)

    with open(FILTERED_JSON) as f:
        dataset = json.load(f)
    identities = sorted(dataset.keys())
    if limit:
        identities = identities[:limit]
    log.info("评估身份数：%d", len(identities))

    occ_types = ["sunglasses", "cup", "glasses"]

    # 结果存储: cfg_name → occ_type → {correct, total}
    results: dict[str, dict] = {
        cfg["name"]: {t: {"correct": 0, "total": 0} for t in occ_types}
        for cfg in CASCADE_CONFIGS
    }
    # 同时记录 B (naive)
    results["B"] = {t: {"correct": 0, "total": 0} for t in occ_types}

    for occ_type in occ_types:
        log.info("=== 遮挡类型: %s ===", occ_type)
        occ_dir = SYNTHETIC_DIR / occ_type
        for identity in identities:
            if identity not in label_set:
                continue
            person_dir = occ_dir / identity
            if not person_dir.exists():
                continue
            for img_path in sorted(person_dir.glob("*.jpg")):
                emb = extract_embedding(img_path)
                if emb is None:
                    continue

                # B naive
                pred_b, _ = cosine_top1(emb, gallery_norm, gallery_labels)
                results["B"][occ_type]["total"] += 1
                if pred_b == identity:
                    results["B"][occ_type]["correct"] += 1

                # Each cascade config
                for cfg in CASCADE_CONFIGS:
                    crop_path = CROPPED_QUERY_DIR / occ_type / identity / img_path.name
                    pred = cascade_predict(
                        emb, crop_path,
                        gallery_norm, gallery_labels,
                        gallery_cropped_norm, gallery_cropped_labels,
                        cfg,
                    )
                    results[cfg["name"]][occ_type]["total"] += 1
                    if pred == identity:
                        results[cfg["name"]][occ_type]["correct"] += 1

    # ── summary ─────────────────────────────────────────────────────────────
    def acc(r, name, occ=None):
        if occ:
            c, t = r[name][occ]["correct"], r[name][occ]["total"]
        else:
            c = sum(r[name][k]["correct"] for k in occ_types)
            t = sum(r[name][k]["total"] for k in occ_types)
        return c / max(t, 1), c, t

    lines = [
        "=== 级联策略消融实验（50 身份抽样） ===",
        "",
        f"{'策略':<12} {'总体准确率':>12}  {'sunglasses':>12}  {'cup':>12}  {'glasses':>12}",
        "-" * 64,
    ]

    all_keys = ["B"] + [c["name"] for c in CASCADE_CONFIGS]
    all_labels = {
        "B": "B Naive",
        "v1": "C v1 (L1_HIGH=0.80, replace)",
        "v2": "C v2 (L1_HIGH=0.50, best-of-two)",
        "v3": "C v3 (L1_HIGH=-1, 禁用L2)",
    }

    summary_data = {}
    for key in all_keys:
        a_all, c_all, t_all = acc(results, key)
        a_sg, _, _ = acc(results, key, "sunglasses")
        a_cup, _, _ = acc(results, key, "cup")
        a_gl, _, _ = acc(results, key, "glasses")
        summary_data[key] = {
            "all": a_all, "c": c_all, "t": t_all,
            "sunglasses": a_sg, "cup": a_cup, "glasses": a_gl,
        }
        lines.append(
            f"{all_labels[key]:<38}  {a_all*100:>6.2f}%  {a_sg*100:>6.2f}%  "
            f"{a_cup*100:>6.2f}%  {a_gl*100:>6.2f}%"
        )
        log.info("%s: 总体=%.2f%%  sg=%.2f%%  cup=%.2f%%  gl=%.2f%%",
                 key, a_all*100, a_sg*100, a_cup*100, a_gl*100)

    lines += [
        "",
        "配置说明：",
        "  v1: L1_HIGH=0.80, L1_LOW=0.40, L2 直接替换 L1",
        "  v2: L1_HIGH=0.50, L1_LOW=0.40, best-of-two（取 L1/L2 中分数更高）",
        "  v3: L1_HIGH=-1.00, L2 禁用（score > -1 恒成立，全走 L1）",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "cascade_ablation.txt"
    out.write_text("\n".join(lines) + "\n")
    log.info("结果写入 %s", out)

    # ── plot ─────────────────────────────────────────────────────────────────
    _plot(summary_data, occ_types)


def _plot(data: dict, occ_types: list) -> None:
    import matplotlib
    import matplotlib.font_manager as fm
    matplotlib.use("Agg")
    for font in ["Arial Unicode MS", "Hei", "Heiti TC", "PingFang SC"]:
        if any(f.name == font for f in fm.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = font
            break
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    C_BLUE   = "#4C72B0"
    C_ORANGE = "#DD8452"
    C_GREEN  = "#55A868"
    C_RED    = "#C44E52"
    C_PURPLE = "#8172B3"
    GREY_BG  = "#F8F9FA"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=GREY_BG)
    fig.suptitle("级联策略消融实验（50 身份抽样）", fontsize=14, fontweight="bold")

    # ── left: overall bar ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(GREY_BG)
    keys   = ["B", "v1", "v2", "v3"]
    labels = ["B\nNaive", "C v1\nL1=0.80\nreplace", "C v2\nL1=0.50\nbest-of-two", "C v3\nL1=−1\n禁用L2"]
    colors = [C_ORANGE, C_RED, C_PURPLE, C_GREEN]
    vals   = [data[k]["all"] * 100 for k in keys]

    bars = ax.bar(labels, vals, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 108)
    ax.set_ylabel("Top-1 准确率 (%)", fontsize=12)
    ax.set_title("四种策略整体准确率", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # ── right: per-type grouped bar ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(GREY_BG)
    import numpy as np
    x = np.arange(len(occ_types))
    w = 0.2
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    for i, (k, col, lbl) in enumerate(zip(keys, colors, ["B", "v1", "v2", "v3"])):
        per = [data[k][t] * 100 for t in occ_types]
        bars2 = ax2.bar(x + offsets[i], per, w, color=col, label=lbl, zorder=3)
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                     f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(occ_types, fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Top-1 准确率 (%)", fontsize=12)
    ax2.set_title("各遮挡类型细分", fontsize=11)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "cascade_ablation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=GREY_BG)
    plt.close(fig)
    log.info("图表保存至 %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50,
                        help="限制身份数（默认 50）")
    args = parser.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
