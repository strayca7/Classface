"""
Generate demo report images and the Markdown demo report.

Produces:
  docs/figures/demo_strip_{identity}.png   — raw | processed | cup | glasses | sunglasses | eye-crop
  docs/figures/demo_occlusion_types.png    — per-type montage grid
  docs/figures/demo_accuracy_full.png      — accuracy comparison with annotation
  docs/figures/demo_cascade_detail.png     — cascade routing diagram
  docs/demo_report.md                      — final Markdown demo report

Usage:
    uv run python scripts/generate_demo_report.py
"""

import logging
import textwrap
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")

# Configure CJK font for macOS
import matplotlib.font_manager as _fm
_CJK_CANDIDATES = ["Arial Unicode MS", "Hei", "Heiti TC", "PingFang SC", "STHeiti"]
for _font in _CJK_CANDIDATES:
    if any(f.name == _font for f in _fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _font
        break

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] demo_report: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

DPI = 150

# ── colour constants ──────────────────────────────────────────────────────────
C_BLUE   = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN  = "#55A868"
C_RED    = "#C44E52"
C_PURPLE = "#8172B3"
GREY_BG  = "#F8F9FA"

# ── demo identities (chosen for visual diversity) ────────────────────────────
DEMO_IDS = [
    "Queen_Latifah",
    "Ethan_Hawke",
    "Carlos_Moya",
    "Greg_Rusedski",
    "Christine_Gregoire",
    "GL_Peiris",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_img(path: Path, size: int = 112) -> np.ndarray:
    """Load BGR image, resize to square, return RGB uint8."""
    img = cv2.imread(str(path))
    if img is None:
        return np.full((size, size, 3), 200, dtype=np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_label(ax: plt.Axes, text: str, color: str = "white",
              bg: str = "#333333") -> None:
    ax.text(0.5, 0.04, text, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc=bg, alpha=0.75))


def add_title_bar(ax: plt.Axes, text: str, bg: str = "#2C3E50") -> None:
    ax.set_title(text, fontsize=8.5, fontweight="bold", color="white",
                 pad=3, backgroundcolor=bg)


# ── Fig 1: identity strips ─────────────────────────────────────────────────


def _first_img(directory: Path) -> Path | None:
    imgs = sorted(directory.glob("*.jpg"))
    return imgs[0] if imgs else None


def _second_img(directory: Path) -> Path | None:
    imgs = sorted(directory.glob("*.jpg"))
    return imgs[1] if len(imgs) > 1 else (imgs[0] if imgs else None)


def generate_identity_strips() -> None:
    """One row per identity: raw | processed | cup | glasses | sunglasses | eye-crop."""
    cols = ["原始图像\n(LFW raw)", "预处理\n(112×112)", "合成 — 水杯\n(cup)",
            "合成 — 眼镜\n(glasses)", "合成 — 墨镜\n(sunglasses)", "眼周裁剪\n(eye-crop)"]
    n_rows = len(DEMO_IDS)
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.8, n_rows * 2.1),
                             facecolor="#1A1A2E")
    fig.suptitle("数据流水线可视化：原始 → 预处理 → 遮挡合成 → 眼周裁剪",
                 fontsize=13, color="white", y=1.01, fontweight="bold")

    col_colors = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, "#888", C_RED]

    for r, identity in enumerate(DEMO_IDS):
        raw_dir  = DATA / "raw"  / "lfw"          / identity
        proc_dir = DATA / "processed" / "lfw"      / identity
        cup_dir  = DATA / "synthetic" / "cup"      / identity
        gl_dir   = DATA / "synthetic" / "glasses"  / identity
        sg_dir   = DATA / "synthetic" / "sunglasses" / identity
        ec_dir   = DATA / "cropped"  / "query" / "cup" / identity

        # Anchor on the first query image in synthetic/cup (never 0001).
        # Use its stem to fetch the matching file in every directory,
        # so all 6 columns show the exact same photo of the same person.
        anchor = _first_img(cup_dir)
        stem = anchor.stem if anchor else None  # e.g. "Queen_Latifah_0002"

        def _by_stem(directory: Path, fallback_fn=_first_img) -> Path | None:
            if stem:
                p = directory / f"{stem}.jpg"
                if p.exists():
                    return p
            return fallback_fn(directory)

        paths = [
            _by_stem(raw_dir),
            _by_stem(proc_dir),
            _by_stem(cup_dir),
            _by_stem(gl_dir),
            _by_stem(sg_dir),
            _by_stem(ec_dir),
        ]

        for c, path in enumerate(paths):
            ax = axes[r][c]
            img = load_img(path) if path and path.exists() else np.full((112, 112, 3), 80, dtype=np.uint8)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(col_colors[c])
                spine.set_linewidth(2)

            if r == 0:
                ax.set_title(cols[c], fontsize=7.5, color=col_colors[c],
                             fontweight="bold", pad=3)
            if c == 0:
                name_short = identity.replace("_", "\n")
                ax.set_ylabel(name_short, fontsize=7, color="white",
                              rotation=0, labelpad=42, va="center")

    fig.tight_layout(pad=0.3)
    out = FIGS / "demo_strip_all.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Fig 2: occlusion type montage ─────────────────────────────────────────────

def generate_occlusion_montage() -> None:
    """3×6 grid: rows = cup/glasses/sunglasses, cols = 6 different identities."""
    occ_types  = ["cup", "glasses", "sunglasses"]
    occ_labels = ["水杯 (cup)", "眼镜 (glasses)", "墨镜 (sunglasses)"]
    occ_colors = [C_ORANGE, C_PURPLE, "#555"]

    n_rows = len(occ_types)
    n_cols = len(DEMO_IDS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.8, n_rows * 2.2),
                             facecolor="#1A1A2E")
    fig.suptitle("三种课堂遮挡类型 — 合成样本展示（6 位身份）",
                 fontsize=13, color="white", y=1.01, fontweight="bold")

    for r, (occ, label, color) in enumerate(zip(occ_types, occ_labels, occ_colors)):
        for c, identity in enumerate(DEMO_IDS):
            ax = axes[r][c]
            syn_dir = DATA / "synthetic" / occ / identity
            path = _second_img(syn_dir)
            img = load_img(path) if path and path.exists() else np.full((112, 112, 3), 80, dtype=np.uint8)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

            if c == 0:
                ax.set_ylabel(label, fontsize=9, color=color,
                              fontweight="bold", rotation=0,
                              labelpad=80, va="center")
            if r == 0:
                ax.set_title(identity.replace("_", "\n"), fontsize=7,
                             color="lightgrey", pad=3)

    fig.tight_layout(pad=0.4)
    out = FIGS / "demo_occlusion_montage.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Fig 3: full accuracy comparison (annotated) ───────────────────────────────

def generate_accuracy_chart() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=GREY_BG)
    fig.suptitle("识别准确率对比实验", fontsize=14, fontweight="bold")

    # ── left: A / B / C bar ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(GREY_BG)
    labels = ["A\n基线（干净图像）", "B\n遮挡 Naive\n全脸 ArcFace", "C\n两级级联 v1\nL1_HIGH=0.8"]
    vals   = [92.65, 89.11, 12.52]
    colors = [C_BLUE, C_ORANGE, C_RED]
    bars = ax.bar(labels, vals, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2,
                f"{v:.2f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    # annotation arrows
    ax.annotate("", xy=(1, 89.11), xytext=(0, 92.65),
                arrowprops=dict(arrowstyle="<->", color="grey", lw=1.5))
    ax.text(0.5, 91, "−3.54pp", ha="center", fontsize=9, color="grey")
    ax.annotate("", xy=(2, 12.52), xytext=(1, 89.11),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5))
    ax.text(1.5, 52, "−76.59pp\n阈值错误", ha="center", fontsize=9, color=C_RED)

    ax.set_ylim(0, 108)
    ax.set_ylabel("Top-1 准确率 (%)", fontsize=12)
    ax.set_title("三组整体准确率（全量，22,452 query）", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # ── right: per-type grouped bar ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(GREY_BG)
    types  = ["sunglasses\n（墨镜）", "cup\n（水杯）", "glasses\n（眼镜）"]
    b_vals = [88.79, 87.85, 90.69]
    c_vals = [8.23,  17.45, 11.89]
    x = np.arange(len(types))
    w = 0.32

    bars_b = ax2.bar(x - w/2, b_vals, w, color=C_ORANGE, label="B — Naive", zorder=3)
    bars_c = ax2.bar(x + w/2, c_vals, w, color=C_RED,    label="C — 级联 v1", zorder=3)

    for bar in list(bars_b) + list(bars_c):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax2.set_ylim(0, 108)
    ax2.set_xticks(x)
    ax2.set_xticklabels(types, fontsize=10)
    ax2.set_ylabel("Top-1 准确率 (%)", fontsize=12)
    ax2.set_title("各遮挡类型细分（全量，n=7,484/type）", fontsize=11)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    out = FIGS / "demo_accuracy_compare.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=GREY_BG)
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Fig 4: cascade optimisation detail ────────────────────────────────────────

def generate_cascade_chart() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=GREY_BG)
    fig.suptitle("级联策略优化分析", fontsize=14, fontweight="bold")

    # ── left: evolution line ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(GREY_BG)
    versions = ["v1\nL1_HIGH=0.8\n(broken)", "v2\nL1_HIGH=0.5\nbest-of-two", "v3\nL1_HIGH=−1\n(禁用 L2)"]
    c_vals   = [12.52, 48.92, 92.70]

    ax.fill_between([0, 1, 2], c_vals, alpha=0.15, color=C_GREEN)
    ax.plot([0, 1, 2], c_vals, marker="o", linewidth=2.5, markersize=10,
            color=C_GREEN, zorder=3, label="C — 级联")
    ax.axhline(92.70, color=C_ORANGE, linestyle="--", linewidth=1.5,
               label="B = 92.70%（50 样本）", zorder=2)
    ax.axhline(92.65, color=C_BLUE, linestyle=":", linewidth=1.5,
               label="A = 92.65%（全量基线）", zorder=2)

    for i, (v, y) in enumerate(zip(versions, c_vals)):
        ax.annotate(f"{y:.2f}%", (i, y),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold", color=C_GREEN)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(versions, fontsize=9.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("C 准确率 (%)", fontsize=12)
    ax.set_title("级联优化三轮迭代（50 身份抽样）", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper left")

    # ── right: v3 per-type breakdown ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(GREY_BG)
    types  = ["sunglasses", "cup", "glasses"]
    b50    = [94.67, 89.94, 93.49]
    c50_v2 = [94.67, 44.97, 93.49]   # type-aware v2 (50-sample)
    c50_v3 = [94.67, 89.94, 93.49]   # v3 = B (50-sample)
    x = np.arange(len(types))
    w = 0.25

    ax2.bar(x - w,   b50,    w, color=C_ORANGE, label="B — Naive",          zorder=3)
    ax2.bar(x,       c50_v2, w, color=C_RED,    label="C — 类型感知 v2",     zorder=3)
    ax2.bar(x + w,   c50_v3, w, color=C_GREEN,  label="C — 禁用L2 v3 (最优)", zorder=3)

    for bars_list in [
        ax2.containers[0], ax2.containers[1], ax2.containers[2]
    ]:
        for bar in bars_list:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                     f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(types, fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("准确率 (%)", fontsize=12)
    ax2.set_title("各遮挡类型 B vs 级联各版本（50 样本）", fontsize=11)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = FIGS / "demo_cascade_detail.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=GREY_BG)
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Fig 5: segmentation comparison ───────────────────────────────────────────

def generate_segmentation_chart() -> None:
    methods = ["YCrCb\n阈值", "GMM\n肤色", "GrabCut\n前景", "Watershed\n分水岭"]
    means   = [51.5, 96.4, 23.6, 35.0]
    stds    = [15.7,  6.0, 14.7, 10.4]
    colors  = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=GREY_BG)
    ax.set_facecolor(GREY_BG)

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=7, color=colors, width=0.5,
                  error_kw={"elinewidth": 1.8, "ecolor": "#555"}, zorder=3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 2,
                f"{m:.1f}%\n±{s:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylim(0, 130)
    ax.set_ylabel("前景像素占比 (%)", fontsize=12)
    ax.set_title("四种传统图像分割方法：前景覆盖率\n（20 张随机抽样，误差棒 = ±1σ）", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGS / "demo_segmentation.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=GREY_BG)
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Markdown report ───────────────────────────────────────────────────────────

REPORT_MD = """\
# 实验演示报告

> **基于"特定场景与非标准遮挡"的人脸识别**  
> 数字图像处理课程设计 · 实验结果展示

---

## 目录

1. [任务场景与流水线](#1-任务场景与流水线)
2. [数据流水线可视化](#2-数据流水线可视化)
3. [遮挡合成样本展示](#3-遮挡合成样本展示)
4. [图像分割对比](#4-图像分割对比)
5. [识别准确率对比实验](#5-识别准确率对比实验)
6. [级联策略优化分析](#6-级联策略优化分析)
7. [关键结论](#7-关键结论)

---

## 1. 任务场景与流水线

**目标**：在智慧课堂/考场/自习室的无感签到场景下，提升以下三类遮挡情况的人脸识别准确率：

| 遮挡类型 | 描述 | 遮挡位置 |
|---------|------|---------|
| **水杯 (cup)** | 喝水时杯口遮挡嘴/鼻区域 | 下半脸 |
| **眼镜 (glasses)** | 普通框架眼镜 | 眼部周围 |
| **墨镜 (sunglasses)** | 深色镜片，遮挡眼部 | 眼部及周围 |

**五阶段流水线**：

```
原始 LFW 图像
    │
    ▼ Phase 1 — 预处理（YCrCb 均衡化 + 眼部对齐 + 112×112 裁剪）
    │
    ▼ Phase 2 — 图像分割（YCrCb 阈值 / GMM / GrabCut / Watershed）
    │
    ▼ Phase 3 — 基线识别（ArcFace 512-d + 余弦相似度 Top-1）
    │
    ▼ Phase 4 — 遮挡数据合成（关键点锚定 + Alpha Blending）
    │
    ▼ Phase 5 — 鲁棒识别评估（三组对比 A / B / C + 级联优化）
```

**数据规模**：

| 指标 | 数量 |
|------|------|
| 有效身份数 | 1,680 |
| Gallery 图像 | 1,680 张（每身份第 1 张） |
| Query 图像 | 7,484 张（干净） |
| 合成遮挡图像 | 22,452 张（3 类型 × 7,484） |

---

## 2. 数据流水线可视化

下图展示 6 位身份从原始图像经过各阶段处理后的样本（每列为一个处理步骤）：

- **原始图像**：LFW-funneled 原始人脸图像（任意尺寸）
- **预处理**：YCrCb 光照均衡化 → Haar 双眼检测 → warpAffine 对齐 → 112×112 裁剪
- **合成 — 水杯**：嘴角中点锚定，随机水杯贴图 Alpha 融合
- **合成 — 眼镜**：双眼中点锚定，随机眼镜贴图 Alpha 融合
- **合成 — 墨镜**：双眼中点锚定，随机墨镜贴图 Alpha 融合
- **眼周裁剪**：从顶至鼻尖区域裁剪，resize 至 112×112（Level-2 特征输入）

![数据流水线可视化](figures/demo_strip_all.png)

---

## 3. 遮挡合成样本展示

三类遮挡在 6 位不同身份上的效果：

![遮挡合成样本](figures/demo_occlusion_montage.png)

**合成参数说明**：

| 类型 | 贴图素材 | 锚点 | 缩放参数 |
|------|---------|------|---------|
| cup（水杯） | 18 种变体 | 嘴角中点 | face_w × 0.70 |
| glasses（眼镜） | 20 种变体 | 双眼中点 | 眼间距 × 2.80 |
| sunglasses（墨镜） | 20 种变体 | 双眼中点 | 眼间距 × 3.00 |

**全量合成统计**：

| 指标 | 数值 |
|------|------|
| 合成总量 | 22,452 张 |
| InsightFace 检测成功率 | 96.9%（7,250/7,484） |
| 检测失败（用固定比例 fallback） | 3.1%（234/7,484） |

---

## 4. 图像分割对比

第二阶段对预处理人脸图像测试了四种传统分割方法，以前景像素占比为评估指标：

![图像分割方法对比](figures/demo_segmentation.png)

| 方法 | 均值 | 标准差 | 特点 |
|------|------|--------|------|
| **YCrCb 阈值** | 51.5% | ±15.7% | 椭圆肤色模型；速度最快；对非标准肤色鲁棒性较弱 |
| **GMM 肤色** | 96.4% | ±6.0% | 数据驱动；覆盖率高（裁剪图几乎全为人脸） |
| **GrabCut** | 23.6% | ±14.7% | 图割能量优化；边界平滑；均匀人脸偏保守 |
| **Watershed** | 35.0% | ±10.4% | 距离变换 + 分水岭；自适应性适中 |

> GMM 覆盖率近 100% 符合预期：预处理已将图像裁剪为以人脸为中心的 112×112，背景极少。

---

## 5. 识别准确率对比实验

### 5.1 三组实验设计

| 组别 | 输入图像 | 识别策略 | 目的 |
|------|---------|---------|------|
| **A — 基线** | 干净图像（7,484） | 全脸 ArcFace 直接检索 | 系统上界 |
| **B — 遮挡 Naive** | 合成遮挡（22,452） | 全脸 ArcFace 直接检索 | 遮挡原始影响 |
| **C — 两级级联** | 合成遮挡（22,452） | L1 全脸 → L2 眼周裁剪 | 改进策略效果 |

### 5.2 全量结果

![识别准确率对比](figures/demo_accuracy_compare.png)

**整体准确率**（全量，1,680 身份）：

| 组别 | 准确率 | 正确数 | 总数 |
|------|--------|--------|------|
| **A — 基线** | **92.65%** | 6,934 | 7,484 |
| **B — 遮挡 Naive** | **89.11%** | 20,007 | 22,452 |
| **C — 两级级联 v1** | **12.52%** | 2,812 | 22,452 |

**各遮挡类型细分**（全量，n=7,484/type）：

| 遮挡类型 | B — Naive | C — 级联 v1 | Δ |
|---------|-----------|------------|---|
| sunglasses（墨镜） | 88.79% | 8.23% | −80.56pp |
| cup（水杯） | 87.85% | 17.45% | −70.40pp |
| glasses（眼镜） | 90.69% | 11.89% | −78.80pp |

### 5.3 C=12.52% 失败根因分析

**原因一：L1_HIGH=0.8 阈值过高**  
ArcFace 对遮挡图像的余弦相似度普遍低于干净图像，约 87% 的 query 落在 [0.4, 0.8] 区间被强制路由至 Level-2。

**原因二：ArcFace 对局部裁剪特征质量低**  
InsightFace 训练于完整对齐的 112×112 人脸，对眼周局部裁剪图：
- 人脸检测通常失败（无完整人脸），退回 `get_feat()` 直接提取
- 未对齐的局部特征与 gallery 特征分布不匹配

**原因三：glasses/sunglasses 眼周被遮挡**  
眼镜/墨镜贴图锚定在双眼，L2 眼周裁剪恰好截取的是遮挡后的眼部区域，失去识别意义。

---

## 6. 级联策略优化分析

![级联策略优化](figures/demo_cascade_detail.png)

### 6.1 三轮优化迭代

| 版本 | L1_HIGH | 策略 | 50 样本 C | 全量 C |
|------|---------|------|-----------|--------|
| **v1（原始）** | 0.8 | L2 best-of-two | — | **12.52%** ✗ |
| **v2** | 0.5 | L2 best-of-two | **48.92%** | — |
| **v3（最优）** | −1 | L2 禁用，全部 L1 | **92.70%** | *≈89.11%* |

### 6.2 各版本在不同遮挡类型上的表现（50 样本）

| 遮挡类型 | B Naive | C v2（类型感知） | C v3（禁用 L2） |
|---------|---------|----------------|----------------|
| sunglasses | 94.67% | 94.67% ✓ | 94.67% ✓ |
| cup | 89.94% | 44.97% ✗ | 89.94% ✓ |
| glasses | 93.49% | 93.49% ✓ | 93.49% ✓ |

### 6.3 核心结论

> **在不重新训练特征提取器的前提下，局部区域级联无法超越全脸识别。**  
> 最优策略 v3（L1_HIGH=−1）等价于直接使用 ArcFace 全脸识别，C ≈ B ≈ **89.11%**。

---

## 7. 关键结论

### ✅ 结论一：ArcFace 对轻度遮挡内置鲁棒性强

A（92.65%）→ B（89.11%）仅下降 **3.54 个百分点**。水杯/眼镜类轻度遮挡不妨碍 ArcFace 从可见区域提取足够的身份信息。

### ❌ 结论二：眼周局部裁剪策略无效（当前架构）

两级级联 v1 准确率仅 12.52%，根源在于 ArcFace 对非对齐局部裁剪特征质量极差，L2 路由只会引入错误。

### 🔑 结论三：阈值是核心超参数

`L1_HIGH` 从 0.8 → −1，C 从 12.52% 提升至 92.70%（50 样本），核心原因是**减少 L2 路由比例**。

### 🚀 结论四：改进方向

| 改进方向 | 预期效果 |
|---------|---------|
| 在含遮挡数据上微调 ArcFace | 提升遮挡场景特征质量 |
| 遮挡感知注意力机制 | 动态关注未被遮挡区域 |
| 图像修复（LaMa/MAT）先填补遮挡再识别 | 对重遮挡场景有效 |
| 采集真实课堂场景数据训练 | 解决合成数据分布偏移 |

---

*生成时间：2026-05-13*  
*数据集：LFW-funneled · 模型：InsightFace buffalo_l (ArcFace R50)*
"""


def write_report() -> None:
    path = ROOT / "docs" / "demo_report.md"
    path.write_text(REPORT_MD, encoding="utf-8")
    log.info("Report written: %s", path)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Generating demo report images → %s", FIGS)

    generate_identity_strips()
    generate_occlusion_montage()
    generate_accuracy_chart()
    generate_cascade_chart()
    generate_segmentation_chart()
    write_report()

    log.info("All done.")
    for p in sorted(FIGS.glob("demo_*.png")):
        log.info("  %s  (%d KB)", p.name, p.stat().st_size // 1024)


if __name__ == "__main__":
    main()
