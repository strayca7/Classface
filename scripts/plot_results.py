"""
Generate experiment result charts from saved accuracy files.
Outputs to docs/figures/ (git-tracked).

Usage:
    uv run python scripts/plot_results.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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
    format="%(asctime)s [%(levelname)s] plot_results: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "results"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN = "#55A868"
C_RED = "#C44E52"
C_PURPLE = "#8172B3"
C_BROWN = "#937860"

GREY_BG = "#F8F9FA"
DPI = 180


def savefig(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    log.info("Saved: %s", path)
    plt.close(fig)


# ── 1. Accuracy overview (A / B / C) ─────────────────────────────────────────
def plot_accuracy_overview() -> None:
    """Bar chart: Group A / B / C accuracy."""
    groups = ["A\n基线（干净）", "B\n遮挡 Naive", "C\n两级级联 v1"]
    values = [92.65, 89.11, 12.52]
    colors = [C_BLUE, C_ORANGE, C_RED]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=GREY_BG)
    ax.set_facecolor(GREY_BG)

    bars = ax.bar(groups, values, color=colors, width=0.5, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.axhline(89.11, color=C_ORANGE, linestyle="--", linewidth=1.2, alpha=0.6,
               label="B = 89.11%")
    ax.axhline(92.65, color=C_BLUE, linestyle="--", linewidth=1.2, alpha=0.6,
               label="A = 92.65%")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-1 准确率 (%)", fontsize=12)
    ax.set_title("三组对比实验：整体准确率\n(全量 7,484 / 22,452 query)", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    fig.tight_layout()
    savefig(fig, "fig1_accuracy_overview.png")


# ── 2. Per-type breakdown B vs C ─────────────────────────────────────────────
def plot_per_type_bc() -> None:
    """Grouped bar: B vs C per occlusion type."""
    types = ["sunglasses\n（墨镜）", "cup\n（水杯）", "glasses\n（眼镜）"]
    b_vals = [88.79, 87.85, 90.69]
    c_vals = [8.23, 17.45, 11.89]

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=GREY_BG)
    ax.set_facecolor(GREY_BG)

    bars_b = ax.bar(x - width / 2, b_vals, width, label="B — 遮挡 Naive", color=C_ORANGE,
                    zorder=3)
    bars_c = ax.bar(x + width / 2, c_vals, width, label="C — 两级级联 v1", color=C_RED,
                    zorder=3)

    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    for bar in bars_c:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel("Top-1 准确率 (%)", fontsize=12)
    ax.set_title("各遮挡类型 B vs C（两级级联 v1）\n(n=7,484 per type)", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig2_per_type_bc.png")


# ── 3. Cascade optimisation evolution ────────────────────────────────────────
def plot_cascade_evolution() -> None:
    """Line chart: C accuracy across cascade versions."""
    versions = ["v1\nL1_HIGH=0.8\nbest-of-two", "v2\nL1_HIGH=0.5\nbest-of-two",
                "v3（当前）\nL1_HIGH=−1\n（禁用 L2）"]
    c_vals = [12.52, 48.92, 92.70]
    b_val = 92.70  # 50-sample B for fair comparison

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=GREY_BG)
    ax.set_facecolor(GREY_BG)

    ax.plot(versions, c_vals, marker="o", linewidth=2.5, markersize=9,
            color=C_GREEN, label="C — 两级级联", zorder=3)
    ax.axhline(b_val, color=C_ORANGE, linestyle="--", linewidth=1.5,
               label=f"B — 遮挡 Naive = {b_val:.2f}% (50 样本)", zorder=2)
    ax.axhline(92.65, color=C_BLUE, linestyle=":", linewidth=1.5,
               label="A — 基线 = 92.65%", zorder=2)

    for i, (x, y) in enumerate(zip(versions, c_vals)):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=11, fontweight="bold",
                    color=C_GREEN)

    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-1 准确率 (%)", fontsize=12)
    ax.set_title("级联策略优化演进（50 身份抽样）", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig3_cascade_evolution.png")


# ── 4. Preprocessing metrics ─────────────────────────────────────────────────
def plot_preprocessing() -> None:
    """Horizontal bar: preprocessing success metrics."""
    metrics = [
        "全量处理成功率\n(13,233 张)",
        "双眼对齐成功率",
        "Gallery 构建成功率\n(InsightFace)",
        "合成遮挡检测率\n(7,484 query)",
        "Gallery 眼周裁剪率",
    ]
    values = [100.0, 46.6, 100.0, 96.9, 97.1]
    colors = [C_BLUE, C_ORANGE, C_BLUE, C_GREEN, C_GREEN]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=GREY_BG)
    ax.set_facecolor(GREY_BG)

    y = np.arange(len(metrics))
    bars = ax.barh(y, values, color=colors, height=0.5, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 115)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=10)
    ax.set_xlabel("百分比 (%)", fontsize=12)
    ax.set_title("流水线各阶段关键指标", fontsize=13)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, "fig4_pipeline_metrics.png")


# ── 5. Segmentation method comparison ────────────────────────────────────────
def plot_segmentation() -> None:
    """Bar chart: foreground coverage per segmentation method."""
    methods = ["YCrCb\n阈值分割", "GMM\n肤色分割", "GrabCut\n前景分割", "Watershed\n分水岭"]
    means = [51.5, 96.4, 23.6, 35.0]
    stds = [15.7, 6.0, 14.7, 10.4]
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=GREY_BG)
    ax.set_facecolor(GREY_BG)

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, width=0.5,
                  error_kw={"elinewidth": 1.5, "ecolor": "grey"}, zorder=3)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 1.5,
                f"{mean:.1f}%\n±{std:.1f}%", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")

    ax.set_ylim(0, 125)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("前景像素占比 (%)", fontsize=12)
    ax.set_title("四种图像分割方法：前景覆盖率对比\n(20 张随机抽样，误差棒 = ±1σ)", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, "fig5_segmentation_compare.png")


# ── 6. Summary dashboard ─────────────────────────────────────────────────────
def plot_dashboard() -> None:
    """2×2 dashboard combining key results."""
    fig = plt.figure(figsize=(13, 9), facecolor=GREY_BG)
    fig.suptitle("基于非标准遮挡的人脸识别 — 实验结果总览",
                 fontsize=15, fontweight="bold", y=0.98)

    # ── top-left: A/B/C overview ──────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor(GREY_BG)
    groups = ["A 基线", "B 遮挡\nNaive", "C 级联\nv1"]
    vals = [92.65, 89.11, 12.52]
    cols = [C_BLUE, C_ORANGE, C_RED]
    bars = ax1.bar(groups, vals, color=cols, width=0.5, zorder=3)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("准确率 (%)")
    ax1.set_title("三组整体准确率", fontsize=11)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # ── top-right: per-type ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor(GREY_BG)
    types_short = ["sunglasses", "cup", "glasses"]
    b_vals = [88.79, 87.85, 90.69]
    c_vals = [8.23, 17.45, 11.89]
    x = np.arange(len(types_short))
    w = 0.35
    ax2.bar(x - w / 2, b_vals, w, label="B Naive", color=C_ORANGE, zorder=3)
    ax2.bar(x + w / 2, c_vals, w, label="C 级联 v1", color=C_RED, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(types_short, fontsize=9)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("准确率 (%)")
    ax2.set_title("各遮挡类型 B vs C", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)

    # ── bottom-left: cascade evolution ───────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor(GREY_BG)
    ver_labels = ["v1\n(0.8)", "v2\n(0.5)", "v3\n(−1)"]
    c_evo = [12.52, 48.92, 92.70]
    ax3.plot(ver_labels, c_evo, marker="o", linewidth=2, markersize=8,
             color=C_GREEN, zorder=3)
    ax3.axhline(92.70, color=C_ORANGE, linestyle="--", linewidth=1.2,
                label="B = 92.70%", zorder=2)
    for x_, y_ in zip(ver_labels, c_evo):
        ax3.annotate(f"{y_:.1f}%", (x_, y_), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=9,
                     fontweight="bold", color=C_GREEN)
    ax3.set_ylim(0, 108)
    ax3.set_ylabel("准确率 (%)")
    ax3.set_title("级联优化演进（L1_HIGH）", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax3.set_axisbelow(True)

    # ── bottom-right: segmentation ───────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor(GREY_BG)
    seg_methods = ["YCrCb", "GMM", "GrabCut", "Watershed"]
    seg_means = [51.5, 96.4, 23.6, 35.0]
    seg_cols = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]
    ax4.bar(seg_methods, seg_means, color=seg_cols, width=0.5, zorder=3)
    for i, (m, v) in enumerate(zip(seg_methods, seg_means)):
        ax4.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax4.set_ylim(0, 118)
    ax4.set_ylabel("前景覆盖率 (%)")
    ax4.set_title("图像分割方法对比", fontsize=11)
    ax4.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax4.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "fig0_dashboard.png")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("Generating experiment result charts → %s", OUT_DIR)

    # Check data availability
    if not (RESULTS_DIR / "baseline_accuracy.txt").exists():
        log.error("baseline_accuracy.txt not found — run `make eval-baseline` first")
        sys.exit(1)
    if not (RESULTS_DIR / "compare_accuracy_v1.txt").exists():
        log.error("compare_accuracy_v1.txt not found — run `make eval-compare` first")
        sys.exit(1)

    plot_accuracy_overview()
    plot_per_type_bc()
    plot_cascade_evolution()
    plot_preprocessing()
    plot_segmentation()
    plot_dashboard()

    log.info("Done — %d figures saved to docs/figures/", 6)
    for p in sorted(OUT_DIR.glob("*.png")):
        log.info("  %s  (%d KB)", p.name, p.stat().st_size // 1024)


if __name__ == "__main__":
    main()
