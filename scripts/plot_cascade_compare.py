"""
生成级联策略全量 vs 50样本对比图表。

数据来源：
  50样本: data/results/cascade_ablation.txt (B/v1/v2/v3)
  全量:   data/results/compare_accuracy_v1.txt (v1)
          data/results/compare_accuracy.txt    (v3, L1_HIGH=-1)

输出：
  docs/figures/cascade_fullrun_compare.png
"""

from pathlib import Path
import re

import matplotlib
import matplotlib.font_manager as fm
matplotlib.use("Agg")
for font in ["Arial Unicode MS", "Hei", "Heiti TC", "PingFang SC"]:
    if any(f.name == font for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = font
        break

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("data/results")
OUT = Path("docs/figures/cascade_fullrun_compare.png")

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    "B":  "#DD8452",   # orange
    "v1": "#C44E52",   # red
    "v2": "#8172B3",   # purple
    "v3": "#55A868",   # green
}
GREY_BG = "#F8F9FA"
OCC_TYPES = ["sunglasses", "cup", "glasses"]
OCC_LABELS = ["Sunglasses\n(墨镜)", "Cup\n(水杯)", "Glasses\n(眼镜)"]

# ── load 50-sample from cascade_ablation.txt ─────────────────────────────────
def load_ablation() -> dict:
    """Returns {key: {all, sunglasses, cup, glasses}} for B/v1/v2/v3."""
    txt = (RESULTS_DIR / "cascade_ablation.txt").read_text()
    data = {}
    key_map = {
        "B Naive": "B",
        "C v1 (L1_HIGH=0.80, replace)": "v1",
        "C v2 (L1_HIGH=0.50, best-of-two)": "v2",
        "C v3 (L1_HIGH=-1, 禁用L2)": "v3",
    }
    for line in txt.splitlines():
        for raw, key in key_map.items():
            if raw in line:
                nums = re.findall(r"(\d+\.\d+)%", line)
                if len(nums) >= 4:
                    data[key] = {
                        "all": float(nums[0]) / 100,
                        "sunglasses": float(nums[1]) / 100,
                        "cup": float(nums[2]) / 100,
                        "glasses": float(nums[3]) / 100,
                    }
    return data


def load_full(path: Path) -> dict:
    """Returns {all, sunglasses, cup, glasses} from compare_accuracy_*.txt."""
    txt = path.read_text()
    result = {}
    m = re.search(r"Group C.*?(\d+\.\d+)%", txt)
    if m:
        result["all"] = float(m.group(1)) / 100
    for occ in OCC_TYPES:
        m = re.search(rf"{occ}.*?C=([\d.]+)", txt)
        if m:
            result[occ] = float(m.group(1))
    return result


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    abl = load_ablation()

    full_v1 = load_full(RESULTS_DIR / "compare_accuracy_v1.txt")
    full_v3 = load_full(RESULTS_DIR / "compare_accuracy.txt")

    # Also load B full-run per-type from v3 file (B col)
    txt_v3 = (RESULTS_DIR / "compare_accuracy.txt").read_text()
    full_B = {"all": 0.8911}
    for occ in OCC_TYPES:
        m = re.search(rf"{occ}.*?B=([\d.]+)", txt_v3)
        if m:
            full_B[occ] = float(m.group(1))

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=GREY_BG)
    fig.suptitle("级联策略 v1/v2/v3 全量 & 50样本对比", fontsize=15, fontweight="bold", y=0.98)

    # Row 1: 整体准确率对比（左=50样本，右=全量）
    # Row 2: 各遮挡类型细分（左=50样本，右=全量）
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.28,
                          left=0.07, right=0.97, top=0.92, bottom=0.10)
    ax_top_l = fig.add_subplot(gs[0, 0])
    ax_top_r = fig.add_subplot(gs[0, 1])
    ax_bot_l = fig.add_subplot(gs[1, 0])
    ax_bot_r = fig.add_subplot(gs[1, 1])

    keys   = ["B", "v1", "v2", "v3"]
    xlbls  = ["B\nNaive", "C v1\n(L1=0.80)", "C v2\n(L1=0.50)", "C v3\n(L1=−1)"]
    colors = [C["B"], C["v1"], C["v2"], C["v3"]]

    def bar_plot(ax, vals_pct, labels, title, footnote=""):
        ax.set_facecolor(GREY_BG)
        has_data = [v is not None for v in vals_pct]
        xs = range(len(vals_pct))
        for i, (v, lbl, col, hd) in enumerate(zip(vals_pct, labels, colors, has_data)):
            if hd:
                bar = ax.bar(i, v, color=col, width=0.55, zorder=3)
                ax.text(i, v + 1.2, f"{v:.2f}%", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
            else:
                ax.bar(i, 95, color="#CCCCCC", width=0.55, zorder=3,
                       hatch="//", edgecolor="#AAAAAA", linewidth=0.8)
                ax.text(i, 50, "未运行", ha="center", va="center",
                        fontsize=10, color="#666666")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Top-1 准确率 (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        if footnote:
            ax.text(0.99, 0.01, footnote, transform=ax.transAxes,
                    fontsize=8, color="#888888", ha="right", va="bottom")

    # 50-sample overall
    vals_50 = [abl[k]["all"] * 100 for k in keys]
    bar_plot(ax_top_l, vals_50, xlbls, "整体准确率（50 身份抽样）", "* 抽样实验，仅供趋势参考")

    # Full-run overall
    full_overall = {
        "B": full_B["all"] * 100 if "all" in full_B else None,
        "v1": full_v1.get("all") * 100 if full_v1.get("all") else None,
        "v2": None,   # not run
        "v3": full_v3.get("all") * 100 if full_v3.get("all") else None,
    }
    vals_full = [full_overall[k] for k in keys]
    bar_plot(ax_top_r, vals_full, xlbls, "整体准确率（全量，1,680 身份）", "⊘ = 未运行")

    # ── per-type grouped bars ─────────────────────────────────────────────────
    def grouped_bar(ax, data_map, title, footnote=""):
        """data_map: {key: {occ: val_or_None}}"""
        ax.set_facecolor(GREY_BG)
        x = np.arange(len(OCC_TYPES))
        n = len(keys)
        w = 0.18
        offsets = np.linspace(-(n - 1) * w / 2, (n - 1) * w / 2, n)

        for i, (k, col) in enumerate(zip(keys, colors)):
            vals = [data_map[k].get(occ) for occ in OCC_TYPES]
            for j, v in enumerate(vals):
                if v is not None:
                    b = ax.bar(x[j] + offsets[i], v * 100, w, color=col,
                               label=k if j == 0 else "", zorder=3)
                    ax.text(x[j] + offsets[i], v * 100 + 0.8,
                            f"{v*100:.1f}", ha="center", va="bottom", fontsize=7.5)
                else:
                    ax.bar(x[j] + offsets[i], 90, w, color="#CCCCCC", zorder=3,
                           hatch="//", edgecolor="#AAAAAA", linewidth=0.5,
                           label="" if j != 0 else "")
                    ax.text(x[j] + offsets[i], 45, "N/A", ha="center", va="center",
                            fontsize=7, color="#888888", rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(OCC_LABELS, fontsize=10)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Top-1 准确率 (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        if footnote:
            ax.text(0.99, 0.01, footnote, transform=ax.transAxes,
                    fontsize=8, color="#888888", ha="right", va="bottom")

        # legend
        patches = [mpatches.Patch(color=C[k], label=f"{'B Naive' if k=='B' else 'C '+k}") for k in keys]
        na_patch = mpatches.Patch(color="#CCCCCC", hatch="//", edgecolor="#AAAAAA", label="未运行")
        ax.legend(handles=patches + [na_patch], fontsize=8, loc="lower right")

    # 50-sample per-type
    abl_per = {k: {occ: abl[k][occ] for occ in OCC_TYPES} for k in keys}
    grouped_bar(ax_bot_l, abl_per, "各遮挡类型（50 身份抽样）", "* 抽样实验")

    # Full-run per-type
    full_per = {
        "B": {occ: full_B.get(occ) for occ in OCC_TYPES},
        "v1": {occ: full_v1.get(occ) for occ in OCC_TYPES},
        "v2": {occ: None for occ in OCC_TYPES},
        "v3": {occ: full_v3.get(occ) for occ in OCC_TYPES},
    }
    grouped_bar(ax_bot_r, full_per, "各遮挡类型（全量，1,680 身份）", "⊘ = 未运行")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=GREY_BG)
    plt.close(fig)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
