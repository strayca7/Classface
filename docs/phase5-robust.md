# Phase 5 — 鲁棒识别：眼周裁剪 + 两级级联

## 概述

第五阶段在第三阶段基线（ArcFace + 余弦相似度）的基础上，针对遮挡场景提出**两级级联识别策略**，将遮挡对准确率的影响降至最低。

核心思路：
1. **Level 1**：对合成遮挡图像进行全脸特征提取，置信度高时直接输出。
2. **Level 2**：置信度不足时，提取**眼周区域（eye-region crop）**再次检索——眼周区域不受水杯/眼镜遮挡的影响。

---

## 模块一：眼周裁剪（`scripts/crop.py`）

### 裁剪区域定义

```
y_top    = bbox_top - PAD_TOP (4 px)
y_bottom = kps[nose_y] + PAD_BOTTOM (4 px)
x_left   = bbox_left - PAD_SIDE (4 px)
x_right  = bbox_right + PAD_SIDE (4 px)
→ resize 到 112 × 112（ArcFace 标准输入）
```

检测失败时退化为图像上方 55%。

### 使用方法

```bash
uv run python scripts/crop.py [--limit N]
```

| 输出 | 说明 |
|------|------|
| `data/cropped/gallery/<identity>/<img>.jpg` | Gallery 眼周裁剪图 |
| `data/cropped/query/<occ_type>/<identity>/<img>.jpg` | Query 眼周裁剪图 |
| `data/features/gallery_cropped.npy` | Gallery 眼周裁剪特征矩阵（N × 512） |
| `data/features/gallery_cropped_labels.json` | 对应身份标签列表 |
| `data/cropped/vis/<identity>.jpg` | 可视化：原图 \| 裁剪图（side-by-side） |

### 特征提取

- 先尝试完整 InsightFace 流水线（检测 → 对齐 → ArcFace）
- 回退：直接调用 recognition model `.get_feat()`
- 保证返回 `(512,)` 一维向量（`.flatten()`）

---

## 模块二：两级级联评估（`scripts/evaluate.py --mode compare`）

### 三组对比实验

| 组别 | 策略 | 数据 |
|-----|------|------|
| A | 基线（全脸，干净图像） | `data/processed/lfw/` |
| B | Naive（全脸，遮挡图像） | `data/synthetic/` |
| C | 两级级联（遮挡图像） | `data/synthetic/` + `data/cropped/query/` |

### 两级级联逻辑

```
L1_HIGH = 0.8,  L1_LOW = 0.4

L1 全脸特征 → cosine_top1(gallery)
    ├─ score > 0.8  → 直接输出 pred_L1
    ├─ 0.4 ≤ score < 0.8 → L2 眼周裁剪 → cosine_top1(gallery_cropped)
    └─ score < 0.4  → 输出 "unknown"（拒识）
```

### 使用方法

```bash
uv run python scripts/evaluate.py --mode compare [--limit N]
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `data/results/eval_compare_v3_full.txt` | A/B/C 总体准确率 + 各遮挡类型细分 |
| `data/results/figures/plot_compare_v3_full.png` | 三组准确率柱状图（300 dpi） |
| `data/results/figures/plot_occlusion_type_v3_full.png` | 各遮挡类型 B vs C 折线图（300 dpi） |

---

## Makefile 目标

```makefile
make crop                         # 眼周裁剪（全量，>10 min，见 TODO.md）
make crop ARGS="--limit 5"        # 调试裁剪
make eval-compare                 # 三组对比实验（全量，>10 min，见 TODO.md）
make eval-compare ARGS="--limit 5"  # 调试评估
```

---

## 数据依赖关系

```
数据准备（Phase 1-2）
    ↓
基线特征提取（Phase 3）→ gallery.npy / gallery_labels.json
    ↓
遮挡合成（Phase 4）→ data/synthetic/{cup,glasses,sunglasses}/
    ↓
眼周裁剪（Phase 5 crop）→ gallery_cropped.npy / data/cropped/query/
    ↓
两级级联评估（Phase 5 compare）→ eval_compare_v3_full.txt + figures/
```

---

## 尚待完成（>10 min，写入 TODO.md）

| 任务 | 命令 | 预计时长 |
|------|------|---------|
| 全量遮挡合成 | `make generate` | ~18.7 min |
| 全量眼周裁剪 | `make crop` | >10 min |
| 全量对比评估 | `make eval-compare` | >10 min |

---

## 依赖

- `insightface[onnxruntime]`（buffalo_l）
- `opencv-python`、`numpy`、`matplotlib`
- Phase 3 gallery features：`data/features/gallery.npy`
- Phase 4 synthetic data：`data/synthetic/`
