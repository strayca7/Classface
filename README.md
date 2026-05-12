# 基于"特定场景与非标准遮挡"的人脸识别

> 数字图像处理课程设计 — 智慧课堂/自习室无感签到系统

## 项目概述

本项目针对**课堂/自习室场景**中的非标准遮挡（手托腮、水杯遮挡、眼镜/墨镜遮挡），设计了一套基于图像处理流水线的人脸识别系统。

与口罩遮挡不同，这类遮挡不规则、位置多变，难以用通用模型直接处理。本系统不依赖图像修复模型，而是通过**眼周动态裁剪 + 两级级联比对**策略，在不修改识别网络结构的前提下，大幅提升遮挡场景下的识别准确率。

**技术流水线**：
```
原始人脸图像
  → 第一阶段：图像预处理（YCrCb 均衡化 + 双眼对齐 + 112×112 裁剪）
  → 第二阶段：图像分割（YCrCb/GMM 肤色 + GrabCut/Watershed 前景）
  → 第三阶段：基线识别（InsightFace ArcFace gallery + 余弦相似度）
  → 第四阶段：遮挡数据合成（InsightFace 5-kps 关键点 + Alpha 掩膜）
  → 第五阶段：两级级联识别 + 三组对比实验
```

详细技术路线见 [`procedures.md`](procedures.md)。

---

## 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)（包管理工具）

---

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd dip

# 安装所有依赖（insightface、opencv、scikit-learn 等）
uv sync
```

---

## 快速开始（完整流水线）

```bash
make download-lfw        # 1. 下载 LFW 数据集（~232MB）
make prepare-dataset     # 2. 生成 gallery/query 分割清单
make preprocess          # 3. 批量预处理（~2min）
make build-gallery       # 4. 提取 gallery 特征（~8min）
make segment-skin        # 5a. 肤色分割
make segment-face        # 5b. 前景分割
make eval-seg            # 5c. 分割方法对比
make generate            # 6. 合成遮挡数据集（~72min）
make eval-baseline       # 7. 基线评估（~32min）
make crop                # 8. 眼周裁剪（~102min）
make eval-compare        # 9. 三组对比实验（~96min）
```

---

## Makefile 命令说明

| 命令 | 说明 | 预计耗时 |
|------|------|---------|
| `make setup` | 创建项目所需目录结构 | <1s |
| `make download-lfw` | 下载 LFW-funneled 数据集并解压 | 网络决定 |
| `make prepare-dataset` | 筛选身份，生成 gallery/query 分割 JSON | <1s |
| `make preprocess` | 批量图像预处理（均衡化 + 对齐 + 112×112） | ~2min |
| `make build-gallery` | 提取 gallery ArcFace 特征，缓存 gallery.npy | ~8min |
| `make segment-skin` | YCrCb 阈值 + GMM 肤色分割 | ~5min |
| `make segment-face` | GrabCut + Watershed 前景分割 | ~5min |
| `make eval-seg` | 四方法对比可视化与统计 | ~1min |
| `make generate` | 合成遮挡图像（cup/glasses/sunglasses） | ~72min |
| `make eval-baseline` | 基线 Top-1 准确率评估 | ~32min |
| `make crop` | 眼周裁剪 + gallery_cropped.npy 预计算 | ~102min |
| `make eval-compare` | 三组对比实验（A 基线/B 遮挡naive/C 两级策略） | ~96min |
| `make clean` | 删除所有生成产物 | <1s |

支持 `ARGS="--limit N"` 快速调试，例如 `make generate ARGS="--limit 10"`。

---

## 实验结果

### 数据集规模

| 数据 | 数量 |
|------|------|
| LFW 原始身份 | 5,749 个，13,233 张 |
| 有效身份（≥2 张） | 1,680 个 |
| Gallery 图像 | 1,680 张 |
| Query 图像（干净） | 7,484 张 |
| 合成遮挡图像 | 22,452 张（3 类 × 7,484） |

### 图像分割方法对比

| 方法 | 平均前景占比 | 说明 |
|------|------------|------|
| YCrCb 阈值（Kovac 椭圆） | 51.5% | 传统阈值，速度最快 |
| GMM 肤色分割 | 96.4% | 几乎整张人脸均为肤色 |
| GrabCut | 23.6% | 前景偏保守 |
| Watershed | 35.0% | 区域增长，噪声较多 |

可视化对比图：`data/results/figures/segmentation_compare.png`

### 人脸识别准确率

| 组别 | 策略 | Top-1 准确率 |
|------|------|------------|
| A — 基线 | 干净图像，全脸 ArcFace | **92.65%** (6934/7484) |
| B — 遮挡 Naive | 遮挡图像，直接全脸检索 | **89.11%** (20007/22452) |
| C — 两级级联 | 遮挡图像，眼周裁剪二次比对 | **12.52%** (2812/22452) |

各遮挡类型详细结果（B vs C）：

| 遮挡类型 | B（Naive） | C（两级级联） | n |
|---------|-----------|-------------|---|
| sunglasses | 88.79% | 8.23% | 7,484 |
| cup | 87.85% | 17.45% | 7,484 |
| glasses | 90.69% | 11.89% | 7,484 |

> **分析**：B 组（89.11%）在遮挡下仍保持较高准确率，表明合成遮挡对全脸特征的干扰有限。C 组两级级联准确率极低（12.52%），主要原因是 L1_HIGH=0.8 阈值过高，大部分遮挡图像的全脸相似度在 0.4–0.8 区间被路由至 L2，而 L2 的眼周裁剪特征（gallery_cropped.npy）与合成遮挡图的眼周区域特征分布不匹配，导致 L2 大量误识。实际应用中需重新调优 L1 阈值或在眼周图像上重新训练特征提取器。

结果写入 `data/results/compare_accuracy.txt`，图表保存至 `data/results/figures/`。

### 查看实验结果

```bash
# 文字结果
cat data/results/baseline_accuracy.txt
cat data/results/compare_accuracy.txt     # eval-compare 完成后可用

# 图表（macOS）
open data/results/figures/accuracy_compare.png   # A/B/C 三组柱状图
open data/results/figures/occlusion_type.png      # 各遮挡类型 B vs C 折线图
open data/results/figures/segmentation_compare.png  # 分割方法对比
```

### 两级级联策略说明

```
Level 1（全脸）：InsightFace ArcFace → cosine_top1(gallery)
    ├─ score > 0.8  → 直接输出身份（高置信）
    ├─ 0.4 ≤ score ≤ 0.8 → Level 2：眼周裁剪 → cosine_top1(gallery_cropped)
    └─ score < 0.4  → 标记 "unknown"（拒识）
```

遮挡（水杯/眼镜/墨镜）主要影响嘴部和眼部区域，但**眼周区域**在这类遮挡下通常保持可见，Level 2 利用这一特点实现稳健识别。

---

## 目录结构

```
dip/
├── data/
│   ├── raw/
│   │   ├── lfw/              # LFW 原始数据（download-lfw 后填充）
│   │   └── lfw_filtered.json # gallery/query 分割清单（1680 身份）
│   ├── processed/lfw/        # 预处理后图像（112×112，YCrCb 均衡化+对齐）
│   ├── overlays/             # 遮挡贴图素材（58 张 PNG，已纳入 git）
│   ├── features/             # InsightFace 特征向量缓存（.npy）
│   │   ├── gallery.npy               # 全脸 gallery (1680, 512)
│   │   ├── gallery_labels.json
│   │   ├── gallery_cropped.npy       # 眼周 gallery (1680, 512)
│   │   └── gallery_cropped_labels.json
│   ├── synthetic/            # 合成遮挡数据集（cup/glasses/sunglasses）
│   ├── cropped/              # 眼周裁剪图像（gallery/query/vis）
│   ├── segmented/            # 分割结果掩膜
│   └── results/              # 评估结果与对比图表
├── docs/
│   ├── phase1-dataset.md         # 数据集准备模块说明
│   ├── phase1-preprocess.md      # 图像预处理模块说明
│   ├── phase2-segmentation.md    # 图像分割模块说明
│   ├── phase3-baseline.md        # 基线识别系统说明
│   ├── phase4-synthesis.md       # 遮挡合成模块说明
│   └── phase5-robust.md          # 鲁棒识别与两级级联说明
├── scripts/
│   ├── prepare_dataset.py    # LFW 筛选与分割
│   ├── preprocess.py         # 图像预处理（均衡化+对齐+裁剪）
│   ├── segment_skin.py       # YCrCb/GMM 肤色分割
│   ├── segment_face.py       # GrabCut/Watershed 前景分割
│   ├── eval_segmentation.py  # 分割方法对比评估
│   ├── validate_segmentation.py  # 分割验证断言
│   ├── build_gallery.py      # gallery 特征提取与缓存
│   ├── evaluate.py           # 基线/对比评估（--mode baseline/compare）
│   ├── generate_cover.py     # 遮挡图像合成
│   └── crop.py               # 眼周裁剪与特征预计算
├── procedures.md     # 完整技术路线与实施步骤
├── TODO.md           # 各阶段任务进度
├── CHANGELOG.md      # 版本变更日志
├── Makefile          # 常用命令
└── pyproject.toml    # 项目依赖
```

---

## 开发规范

- **提交信息**：遵循 [Conventional Commits](https://www.conventionalcommits.org/)，格式 `<type>(<scope>): <description>`
- **代码格式**：`uv tool run ruff check scripts/` + `uv fmt`
- **版本管理**：语义化版本（SemVer），见 `VERSION` 文件
- **日志格式**：`%(asctime)s [%(levelname)s] %(name)s: %(message)s`

---

## 文档

| 文档 | 内容 |
|------|------|
| [`docs/phase1-dataset.md`](docs/phase1-dataset.md) | LFW 数据集准备与 gallery/query 分割 |
| [`docs/phase1-preprocess.md`](docs/phase1-preprocess.md) | 直方图均衡化、人脸对齐、裁剪策略 |
| [`docs/phase2-segmentation.md`](docs/phase2-segmentation.md) | 肤色/前景分割方法与定量对比 |
| [`docs/phase3-baseline.md`](docs/phase3-baseline.md) | ArcFace 特征提取与余弦相似度基线 |
| [`docs/phase4-synthesis.md`](docs/phase4-synthesis.md) | 关键点定位与 alpha blending 合成 |
| [`docs/phase5-robust.md`](docs/phase5-robust.md) | 眼周裁剪、两级级联策略、对比实验 |

---

## 贡献指南

1. Fork 本仓库，基于 `main` 分支新建特性分支（如 `feat/phase2-baseline`）
2. 修改代码后运行 `uv fmt` 格式化
3. 提交前确保代码可正常运行
4. 提交信息遵循 Conventional Commits 规范
5. 提交 PR 并描述变更内容
