# 基于"特定场景与非标准遮挡"的人脸识别

> 数字图像处理课程设计 — 智慧课堂/自习室无感签到系统

## 项目概述

本项目针对**课堂/自习室场景**中的非标准遮挡（手托腮、水杯遮挡、眼镜/墨镜遮挡），设计了一套基于图像处理流水线的人脸识别系统。

**技术流水线**：
```
原始人脸图像
  → 第一阶段：图像预处理（YCrCb 均衡化 + 双眼对齐 + 112×112 裁剪）
  → 第二阶段：图像分割（深度学习 — ResUNet：ResNet-18 编码器 + U-Net 解码器）
  → 第三阶段：基线识别（InsightFace ArcFace gallery + 余弦相似度）
  → 第四阶段：遮挡数据合成（InsightFace 5-kps 关键点 + Alpha 掩膜）
  → 第五阶段：两级级联识别 + 三组对比实验（A基线/B遮挡Naive/C级联）
```

详细技术路线见 [`procedures.md`](procedures.md)。

---

## 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)（包管理工具）
- GPU（可选，强烈推荐用于深度学习训练）：
  - Apple Silicon M3/M4：MPS 自动启用（无需额外配置）
  - NVIDIA GPU（RTX 4060 等）：需安装 CUDA 版 torch（见下方安装说明）

---

## 安装

```bash
git clone <repo-url>
cd dip
uv sync          # 安装所有依赖（torch、insightface、opencv 等）
```

**NVIDIA GPU（CUDA 加速）**：`uv sync` 后需额外安装 CUDA 版 torch：

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证 GPU 是否正确识别：

```bash
uv run python -c "import sys; sys.path.insert(0,'scripts'); from dl_model import get_device; print(get_device())"
# Apple M3  → device(type='mps')
# RTX 4060  → device(type='cuda')
# 无 GPU    → device(type='cpu')
```

---

## 快速开始（完整流水线）

> 各步骤预计时间以 Apple M3（MPS）/ NVIDIA RTX 4060（CUDA）/ 纯 CPU 分别标注。

```bash
make download-lfw        # 1. 下载 LFW 数据集（~232MB，网络决定）
make prepare-dataset     # 2. 生成 gallery/query 分割清单（<1s）
make preprocess          # 3. 批量预处理（M3≈2min / RTX≈1min / CPU≈2min）

# ── 第二阶段：深度学习分割 ──────────────────────────────────────
make segment-face        # 4. 生成 ResUNet 训练伪标签（GrabCut，CPU≈5min）
make dl-train            # 5. 训练 ResUNet（M3≈15min / RTX4060≈4min / CPU≈90min）
make dl-segment          # 6. 批量推理，输出 dl_unet 掩膜（M3≈5min / RTX≈2min / CPU≈15min）
make dl-eval-seg         # 7. DL vs 传统方法对比评估（≈1min）

# ── 第三–五阶段：识别系统 ───────────────────────────────────────
make build-gallery       # 8. 提取 gallery 特征（M3≈8min / RTX≈2min / CPU≈8min）
make generate            # 9. 合成遮挡数据集（M3≈72min / RTX≈20min）
make eval-baseline       # 10. 基线评估（M3≈32min / RTX≈8min）
make crop                # 11. 眼周裁剪（M3≈102min / RTX≈25min）
make eval-compare        # 12. 三组对比实验（M3≈137min / RTX≈35min）
```

支持 `ARGS="--limit N"` 快速调试，例如 `make dl-train ARGS="--epochs 3 --limit 200"`。

### 仅运行深度学习分割（最小步骤）

```bash
make download-lfw && make prepare-dataset && make preprocess
make segment-face   # 生成 ResUNet 训练伪标签（GrabCut）
make dl-train       # 训练 U-Net（M3≈15min）
make dl-segment     # 批量推理
make dl-eval-seg    # 对比评估
```

---

## Makefile 命令说明

### 主流程命令

| 命令 | 说明 | Apple M3 | RTX 4060 | CPU |
|------|------|----------|----------|-----|
| `make setup` | 创建项目目录结构 | <1s | <1s | <1s |
| `make download-lfw` | 下载 LFW-funneled 数据集并解压 | 网络决定 | — | — |
| `make prepare-dataset` | 筛选身份，生成 gallery/query 分割 JSON | <1s | — | — |
| `make preprocess` | 批量图像预处理（均衡化+对齐+112×112） | ~2min | ~1min | ~2min |
| `make segment-face` | **生成 ResUNet 训练伪标签**（GrabCut） | ~5min | — | ~5min |
| `make dl-train` | **训练 ResUNet**（20 epoch，GrabCut 伪标签） | **~15min** | **~4min** | ~90min |
| `make dl-segment` | **批量推理**，输出 data/segmented/dl_unet/ | **~5min** | **~2min** | ~15min |
| `make dl-eval-seg` | **DL vs 传统对比评估**（IoU/Dice/前景占比） | ~1min | ~1min | ~1min |
| `make build-gallery` | 提取 gallery ArcFace 特征，缓存 gallery.npy | ~8min | ~2min | ~8min |
| `make generate` | 合成遮挡图像（cup/glasses/sunglasses） | ~72min | ~20min | ~72min |
| `make eval-baseline` | 基线 Top-1 准确率评估 | ~32min | ~8min | ~32min |
| `make crop` | 眼周裁剪 + gallery_cropped.npy 预计算 | ~102min | ~25min | ~102min |
| `make eval-compare` | 三组对比实验（A基线/B遮挡naive/C两级策略） | ~137min | ~35min | ~137min |
| `make clean` | 删除所有生成产物 | <1s | — | — |

### 传统方法命令（历史对比数据，已完成）

> 以下命令对应传统机器学习分割方法，实验结果见"图像分割方法对比"表格。无需重新运行。

| 命令 | 说明 |
|------|------|
| `make segment-skin` | YCrCb 阈值 + GMM 肤色分割（结果已记录） |
| `make eval-seg` | 传统四方法对比可视化与统计（结果已记录） |

---

## 实验结果

### 数据集规模

| 数据 | 数量 |
|------|------|
| LFW 原始身份 | 5,749 个，13,233 张 |
| 有效身份（≥2 张） | 1,680 个 |
| Gallery 图像 | 1,680 张（每位身份第 1 张） |
| Query 图像（干净） | 7,484 张（每位身份第 2 张起） |
| 合成遮挡图像 | 22,452 张（3 类 × 7,484） |

### 图像预处理

| 指标 | 数值 |
|------|------|
| 处理成功率 | 13,233/13,233（100%） |
| 双眼对齐率 | 46.6%（其余退化为中心裁剪） |
| 对齐旋转角 \|angle\| 均值 | 4.0°（最大保护 20°） |
| 输出尺寸 | 112×112×3，全部成功 |

### 图像分割方法对比

**当前方法（深度学习）：**

| 方法 | 前景占比 | IoU vs GrabCut | Dice vs GrabCut |
|------|----------|----------------|-----------------|
| **ResUNet（深度学习）** | *(待运行 `make dl-train && make dl-segment && make dl-eval-seg`)* | — | — |

**传统方法历史数据（保留作对比）：**

| 方法 | 平均前景占比 | 特点 |
|------|------------|------|
| YCrCb 阈值（Kovac 椭圆） | 51.5% | 速度最快，边界较粗糙 |
| GMM 肤色分割 | 96.4% | 几乎整张人脸均判为肤色，召回率高 |
| GrabCut（用作伪标签来源） | 23.6% | 前景保守，边缘精细 |
| Watershed | 35.0% | 区域增长，噪声较多 |

可视化对比图：`data/results/figures/dl_segmentation_compare.png`（DL vs 传统 6 列对比）

### 人脸识别准确率

#### 三组对比实验

| 组别 | 策略 | Top-1 准确率 | 样本数 |
|------|------|------------|--------|
| **A — 基线** | 干净图像，全脸 ArcFace | **92.65%** (6934/7484) | 7,484 |
| **B — 遮挡 Naive** | 遮挡图像，直接全脸识别 | **89.11%** (20007/22452) | 22,452 |
| **C — 两级级联（优化版）** | 遮挡图像，L1_HIGH=-1（等效 B） | **89.11%** | 22,452 |

#### 各遮挡类型详细（B 组）

| 遮挡类型 | Top-1 准确率 | 样本数 | 说明 |
|---------|------------|--------|------|
| sunglasses（墨镜） | 88.79% | 7,484 | 覆盖眼部，干扰最大 |
| cup（水杯） | 87.85% | 7,484 | 覆盖嘴/下颌，干扰中等 |
| glasses（眼镜） | 90.69% | 7,484 | 镜框较细，干扰最小 |

#### 级联策略演进对比

| 版本 | L1_HIGH | C 准确率 | 说明 |
|------|---------|---------|------|
| v1（原始） | 0.8 | **12.52%** | 87% 图像被错误路由至 L2，L2 特征质量差 |
| v2（best-of-two） | 0.5 | ~48.92%* | L2 仍引入噪声 |
| v3（优化版，当前） | -1 | **89.11%** | 禁用 L2，全部走 L1，消除级联劣化 |

*50 身份抽样结果，仅供参考。

### 关键发现与分析

**1. ArcFace 对轻度遮挡具有内置鲁棒性**
A → B 准确率仅下降 **3.54pp**（92.65% → 89.11%），说明 ArcFace 512-d 特征在局部遮挡下仍保留足够身份信息。课堂/自习室场景中的水杯、眼镜等遮挡属"轻度遮挡"，不足以严重损害全脸特征。

**2. 眼周局部裁剪不能改善识别（架构限制）**
- 原设计假设：眼周区域在遮挡下保持清晰 → 眼周特征更纯净 → 提升准确率
- 实际结论：InsightFace ArcFace 训练于标准对齐的完整人脸，在局部裁剪（112×112 眼周）上特征质量显著下降，路由至 L2 反而劣化结果
- glasses/sunglasses 贴图直接覆盖眼部，L2 眼周裁剪引入遮挡物特征，加剧误识

**3. 级联阈值设置的重要性**
v1 的 L1_HIGH=0.8 使 87% 的图像被路由至质量较低的 L2，导致 C=12.52%。**阈值是级联策略的核心超参数**；实际部署时应在验证集上调优。

**4. 改进方向**
- 针对各遮挡类型训练专用局部特征提取器（如在遮挡人脸数据上微调 ArcFace）
- 对遮挡区域做图像修复（inpainting）后再识别
- 在眼部/口部分别提取特征并加权融合

### 查看实验结果

```bash
# 文字结果
cat data/results/baseline_accuracy.txt
cat data/results/compare_accuracy.txt
cat data/results/dl_segmentation_stats.txt   # DL 分割评估结果（dl-eval-seg 后生成）

# 图表（macOS）
open data/results/figures/accuracy_compare.png           # A/B/C 三组柱状图
open data/results/figures/occlusion_type.png             # 各遮挡类型 B vs C 折线图
open data/results/figures/dl_segmentation_compare.png    # DL vs 传统 6 列对比（dl-eval-seg 后生成）
open data/results/figures/dl_fg_ratio_compare.png        # 各方法前景占比柱状图（dl-eval-seg 后生成）
```

### 两级级联策略说明（优化版）

```
遮挡图像（water cup / glasses / sunglasses）
  → Level 1：全脸 InsightFace ArcFace → cosine_top1(gallery.npy)
       ├─ score > L1_HIGH（默认 -1，即全部）→ 直接输出身份
       └─ score ≤ L1_HIGH → Level 2（仅检测完全失败时触发，实际极少）
              眼周裁剪 112×112 → cosine_top1(gallery_cropped.npy) → 输出身份
```

> **设计说明**：L1_HIGH=-1 在当前架构下为最优设置，原因见"关键发现"第 2 条。
> 若将来替换为在局部人脸上微调的特征提取器，可将 L1_HIGH 调回 0.5~0.8。

---

## 深度学习分割（Phase 2 重构）

### 架构：ResUNet

```
输入：3×112×112 BGR 图像（ImageNet 归一化）
  │
  ├─ Encoder（ResNet-18，ImageNet 预训练）
  │    stem  → 64×56×56
  │    layer1 → 64×28×28   ┐
  │    layer2 → 128×14×14  │ skip connections
  │    layer3 → 256×7×7    │
  │    layer4 → 512×4×4    ┘  (bottleneck)
  │
  └─ Decoder（双线性上采样 + 跳跃连接 + ConvBNReLU×2）
       4×4 → 7×7 → 14×14 → 28×28 → 56×56 → 112×112
  │
输出：1×112×112 sigmoid 概率图 → 阈值 0.5 → 二值掩膜
```

### 训练策略

| 项目 | 配置 |
|------|------|
| 伪标签来源 | GrabCut 输出（`data/segmented/grabcut/`，无需人工标注） |
| 训练/验证分割 | 80% / 20%，从 1,680 gallery 图像随机划分 |
| 数据增强 | 随机水平翻转 + 亮度抖动 ±20 |
| 损失函数 | BCE + Dice（各 0.5 权重） |
| 优化器 | Adam（lr=1e-4）+ CosineAnnealingLR |
| 训练轮次 | 默认 20 epoch（`--epochs N` 可调） |
| Checkpoint | 每 epoch 保存最优 val Dice → `data/features/unet_ckpt.pth` |

### 设备自动检测

```
get_device() 优先级：CUDA（NVIDIA RTX 4060）> MPS（Apple M3）> CPU
```

| 平台 | 训练时间（20 epoch，全量）| 推理时间（13,233 张）|
|------|----------------------|------------------|
| Apple M3（MPS） | ~15 min | ~5 min |
| NVIDIA RTX 4060（CUDA） | ~4 min | ~2 min |
| CPU（无 GPU） | ~90 min | ~15 min |

### 运行

```bash
# 前置：生成 ResUNet 训练伪标签（GrabCut）
make segment-face

# 训练
make dl-train                            # 全量，20 epoch
make dl-train ARGS="--epochs 3 --limit 200"  # 快速调试

# 推理
make dl-segment

# 评估（DL vs 传统方法）
make dl-eval-seg
```

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
│   │   ├── gallery_cropped_labels.json
│   │   └── unet_ckpt.pth             # ResUNet 最优训练权重（dl-train 后生成）
│   ├── synthetic/            # 合成遮挡数据集（cup/glasses/sunglasses）
│   ├── cropped/              # 眼周裁剪图像（gallery/query/vis）
│   ├── segmented/            # 分割结果掩膜
│   │   ├── dl_unet/          # ResUNet 深度学习分割输出（主方法）
│   │   ├── grabcut/          # GrabCut（用作 U-Net 训练伪标签）
│   │   ├── skin_ycrcb/       # YCrCb 阈值分割（历史对比）
│   │   ├── skin_gmm/         # GMM 肤色分割（历史对比）
│   │   └── watershed/        # Watershed 分割（历史对比）
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
│   ├── segment_skin.py       # YCrCb/GMM 肤色分割（传统）
│   ├── segment_face.py       # GrabCut/Watershed 前景分割（传统，兼作伪标签）
│   ├── eval_segmentation.py  # 传统分割方法对比评估
│   ├── validate_segmentation.py  # 分割验证断言
│   ├── dl_model.py           # ResUNet 模型（ResNet-18 encoder + U-Net decoder）
│   ├── dl_train.py           # 深度学习训练循环（GrabCut 伪标签 + checkpoint）
│   ├── dl_segment.py         # 深度学习批量推理
│   ├── dl_eval_segmentation.py  # DL vs 传统方法对比评估
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
| [`docs/phase2-segmentation.md`](docs/phase2-segmentation.md) | 传统肤色/前景分割方法与定量对比 |
| [`docs/phase2-dl-segmentation.md`](docs/phase2-dl-segmentation.md) | ResUNet 深度学习分割：架构、训练策略与实验结果 |
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
