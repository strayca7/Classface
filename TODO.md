# TODO

## 第一阶段：数据准备与图像预处理 ✅

- [x] **搭建项目骨架**
    - [x] 创建目录：`data/raw/lfw/`、`data/processed/lfw/`、`data/overlays/`、`data/features/`、`data/results/figures/`、`scripts/`
    - [x] `pyproject.toml` 添加依赖：`mediapipe`、`insightface`、`onnxruntime`、`matplotlib`
    - [x] 运行 `uv sync` 安装所有依赖

- [x] **下载并整理 LFW 数据集**
    - [x] 从 figshare 下载 LFW-funneled（232MB）并解压至 `data/raw/lfw/`（5749 个身份，13233 张图像）
    - [x] 运行 `scripts/prepare_dataset.py` 筛选 ≥2 张图像的身份，生成 `data/raw/lfw_filtered.json`
        - 结果：1680 个身份，gallery 1680 张，query 7484 张

- [x] **预处理干净人脸数据**
    - [x] 运行 `make preprocess` 批量预处理，输出至 `data/processed/lfw/`
        - 处理结果：13233 张，成功 13233，失败 0
        - 对齐率：46.6%（OpenCV Haar 级联检测到双眼时做仿射对齐，否则中心裁剪）
        - 对齐旋转角 |angle| 均值：4.0°（远小于 15°，合理✓）
        - 输出尺寸：全部为 112×112×3 ✓

## 第二阶段：图像分割（深度学习方法）⬜

> 传统方法（YCrCb/GMM/GrabCut/Watershed）已完成并保留作为对比基线，本阶段用 PyTorch U-Net 实现端到端深度学习分割。

### 准备

- [ ] **更新依赖**：在 `pyproject.toml` 中添加 `torch>=2.2.0`、`torchvision>=0.17.0`
    - 运行 `uv sync` 安装（macOS 自动启用 MPS 支持）
    - ⚠️ NVIDIA 平台需额外安装 CUDA 版本，见"如何开始运行"章节

### 2a 模型与训练

- [ ] **实现 U-Net 模型**（`scripts/dl_model.py`）
    - Encoder：ResNet-18（ImageNet 预训练）+ 4 级跳跃特征；Decoder：双线性上采样逐级恢复至 112×112
    - `get_device()` 自动检测 CUDA → MPS → CPU

- [ ] **实现训练脚本**（`scripts/dl_train.py`）
    - 输入：`data/processed/lfw/`；伪标签：`data/segmented/grabcut/`
    - BCE + Dice 损失，Adam 优化，默认 20 epoch
    - 最优权重保存至 `data/features/unet_ckpt.pth`

### 2b 批量推理

- [ ] **实现推理脚本**（`scripts/dl_segment.py`）
    - 加载 `unet_ckpt.pth`，推理所有 LFW 图像
    - 输出掩膜至 `data/segmented/dl_unet/`

### 2c 对比实验

- [ ] **实现对比评估脚本**（`scripts/dl_eval_segmentation.py`）
    - 6 列可视化（原图 | YCrCb | GMM | GrabCut | Watershed | U-Net）
    - IoU / Dice / 前景占比对比，保存至 `data/results/figures/plot_seg_dl_compare.png`

### 工程规范

- [ ] **更新 Makefile**：新增 `dl-train`、`dl-segment`、`dl-eval-seg`
- [ ] **文档**：`docs/phase2-dl-segmentation.md`（实验后写入）
- [ ] **提交**：`feat(segment): replace traditional ML with U-Net deep learning segmentation`

---

## 第三阶段：基线识别系统验证 ✅

- [x] **实现 gallery 特征库构建**（`scripts/build_gallery.py`）
    - InsightFace buffalo_l，上采样 112→320 后送入完整流水线（检测→对齐→ArcFace）
    - 输出 `data/features/gallery.npy`（1680, 512）和 `gallery_labels.json`
    - 全量运行：成功 1680，失败 0，耗时 ~503s
- [x] **实现基线评估**（`scripts/evaluate.py --mode baseline`）
    - 向量化余弦相似度 1:N 检索，Top-1 准确率
    - 验证结果（50 身份抽样）：**97.04%** > 95% 预期 ✓
- [x] **新增 Makefile 命令**：`build-gallery`、`eval-baseline`
- [x] **文档**：`docs/phase3-baseline.md`
- [x] **提交**：`feat(recognize): add baseline recognition with InsightFace gallery`

---

## 第四阶段：非标准遮挡数据合成 ✅

- [x] **贴图素材**：真实 PNG 资产 58 张（cup×18 / glasses×20 / sunglasses×20），存于 `data/overlays/`，已纳入 git
- [x] **实现遮挡合成**（`scripts/generate_cover.py`）
    - InsightFace 5-kps 定位（上采样 112→320）→ alpha blending
    - cup：嘴角中点锚点，宽 = face_w × 0.70
    - glasses / sunglasses：双眼中点锚点，宽 = 眼间距 × 2.8 / 3.0
    - 全量运行结果：7484 query × 3 类 = **22,452 张**合成图，耗时 **72 min**
    - 人脸检测率：96.9%（7250/7484）；失败时退化为固定比例
    - 输出：`data/synthetic/{cup,glasses,sunglasses}/`

---

## 第五阶段：遮挡鲁棒识别与对比实验

### 已完成

- [x] **预处理（重跑）**：`make preprocess`，13,233 张，耗时 1m33s
- [x] **Gallery 特征库（重建）**：`make build-gallery`，1680 张，耗时 8m22s，输出 `data/features/gallery.npy`
- [x] **基线评估（全量）**：`make eval-baseline`
    - 全量 7484 query，**Top-1 = 92.65%**（6934/7484），耗时 31m49s
    - 输出：`data/results/eval_baseline_full.txt`
- [x] **眼周裁剪**：`make crop`
    - Gallery：1680 张，detect_ok=1632（97.1%）；Query：22,452 张，detect_ok=21,036（93.7%）
    - 耗时：102 min；输出：`data/cropped/`、`data/features/gallery_cropped.npy`
- [x] **对比实验 v1**（`evaluate.py` L1_HIGH=0.8，已有结果）
    - A=92.65%，B=89.11%，C=12.52%（级联阈值设置错误，已分析）

### 全部完成 ✅

- [x] **eval-compare v3 全量运行**：`make eval-compare`（L1_HIGH=-1）
    - 耗时：约 97 min（07:22→08:55），3 类型 × 7,484 query
    - 最终结果：**A=92.65%  B=89.11%  C=89.11%（C=B ✅）**
    - per-type：sunglasses B=C=88.79%，cup B=C=87.85%，glasses B=C=90.69%

- [x] **更新所有报告文档与图表**：`make update-reports`
    - 已更新：README.md、CHANGELOG.md、docs/report.md、docs/demo_report.md
    - 已重新生成：`docs/figures/` 下所有图表

### 已完成（本次会话）

- [x] **docs/report.md**：项目汇报报告，含各阶段原理、执行计划与实验结果（627 行）
- [x] **docs/demo_report.md**：演示报告，含流水线可视化、遮挡合成展示、结果对比图表
- [x] **docs/figures/**：6 张演示图表（strip/montage/accuracy/cascade/segmentation）
- [x] **scripts/plot_results.py**：从实验结果生成 6 张通用图表（修复 CJK 字体）
- [x] **scripts/generate_demo_report.py**：生成演示图片 + demo_report.md（修复人脸对齐 bug）
- [x] **scripts/update_reports.py**：eval-compare 完成后自动更新所有报告的脚本
- [x] **Makefile**：新增 `plot-results`、`demo-report`、`update-reports` 三个目标

---

## 如何开始运行（第二阶段深度学习分割）

### 第 0 步：安装 PyTorch

**macOS（Apple Silicon M3）——MPS 加速：**
```bash
uv sync   # pyproject.toml 中已添加 torch + torchvision，直接安装
```

**Windows / Linux（NVIDIA RTX 4060）——CUDA 加速：**
```bash
# 先用 uv 安装其他依赖，再手动覆盖 torch 为 CUDA 版本
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证设备检测：
```bash
uv run python -c "from scripts.dl_model import get_device; print(get_device())"
# Apple M3  → device(type='mps')
# RTX 4060  → device(type='cuda')
# 无 GPU    → device(type='cpu')
```

### 第 1 步：训练 U-Net（约 10~30 min，视硬件而定）

```bash
make dl-train
# 或指定 epoch 数调试（推荐先用少量验证流水线）：
make dl-train ARGS="--epochs 3 --limit 200"
```

训练完成后：
- 最优权重保存至 `data/features/unet_ckpt.pth`
- 终端输出每 epoch 的 train/val Dice

### 第 2 步：批量推理

```bash
make dl-segment
# 输出：data/segmented/dl_unet/<person_name>/<img>.png
```

快速调试（仅处理前 100 张）：
```bash
make dl-segment ARGS="--limit 100"
```

### 第 3 步：对比评估

```bash
make dl-eval-seg
# 输出：
#   data/results/figures/plot_seg_dl_compare.png（6 列对比图）
#   data/results/eval_seg_dl_stats.txt（各方法 IoU/Dice 数据）
```

### 完整一键运行

```bash
make dl-train && make dl-segment && make dl-eval-seg
```

### 注意事项

- GrabCut 伪标签（`data/segmented/grabcut/`）须已存在，否则先运行 `make segment-face`
- Apple M3 上 MPS 训练速度约为 CPU 的 3~5×；RTX 4060 CUDA 约为 CPU 的 20~50×
- checkpoint 文件 `data/features/unet_ckpt.pth` 已在 `.gitignore` 中，不会提交至 git

---

## 待补充实验数据（报告存在空缺，择时运行）

以下两项实验数据在 `docs/report.md` 中标记为 `—`（未运行），补充后需手动将结果填入报告对应表格。

---

### ✅ 补充实验 1：YCrCb / GMM 分割掩膜的 IoU / Dice 指标（已完成 2026-06-08）

**结果**：已运行 `make segment-skin && make dl-eval-seg`，数据已填入报告。

| 方法 | 前景占比 | IoU vs GrabCut | Dice vs GrabCut |
|------|---------|--------------|----------------|
| YCrCb | 54.7% | 0.499 | 0.633 |
| GMM | 93.7% | 0.308 | 0.456 |

数据来源：`data/results/eval_seg_dl_stats.txt`

---

### 补充实验 2：C v2 全量对比实验（L1_HIGH=0.5，best-of-two）

**缺失位置**：`docs/report.md` §七（续）两张汇总表中 `C v2（全量）` 列均为 `—`

**缺失原因**：v2 全量运行预计耗时 >97 min（CPU），此前未运行

**预计耗时**：约 100–130 min（CPU）；RTX 4060 约 10–15 min

**注意**：`eval_cascade_ablation.py` 在同一个循环内同时跑 v1/v2/v3，运行后将**覆盖** `data/results/eval_cascade_v123_50s.txt`（目前存有 50 样本结果）。建议先备份：

```bash
# 可选：备份当前 50 样本结果
cp data/results/eval_cascade_v123_50s.txt data/results/eval_cascade_v123_50s_backup.txt
```

**运行命令**（取消 50 人限制，对全部 1680 身份运行 v1/v2/v3）：

```bash
uv run python scripts/eval_cascade_ablation.py --limit 0
# --limit 0 等价于无限制（代码逻辑：if limit: identities = identities[:limit]，0 为 falsy 跳过截断）
# 输出：
#   data/results/eval_cascade_v123_50s.txt     ← 全量 v1/v2/v3 数值
#   data/results/figures/plot_cascade_v123_50s.png（更新）
```

**运行后操作**：查看 `data/results/eval_cascade_v123_50s.txt` 中 v2 全量数据，填入 `docs/report.md` §七（续）以下位置：

```
### 整体准确率汇总  →  C v2 全量 列填入实际数值（预计 ~45–50%）
### 各遮挡类型全量细分  →  C v2（全量）列填入 sunglasses/cup/glasses 三行数值
```
