# 技术路线与实施步骤

## 技术栈

| 类别 | 工具 / 库 | 用途 |
|------|-----------|------|
| 语言 | Python 3.13+ | 全栈 |
| 包管理 | `uv` | 依赖安装与脚本运行（`uv sync` / `uv run`） |
| 图像处理 | `opencv-python` | 仿射变换、直方图均衡、掩膜融合、裁剪 |
| 关键点检测 | `mediapipe` | 人脸 468 关键点（FaceMesh） |
| 人脸识别模型 | `insightface`（ArcFace backbone） | 512 维特征向量提取（预训练，无需额外训练） |
| 深度学习框架 | `torch` + `torchvision` | U-Net 模型定义、训练、推理；自动检测 CUDA / MPS / CPU |
| 数值计算 | `numpy` | 向量化余弦相似度计算 |
| 数据集 | LFW (Labeled Faces in the Wild) | 干净人脸数据，≥2 张图像的身份共 1,680 位 |
| 可视化 | `matplotlib` | 准确率对比图表 |
| 构建 | `Makefile` | 统一封装常用命令 |
| 格式化 | `ruff` (via `uv tool run`) | Python 代码格式化 |

---

## 流水线总览

```
【LFW 原始图像（干净人脸）】
    │
    ▼
【第一阶段】预处理与人脸对齐 ✅
    │  直方图均衡化（YCrCb/Y 通道）→ 双眼中心对齐（warpAffine）→ 统一 112×112
    ▼
【第二阶段】图像分割（深度学习方法）⬜
    │  U-Net（PyTorch，4 级编解码器 + 跳跃连接）
    │  设备：CUDA（NVIDIA RTX 4060）> MPS（Apple M3）> CPU 自动选择
    │  训练策略：GrabCut 输出作为伪标签（弱监督）
    │  推理输出：data/segmented/dl_unet/，二值掩膜
    │  方法对比：DL U-Net vs YCrCb vs GMM vs GrabCut vs Watershed（IoU / Dice）
    ▼
【第三阶段】基线识别系统验证（干净数据）✅
    │  InsightFace 512-d 特征提取 → gallery 缓存 → 余弦相似度 → 基线 Top-1 = 97.04%
    ▼
【第四阶段】非标准遮挡数据合成 ✅
    │  InsightFace 5-kps 关键点定位 → Alpha 掩膜融合 → 合成遮挡数据集（cup/glasses/sunglasses）
    ▼
【第五阶段】遮挡鲁棒识别：动态局部裁剪 + 两级级联 ✅
       遮挡图像 → Level 1 全局比对
       ├─ 得分 > 0.8  → 直接输出身份
       ├─ 得分 0.4~0.8 → Level 2 裁剪眼周区域 → 二次比对，输出身份
       └─ 得分 < 0.4  → 标记"无法识别"
       └─ 对比实验：A=92.65% / B=89.11% / C=12.52%（级联因阈值问题未能提升，见 Phase 5 分析）
```

---

## 第一阶段：数据准备与图像预处理 ✅

**目标**：搭建项目骨架，下载 LFW，对干净人脸执行标准预处理，输出统一格式的图像供后续所有阶段复用。

**核心 API**：`cv2.cvtColor`、`cv2.equalizeHist`、`np.arctan2`、`cv2.getRotationMatrix2D`、`cv2.warpAffine`、`cv2.resize`、`mediapipe.solutions.face_mesh`

### 步骤

- [x] **搭建项目骨架**
    - 创建目录：`data/raw/lfw/`、`data/processed/`、`data/overlays/`、`data/features/`、`data/results/`
    - `pyproject.toml` 添加依赖：`opencv-python`、`mediapipe`、`insightface`、`numpy`、`matplotlib`、`Pillow`
    - 运行 `uv sync` 安装所有依赖

- [x] **下载并整理 LFW 数据集**
    - 下载 LFW-funneled 版本（232 MB），解压至 `data/raw/lfw/<person_name>/<image>.jpg`（5,749 个身份，13,233 张）
    - 筛选出 ≥2 张图像的身份，生成 `data/raw/lfw_filtered.json`：**1,680 个身份**，gallery 1,680 张，query 7,484 张
    - gallery/query 分割：每位身份取第 1 张为 gallery，其余为 query

- [x] **实现光照归一化**（`scripts/preprocess.py`）
    - 转换至 YCrCb：`cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)`
    - 对 Y 通道执行 `cv2.equalizeHist`，保留色彩信息后转回 BGR

- [x] **实现人脸对齐**
    - OpenCV Haar 级联检测双眼，计算倾斜角并用 `cv2.warpAffine` 旋转摆正
    - 对齐率：46.6%（检测不到双眼时退化为中心裁剪）；对齐旋转角均值 4.0°

- [x] **统一输出尺寸**
    - 以双眼中心为基准裁剪人脸区域，`cv2.resize` 至 112×112；全部 13,233 张成功，0 失败

- [x] **工程规范**
    - 日志格式：`%(asctime)s [%(levelname)s] %(name)s: %(message)s`
    - Makefile：`preprocess` → `uv run python scripts/preprocess.py`
    - 提交：`feat(preprocess): add histogram equalization and face alignment`

---

## 第二阶段：图像分割（深度学习方法）⬜

**目标**：用 PyTorch U-Net 替代传统机器学习方法（GMM / GrabCut / Watershed），实现端到端人脸前景分割，并通过对比实验定量验证深度学习方案的优势。

**核心技术**：U-Net（4 级编解码器 + 跳跃连接）、GrabCut 伪标签、BCE + Dice 复合损失函数

**设备适配**：`get_device()` 函数自动检测——CUDA（NVIDIA RTX 4060）→ MPS（Apple M3）→ CPU，代码无需手动修改。

### 2a 模型与训练

- [ ] **实现 U-Net 模型**（`scripts/dl_model.py`）
    - Encoder：ResNet-18（ImageNet 预训练，torchvision），提取 4 级特征图 [56², 28², 14², 7², 4²]
    - Decoder：双线性上采样 + 跳跃连接 + Conv-BN-ReLU × 2，逐级恢复空间分辨率至 112×112
    - 输出：`1×112×112`，Sigmoid 激活，二值掩膜
    - `get_device()` 统一入口：CUDA → MPS → CPU，打印当前设备信息

- [ ] **实现训练脚本**（`scripts/dl_train.py`）
    - 数据集：`data/processed/lfw/`（输入）+ `data/segmented/grabcut/`（GrabCut 伪标签）
    - 训练/验证分割：80% / 20%（从 1,680 gallery 图像中随机划分）
    - 批次大小：GPU 模式 16，CPU 模式 4（自动调整）
    - 损失函数：BCE Loss + Dice Loss（各权重 0.5）
    - 优化器：Adam（lr=1e-4），余弦退火调度（CosineAnnealingLR）
    - 训练轮次：默认 20 epoch（`--epochs N` 可调）
    - Checkpoint：每 epoch 保存最优模型至 `data/features/unet_ckpt.pth`
    - 支持 `--epochs N` 参数；自动在训练结束后打印最优 val Dice

### 2b 批量推理

- [ ] **实现推理脚本**（`scripts/dl_segment.py`）
    - 加载 `data/features/unet_ckpt.pth`，对所有 `data/processed/lfw/` 图像推理
    - 阈值 0.5 二值化，保存掩膜至 `data/segmented/dl_unet/<person_name>/<img>.png`
    - 支持 `--limit N` 调试参数

### 2c 方法对比实验

- [ ] **实现对比评估脚本**（`scripts/dl_eval_segmentation.py`）
    - 随机抽取 20 张图像，6 列并排可视化（原图 | YCrCb | GMM | GrabCut | Watershed | U-Net）
    - 以 GrabCut 掩膜为参考，计算各方法 IoU / Dice / 前景占比
    - 保存对比图至 `data/results/figures/dl_segmentation_compare.png`（300 dpi）
    - 统计结果输出至 `data/results/dl_segmentation_stats.txt`

### 工程规范

- [ ] **更新 Makefile**：新增 `dl-train`、`dl-segment`、`dl-eval-seg` 目标
- [ ] **更新依赖**（`pyproject.toml`）：添加 `torch>=2.2.0`、`torchvision>=0.17.0`
- [ ] **文档**：`docs/phase2-dl-segmentation.md`
- [ ] **提交**：`feat(segment): replace traditional ML with U-Net deep learning segmentation`

---

## 第三阶段：基线识别系统验证（干净数据）✅

**目标**：在无遮挡的 LFW 预处理图像上，完整跑通特征提取 → gallery 建库 → 余弦相似度比对的识别链路，获得**基线 Top-1 准确率**，验证整条流水线的正确性。

**核心 API**：`insightface.app.FaceAnalysis`、`numpy.dot`、`numpy.linalg.norm`、`numpy.save` / `numpy.load`

### 步骤

- [x] **初始化 InsightFace 模型**（`scripts/build_gallery.py` / `scripts/evaluate.py`）
    - `FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])`
    - `app.prepare(ctx_id=0, det_size=(640, 640))`
    - 输入图像先上采样至 320×320，确保检测器 anchor 正常计算

- [x] **预计算并缓存 gallery 特征**（`scripts/build_gallery.py`）
    - 遍历 1,680 个身份的 gallery 图像，提取 512-d ArcFace 嵌入
    - 保存：`data/features/gallery.npy`（1680, 512）、`data/features/gallery_labels.json`
    - 全量耗时 ~503s（CPU），0 失败

- [x] **实现向量化余弦相似度**
    ```python
    G_norm = G / np.linalg.norm(G, axis=1, keepdims=True)
    scores = G_norm @ (q / np.linalg.norm(q))
    pred_id = gallery_labels[np.argmax(scores)]
    ```

- [x] **编写基线评估脚本**（`scripts/evaluate.py --mode baseline`）
    - 遍历所有 query 图像（每位身份第 2 张起）
    - 输出至 `data/results/baseline_accuracy.txt`
    - 验证结果（50 身份抽样）：**Top-1 = 97.04%** > 95% ✓

- [x] **工程规范**
    - 日志：记录每次比对的最高得分、预测身份与真实身份
    - Makefile：`build-gallery`、`eval-baseline`（支持 `ARGS="--limit N"` 调试）
    - 提交：`feat(recognize): add baseline recognition with InsightFace gallery`

---

## 第四阶段：非标准遮挡数据合成 ✅

**目标**：在第一阶段预处理后的干净图像上，通过关键点定位 + Alpha 掩膜融合，自动合成课堂场景下的遮挡图像，构建专属测试集。

**核心 API**：`insightface.app.FaceAnalysis`（5-kps 关键点）、`PIL.Image`、`cv2.resize`、Alpha blending（numpy）

### 步骤

- [x] **准备贴图素材**
    - 真实拍摄 PNG 贴图（带 Alpha 通道），共 3 类 58 张，存入 `data/overlays/`（纳入 git）：
        - 水杯 `cup_*.png`：18 张
        - 普通眼镜 `glasses_*.png`：20 张
        - 墨镜 `sunglasses_*.png`：20 张
    - 运行时通过 `glob(f"{type}_*.png")` 自动发现，每张 query 随机取一个变体

- [x] **实现关键点定位**（`scripts/generate_cover.py`）
    - 使用 InsightFace buffalo_l 5-kps（上采样 112→320 后检测，坐标映射回 112 空间）
    - 遮挡锚点：
        - cup → 嘴角中点 `(kps[3]+kps[4])/2`，宽度 = face_w × 0.70
        - glasses / sunglasses → 双眼中点 `(kps[0]+kps[1])/2`，宽度 = 眼间距 × 2.8 / 3.0
    - 检测失败时退化为固定比例 fallback（112×112 经验坐标）

- [x] **实现 Alpha 掩膜融合合成**
    - 根据锚点距离缩放贴图至目标宽度（PIL LANCZOS），保持宽高比
    - 以锚点为中心 alpha blending（`fg × α + bg × (1−α)`）

- [x] **批量生成合成数据集**
    - 输入源：`data/processed/lfw/`（预处理后的干净图像）
    - 每张 query 图像为每种遮挡类型各生成一张，保存至 `data/synthetic/{cup,glasses,sunglasses}/<person_name>/`
    - 输出生成统计日志（总数、各遮挡类型数量、人脸检测率）
    - ⚠️ 全量合成约 18.7 min，待手动运行：`make generate`

- [x] **工程规范**
    - 日志：记录进度、detect_ok/fail、每类生成数量
    - Makefile：`generate` → `uv run python scripts/generate_cover.py`（`gen-overlays` 已废弃，贴图为真实资产）
    - 文档：`docs/phase4-synthesis.md`
    - 提交：`feat(overlays): replace generated assets with real image cutouts (cup/glasses/sunglasses)`

---

## 第五阶段：遮挡鲁棒识别与对比实验 ✅

**目标**：实现"动态局部裁剪 + 两级级联"识别策略，与基线对比，定量证明策略对遮挡场景的提升效果。

**实际实现说明**：眼周裁剪使用 InsightFace 5-kps 关键点（而非 MediaPipe FaceMesh），两级级联逻辑集成在 `scripts/evaluate.py --mode compare` 中（无独立 recognize.py）。

**核心 API**：InsightFace 5-kps、NumPy 数组切片、`cv2.hconcat`、`matplotlib.pyplot`

### 步骤

#### 5.1 动态局部裁剪

- [x] **计算眼周区域**（`scripts/crop.py`）
    - 使用 InsightFace 5-kps：y=[bbox_top−4, nose_y+4]，x=[bbox_left−4, bbox_right+4]
    - 检测失败时退化为固定比例裁剪（上 1/3 区域）

- [x] **实现裁剪函数**
    - `img[y1:y2, x1:x2]` 裁剪，`np.clip` 防止越界，`cv2.resize` 至 112×112
    - 对 gallery（干净）和 query（遮挡合成图）执行相同裁剪，分别保存

- [x] **预计算眼周 gallery 特征**
    - 基于 `data/cropped/gallery/` 提取特征，保存至 `data/features/gallery_cropped.npy`（1680, 512）
    - detect_ok=1632/1680（97.1%）

- [x] **可视化验证**
    - 随机抽取样本，orig+cropped 拼接，保存至 `data/cropped/vis/`

#### 5.2 两级级联识别

- [x] **实现两级级联逻辑**（`scripts/evaluate.py --mode compare`）
    - **Level 1（全局）**：完整图像 → ArcFace → gallery.npy 比对
        - 最高得分 > 0.8（L1_HIGH）：直接输出身份
    - **Level 2（局部触发）**：0.4 ≤ 得分 < 0.8 → 读取预裁剪眼周图 → gallery_cropped.npy 二次比对
    - 得分 < 0.4（L1_LOW）：标记"无法识别"

#### 5.3 对比实验

- [x] **三组对比测试**（`scripts/evaluate.py --mode compare`）
    - **组 A（基线）**：干净 LFW query → 全脸识别（复用 baseline_accuracy.txt）
    - **组 B（遮挡 naive）**：合成遮挡图 → 直接全脸识别
    - **组 C（两级策略）**：合成遮挡图 → 两级级联识别
    - 全量实验耗时 **136m44s**（CPU）

- [x] **实验结果**
    | 组别 | 准确率 | 样本数 |
    |------|--------|--------|
    | A — 基线 | **92.65%** | 7,484 |
    | B — 遮挡 Naive | **89.11%** | 22,452 |
    | C — 两级级联 | **12.52%** | 22,452 |

    各遮挡类型：sunglasses B=88.79%/C=8.23%，cup B=87.85%/C=17.45%，glasses B=90.69%/C=11.89%

    > **分析**：C 组准确率极低，根本原因是：① L1_HIGH=0.8 阈值过高，大部分遮挡图像（即使 B 能正确识别的）得分落在 0.4~0.8 被路由至 L2；② glasses/sunglasses 贴图直接覆盖眼周区域，L2 的眼周裁剪无法获得干净特征，导致大量误识。B 组（89.11%）证明 ArcFace 全脸特征对局部遮挡本身具备一定鲁棒性（相对基线仅下降 3.54pp）。如需改进，应针对各遮挡类型动态选择裁剪区域（水杯→眼周有效，眼镜→嘴/下颌区域）并调低 L1_HIGH。

- [x] **绘制对比图表**（`matplotlib`）
    - `data/results/figures/accuracy_compare.png`：三组准确率柱状图（300 dpi）
    - `data/results/figures/occlusion_type.png`：各遮挡类型 B vs C 折线图（300 dpi）

- [x] **工程规范**
    - 日志：记录遮挡类型处理进度、最终三组准确率
    - Makefile：`crop`、`eval-compare`（`eval-baseline` 为 baseline 模式）
    - 文档：`docs/phase5-robust.md`
    - 提交：`docs(procedures): mark phase 5 complete with experimental results`
