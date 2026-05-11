# 技术路线与实施步骤

## 技术栈

| 类别 | 工具 / 库 | 用途 |
|------|-----------|------|
| 语言 | Python 3.13+ | 全栈 |
| 包管理 | `uv` | 依赖安装与脚本运行（`uv sync` / `uv run`） |
| 图像处理 | `opencv-python` | 仿射变换、直方图均衡、掩膜融合、裁剪 |
| 关键点检测 | `mediapipe` | 人脸 468 关键点（FaceMesh） |
| 人脸识别模型 | `insightface`（ArcFace backbone） | 512 维特征向量提取（预训练，无需额外训练） |
| 机器学习 | `scikit-learn` | GMM 肤色分割（`GaussianMixture`） |
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
【第二阶段】图像分割（传统机器学习方法）✅
    │  肤色分割：YCrCb 阈值（Kovac 椭圆模型）+ GMM
    │  前景分割：GrabCut + Watershed
    │  方法对比：前景占比 YCrCb 51.5% / GMM 96.4% / GrabCut 23.6% / Watershed 35.0%
    ▼
【第三阶段】基线识别系统验证（干净数据）✅
    │  InsightFace 512-d 特征提取 → gallery 缓存 → 余弦相似度 → 基线 Top-1 = 97.04%
    ▼
【第四阶段】非标准遮挡数据合成 ✅
    │  InsightFace 5-kps 关键点定位 → Alpha 掩膜融合 → 合成遮挡数据集（cup/glasses/sunglasses）
    ▼
【第五阶段】遮挡鲁棒识别：动态局部裁剪 + 两级级联
       遮挡图像 → Level 1 全局比对
       ├─ 得分 > 0.8  → 直接输出身份
       ├─ 得分 0.4~0.8 → Level 2 裁剪眼周区域 → 二次比对，输出身份
       └─ 得分 < 0.4  → 标记"无法识别"
       └─ 对比实验：基线准确率 vs 遮挡后 vs 两级策略提升
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

## 第二阶段：图像分割（传统机器学习方法）✅

**目标**：在预处理后的人脸图像上，用传统方法实现肤色分割与人脸前景分割，为后续遮挡鲁棒识别提供先验掩膜，并通过定量对比验证各方法效果。

**核心 API**：`cv2.cvtColor`、`cv2.morphologyEx`、`cv2.grabCut`、`cv2.watershed`、`cv2.distanceTransform`、`sklearn.mixture.GaussianMixture`

### 2a 肤色分割

- [x] **实现 YCrCb 颜色阈值分割**（`scripts/segment_skin.py`）
    - 转换至 YCrCb，应用 Kovac 经典椭圆模型：Cr ∈ [133, 173]、Cb ∈ [77, 127]
    - 形态学后处理（开运算 + 闭运算）去除噪点
    - 输出掩膜至 `data/segmented/skin_ycrcb/`；实测平均前景占比 **51.5%**

- [x] **实现 GMM 肤色分割**（同脚本）
    - 采样皮肤像素（图像中心 20×20）与背景像素（四角 10×10），共 ~400K 像素
    - `sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full', n_init=3)`
    - 模型缓存至 `data/features/gmm_skin.pkl`，自动复用
    - 输出掩膜至 `data/segmented/skin_gmm/`；实测平均前景占比 **96.4%**（人脸区域几乎全为皮肤）

### 2b 人脸前景分割

- [x] **实现 GrabCut 前景分割**（`scripts/segment_face.py`）
    - 初始矩形 `rect=(10, 10, 92, 92)`（留 10px 余量），`iterCount=5`
    - 提取 `GC_FGD | GC_PR_FGD` 前景掩膜
    - 输出至 `data/segmented/grabcut/`；实测平均前景占比 **23.6%**

- [x] **实现 Watershed 分割**（同脚本）
    - 灰度化 → Otsu 阈值 → `cv2.distanceTransform` → 峰值标记 → `cv2.watershed`
    - 输出前景掩膜至 `data/segmented/watershed/`；实测平均前景占比 **35.0%**

### 2c 方法对比实验

- [x] **编写对比可视化脚本**（`scripts/eval_segmentation.py`）
    - 随机抽取 20 张图像，5 列并排（原图 | YCrCb | GMM | GrabCut | Watershed）
    - 保存对比图至 `data/results/figures/segmentation_compare.png`（300 dpi）
    - 统计结果输出至 `data/results/segmentation_stats.txt`

### 工程规范

- [x] **新增 Makefile 命令**：`segment-skin`、`segment-face`、`eval-seg`、`validate-seg`
- [x] **验证脚本**（`scripts/validate_segmentation.py`）：6 项断言，验证通过 ✓
- [x] **文档**：`docs/phase2-segmentation.md`
- [x] **提交**：`feat(segment): add skin color and face foreground segmentation`

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

## 第五阶段：遮挡鲁棒识别与对比实验

**目标**：实现"动态局部裁剪 + 两级级联"识别策略，与基线对比，定量证明策略对遮挡场景的提升效果。

**核心 API**：MediaPipe FaceMesh 关键点、NumPy 数组切片、`cv2.hconcat`、`matplotlib.pyplot`

### 步骤

#### 5.1 动态局部裁剪

- [ ] **计算眼周黄金区域**（`scripts/crop.py`）
    - 上边界：眉毛上方关键点 #70（左）/ #105（右），留 10px 余量
    - 下边界：鼻梁中部关键点 #6，留 5px 余量
    - 左右边界：脸部轮廓关键点 #234（左）/ #454（右）

- [ ] **实现裁剪函数**
    - `img[y1:y2, x1:x2]` 执行裁剪，`np.clip` 防止越界
    - 对 `data/processed/lfw/`（gallery 底库）和 `data/synthetic/`（遮挡测试集）执行相同裁剪，分别保存至 `data/cropped/gallery/` 和 `data/cropped/query/`

- [ ] **预计算眼周 gallery 特征**
    - 基于 `data/cropped/gallery/` 提取眼周特征，保存至 `data/features/gallery_cropped.npy`

- [ ] **可视化验证**
    - 随机抽取 10 组，`cv2.hconcat([orig, cropped])` 拼接，保存至 `data/cropped/vis/`

#### 5.2 两级级联识别

- [ ] **实现两级级联逻辑**（`scripts/recognize.py`）
    - **Level 1（全局）**：完整对齐图像 → InsightFace 特征 → 与 `gallery.npy` 比对
        - 最高得分 > 0.8：直接输出身份
    - **Level 2（局部触发）**：得分 0.4~0.8 → 裁剪眼周区域 → 与 `gallery_cropped.npy` 二次比对，输出结果
    - 得分 < 0.4：标记"无法识别"

#### 5.3 对比实验

- [ ] **编写三组对比测试**（`scripts/evaluate.py --mode compare`）
    - **组 A（基线）**：干净 LFW query 图像 → 全脸识别（第三阶段结果，复用）
    - **组 B（遮挡 naive）**：合成遮挡图像 → 直接全脸识别（不使用任何遮挡策略）
    - **组 C（两级策略）**：合成遮挡图像 → 两级级联识别

- [ ] **绘制对比图表**（`matplotlib`）
    - 图1：三组准确率柱状图（A vs B vs C），保存 `data/results/figures/accuracy_compare.png`
    - 图2：不同遮挡类型（水杯/手/书本）在组 B vs 组 C 下的准确率折线图，保存 `data/results/figures/occlusion_type.png`
    - 图分辨率 300 dpi

- [ ] **工程规范**
    - 日志：记录每次识别的触发级别（L1/L2）、得分、最终身份、耗时（ms）
    - Makefile：`crop` → `uv run python scripts/crop.py`；`recognize` → `uv run python scripts/recognize.py`；`eval` → `uv run python scripts/evaluate.py`
    - 提交：`feat(recognize): add two-stage cascade recognition with occlusion robustness`
