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
【第三阶段】基线识别系统验证（干净数据）
    │  InsightFace 512-d 特征提取 → gallery 缓存 → 余弦相似度 → 基线 Top-1 准确率
    ▼
【第四阶段】非标准遮挡数据合成
    │  MediaPipe 关键点定位 → 仿射变换贴图 → Alpha 掩膜融合 → 合成遮挡数据集
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

## 第三阶段：基线识别系统验证（干净数据）

**目标**：在无遮挡的 LFW 预处理图像上，完整跑通特征提取 → gallery 建库 → 余弦相似度比对的识别链路，获得**基线 Top-1 准确率**，验证整条流水线的正确性。

**核心 API**：`insightface.app.FaceAnalysis`、`numpy.dot`、`numpy.linalg.norm`、`numpy.save` / `numpy.load`

### 步骤

- [ ] **初始化 InsightFace 模型**（`scripts/recognize.py`）
    - `app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])`
    - `app.prepare(ctx_id=0, det_size=(112, 112))`
    - 对单张图像调用 `app.get(img)` 提取 512 维 `embedding`

- [ ] **预计算并缓存 gallery 特征**（`scripts/build_gallery.py`）
    - 遍历 `data/processed/lfw/` 中每位身份的 gallery 图像（第 1 张）
    - 提取 512 维特征向量，堆叠为矩阵 `(N, 512)`
    - 保存：`data/features/gallery.npy`（特征矩阵）、`data/features/gallery_labels.json`（身份标签列表）

- [ ] **实现向量化余弦相似度**
    ```python
    # 查询向量 q (512,)，底库矩阵 G (N, 512)
    G_norm = G / np.linalg.norm(G, axis=1, keepdims=True)
    scores = G_norm @ (q / np.linalg.norm(q))
    pred_id = gallery_labels[np.argmax(scores)]
    ```

- [ ] **编写基线评估脚本**（`scripts/evaluate.py`）
    - 遍历 `data/processed/lfw/` 中所有 query 图像（每位身份第 2 张起）
    - 计算 Top-1 准确率，输出至 `data/results/baseline_accuracy.txt`
    - 预期基线准确率（干净数据）> 95%

- [ ] **工程规范**
    - 日志：记录每次比对的最高得分、预测身份与真实身份（是否匹配）
    - Makefile：`build-gallery` → `uv run python scripts/build_gallery.py`；`eval-baseline` → `uv run python scripts/evaluate.py --mode baseline`
    - 提交：`feat(recognize): add baseline recognition system with gallery caching`

---

## 第四阶段：非标准遮挡数据合成

**目标**：在第一阶段预处理后的干净图像上，通过关键点定位 + 仿射变换 + Alpha 掩膜，自动合成课堂场景下的遮挡图像，构建专属测试集。

**核心 API**：`mediapipe.solutions.face_mesh`、`cv2.getRotationMatrix2D`、`cv2.warpAffine`、`cv2.split`、`cv2.bitwise_and`、`cv2.add`

### 步骤

- [ ] **准备贴图素材**
    - 收集 ≥3 类带 Alpha 通道的 PNG 贴图：水杯（`cup_01.png`）、托腮手（`hand_01.png`）、书本（`book_01.png`），存入 `data/overlays/`
    - 每类至少 2~3 个变体，增加多样性

- [ ] **实现关键点定位**（`scripts/generate_cover.py`）
    - 使用 `mediapipe.solutions.face_mesh` 获取面部关键点
    - 遮挡锚点：嘴部 #13/#14（水杯/书本）、下巴 #152（手托腮）

- [ ] **实现仿射变换贴图合成**
    - 根据锚点关键点间距离计算贴图目标尺寸（自适应人脸大小，关键点距离比例缩放）
    - `cv2.getRotationMatrix2D` + `cv2.warpAffine` 对贴图旋转/缩放至目标姿态
    - `cv2.split(overlay)` 分离 Alpha 通道，生成前景掩膜与背景掩膜
    - `cv2.bitwise_and` + `cv2.add` 将贴图融合到人脸图像

- [ ] **批量生成合成数据集**
    - 输入源：`data/processed/lfw/`（预处理后的干净图像）
    - 每张图像随机叠加 1~2 种遮挡类型，保存至 `data/synthetic/<occlusion_type>/<person_name>/`
    - 输出生成统计日志（总数、各遮挡类型比例）

- [ ] **工程规范**
    - 日志：记录每张图像的遮挡类型、贴图锚点坐标、缩放比例
    - Makefile：`generate` → `uv run python scripts/generate_cover.py`
    - 提交：`feat(data): add classroom occlusion synthesis pipeline`

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
