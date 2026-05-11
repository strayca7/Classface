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

## 第二阶段：图像分割（传统机器学习方法）✅

### 2a 肤色分割

- [x] **实现 YCrCb/HSV 颜色阈值分割**（`scripts/segment_skin.py`）
    - 将预处理图像转换至 YCrCb 与 HSV 色彩空间
    - YCrCb 范围：Cr ∈ [133, 173]、Cb ∈ [77, 127]（Kovac 经典椭圆模型）
    - 形态学后处理：`cv2.morphologyEx` 去噪（开运算 + 闭运算）
    - 输出掩膜至 `data/segmented/skin_ycrcb/`
- [x] **实现 GMM 肤色分割**（同脚本）
    - 从 LFW 样本中采样皮肤像素（中心区域）与背景像素（四角）
    - 训练 `sklearn.mixture.GaussianMixture`（n_components=2，covariance_type='full'）
    - 模型缓存至 `data/features/gmm_skin.pkl`，对每张图像逐像素分类
    - 输出掩膜至 `data/segmented/skin_gmm/`

### 2b 人脸前景分割

- [x] **实现 GrabCut 前景分割**（`scripts/segment_face.py`）
    - 以预处理图像的人脸边界框（留 10px 余量）初始化矩形区域
    - `cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)`
    - 提取前景掩膜（GC_FGD | GC_PR_FGD），输出至 `data/segmented/grabcut/`
- [x] **实现 Watershed 分割**（同脚本）
    - 灰度化 → Otsu 阈值 → 距离变换（`cv2.distanceTransform`）→ 峰值标记
    - `cv2.watershed` 执行区域增长，分离前景/背景
    - 输出前景掩膜至 `data/segmented/watershed/`

### 2c 方法对比实验

- [x] **编写对比可视化脚本**（`scripts/eval_segmentation.py`）
    - 随机抽取 20 张图像，5 列并排可视化（原图 | YCrCb | GMM | GrabCut | Watershed）
    - 计算平均前景像素占比，输出至 `data/results/segmentation_stats.txt`
    - 保存对比图至 `data/results/figures/segmentation_compare.png`（300 dpi，~1 MB）
    - 实测结果：YCrCb 51.5%、GMM 96.4%、GrabCut 23.6%、Watershed 35.0%

### 工程规范

- [x] **新增 Makefile 命令**：`segment-skin`、`segment-face`、`eval-seg`、`validate-seg`
- [x] **编写验证脚本**（`scripts/validate_segmentation.py`）：6 项断言，验证通过 ✓
- [x] **文档**：`docs/phase2-segmentation.md`（验证通过后自动写入）
- [x] **提交**：`feat(segment): add skin color and face foreground segmentation`

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
    - 注：比 Phase 3 抽样（97.04%）略低，全量包含更多难例
    - 输出：`data/results/baseline_accuracy.txt`
- [x] **眼周裁剪**：`make crop`
    - Gallery：1680 张，detect_ok=1632 / fail=48（97.1%）
    - Query：22,452 张，detect_ok=21,036 / fail=1,416（93.7%）
    - 耗时：**102 min**
    - 输出：`data/cropped/gallery/`、`data/cropped/query/`、`data/cropped/vis/`、`data/features/gallery_cropped.npy`

### 待运行（下次继续）

- [ ] **对比实验（eval-compare）**：运行命令 `make eval-compare`
    - 三组对比：A 基线(干净) / B 遮挡naive / C 两级级联
    - 两级级联：L1_HIGH=0.8 → 直接输出；0.4≤score<0.8 → 眼周裁剪二次比对；<0.4 → unknown
    - 预计耗时：**~96 min**
    - 前置条件：`data/features/gallery.npy` ✓、`data/features/gallery_cropped.npy` ✓、`data/synthetic/` ✓、`data/cropped/query/` ✓
    - 输出：`data/results/compare_accuracy.txt`、`data/results/figures/accuracy_compare.png`、`data/results/figures/occlusion_type.png`
