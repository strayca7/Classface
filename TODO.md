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

## 第三阶段：基线识别系统验证

> 待第二阶段完成后展开

- [ ] **建立 gallery 特征库**
    - [ ] 运行 `scripts/build_gallery.py` 提取 InsightFace 特征，保存至 `data/features/`
- [ ] **评估干净数据基线准确率**
    - [ ] 运行 `scripts/evaluate.py --mode baseline`，预期 Top-1 > 95%

## 第四阶段：非标准遮挡数据合成

> 待第三阶段完成后展开

## 第五阶段：遮挡鲁棒识别与对比实验

> 待第四阶段完成后展开
