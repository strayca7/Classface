# TODO

## 第一阶段：数据准备与图像预处理 ✅

- [x] **搭建项目骨架**
    - [x] 创建目录：`data/raw/lfw/`、`data/processed/lfw/`、`data/overlays/`、`data/features/`、`data/results/figures/`、`tools/`
    - [x] `pyproject.toml` 添加依赖：`mediapipe`、`insightface`、`onnxruntime`、`matplotlib`
    - [x] 运行 `uv sync` 安装所有依赖

- [x] **下载并整理 LFW 数据集**
    - [x] 从 figshare 下载 LFW-funneled（232MB）并解压至 `data/raw/lfw/`（5749 个身份，13233 张图像）
    - [x] 运行 `tools/prepare_dataset.py` 筛选 ≥2 张图像的身份，生成 `data/raw/lfw_filtered.json`
        - 结果：1680 个身份，gallery 1680 张，query 7484 张

- [x] **预处理干净人脸数据**
    - [x] 运行 `make preprocess` 批量预处理，输出至 `data/processed/lfw/`
        - 处理结果：13233 张，成功 13233，失败 0
        - 对齐率：46.6%（OpenCV Haar 级联检测到双眼时做仿射对齐，否则中心裁剪）
        - 对齐旋转角 |angle| 均值：4.0°（远小于 15°，合理✓）
        - 输出尺寸：全部为 112×112×3 ✓

## 第二阶段：基线识别系统验证

> 开始进行

- [ ] **建立 gallery 特征库**
    - [ ] 运行 `tools/build_gallery.py` 提取 InsightFace 特征，保存至 `data/features/`
- [ ] **评估干净数据基线准确率**
    - [ ] 运行 `tools/evaluate.py --mode baseline`，预期 Top-1 > 95%

## 第三阶段：非标准遮挡数据合成

> 待第二阶段完成后展开

## 第四阶段：遮挡鲁棒识别与对比实验

> 待第三阶段完成后展开
