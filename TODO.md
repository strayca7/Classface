# TODO

## 第一阶段：数据准备与图像预处理

- [x] **搭建项目骨架**
    - [x] 创建目录：`data/raw/lfw/`、`data/processed/lfw/`、`data/overlays/`、`data/features/`、`data/results/figures/`、`tools/`
    - [x] `pyproject.toml` 添加依赖：`mediapipe`、`insightface`、`onnxruntime`、`matplotlib`
    - [ ] 联网后运行 `uv sync` 安装所有依赖

- [ ] **下载并整理 LFW 数据集**
    - [ ] 运行 `make download-lfw` 下载 LFW-funneled（~200MB）并解压至 `data/raw/lfw/`
    - [ ] 运行 `uv run python tools/prepare_dataset.py` 筛选 ≥2 张图像的身份，生成 `data/raw/lfw_filtered.json`

- [ ] **预处理干净人脸数据**
    - [ ] 运行 `make preprocess` 执行批量预处理，输出至 `data/processed/lfw/`
    - [ ] 检查日志，验证对齐旋转角度分布合理（|angle| < 15° 为主）

## 第二阶段：基线识别系统验证

> 待第一阶段完成后展开

## 第三阶段：非标准遮挡数据合成

> 待第二阶段完成后展开

## 第四阶段：遮挡鲁棒识别与对比实验

> 待第三阶段完成后展开
