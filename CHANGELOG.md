# Changelog

## [Unreleased]

### Added
- `tools/preprocess.py`：图像预处理流水线（YCrCb 直方图均衡化 + MediaPipe 双眼对齐 + 112×112 裁剪）
- `tools/prepare_dataset.py`：LFW 数据集筛选与 gallery/query 分割工具
- `TODO.md`：第一阶段任务清单
- `Makefile` 新增目标：`setup`、`download-lfw`、`prepare-dataset`、`preprocess`
- `pyproject.toml` 新增依赖：`mediapipe`、`insightface`、`onnxruntime`、`matplotlib`
- 项目目录结构：`data/`、`tools/`
