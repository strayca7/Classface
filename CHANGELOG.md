# Changelog

## [Unreleased]

### Added
- `tools/preprocess.py`：图像预处理流水线（YCrCb 直方图均衡化 + MediaPipe 双眼对齐 + 112×112 裁剪）
- `tools/prepare_dataset.py`：LFW 数据集筛选与 gallery/query 分割工具
- `doc/phase1-preprocess.md`：预处理模块原理与代码说明
- `doc/phase1-dataset.md`：LFW 数据集准备模块说明
- `README.md`：项目概述、安装指南、使用说明、目录结构、贡献指南
- `VERSION`：语义化版本文件，当前版本 0.1.0
- `TODO.md`：第一阶段任务清单
- `Makefile` 目标：`setup`、`download-lfw`、`prepare-dataset`、`preprocess`
- `pyproject.toml` 依赖：`mediapipe`、`insightface`、`onnxruntime`、`matplotlib`

### Fixed
- `tools/prepare_dataset.py`：修复 `build_split` 中 `relative_to(Path("."))` 在绝对路径下崩溃的问题，改为相对于 `lfw_dir`
- `tools/preprocess.py`：修复 `align_face` 返回类型注解错误（`np.ndarray` → 正确的 3 元组类型）
