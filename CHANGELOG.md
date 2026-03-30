# Changelog

## [Unreleased]

### Added
- `tools/preprocess.py`：图像预处理流水线（YCrCb 直方图均衡化 + OpenCV Haar 双眼对齐 + 112×112 裁剪）
- `tools/prepare_dataset.py`：LFW 数据集筛选与 gallery/query 分割工具
- `doc/phase1-preprocess.md`：预处理模块原理与代码说明
- `doc/phase1-dataset.md`：LFW 数据集准备模块说明
- `README.md`：项目概述、安装指南、使用说明、目录结构、贡献指南
- `VERSION`：语义化版本文件，当前版本 0.1.0
- `TODO.md`：阶段任务清单
- `Makefile` 目标：`setup`、`download-lfw`、`prepare-dataset`、`preprocess`
- LFW-funneled 数据集：5749 个身份，13233 张图像（解压至 `data/raw/lfw/`）
- `data/raw/lfw_filtered.json`：1680 个满足 ≥2 张的身份，gallery 1680 张，query 7484 张
- 预处理产物 `data/processed/lfw/`：13233 张 112×112 图像，对齐率 46.6%，|angle| 均值 4.0°

### Fixed
- `tools/prepare_dataset.py`：修复 `relative_to(Path("."))` 在绝对路径下崩溃，改为相对于 `lfw_dir`
- `tools/preprocess.py`：修复 `align_face` 返回类型注解错误；将 MediaPipe（0.10.x 删除了 `solutions` API）替换为 OpenCV Haar 级联；新增 |angle| > 20° 异常降级保护

### Changed
- `Makefile`：`download-lfw` 目标更新为从 figshare 镜像下载（UMass 服务器 SSL 问题）
