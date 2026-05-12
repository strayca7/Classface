# Changelog

## [Unreleased]

### Added
- `scripts/preprocess.py`：图像预处理流水线（YCrCb 直方图均衡化 + OpenCV Haar 双眼对齐 + 112×112 裁剪）
- `scripts/prepare_dataset.py`：LFW 数据集筛选与 gallery/query 分割工具
- `docs/phase1-preprocess.md`：预处理模块原理与代码说明
- `docs/phase1-dataset.md`：LFW 数据集准备模块说明
- `README.md`：项目概述、安装指南、使用说明、目录结构、贡献指南
- `VERSION`：语义化版本文件，当前版本 0.1.0
- `TODO.md`：阶段任务清单
- `Makefile` 目标：`setup`、`download-lfw`、`prepare-dataset`、`preprocess`
- LFW-funneled 数据集：5749 个身份，13233 张图像（解压至 `data/raw/lfw/`）
- `data/raw/lfw_filtered.json`：1680 个满足 ≥2 张的身份，gallery 1680 张，query 7484 张
- 预处理产物 `data/processed/lfw/`：13233 张 112×112 图像，对齐率 46.6%，|angle| 均值 4.0°
- `scripts/segment_skin.py`：YCrCb 阈值分割（Kovac 椭圆模型）+ GMM 肤色分割
- `scripts/segment_face.py`：GrabCut + Watershed 人脸前景分割
- `scripts/eval_segmentation.py`：四方法对比可视化与统计
- `scripts/validate_segmentation.py`：6 项断言验证脚本
- `docs/phase2-segmentation.md`：图像分割模块文档
- `scripts/build_gallery.py`：InsightFace ArcFace gallery 特征提取与缓存
- `scripts/evaluate.py`：支持 `--mode baseline` 和 `--mode compare` 双模式评估
- `docs/phase3-baseline.md`：基线识别系统文档
- `data/overlays/`：真实 PNG 贴图资产 58 张（cup×18 / glasses×20 / sunglasses×20），已纳入 git
- `scripts/generate_cover.py`：InsightFace 5-kps 关键点定位 + alpha blending 遮挡合成
- `scripts/crop.py`：眼周区域裁剪 + gallery_cropped.npy 特征预计算
- `docs/phase4-synthesis.md`：遮挡合成模块文档
- `docs/phase5-robust.md`：鲁棒识别与两级级联文档

### 实验结果（2026-05-11 全量运行）

| 步骤 | 指标 | 数值 |
|------|------|------|
| 预处理 | 成功率 | 13,233/13,233（100%） |
| 预处理 | 对齐率 | 46.6%，\|angle\| 均值 4.0° |
| Gallery 构建 | 特征维度 | (1680, 512) float32 |
| 遮挡合成 | 合成总量 | 22,452 张（7484×3 类） |
| 遮挡合成 | 人脸检测率 | 96.9%（7250/7484） |
| 基线评估（全量） | Top-1 准确率 | **92.65%**（6934/7484） |
| 眼周裁剪 Gallery | 检测率 | 97.1%（1632/1680） |
| 眼周裁剪 Query | 检测率 | 93.7%（21,036/22,452） |
| eval-compare A（基线） | Top-1 准确率 | **92.65%**（6934/7484） |
| eval-compare B（遮挡 Naive） | Top-1 准确率 | **89.11%**（20007/22452） |
| eval-compare C（两级级联） | Top-1 准确率 | **12.52%**（2812/22452） |
| eval-compare 各类型 B/C | sunglasses | B=88.79% / C=8.23% |
| eval-compare 各类型 B/C | cup | B=87.85% / C=17.45% |
| eval-compare 各类型 B/C | glasses | B=90.69% / C=11.89% |

### Fixed
- `scripts/prepare_dataset.py`：修复 `relative_to(Path("."))` 在绝对路径下崩溃，改为相对于 `lfw_dir`
- `scripts/preprocess.py`：修复 `align_face` 返回类型注解错误；将 MediaPipe（0.10.x 删除了 `solutions` API）替换为 OpenCV Haar 级联；新增 |angle| > 20° 异常降级保护
- `scripts/crop.py`：去除无用变量 `scale`，修正 occ_types 为 cup/glasses/sunglasses

### Changed
- `Makefile`：`download-lfw` 目标更新为从 figshare 镜像下载（UMass 服务器 SSL 问题）
- `scripts/generate_cover.py`：遮挡类型从 cup/hand/book 改为 cup/glasses/sunglasses；锚点逻辑重写
- `Makefile`：`gen-overlays` 废弃（贴图已改用真实 PNG 资产）
