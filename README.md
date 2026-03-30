# 基于"特定场景与非标准遮挡"的人脸识别

> 数字图像处理课程设计 — 智慧课堂/自习室无感签到系统

## 项目概述

本项目针对**课堂/自习室场景**中的非标准遮挡（手托腮、水杯遮挡、书本遮挡、趴桌露半脸），设计了一套基于图像处理流水线的人脸识别系统。

与口罩遮挡不同，这类遮挡不规则、位置多变，难以用通用模型直接处理。本系统不依赖图像修复模型，而是通过**动态局部裁剪 + 两级级联比对**策略，在不修改识别网络结构的前提下，大幅提升遮挡场景下的识别准确率。

**技术流水线**：
```
原始人脸图像
  → 第一阶段：图像预处理（均衡化 + 对齐 + 裁剪）
  → 第二阶段：基线识别验证（干净数据建库 + 余弦相似度）
  → 第三阶段：遮挡数据合成（仿射变换 + Alpha 掩膜）
  → 第四阶段：两级级联识别 + 对比实验
```

详细技术路线见 [`procedures.md`](procedures.md)。

---

## 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)（包管理工具）

---

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd dip

# 安装所有依赖（含 mediapipe、insightface、opencv 等）
uv sync
```

---

## 快速开始

### 1. 创建项目目录结构

```bash
make setup
```

### 2. 下载 LFW 数据集（约 200MB，需联网）

```bash
make download-lfw
```

### 3. 生成数据集分割清单

```bash
make prepare-dataset
# 输出：data/raw/lfw_filtered.json
```

### 4. 批量预处理人脸图像

```bash
make preprocess
# 输出：data/processed/lfw/（112×112，均衡化+对齐）
```

---

## Makefile 命令说明

| 命令 | 说明 |
|------|------|
| `make setup` | 创建项目所需目录结构 |
| `make download-lfw` | 下载 LFW-funneled 数据集并解压 |
| `make prepare-dataset` | 筛选身份，生成 gallery/query 分割 JSON |
| `make preprocess` | 批量图像预处理（均衡化 + 对齐 + 112×112 裁剪） |
| `make cover` | 生成遮挡合成样例图（开发测试用） |
| `make clean` | 删除所有生成产物（processed/、features/ 等） |

---

## 目录结构

```
dip/
├── data/
│   ├── raw/lfw/          # LFW 原始数据（download-lfw 后填充）
│   ├── processed/lfw/    # 预处理后图像（112×112）
│   ├── overlays/         # 遮挡贴图素材（PNG，带 Alpha 通道）
│   ├── features/         # InsightFace 特征向量缓存（.npy）
│   ├── synthetic/        # 合成遮挡数据集
│   └── results/          # 评估结果、对比图表
├── docs/
│   ├── phase1-dataset.md     # 数据集准备模块说明
│   └── phase1-preprocess.md  # 图像预处理模块说明
├── scripts/
│   ├── prepare_dataset.py    # LFW 筛选与分割
│   └── preprocess.py         # 图像预处理（均衡化+对齐+裁剪）
├── procedures.md     # 完整技术路线与实施步骤
├── TODO.md           # 各阶段任务进度
├── CHANGELOG.md      # 版本变更日志
├── Makefile          # 常用命令
└── pyproject.toml    # 项目依赖
```

---

## 开发规范

- **提交信息**：遵循 [Conventional Commits](https://www.conventionalcommits.org/)，格式 `<type>(<scope>): <description>`
- **代码格式**：`uv fmt`（Python，基于 ruff）
- **版本管理**：语义化版本（SemVer），见 `VERSION` 文件
- **日志格式**：`%(asctime)s [%(levelname)s] %(name)s: %(message)s`

---

## 文档

每个阶段的模块说明和代码解释位于 `docs/` 目录：

- [`docs/phase1-dataset.md`](docs/phase1-dataset.md)：LFW 数据集准备与 gallery/query 分割
- [`docs/phase1-preprocess.md`](docs/phase1-preprocess.md)：直方图均衡化、人脸对齐、裁剪策略

---

## 贡献指南

1. Fork 本仓库，基于 `main` 分支新建特性分支（如 `feat/phase2-baseline`）
2. 修改代码后运行 `uv fmt` 格式化
3. 提交前确保代码可正常运行
4. 提交信息遵循 Conventional Commits 规范
5. 提交 PR 并描述变更内容
