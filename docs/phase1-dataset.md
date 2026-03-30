# 第一阶段：数据集准备模块说明

**对应脚本**：`scripts/prepare_dataset.py`  
**Makefile 入口**：`make prepare-dataset`

---

## 模块职责

扫描 LFW 目录，筛选出满足条件的身份，生成 gallery/query 分割清单（JSON 文件），供第二阶段建库和评估使用。

```
data/raw/lfw/（LFW 原始目录）
    │
    ├─ 过滤：保留图像数 ≥ 2 的身份（约 1,680 位）
    ├─ 分割：每位身份第 1 张为 gallery，其余为 query
    └─ 输出：data/raw/lfw_filtered.json
```

---

## 数据集介绍：LFW

**LFW（Labeled Faces in the Wild）** 是人脸识别领域最广泛使用的公开数据集之一，由麻省大学阿默斯特分校发布：

- 包含 13,233 张图像，5,749 个不同身份
- 图像来源：新闻照片，包含真实世界的光照、姿态、表情变化
- 使用 **LFW-funneled** 版本：图像经过预对齐，人脸区域已大致居中
- 下载地址：`https://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz`（约 200MB）

### 目录结构

```
data/raw/lfw/
├── Aaron_Eckhart/
│   └── Aaron_Eckhart_0001.jpg
├── Aaron_Guiel/
│   └── Aaron_Guiel_0001.jpg
├── ...
└── Zydrunas_Ilgauskas/
    ├── Zydrunas_Ilgauskas_0001.jpg
    └── Zydrunas_Ilgauskas_0002.jpg
```

---

## 为什么筛选"≥ 2 张图像"的身份？

人脸识别的验证需要 **gallery（底库）** 和 **query（查询集）** 各自独立：

- **gallery**：每位身份至少 1 张"干净"图像用于建库
- **query**：至少还剩 1 张用于查询，计算 Top-1 准确率

仅有 1 张图像的身份（约 4,069 位）无法同时提供 gallery 和 query，因此排除在外。

---

## 分割策略

| 字段 | 说明 |
|------|------|
| `gallery` | 每位身份**第 1 张**图像（按文件名排序），用于建库 |
| `query` | 第 2 张起所有图像，用于查询评估 |

这是 1-shot 评估协议：用 1 张注册照识别任意后续图像，贴近真实签到场景（学生入学时只拍一张证件照）。

---

## 输出格式

**`data/raw/lfw_filtered.json`**（路径相对于 `data/raw/lfw/`）：

```json
{
  "Aaron_Eckhart": {
    "gallery": ["Aaron_Eckhart/Aaron_Eckhart_0001.jpg"],
    "query":   []
  },
  "Zydrunas_Ilgauskas": {
    "gallery": ["Zydrunas_Ilgauskas/Zydrunas_Ilgauskas_0001.jpg"],
    "query":   ["Zydrunas_Ilgauskas/Zydrunas_Ilgauskas_0002.jpg"]
  }
}
```

**路径使用说明**：

```python
import json
from pathlib import Path

LFW_DIR = Path("data/raw/lfw")
with open("data/raw/lfw_filtered.json") as f:
    split = json.load(f)

# 还原完整路径
for person, data in split.items():
    gallery_paths = [LFW_DIR / p for p in data["gallery"]]
    query_paths   = [LFW_DIR / p for p in data["query"]]
```

---

## 关键代码

```python
def build_split(lfw_dir: Path, min_images: int = 2) -> dict:
    split = {}
    for person_dir in sorted(lfw_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted(person_dir.glob("*.jpg"))
        if len(images) < min_images:
            continue
        # 路径相对于 lfw_dir，方便在不同机器复用 JSON
        rel_images = [str(img.relative_to(lfw_dir)) for img in images]
        split[person_dir.name] = {
            "gallery": rel_images[:1],   # 第 1 张作为 gallery
            "query":   rel_images[1:],   # 其余作为 query
        }
    return split
```

---

## 统计结果（预期）

| 指标 | 值 |
|------|----|
| LFW 总身份数 | 5,749 |
| 满足 ≥2 张的身份数 | ~1,680 |
| gallery 图像总数 | ~1,680 |
| query 图像总数 | ~5,800 |

---

## 使用方式

```bash
# 前提：已运行 make download-lfw
make prepare-dataset

# 或手动指定（当前版本路径硬编码，如需修改请直接编辑脚本）
uv run python scripts/prepare_dataset.py
```
