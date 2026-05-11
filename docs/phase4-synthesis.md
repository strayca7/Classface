# Phase 4 — 遮挡图像合成

## 概述

第四阶段为后续的鲁棒人脸识别实验构建训练/测试所需的**带遮挡合成数据集**。  
使用真实拍摄的 PNG 贴图资产，通过 InsightFace 关键点定位，以 alpha blending 方式将遮挡物叠加到 LFW 人脸图像上，生成三类课堂场景遮挡：水杯（cup）、眼镜（glasses）、墨镜（sunglasses）。

---

## 遮挡类型与关键点锚定

| 遮挡类型   | 锚点位置          | 参考宽度          | 缩放系数 |
|----------|-----------------|-----------------|--------|
| cup      | 嘴角中点          | 人脸检测框宽度     | 0.70   |
| glasses  | 双眼中点          | 两眼间距          | 2.80   |
| sunglasses | 双眼中点        | 两眼间距          | 3.00   |

InsightFace 5-kps 索引：  
`[0]` 左眼 · `[1]` 右眼 · `[2]` 鼻尖 · `[3]` 嘴左角 · `[4]` 嘴右角

如果 InsightFace 对某张图片检测失败，退化为固定比例（112×112 空间内的经验坐标）。

---

## 贴图资产

贴图位于 `data/overlays/`，已纳入 git 版本管理。

| 类型       | 文件数量 | 文件名规律            |
|----------|--------|--------------------|
| cup      | 18     | cup_1 … cup_20（缺 11, 17） |
| glasses  | 20     | glasses_1 … glasses_20 |
| sunglasses | 20   | sunglasses_1 … sunglasses_20 |

运行时通过 `glob(f"{occ_type}_*.png")` 自动发现所有变体，每张查询图像随机选取一个变体。

---

## 主要脚本：`scripts/generate_cover.py`

```
uv run python scripts/generate_cover.py [--types cup,glasses,sunglasses] [--limit N]
```

| 参数      | 默认值                         | 说明                      |
|---------|------------------------------|--------------------------|
| `--types` | `cup,glasses,sunglasses`     | 需要合成的遮挡类型（逗号分隔） |
| `--limit` | 无（处理所有身份）               | 调试时限制处理身份数量         |

**输出目录**：`data/synthetic/{cup,glasses,sunglasses}/<identity>/<img>.jpg`

### 关键函数

| 函数 | 作用 |
|------|------|
| `discover_variants(occ_type)` | 扫描 `data/overlays/` 获取该类型所有贴图文件名 |
| `get_face_info(app, img)` | 上采样 112→320，调用 InsightFace 检测，返回 bbox/kps（112 空间） |
| `alpha_blend(base, overlay, cx, cy)` | 以 (cx, cy) 为中心 alpha blending |
| `synthesize_image(...)` | 根据遮挡类型选取锚点，缩放贴图，调用 `alpha_blend` |

### 流程图

```
LFW 查询图像
    │
    ▼
InsightFace 检测（上采样 112→320）
    │   检测失败 ──► 固定比例 fallback
    ▼
提取 bbox + 5-kps
    │
    ▼
选取遮挡类型 & 随机贴图变体
    │
    ▼
缩放贴图 → Alpha Blending → 保存合成图
```

---

## Makefile 目标

```makefile
make generate          # 全量合成（≈18.7 min，已写入 TODO.md）
make generate ARGS="--limit 5"   # 快速冒烟测试
```

---

## 验证

冒烟测试（`--limit 3`）结果：
- 处理 5 张查询图像，100% 人脸检测成功
- 生成 `data/synthetic/cup/`、`glasses/`、`sunglasses/` 各 5 张

```
2026-05-11 [INFO] generate_cover: Progress: 5/5 | detect_ok=5 fail=0
2026-05-11 [INFO] generate_cover: cup: 5 images
2026-05-11 [INFO] generate_cover: glasses: 5 images
2026-05-11 [INFO] generate_cover: sunglasses: 5 images
2026-05-11 [INFO] generate_cover: FaceDetect rate: 100.0% (5/5)
```

---

## 依赖

- `insightface[onnxruntime]`（buffalo_l 模型）
- `opencv-python`、`Pillow`
- LFW 预处理数据：`data/processed/lfw/`
- 过滤后身份列表：`data/raw/lfw_filtered.json`
