# 第三阶段：基线识别系统

**对应脚本**：`scripts/build_gallery.py`、`scripts/evaluate.py`  
**Makefile 入口**：`make build-gallery`、`make eval-baseline`

---

## 模块职责

在无遮挡的 LFW 预处理图像上，建立 InsightFace ArcFace 特征库，并通过向量化余弦相似度实现
1:N 人脸检索，获得基线 Top-1 准确率，验证整条识别流水线的正确性。

```
预处理图像 112×112 (data/processed/lfw/)
    │
    ▼
【build_gallery.py】
    上采样 112→320 → InsightFace buffalo_l 检测+对齐 → ArcFace 512-d 嵌入
    │
    ├─ data/features/gallery.npy          (1680, 512) float32
    └─ data/features/gallery_labels.json  ["Aaron_Peirsol", ...]
    │
    ▼
【evaluate.py --mode baseline】
    query 图像 → 同样流水线 → 余弦相似度检索 → Top-1 预测
    │
    └─ data/results/baseline_accuracy.txt
```

---

## 特征提取

### InsightFace buffalo_l 模型

- **检测模型**：`det_10g.onnx`（RetinaFace，标准推理分辨率 640×640）
- **识别模型**：`w600k_r50.onnx`（ArcFace R50，输入 112×112，输出 512-d L2 归一化嵌入）
- 模型自动下载至 `~/.insightface/models/buffalo_l/`（首次运行，约 500 MB）

### 上采样策略

LFW 预处理图像为 112×112，直接送入检测器时 anchor 尺度不匹配会导致 `ValueError`。
解决方案：先双线性插值上采样至 320×320，送入完整流水线后 ArcFace 自动完成对齐裁剪。

```python
img_large = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
faces = app.get(img_large)
embedding = faces[0].embedding  # (512,) float32，已 L2 归一化
```

若检测失败（极少数低质量图像），退回到识别模型的 `get_feat()` 直接提取。

---

## Gallery 构建（`scripts/build_gallery.py`）

### 数据来源

读取 `data/raw/lfw_filtered.json`（结构：`{identity: {gallery: [...], query: [...]}}`），
取每位身份的第 1 张图像作为 gallery，共 **1,680 张**。

### 核心代码

```python
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = []
for identity in identities:
    img_path = PROCESSED_DIR / dataset[identity]["gallery"][0]
    img_large = cv2.resize(cv2.imread(str(img_path)), (320, 320))
    faces = app.get(img_large)
    embeddings.append(faces[0].embedding.astype(np.float32))

gallery_matrix = np.stack(embeddings)  # (1680, 512)
np.save("data/features/gallery.npy", gallery_matrix)
```

### 运行结果

| 指标 | 值 |
|------|----|
| 处理身份数 | 1,680 |
| 成功提取 | 1,680（0 失败） |
| 输出形状 | (1680, 512) float32 |
| 耗时 | ~503s（CPU） |

---

## 余弦相似度检索

```python
# 预归一化 gallery（只算一次）
G_norm = gallery_matrix / np.linalg.norm(gallery_matrix, axis=1, keepdims=True)

# 查询（每次）
q_norm = embedding / np.linalg.norm(embedding)
scores = G_norm @ q_norm          # (N,) 余弦相似度
pred_id = labels[np.argmax(scores)]
top_score = float(scores.max())
```

余弦相似度范围 [-1, 1]，ArcFace 嵌入已 L2 归一化，实际得分集中在 [0.2, 1.0]。

---

## 基线评估（`scripts/evaluate.py --mode baseline`）

遍历所有 query 图像（每位身份第 2 张起，共 7,484 张），计算 Top-1 准确率：

```
Top-1 Accuracy = 最高得分对应身份与真实身份匹配数 / 有效 query 总数
```

### 验证结果（50 身份抽样，169 query）

| 指标 | 值 |
|------|----|
| Top-1 准确率 | **97.04%** |
| 正确匹配 | 164 / 169 |
| 耗时 | ~67s（CPU） |

> 97.04% 超过预期基线 95%，流水线正确性验证通过 ✓

---

## 使用方式

```bash
# 构建 gallery 特征库（首次需下载模型，约 500 MB）
make build-gallery

# 调试模式（只处理前 N 个身份）
make build-gallery ARGS="--limit 50"

# 基线准确率评估
make eval-baseline

# 调试模式
make eval-baseline ARGS="--limit 50"
```

---

## 日志格式

```
2026-05-11 16:14:47 [INFO] build_gallery: 特征提取完成：成功 1680，失败 0，耗时 503.3s
2026-05-11 16:14:47 [INFO] build_gallery: 保存 gallery.npy: 形状 (1680, 512)
2026-05-11 16:19:25 [INFO] evaluate: 基线评估完成：Top-1=0.9704（164/169），耗时 66.7s
2026-05-11 16:19:25 [INFO] evaluate: 基线准确率达标 ✓ (97.04%)
```
