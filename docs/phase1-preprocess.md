# 第一阶段：图像预处理模块说明

**对应脚本**：`scripts/preprocess.py`  
**Makefile 入口**：`make preprocess`

---

## 模块职责

对原始 LFW 人脸图像执行三步标准化处理，输出统一格式（112×112 BGR）的图像，供第二阶段特征提取直接使用：

```
原始图像（任意尺寸）
    │
    ├─ 1. YCrCb 直方图均衡化（光照归一化）
    ├─ 2. MediaPipe 双眼对齐（warpAffine 仿射旋转）
    └─ 3. 以眼睛中心为基准裁剪 + resize 至 112×112
```

---

## 步骤一：直方图均衡化（光照归一化）

### 原理

直方图均衡化将图像的像素灰度值重新分布，使其尽量均匀铺满 [0, 255]，从而增强对比度、消除光照不均匀的影响。

**为什么用 YCrCb 而非直接对 BGR 操作？**

- BGR 三通道独立均衡化会改变颜色比例，导致色偏（偏紫/偏绿）。
- YCrCb 将亮度（Y）与色度（Cr、Cb）分离。仅对 Y 通道均衡化，再还原为 BGR，可保留原始色彩信息，只改变亮度分布。

### 关键代码

```python
def equalize_hist_color(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # BGR → YCrCb
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # 仅均衡 Y 通道
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)  # 还原 BGR
```

---

## 步骤二：人脸对齐（仿射变换）

### 原理

人在拍照时头部会有倾斜，导致双眼连线不水平。后续裁剪和特征提取对倾斜敏感。通过仿射旋转将双眼连线校正至水平，确保所有图像中人脸姿态一致。

**关键点选取**：使用 MediaPipe FaceMesh 468 点中的：
- 左眼中心：关键点 `#33`（左眼瞳孔附近）
- 右眼中心：关键点 `#263`（右眼瞳孔附近）

### 旋转角度计算

```
        右眼 (rx, ry)
       /
      / dy = ry - ly
     /
左眼 (lx, ly) ── dx = rx - lx ──>

angle = arctan2(dy, dx)   （正值 = 人脸向右倾斜）
```

旋转矩阵以双眼中点为旋转中心、缩放比 1.0（不拉伸）：

```python
def align_face(img, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))          # 倾斜角（度）

    eye_center = ((left_eye[0] + right_eye[0]) / 2,
                  (left_eye[1] + right_eye[1]) / 2)

    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)   # 2×3 仿射矩阵
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR)       # 双线性插值
    return aligned, angle, eye_center
```

### 降级策略

当 mediapipe 未安装或关键点检测失败时，自动降级为**中心正方形裁剪**，不做旋转：

```python
try:
    import mediapipe as mp
    _USE_MEDIAPIPE = True
except ImportError:
    _USE_MEDIAPIPE = False  # 静默降级，只记录 WARNING 日志
```

---

## 步骤三：裁剪与 Resize

以双眼中心为锚点，取图像较短边 80% 的正方形区域裁出人脸，再 resize 至 112×112。

选 112×112 的原因：InsightFace ArcFace 模型的标准输入尺寸为 112×112。

```python
def crop_and_resize(img, eye_center, output_size=112):
    h, w = img.shape[:2]
    cx, cy = eye_center
    side = int(min(h, w) * 0.8)                         # 正方形边长
    x1 = int(np.clip(cx - side // 2, 0, w - side))     # 左边界（夹紧防越界）
    y1 = int(np.clip(cy - side // 2, 0, h - side))     # 上边界
    cropped = img[y1:y1+side, x1:x1+side]
    return cv2.resize(cropped, (output_size, output_size),
                      interpolation=cv2.INTER_LINEAR)
```

---

## 完整处理流程

```python
def process_image(src_path):
    img = cv2.imread(str(src_path))           # 读取图像
    if img is None: return None, stats        # 读取失败则跳过

    img = equalize_hist_color(img)            # 光照归一化

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    eye_centers = get_eye_centers(img_rgb)    # MediaPipe 关键点检测

    if eye_centers:
        img, angle, eye_center = align_face(img, *eye_centers)  # 仿射对齐
        output = crop_and_resize(img, eye_center)               # 裁剪+resize
    else:
        output = center_crop_resize(img)      # 降级：中心裁剪

    return output, stats
```

---

## 日志输出示例

```
2026-03-30 10:00:01,234 [INFO] preprocess: 开始批量预处理：13233 张图像，输出至 data/processed/lfw
2026-03-30 10:02:15,678 [INFO] preprocess: [500/13233] aligned=True angle=-2.3° hist_mean 81.2→127.4
2026-03-30 10:15:30,001 [INFO] preprocess: 批量预处理完成：13201 成功，32 失败（共 13233）
```

---

## 使用方式

```bash
# 默认：data/raw/lfw → data/processed/lfw
make preprocess

# 自定义路径
uv run python scripts/preprocess.py --src data/raw/lfw --dst data/processed/lfw
```
