"""Synthesize occluded face images using overlay assets and InsightFace keypoints.

Uses InsightFace 5-kps (left-eye, right-eye, nose, mouth-left, mouth-right)
and bounding box to anchor three occlusion types:

  cup   – overlay centred at mouth midpoint, scaled to face width × 0.7
  hand  – overlay centred between mouth and bbox bottom (chin area)
  book  – solid rectangle covering lower 45% of face (from nose-level to bbox bottom)

Fallback to fixed proportions if InsightFace detects no face.

Usage:
    uv run python scripts/generate_cover.py [--types cup,hand,book] [--limit N]
"""

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_cover")

DATA_ROOT = Path("data/processed/lfw")
OVERLAY_DIR = Path("data/overlays")
SYNTH_ROOT = Path("data/synthetic")
LFW_JSON = Path("data/raw/lfw_filtered.json")

# anchor indices in InsightFace 5-kps: [left_eye, right_eye, nose, mouth_l, mouth_r]
KPS_LEFT_EYE = 0
KPS_RIGHT_EYE = 1
KPS_NOSE = 2
KPS_MOUTH_L = 3
KPS_MOUTH_R = 4

OVERLAY_VARIANTS = {
    "cup": ["cup_01.png", "cup_02.png", "cup_03.png", "cup_04.png", "cup_05.png"],
    "hand": ["hand_01.png", "hand_02.png", "hand_03.png", "hand_04.png", "hand_05.png"],
    "book": ["book_01.png", "book_02.png", "book_03.png", "book_04.png", "book_05.png"],
}


def load_overlay(name: str) -> np.ndarray:
    """Load RGBA overlay as (H, W, 4) uint8 numpy array."""
    path = OVERLAY_DIR / name
    img = Image.open(path).convert("RGBA")
    return np.array(img)


def alpha_blend(base: np.ndarray, overlay_rgba: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Alpha-blend overlay (H_o, W_o, 4) centred at (cx, cy) onto base (H, W, 3)."""
    H, W = base.shape[:2]
    oh, ow = overlay_rgba.shape[:2]
    x1 = cx - ow // 2
    y1 = cy - oh // 2
    x2 = x1 + ow
    y2 = y1 + oh

    # Clip to image bounds
    ox1 = max(0, -x1)
    oy1 = max(0, -y1)
    ox2 = ow - max(0, x2 - W)
    oy2 = oh - max(0, y2 - H)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    if x1 >= x2 or y1 >= y2:
        return base

    patch = overlay_rgba[oy1:oy2, ox1:ox2]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    fg = patch[:, :, :3].astype(np.float32)
    bg = base[y1:y2, x1:x2].astype(np.float32)
    blended = (fg * alpha + bg * (1 - alpha)).clip(0, 255).astype(np.uint8)
    result = base.copy()
    result[y1:y2, x1:x2] = blended
    return result


def apply_book_cover(base: np.ndarray, y_start: int) -> np.ndarray:
    """Cover lower face with a semi-transparent colored rectangle (book)."""
    H, W = base.shape[:2]
    y_start = max(0, min(y_start, H))
    # Pick random book color (blue or red) reproducibly by image hash
    color = (45, 85, 160) if random.random() < 0.5 else (175, 40, 40)
    result = base.copy()
    overlay = np.zeros((H - y_start, W, 3), dtype=np.uint8)
    overlay[:] = color
    alpha = 0.88
    result[y_start:H] = cv2.addWeighted(overlay, alpha, result[y_start:H], 1 - alpha, 0)
    return result


def get_face_info(app, img_112: np.ndarray):
    """
    Upsample 112x112 → 320x320, run InsightFace detection, return (bbox, kps) in
    original (112×112) coordinates. Returns (None, None) if no face found.
    """
    scale = 320 / 112
    img_large = cv2.resize(img_112, (320, 320), interpolation=cv2.INTER_LINEAR)
    faces = app.get(img_large)
    if not faces:
        return None, None
    face = faces[0]
    bbox = (face.bbox / scale).astype(np.float32)   # [x1,y1,x2,y2]
    kps = (face.kps / scale).astype(np.float32)     # (5,2)
    return bbox, kps


def synthesize_image(
    base_bgr: np.ndarray,
    occ_type: str,
    overlay: np.ndarray,
    bbox,
    kps,
) -> np.ndarray:
    """Return synthesized BGR image with occlusion applied."""
    H, W = base_bgr.shape[:2]

    if kps is not None:
        mouth_cx = int((kps[KPS_MOUTH_L][0] + kps[KPS_MOUTH_R][0]) / 2)
        mouth_cy = int((kps[KPS_MOUTH_L][1] + kps[KPS_MOUTH_R][1]) / 2)
        face_w = float(bbox[2] - bbox[0])
        chin_cy = int((mouth_cy + bbox[3]) / 2)
        nose_y = int(kps[KPS_NOSE][1])
    else:
        # Fallback fixed proportions
        mouth_cx, mouth_cy = W // 2, int(H * 0.72)
        face_w = float(W * 0.8)
        chin_cy = int(H * 0.87)
        nose_y = int(H * 0.55)

    # Scale overlay to 70% of face width
    target_w = max(20, int(face_w * 0.70))
    oh, ow = overlay.shape[:2]
    target_h = int(oh * target_w / ow)
    overlay_resized = np.array(
        Image.fromarray(overlay).resize((target_w, target_h), Image.LANCZOS)
    )

    base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)

    if occ_type == "cup":
        result_rgb = alpha_blend(base_rgb, overlay_resized, mouth_cx, mouth_cy)
    elif occ_type == "hand":
        result_rgb = alpha_blend(base_rgb, overlay_resized, mouth_cx, chin_cy)
    else:  # book
        result_rgb = apply_book_cover(base_rgb, nose_y)

    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize occluded face images")
    parser.add_argument("--types", default="cup,hand,book", help="Comma-separated occlusion types")
    parser.add_argument("--limit", type=int, default=None, help="Max identities to process")
    args = parser.parse_args()

    occ_types = [t.strip() for t in args.types.split(",")]
    logger.info("Occlusion types: %s", occ_types)

    # Load ALL overlay variants for random selection
    all_overlays: dict[str, list[np.ndarray]] = {}
    for t in occ_types:
        if t != "book":
            all_overlays[t] = [load_overlay(v) for v in OVERLAY_VARIANTS[t]]

    # Load identity map
    with open(LFW_JSON) as f:
        identity_map = json.load(f)

    identities = list(identity_map.keys())
    if args.limit:
        identities = identities[: args.limit]
    logger.info("Processing %d identities", len(identities))

    # Initialize InsightFace
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    stats: dict[str, int] = {t: 0 for t in occ_types}
    stats["detect_ok"] = 0
    stats["detect_fail"] = 0
    total_queries = sum(len(identity_map[i]["query"]) for i in identities)
    processed = 0

    for identity in identities:
        query_paths = identity_map[identity]["query"]
        for rel_path in query_paths:
            img_path = DATA_ROOT / rel_path
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("Cannot read %s", img_path)
                continue

            bbox, kps = get_face_info(app, img_bgr)
            if kps is not None:
                stats["detect_ok"] += 1
            else:
                stats["detect_fail"] += 1

            img_name = Path(rel_path).name
            for t in occ_types:
                # Randomly pick a variant for diversity
                variants = all_overlays.get(t)
                overlay = random.choice(variants) if variants else np.zeros((1, 1, 4), dtype=np.uint8)
                synth = synthesize_image(img_bgr, t, overlay, bbox, kps)
                out_dir = SYNTH_ROOT / t / identity
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / img_name), synth)
                stats[t] += 1

            processed += 1
            if processed % 200 == 0 or processed == total_queries:
                logger.info(
                    "Progress: %d/%d | detect_ok=%d fail=%d",
                    processed,
                    total_queries,
                    stats["detect_ok"],
                    stats["detect_fail"],
                )

    logger.info("=== Synthesis complete ===")
    for t in occ_types:
        logger.info("  %s: %d images", t, stats[t])
    detect_total = stats["detect_ok"] + stats["detect_fail"]
    if detect_total:
        rate = 100 * stats["detect_ok"] / detect_total
        logger.info("  FaceDetect rate: %.1f%% (%d/%d)", rate, stats["detect_ok"], detect_total)


if __name__ == "__main__":
    main()
