"""Synthesize occluded face images using overlay assets and InsightFace keypoints.

Uses InsightFace 5-kps (left-eye, right-eye, nose, mouth-left, mouth-right)
and bounding box to anchor three occlusion types:

  cup        – overlay centred at mouth midpoint, scaled to face width × 0.7
  glasses    – overlay centred at eye midpoint, scaled to inter-eye distance × 2.8
  sunglasses – same anchor as glasses, scaled to inter-eye distance × 3.0

Overlay assets are discovered automatically from data/overlays/ by prefix
(cup_*.png, glasses_*.png, sunglasses_*.png).

Fallback to fixed proportions if InsightFace detects no face.

Usage:
    uv run python scripts/generate_cover.py [--types cup,glasses,sunglasses] [--limit N]
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

# InsightFace 5-kps indices
KPS_LEFT_EYE = 0
KPS_RIGHT_EYE = 1
KPS_NOSE = 2
KPS_MOUTH_L = 3
KPS_MOUTH_R = 4

# Scale factors: overlay width = face_ref × scale
SCALE = {
    "cup": 0.70,       # 70% of face width, anchored at mouth
    "glasses": 2.8,    # 2.8× inter-eye distance, anchored at eye midpoint
    "sunglasses": 3.0, # 3.0× inter-eye distance, anchored at eye midpoint
}


def discover_variants(occ_type: str) -> list[str]:
    """Return sorted list of overlay filenames matching <occ_type>_*.png."""
    files = sorted(OVERLAY_DIR.glob(f"{occ_type}_*.png"))
    names = [f.name for f in files]
    if not names:
        raise FileNotFoundError(f"No overlay files found for type '{occ_type}' in {OVERLAY_DIR}")
    return names


def load_overlay(name: str) -> np.ndarray:
    """Load RGBA overlay as (H, W, 4) uint8 numpy array."""
    img = Image.open(OVERLAY_DIR / name).convert("RGBA")
    return np.array(img)


def alpha_blend(base: np.ndarray, overlay_rgba: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Alpha-blend overlay centred at (cx, cy) onto base (H, W, 3) RGB image."""
    H, W = base.shape[:2]
    oh, ow = overlay_rgba.shape[:2]
    x1 = cx - ow // 2
    y1 = cy - oh // 2
    x2, y2 = x1 + ow, y1 + oh

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
    result = base.copy()
    result[y1:y2, x1:x2] = (fg * alpha + bg * (1 - alpha)).clip(0, 255).astype(np.uint8)
    return result


def get_face_info(app, img_112: np.ndarray):
    """Upsample 112→320, detect face, return (bbox, kps) in 112-space. (None, None) on failure."""
    scale = 320 / 112
    img_large = cv2.resize(img_112, (320, 320), interpolation=cv2.INTER_LINEAR)
    faces = app.get(img_large)
    if not faces:
        return None, None
    face = faces[0]
    return (face.bbox / scale).astype(np.float32), (face.kps / scale).astype(np.float32)


def synthesize_image(
    base_bgr: np.ndarray,
    occ_type: str,
    overlay: np.ndarray,
    bbox,
    kps,
) -> np.ndarray:
    """Return synthesized BGR image with occlusion applied."""
    H, W = base_bgr.shape[:2]
    scale = SCALE[occ_type]

    if kps is not None:
        eye_cx = int((kps[KPS_LEFT_EYE][0] + kps[KPS_RIGHT_EYE][0]) / 2)
        eye_cy = int((kps[KPS_LEFT_EYE][1] + kps[KPS_RIGHT_EYE][1]) / 2)
        eye_dist = abs(float(kps[KPS_RIGHT_EYE][0] - kps[KPS_LEFT_EYE][0]))
        mouth_cx = int((kps[KPS_MOUTH_L][0] + kps[KPS_MOUTH_R][0]) / 2)
        mouth_cy = int((kps[KPS_MOUTH_L][1] + kps[KPS_MOUTH_R][1]) / 2)
        face_w = float(bbox[2] - bbox[0])
    else:
        # Fallback: fixed proportions for 112×112 LFW images
        eye_cx, eye_cy = W // 2, int(H * 0.38)
        eye_dist = W * 0.28
        mouth_cx, mouth_cy = W // 2, int(H * 0.72)
        face_w = W * 0.8

    if occ_type == "cup":
        ref_w = face_w
        cx, cy = mouth_cx, mouth_cy
    else:  # glasses / sunglasses
        ref_w = eye_dist
        cx, cy = eye_cx, eye_cy

    target_w = max(20, int(ref_w * scale))
    oh, ow = overlay.shape[:2]
    target_h = max(1, int(oh * target_w / ow))
    overlay_resized = np.array(
        Image.fromarray(overlay).resize((target_w, target_h), Image.LANCZOS)
    )

    base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = alpha_blend(base_rgb, overlay_resized, cx, cy)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize occluded face images")
    parser.add_argument(
        "--types", default="cup,glasses,sunglasses",
        help="Comma-separated occlusion types (cup / glasses / sunglasses)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max identities to process")
    args = parser.parse_args()

    occ_types = [t.strip() for t in args.types.split(",")]
    logger.info("Occlusion types: %s", occ_types)

    # Discover and load all overlay variants per type
    all_overlays: dict[str, list[np.ndarray]] = {}
    for t in occ_types:
        variants = discover_variants(t)
        all_overlays[t] = [load_overlay(v) for v in variants]
        logger.info("  %s: %d variants loaded", t, len(variants))

    with open(LFW_JSON) as f:
        identity_map = json.load(f)

    identities = list(identity_map.keys())
    if args.limit:
        identities = identities[: args.limit]
    logger.info("Processing %d identities", len(identities))

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    stats: dict[str, int] = {t: 0 for t in occ_types}
    stats["detect_ok"] = 0
    stats["detect_fail"] = 0
    total_queries = sum(len(identity_map[i]["query"]) for i in identities)
    processed = 0

    for identity in identities:
        for rel_path in identity_map[identity]["query"]:
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
                overlay = random.choice(all_overlays[t])
                synth = synthesize_image(img_bgr, t, overlay, bbox, kps)
                out_dir = SYNTH_ROOT / t / identity
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / img_name), synth)
                stats[t] += 1

            processed += 1
            if processed % 200 == 0 or processed == total_queries:
                logger.info(
                    "Progress: %d/%d | detect_ok=%d fail=%d",
                    processed, total_queries,
                    stats["detect_ok"], stats["detect_fail"],
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
