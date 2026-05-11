"""Crop eye-region from face images using InsightFace 5-kps.

Eye region defined as:
  y: [bbox_top - pad_top, kps_nose_y + pad_bottom]
  x: [bbox_left - pad_side, bbox_right + pad_side]
Then resized to 112×112 for ArcFace input.

Processes gallery images and synthetic query images.
Precomputes gallery_cropped.npy feature embeddings.

Usage:
    uv run python scripts/crop.py [--limit N]
"""

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crop")

DATA_ROOT = Path("data/processed/lfw")
SYNTH_ROOT = Path("data/synthetic")
CROP_ROOT = Path("data/cropped")
FEAT_DIR = Path("data/features")
LFW_JSON = Path("data/raw/lfw_filtered.json")

KPS_NOSE = 2
PAD_TOP = 4      # px in original 112 space
PAD_BOTTOM = 4
PAD_SIDE = 4


def get_face_info(app, img_112: np.ndarray):
    """Upsample 112→320, detect, return (bbox, kps) in 112-space."""
    scale = 320 / 112
    img_large = cv2.resize(img_112, (320, 320), interpolation=cv2.INTER_LINEAR)
    faces = app.get(img_large)
    if not faces:
        return None, None
    face = faces[0]
    bbox = (face.bbox / scale).astype(np.float32)
    kps = (face.kps / scale).astype(np.float32)
    return bbox, kps


def crop_eye_region(img_112: np.ndarray, bbox, kps) -> np.ndarray:
    """Extract eye region; fall back to upper 55% if no detection."""
    H, W = img_112.shape[:2]
    if kps is not None:
        nose_y = int(kps[KPS_NOSE][1])
        x1 = max(0, int(bbox[0]) - PAD_SIDE)
        y1 = max(0, int(bbox[1]) - PAD_TOP)
        x2 = min(W, int(bbox[2]) + PAD_SIDE)
        y2 = min(H, nose_y + PAD_BOTTOM)
    else:
        x1, y1, x2, y2 = 0, 0, W, int(H * 0.55)

    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, W, int(H * 0.55)

    crop = img_112[y1:y2, x1:x2]
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)


def extract_feature(app, img_112: np.ndarray):
    """Extract 512-d embedding (1-D, shape (512,)); return None on failure."""
    img_large = cv2.resize(img_112, (320, 320), interpolation=cv2.INTER_LINEAR)
    faces = app.get(img_large)
    if faces:
        return faces[0].embedding.astype(np.float32).flatten()
    # Direct feature extraction fallback
    try:
        feat = app.models["recognition"].get_feat(img_112)
        return feat.astype(np.float32).flatten()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop eye regions for robust recognition")
    parser.add_argument("--limit", type=int, default=None, help="Max identities to process")
    args = parser.parse_args()

    with open(LFW_JSON) as f:
        identity_map = json.load(f)

    identities = list(identity_map.keys())
    if args.limit:
        identities = identities[: args.limit]
    logger.info("Processing %d identities", len(identities))

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # --- Gallery crops ---
    logger.info("Cropping gallery images...")
    gallery_embeddings = []
    gallery_labels = []
    n_detect_ok = n_detect_fail = 0

    for identity in identities:
        gallery_path = identity_map[identity]["gallery"][0]
        img_path = DATA_ROOT / gallery_path
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        bbox, kps = get_face_info(app, img)
        if kps is not None:
            n_detect_ok += 1
        else:
            n_detect_fail += 1
        crop = crop_eye_region(img, bbox, kps)

        # Save crop
        out_dir = CROP_ROOT / "gallery" / identity
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / Path(gallery_path).name), crop)

        # Extract feature from crop
        feat = extract_feature(app, crop)
        if feat is not None:
            gallery_embeddings.append(feat)
            gallery_labels.append(identity)

    gallery_arr = np.array(gallery_embeddings, dtype=np.float32)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEAT_DIR / "gallery_cropped.npy", gallery_arr)
    with open(FEAT_DIR / "gallery_cropped_labels.json", "w") as f:
        json.dump(gallery_labels, f)

    logger.info(
        "Gallery: %d crops saved, %d embeddings (detect_ok=%d fail=%d)",
        len(gallery_labels), len(gallery_embeddings), n_detect_ok, n_detect_fail,
    )

    # --- Synthetic query crops ---
    occ_types = ["cup", "glasses", "sunglasses"]
    n_detect_ok = n_detect_fail = 0
    total_query = 0

    for occ_type in occ_types:
        for identity in identities:
            synth_dir = SYNTH_ROOT / occ_type / identity
            if not synth_dir.exists():
                continue
            for img_path in sorted(synth_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                bbox, kps = get_face_info(app, img)
                if kps is not None:
                    n_detect_ok += 1
                else:
                    n_detect_fail += 1
                crop = crop_eye_region(img, bbox, kps)

                out_dir = CROP_ROOT / "query" / occ_type / identity
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / img_path.name), crop)
                total_query += 1

        logger.info("  %s query crops done", occ_type)

    logger.info(
        "Query: %d total crops (detect_ok=%d fail=%d)",
        total_query, n_detect_ok, n_detect_fail,
    )

    # --- Visualizations (random 10 gallery comparisons) ---
    vis_dir = CROP_ROOT / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = random.sample(gallery_labels[: len(gallery_labels)], min(10, len(gallery_labels)))
    for identity in sample_ids:
        gallery_path = identity_map[identity]["gallery"][0]
        orig = cv2.imread(str(DATA_ROOT / gallery_path))
        crop_path = next((CROP_ROOT / "gallery" / identity).iterdir(), None)
        if orig is None or crop_path is None:
            continue
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        # side by side: original | crop (both 112×112)
        orig_r = cv2.resize(orig, (112, 112))
        vis = np.hstack([orig_r, crop])
        cv2.imwrite(str(vis_dir / f"{identity}.jpg"), vis)

    logger.info("Visualizations saved to %s", vis_dir)
    logger.info("=== crop.py complete ===")


if __name__ == "__main__":
    main()
