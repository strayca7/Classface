"""
Batch inference: run trained ResUNet on all processed LFW images.

Usage:
  uv run python scripts/dl_segment.py [--limit N]

Inputs:
  data/features/unet_ckpt.pth   — trained checkpoint
  data/processed/lfw/           — preprocessed 112×112 LFW images

Outputs:
  data/segmented/dl_unet/<person_name>/<stem>.png  — binary uint8 masks (0 or 255)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dl_model import ResUNet
from device_utils import get_device
from dl_train import _to_tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dl_segment")

THRESHOLD = 0.5


def segment_all(args: argparse.Namespace) -> None:
    device = get_device()
    root = Path(__file__).parent.parent
    ckpt_path = root / "data/features/unet_ckpt.pth"

    if not ckpt_path.exists():
        logger.error(
            "Checkpoint not found: %s\nRun `make dl-train` first.", ckpt_path
        )
        sys.exit(1)

    model = ResUNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    logger.info("Loaded checkpoint: %s", ckpt_path)

    with open(root / "data/raw/lfw_filtered.json") as f:
        db = json.load(f)

    out_root = root / "data/segmented/dl_unet"
    identities = list(db.keys())
    if args.limit:
        identities = identities[:args.limit]

    total = ok = fail = 0
    for i, identity in enumerate(identities):
        all_imgs = db[identity]["gallery"] + db[identity]["query"]
        for rel_path in all_imgs:
            img_p = root / "data/processed/lfw" / rel_path
            if not img_p.exists():
                continue

            img = cv2.imread(str(img_p))
            if img is None:
                fail += 1
                continue
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112))

            x = _to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x)           # (1, 1, 112, 112)

            mask_np = (pred[0, 0].cpu().numpy() > THRESHOLD).astype(np.uint8) * 255

            # Mirror directory structure: dl_unet/<person>/<stem>.png
            parts = Path(rel_path).parts
            out_dir = out_root / parts[-2]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_p = out_dir / (Path(parts[-1]).stem + ".png")
            cv2.imwrite(str(out_p), mask_np)
            ok += 1
            total += 1

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d identities processed", i + 1, len(identities))

    logger.info("Done. Total=%d  OK=%d  Failed=%d", total, ok, fail)
    logger.info("Masks saved to: %s", out_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference with trained ResUNet")
    p.add_argument("--limit", type=int, default=None, help="Limit identities (debug)")
    return p.parse_args()


if __name__ == "__main__":
    segment_all(parse_args())
