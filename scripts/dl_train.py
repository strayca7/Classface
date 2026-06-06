"""
Train ResUNet on LFW processed images with GrabCut pseudo-labels.

Usage:
  uv run python scripts/dl_train.py [--epochs N] [--batch N] [--limit N] [--lr F]

Outputs:
  data/features/unet_ckpt.pth   — best checkpoint (by val Dice)
  data/results/dl_train_log.txt — per-epoch loss & Dice history
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))
from dl_model import ResUNet, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dl_train")

# ── ImageNet normalisation constants ──────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 112×112 → normalised float32 tensor (3, 112, 112)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


class FaceSegDataset(Dataset):
    """Pairs processed LFW images with GrabCut pseudo-label masks."""

    def __init__(self, pairs: list[tuple[Path, Path]], augment: bool = False) -> None:
        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((112, 112), dtype=np.uint8)

        # Resize to 112×112 if needed
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        if mask.shape != (112, 112):
            mask = cv2.resize(mask, (112, 112), interpolation=cv2.INTER_NEAREST)

        # Binarise mask (GrabCut outputs 255 for foreground)
        bin_mask = (mask > 127).astype(np.float32)

        if self.augment:
            img, bin_mask = _augment(img, bin_mask)

        x = _to_tensor(img)
        y = torch.from_numpy(bin_mask).unsqueeze(0)  # (1, 112, 112)
        return x, y


def _augment(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal flip + slight brightness jitter."""
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    # Brightness jitter ±20
    delta = random.randint(-20, 20)
    img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return img, mask


# ── Loss ──────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.view(-1)
    tgt_flat = target.view(-1)
    intersection = (pred_flat * tgt_flat).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred_flat.sum() + tgt_flat.sum() + eps)


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    p = (pred > threshold).float()
    t = target.float()
    inter = (p * t).sum().item()
    union = p.sum().item() + t.sum().item()
    return (2 * inter + 1e-6) / (union + 1e-6)


# ── Build dataset ─────────────────────────────────────────────────────────────

def build_pairs(
    lfw_json: Path,
    processed_dir: Path,
    grabcut_dir: Path,
    limit: int | None = None,
) -> list[tuple[Path, Path]]:
    """Collect (image_path, mask_path) pairs where both files exist."""
    with open(lfw_json) as f:
        db = json.load(f)

    pairs: list[tuple[Path, Path]] = []
    identities = list(db.keys())
    if limit:
        identities = identities[:limit]

    for identity in identities:
        all_imgs = db[identity]["gallery"] + db[identity]["query"]
        for rel_path in all_imgs:
            img_p = processed_dir / rel_path
            # GrabCut masks are stored as <person>/<stem>.png
            parts = Path(rel_path).parts
            mask_p = grabcut_dir / parts[-2] / (Path(parts[-1]).stem + ".png")
            if img_p.exists() and mask_p.exists():
                pairs.append((img_p, mask_p))

    logger.info("Dataset: %d valid pairs", len(pairs))
    return pairs


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = get_device()

    # Auto-adjust batch size for CPU
    batch_size = args.batch
    if device.type == "cpu" and batch_size > 4:
        batch_size = 4
        logger.info("CPU detected — reducing batch size to %d", batch_size)

    root = Path(__file__).parent.parent
    pairs = build_pairs(
        lfw_json=root / "data/raw/lfw_filtered.json",
        processed_dir=root / "data/processed/lfw",
        grabcut_dir=root / "data/segmented/grabcut",
        limit=args.limit,
    )

    if not pairs:
        logger.error(
            "No valid pairs found. Ensure GrabCut masks exist "
            "(run `make segment-face` first)."
        )
        sys.exit(1)

    random.shuffle(pairs)
    split = int(0.8 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    logger.info("Train: %d  Val: %d", len(train_pairs), len(val_pairs))

    train_loader = DataLoader(
        FaceSegDataset(train_pairs, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        FaceSegDataset(val_pairs, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ResUNet(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    ckpt_path = root / "data/features/unet_ckpt.pth"
    log_path = root / "data/results/dl_train_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0
    log_lines: list[str] = ["epoch,train_loss,val_loss,val_dice"]

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = combined_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_pairs)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss_sum += combined_loss(pred, y).item() * x.size(0)
                val_dice_sum += dice_score(pred, y) * x.size(0)
        val_loss = val_loss_sum / len(val_pairs)
        val_dice = val_dice_sum / len(val_pairs)

        scheduler.step()
        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_dice=%.4f",
            epoch, args.epochs, train_loss, val_loss, val_dice,
        )
        log_lines.append(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  ✓ Saved best checkpoint (val_dice=%.4f)", best_val_dice)

    logger.info("Training complete. Best val Dice: %.4f", best_val_dice)
    logger.info("Checkpoint: %s", ckpt_path)

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    logger.info("Training log: %s", log_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResUNet face segmentation model")
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    p.add_argument("--batch", type=int, default=16, help="Batch size (auto-reduced to 4 on CPU)")
    p.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of identities (debug)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
