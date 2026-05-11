"""Generate programmatic RGBA overlay assets (cup, hand, book) for occlusion synthesis.

Each overlay is 80×80 RGBA PNG with 5px feathered edge only; main body is fully opaque
(alpha=255) so synthesized occlusions are realistic. Saved to data/overlays/.
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gen_overlays")

OUT_DIR = Path("data/overlays")
SIZE = 80
FEATHER = 5  # small radius — edges only, core stays opaque


def feather_alpha(img: Image.Image, radius: int = FEATHER) -> Image.Image:
    """Apply Gaussian feather to alpha channel edges (core remains opaque)."""
    r, g, b, a = img.split()
    a_blurred = a.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.merge("RGBA", (r, g, b, a_blurred))


# ---------------------------------------------------------------------------
# Cup variants
# ---------------------------------------------------------------------------

def make_cup_01() -> Image.Image:
    """Dark gray ceramic mug (trapezoid body + ellipse rim)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([(20, 28), (60, 28), (65, 76), (15, 76)], fill=(55, 55, 60, 255))
    draw.ellipse([16, 22, 64, 36], fill=(75, 75, 80, 255))
    # handle
    draw.arc([60, 38, 76, 60], start=320, end=220, fill=(75, 75, 80, 255), width=4)
    # highlight strip
    draw.rectangle([26, 36, 30, 68], fill=(90, 90, 95, 255))
    return feather_alpha(img)


def make_cup_02() -> Image.Image:
    """White ceramic mug (fully opaque)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([(22, 28), (58, 28), (62, 74), (18, 74)], fill=(235, 235, 238, 255))
    draw.ellipse([20, 22, 60, 35], fill=(220, 220, 224, 255))
    draw.arc([58, 38, 74, 60], start=320, end=220, fill=(200, 200, 205, 255), width=4)
    draw.rectangle([30, 36, 34, 65], fill=(255, 255, 255, 255))
    return feather_alpha(img)


def make_cup_03() -> Image.Image:
    """Tall blue-green water bottle (narrow cylinder)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # body
    draw.rectangle([28, 18, 52, 76], fill=(30, 120, 140, 255))
    # cap
    draw.rectangle([30, 10, 50, 20], fill=(20, 80, 100, 255))
    # ellipse top
    draw.ellipse([28, 15, 52, 22], fill=(20, 80, 100, 255))
    # label band
    draw.rectangle([28, 40, 52, 55], fill=(50, 160, 180, 255))
    # highlight
    draw.rectangle([31, 22, 35, 68], fill=(80, 180, 200, 255))
    return feather_alpha(img)


def make_cup_04() -> Image.Image:
    """Thermos / insulated travel mug (dark cylinder + metallic cap)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([24, 20, 56, 76], fill=(35, 35, 38, 255))
    # metallic cap
    draw.rectangle([22, 12, 58, 22], fill=(160, 160, 165, 255))
    draw.ellipse([22, 8, 58, 18], fill=(170, 170, 175, 255))
    # brand stripe
    draw.rectangle([24, 44, 56, 50], fill=(200, 60, 40, 255))
    # highlight
    draw.rectangle([27, 24, 31, 70], fill=(70, 70, 75, 255))
    return feather_alpha(img)


def make_cup_05() -> Image.Image:
    """Disposable paper cup (cream cone with horizontal lines)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # tapered body (wider at top)
    draw.polygon([(18, 20), (62, 20), (55, 76), (25, 76)], fill=(240, 225, 200, 255))
    # rim
    draw.ellipse([16, 14, 64, 28], fill=(220, 205, 180, 255))
    # horizontal texture lines
    for y in [32, 42, 52, 62]:
        draw.line([(22, y), (58, y)], fill=(200, 185, 160, 255), width=1)
    # highlight
    draw.rectangle([26, 28, 30, 68], fill=(255, 245, 225, 255))
    return feather_alpha(img)


# ---------------------------------------------------------------------------
# Hand variants
# ---------------------------------------------------------------------------

def make_hand_01() -> Image.Image:
    """Light skin-tone palm resting on chin (fully opaque)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([6, 18, 74, 74], fill=(220, 185, 150, 255))
    # thumb
    draw.ellipse([0, 10, 28, 40], fill=(215, 180, 145, 255))
    # knuckle lines
    draw.line([(25, 22), (28, 50)], fill=(190, 155, 120, 255), width=2)
    draw.line([(40, 20), (42, 50)], fill=(190, 155, 120, 255), width=2)
    draw.line([(55, 22), (56, 50)], fill=(190, 155, 120, 255), width=2)
    return feather_alpha(img)


def make_hand_02() -> Image.Image:
    """Wide palm, spread fingers (light skin)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 16, 80, 72], fill=(215, 178, 142, 255))
    for x in [20, 35, 50, 65]:
        draw.line([(x, 16), (x + 2, 52)], fill=(185, 148, 112, 255), width=3)
    return feather_alpha(img)


def make_hand_03() -> Image.Image:
    """Dark skin-tone palm (deeper brown)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([6, 18, 74, 74], fill=(130, 85, 50, 255))
    draw.ellipse([0, 10, 28, 40], fill=(125, 80, 45, 255))
    draw.line([(25, 22), (28, 50)], fill=(100, 60, 30, 255), width=2)
    draw.line([(40, 20), (42, 50)], fill=(100, 60, 30, 255), width=2)
    draw.line([(55, 22), (56, 50)], fill=(100, 60, 30, 255), width=2)
    return feather_alpha(img)


def make_hand_04() -> Image.Image:
    """Side-view hand (tilted ellipse blocking lower face)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # tilted oval (side palm view)
    draw.ellipse([2, 24, 62, 72], fill=(215, 178, 142, 255))
    # thumb sticking out sideways
    draw.ellipse([50, 12, 78, 42], fill=(210, 173, 137, 255))
    draw.line([(8, 30), (10, 65)], fill=(185, 148, 112, 255), width=2)
    draw.line([(20, 26), (22, 65)], fill=(185, 148, 112, 255), width=2)
    return feather_alpha(img)


def make_hand_05() -> Image.Image:
    """Spread fingers covering lower face (finger silhouettes)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # palm base
    draw.ellipse([8, 44, 72, 78], fill=(218, 182, 148, 255))
    # five fingers extending upward
    finger_x = [12, 24, 36, 50, 62]
    finger_w = [10, 10, 11, 10, 9]
    for x, w in zip(finger_x, finger_w, strict=True):
        draw.rounded_rectangle([x, 8, x + w, 52], radius=5, fill=(215, 178, 142, 255))
    return feather_alpha(img)


# ---------------------------------------------------------------------------
# Book variants
# ---------------------------------------------------------------------------

def make_book_01() -> Image.Image:
    """Blue hardcover book (fully opaque)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 8, 75, 74], fill=(40, 80, 165, 255))
    draw.rectangle([5, 8, 13, 74], fill=(28, 58, 128, 255))
    draw.rectangle([18, 20, 68, 26], fill=(200, 215, 245, 255))
    draw.rectangle([18, 32, 58, 37], fill=(200, 215, 245, 255))
    draw.rectangle([18, 43, 50, 47], fill=(180, 195, 225, 255))
    return feather_alpha(img)


def make_book_02() -> Image.Image:
    """Red hardcover book (fully opaque)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 8, 75, 74], fill=(180, 35, 35, 255))
    draw.rectangle([5, 8, 13, 74], fill=(140, 25, 25, 255))
    draw.rectangle([18, 20, 68, 26], fill=(245, 205, 205, 255))
    draw.rectangle([18, 32, 58, 37], fill=(245, 205, 205, 255))
    draw.rectangle([18, 43, 50, 47], fill=(225, 185, 185, 255))
    return feather_alpha(img)


def make_book_03() -> Image.Image:
    """Green hardcover book with visible spine."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 8, 75, 74], fill=(30, 110, 60, 255))
    draw.rectangle([5, 8, 13, 74], fill=(20, 80, 42, 255))
    draw.rectangle([18, 20, 68, 26], fill=(185, 230, 195, 255))
    draw.rectangle([18, 32, 58, 37], fill=(185, 230, 195, 255))
    draw.rectangle([18, 43, 50, 47], fill=(165, 210, 175, 255))
    return feather_alpha(img)


def make_book_04() -> Image.Image:
    """Black heavy textbook with gold spine text lines."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 8, 75, 74], fill=(22, 22, 24, 255))
    draw.rectangle([5, 8, 14, 74], fill=(12, 12, 14, 255))
    # gold lines on spine
    for y in [20, 30, 55, 65]:
        draw.line([(6, y), (13, y)], fill=(200, 170, 60, 255), width=2)
    draw.rectangle([18, 20, 68, 25], fill=(180, 150, 50, 255))
    draw.rectangle([18, 50, 65, 54], fill=(160, 130, 40, 255))
    return feather_alpha(img)


def make_book_05() -> Image.Image:
    """Cream-white notebook with spiral binding."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([12, 8, 75, 74], fill=(245, 240, 225, 255))
    # spiral binding
    for y in range(14, 70, 8):
        draw.ellipse([6, y, 16, y + 6], outline=(160, 160, 165, 255), width=2)
    # ruled lines
    for y in [28, 36, 44, 52, 60, 68]:
        draw.line([(18, y), (72, y)], fill=(180, 190, 210, 255), width=1)
    # red margin line
    draw.line([(26, 10), (26, 74)], fill=(220, 80, 80, 255), width=2)
    return feather_alpha(img)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "cup_01.png": make_cup_01,
    "cup_02.png": make_cup_02,
    "cup_03.png": make_cup_03,
    "cup_04.png": make_cup_04,
    "cup_05.png": make_cup_05,
    "hand_01.png": make_hand_01,
    "hand_02.png": make_hand_02,
    "hand_03.png": make_hand_03,
    "hand_04.png": make_hand_04,
    "hand_05.png": make_hand_05,
    "book_01.png": make_book_01,
    "book_02.png": make_book_02,
    "book_03.png": make_book_03,
    "book_04.png": make_book_04,
    "book_05.png": make_book_05,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove stale overlays no longer in registry
    for old in OUT_DIR.glob("*.png"):
        if old.name not in GENERATORS:
            old.unlink()
            logger.info("Removed stale overlay: %s", old.name)
    for fname, fn in GENERATORS.items():
        img = fn()
        out_path = OUT_DIR / fname
        img.save(out_path)
        logger.info("Saved %s (%dx%d RGBA)", out_path, img.width, img.height)
    logger.info("Done — %d overlay assets saved to %s", len(GENERATORS), OUT_DIR)


if __name__ == "__main__":
    main()
