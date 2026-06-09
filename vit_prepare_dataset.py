"""
data/prepare_dataset.py
-----------------------
Two modes:
  1. SYNTHETIC : Generates a synthetic retinal fundus dataset that mirrors
                 APTOS 2019 structure: 3,662 images across 5 DR grades.
                 Creates actual PNG images with realistic circular fundus
                 appearance and grade-dependent visual features.
  2. REAL      : Instructions for downloading the APTOS 2019 dataset from Kaggle.

Diabetic Retinopathy Grades
-----------------------------
  0 — No DR
  1 — Mild
  2 — Moderate
  3 — Severe
  4 — Proliferative DR

Class distribution (mirrors APTOS 2019)
  0: 49.2%  |  1: 7.4%  |  2: 27.0%  |  3: 6.0%  |  4: 10.4%

Usage
-----
  # Synthetic (generates real PNG images — no download needed)
  python data/prepare_dataset.py --mode synthetic --out_dir data/processed

  # Real APTOS (requires Kaggle API key)
  python data/prepare_dataset.py --mode real
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# APTOS 2019 class distribution
GRADE_PROBS = [0.492, 0.074, 0.270, 0.060, 0.104]
GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
N_IMAGES    = 3662
IMG_SIZE    = 224   # ViT-B/16 input size


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fundus image generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_fundus_image(grade: int, img_size: int = 224, rng=None) -> Image.Image:
    """
    Generate a synthetic retinal fundus image for a given DR grade.

    Visual cues per grade:
      0 — Clean orange-red disc, clear vessels, no lesions
      1 — Mild: 1-2 small microaneurysms (dark dots)
      2 — Moderate: multiple MA + hard exudates (bright yellow dots)
      3 — Severe: haemorrhages, venous beading, IRMA
      4 — Proliferative: neovascularisation, large haemorrhages, fibrosis
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Background: dark circular fundus ──────────────────────────────────
    img = Image.new("RGB", (img_size, img_size), color=(10, 5, 5))
    draw = ImageDraw.Draw(img)

    cx, cy = img_size // 2, img_size // 2
    radius = int(img_size * 0.46)

    # Fundus background (reddish-orange retina)
    base_r = int(rng.integers(120, 150))
    base_g = int(rng.integers(40, 70))
    base_b = int(rng.integers(20, 45))
    draw.ellipse([(cx-radius, cy-radius), (cx+radius, cy+radius)],
                  fill=(base_r, base_g, base_b))

    # ── Optic disc (bright yellow-white circle) ───────────────────────────
    disc_cx = cx + int(rng.integers(-20, 20))
    disc_cy = cy + int(rng.integers(-10, 10))
    disc_r  = int(img_size * 0.07)
    draw.ellipse([(disc_cx-disc_r, disc_cy-disc_r),
                  (disc_cx+disc_r, disc_cy+disc_r)],
                  fill=(230, 210, 170))

    # ── Blood vessels (dark branching lines) ──────────────────────────────
    n_vessels = int(rng.integers(4, 8))
    for _ in range(n_vessels):
        angle  = rng.uniform(0, 2 * np.pi)
        length = int(rng.integers(radius // 3, radius - 5))
        end_x  = int(disc_cx + length * np.cos(angle))
        end_y  = int(disc_cy + length * np.sin(angle))
        vessel_color = (int(base_r * 0.4), int(base_g * 0.4), int(base_b * 0.4))
        width  = int(rng.integers(1, 3))
        draw.line([(disc_cx, disc_cy), (end_x, end_y)],
                   fill=vessel_color, width=width)

    # ── Grade-specific lesions ─────────────────────────────────────────────

    if grade >= 1:
        # Microaneurysms: small dark red dots
        n_ma = int(rng.integers(1, 3 + grade * 2))
        for _ in range(n_ma):
            mx = int(cx + rng.integers(-radius + 10, radius - 10))
            my = int(cy + rng.integers(-radius + 10, radius - 10))
            mr = int(rng.integers(2, 4))
            draw.ellipse([(mx-mr, my-mr), (mx+mr, my+mr)],
                          fill=(100, 20, 20))

    if grade >= 2:
        # Hard exudates: bright yellow-white dots
        n_ex = int(rng.integers(3, 8 + grade * 3))
        for _ in range(n_ex):
            ex = int(cx + rng.integers(-radius + 10, radius - 10))
            ey = int(cy + rng.integers(-radius + 10, radius - 10))
            er = int(rng.integers(2, 5))
            draw.ellipse([(ex-er, ey-er), (ex+er, ey+er)],
                          fill=(240, 230, 150))

        # Cotton wool spots: soft white patches
        n_cw = int(rng.integers(1, 4))
        for _ in range(n_cw):
            cwx = int(cx + rng.integers(-radius + 15, radius - 15))
            cwy = int(cy + rng.integers(-radius + 15, radius - 15))
            cwr = int(rng.integers(4, 9))
            draw.ellipse([(cwx-cwr, cwy-cwr), (cwx+cwr, cwy+cwr)],
                          fill=(200, 195, 180))

    if grade >= 3:
        # Haemorrhages: larger dark red blotches
        n_hm = int(rng.integers(4, 12))
        for _ in range(n_hm):
            hx = int(cx + rng.integers(-radius + 10, radius - 10))
            hy = int(cy + rng.integers(-radius + 10, radius - 10))
            hr = int(rng.integers(4, 10))
            draw.ellipse([(hx-hr, hy-hr), (hx+hr, hy+hr)],
                          fill=(60, 5, 5))

        # Venous beading: irregular vessel width variations
        for _ in range(3):
            angle  = rng.uniform(0, 2 * np.pi)
            for seg in range(5):
                sx = int(disc_cx + (radius * 0.5 + seg * 8) * np.cos(angle))
                sy = int(disc_cy + (radius * 0.5 + seg * 8) * np.sin(angle))
                bw = int(rng.integers(2, 5))
                draw.ellipse([(sx-bw, sy-bw), (sx+bw, sy+bw)],
                              fill=(int(base_r*0.3), 5, 5))

    if grade >= 4:
        # Neovascularisation: fine new vessel network near disc
        for _ in range(6):
            angle  = rng.uniform(0, 2 * np.pi)
            dist   = int(rng.integers(disc_r + 2, disc_r + 20))
            nx     = int(disc_cx + dist * np.cos(angle))
            ny     = int(disc_cy + dist * np.sin(angle))
            draw.line([(disc_cx, disc_cy), (nx, ny)],
                       fill=(180, 50, 50), width=1)

        # Fibrous proliferation: white-grey patches
        for _ in range(3):
            fx = int(cx + rng.integers(-radius + 20, radius - 20))
            fy = int(cy + rng.integers(-radius + 20, radius - 20))
            fr = int(rng.integers(8, 18))
            draw.ellipse([(fx-fr, fy-fr), (fx+fr, fy+fr)],
                          fill=(170, 160, 150))

    # Smooth the image (real fundus images aren't pixel-perfect)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    # Add mild noise
    arr  = np.array(img, dtype=np.float32)
    arr += rng.normal(0, 4, arr.shape)
    arr  = np.clip(arr, 0, 255).astype(np.uint8)
    img  = Image.fromarray(arr)

    return img


def generate_synthetic_dataset(out_dir: Path, n_images: int = 3662,
                                 img_size: int = IMG_SIZE, seed: int = 42):
    rng    = np.random.default_rng(seed)
    grades = rng.choice(len(GRADE_PROBS), size=n_images, p=GRADE_PROBS)

    # ── Create directory structure ────────────────────────────────────────
    for split in ["train", "val", "test"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)

    # ── Assign splits (stratified) ────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    idx      = np.arange(n_images)
    tr_idx, te_idx = train_test_split(idx, test_size=0.15,
                                       stratify=grades, random_state=seed)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.12,
                                       stratify=grades[tr_idx], random_state=seed)

    split_map = {}
    for i in tr_idx: split_map[i] = "train"
    for i in va_idx: split_map[i] = "val"
    for i in te_idx: split_map[i] = "test"

    print(f"Generating {n_images:,} synthetic fundus images "
          f"({img_size}×{img_size} px)...")

    records = []
    for i in range(n_images):
        grade  = int(grades[i])
        split  = split_map[i]
        fname  = f"img_{i:05d}.png"
        fpath  = out_dir / split / "images" / fname

        img = generate_fundus_image(grade, img_size, rng)
        img.save(fpath, "PNG")

        records.append({"id_code": f"img_{i:05d}", "diagnosis": grade,
                         "split": split, "filename": fname})

        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1:,}/{n_images:,}...")

    # ── Save labels CSVs ──────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "all_labels.csv", index=False)
    for split in ["train","val","test"]:
        df[df["split"]==split][["id_code","diagnosis"]].to_csv(
            out_dir / f"{split}_labels.csv", index=False)

    meta = {
        "n_total": n_images, "img_size": img_size,
        "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
        "class_dist": {str(g): int((grades==g).sum()) for g in range(5)},
        "grade_names": GRADE_NAMES,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SYNTHETIC] Saved {n_images:,} fundus images → {out_dir}")
    for g in range(5):
        print(f"  Grade {g} ({GRADE_NAMES[g]}): {(grades==g).sum():,} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     choices=["synthetic","real"], default="synthetic")
    parser.add_argument("--out_dir",  default="data/processed")
    parser.add_argument("--n_images", type=int, default=3662)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.mode == "synthetic":
        generate_synthetic_dataset(out_dir, args.n_images, args.img_size, args.seed)
    else:
        print("Real APTOS 2019: kaggle competitions download -c aptos2019-blindness-detection")
        print("Then re-run with --mode synthetic to test the full pipeline.")
