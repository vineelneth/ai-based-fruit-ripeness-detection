"""
Generate demo.gif — animated demo showing live model predictions.
Run with: .venv/Scripts/python create_demo_gif.py
"""

import io
import os
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image as kimage

MODEL_PATH = "fruit_freshness_regression.keras"
IMAGE_DIR = "dataset/all_images"
OUTPUT_GIF = "demo.gif"
FRAME_DURATION_MS = 2200
TRANSITION_MS = 80
N_SAMPLES = 6
IMG_SIZE = (192, 192)
GIF_WIDTH = 520


def load_samples(image_dir: str, n: int, seed: int = 7) -> list[str]:
    all_imgs = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    random.seed(seed)
    # Pick balanced mix: apples and bananas
    apples = [f for f in all_imgs if "apple" in f.lower()]
    bananas = [f for f in all_imgs if "banana" in f.lower()]
    chosen = random.sample(apples, min(n // 2, len(apples))) + \
             random.sample(bananas, min(n - n // 2, len(bananas)))
    random.shuffle(chosen)
    return chosen[:n]


def predict(model, img_path: str) -> tuple[float, float]:
    img = kimage.load_img(img_path, target_size=IMG_SIZE)
    arr = np.expand_dims(kimage.img_to_array(img), axis=0) / 255.0
    preds = model.predict(arr, verbose=0)[0]
    return float(np.clip(preds[0], 0, 100)), float(max(0.0, preds[1]))


def ripeness_color(pct: float) -> str:
    if pct >= 85:
        return "#e05252"   # red — fully ripe/overripe
    elif pct >= 60:
        return "#f0a500"   # amber — ripe
    else:
        return "#4caf50"   # green — unripe/ripening


def render_frame(img_path: str, ripeness: float, days: float) -> Image.Image:
    display_img = kimage.load_img(img_path)
    fig, ax = plt.subplots(figsize=(4.5, 5.2), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.imshow(display_img)
    ax.axis("off")

    color = ripeness_color(ripeness)
    status = "Fully Ripe" if ripeness >= 85 else ("Ripening" if ripeness >= 60 else "Unripe")
    fruit = "Apple" if "apple" in os.path.basename(img_path).lower() else "Banana"

    # Top label bar
    ax.text(0.5, 1.04, f"{fruit}  •  {status}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color,
            fontfamily="monospace")

    # Ripeness bar background
    bar_y = -0.13
    bar_h = 0.055
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, bar_y), 1.0, bar_h,
        boxstyle="round,pad=0.01", linewidth=0,
        transform=ax.transAxes, clip_on=False,
        facecolor="#2e2e4a"
    ))
    # Ripeness fill
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, bar_y), ripeness / 100, bar_h,
        boxstyle="round,pad=0.01", linewidth=0,
        transform=ax.transAxes, clip_on=False,
        facecolor=color, alpha=0.85
    ))
    ax.text(0.5, bar_y + bar_h / 2,
            f"Ripeness: {ripeness:.1f}%",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white",
            fontfamily="monospace")

    # Days badge
    ax.text(0.5, bar_y - 0.07,
            f"Est. {days:.1f} days to peak ripeness",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#aaaacc", fontfamily="monospace")

    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    frame = Image.open(buf).convert("RGB")
    # Uniform width for GIF
    aspect = frame.height / frame.width
    frame = frame.resize((GIF_WIDTH, int(GIF_WIDTH * aspect)), Image.LANCZOS)
    return frame


def make_title_frame(width: int, height: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    ax.text(0.5, 0.62, "Fruit Ripeness Detection",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=16, fontweight="bold", color="#e0e0ff",
            fontfamily="monospace")
    ax.text(0.5, 0.45, "MobileNetV2 · Multi-output Regression",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#8888aa", fontfamily="monospace")
    ax.text(0.5, 0.28, "Predicts ripeness % + days to peak",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#666688", fontfamily="monospace")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    frame = Image.open(buf).convert("RGB").resize((width, height), Image.LANCZOS)
    return frame


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)

    print("Selecting sample images...")
    samples = load_samples(IMAGE_DIR, N_SAMPLES)

    frames: list[Image.Image] = []
    durations: list[int] = []

    # Predict all samples first so we know the final frame size
    results = []
    for fname in samples:
        path = os.path.join(IMAGE_DIR, fname)
        rip, days = predict(model, path)
        results.append((path, rip, days))
        print(f"  {fname[:50]:<50}  ripeness={rip:.1f}%  days={days:.1f}")

    # Build frames
    prediction_frames = [render_frame(p, r, d) for p, r, d in results]
    frame_h = prediction_frames[0].height

    title = make_title_frame(GIF_WIDTH, frame_h)
    frames.append(title)
    durations.append(1800)

    for pf in prediction_frames:
        frames.append(pf)
        durations.append(FRAME_DURATION_MS)

    print(f"\nSaving {OUTPUT_GIF} ({len(frames)} frames)...")
    frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"Done! {OUTPUT_GIF}  ({os.path.getsize(OUTPUT_GIF) // 1024} KB)")


if __name__ == "__main__":
    main()
