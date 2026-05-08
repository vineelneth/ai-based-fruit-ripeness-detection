"""
Run ripeness predictions on fruit images.

GUI mode (default): opens a file-picker dialog; loops until you click Cancel.
CLI mode:           pass --image <path> to predict a single image non-interactively.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image


def load_model(model_path: str):
    if not os.path.exists(model_path):
        sys.exit(f"Error: model file not found at '{model_path}'")
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}\n")
    return model


def predict_fruit(model, img_path: str, show_plot: bool = True) -> dict:
    img = image.load_img(img_path, target_size=(192, 192))
    arr = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

    preds = model.predict(arr, verbose=0)[0]
    ripeness_pct = float(np.clip(preds[0], 0, 100))
    days_to_ripe = float(max(0.0, preds[1]))

    status = "Fully Ripe / Overripe" if ripeness_pct >= 95 else "Unripe / Ripening"
    title_color = "red" if ripeness_pct >= 95 else "green"

    print(f"--- {os.path.basename(img_path)} ---")
    print(f"Status:       {status}")
    print(f"Ripeness:     {ripeness_pct:.1f}%")
    print(f"Days to Ripe: {days_to_ripe:.1f} days\n")

    if show_plot:
        display_img = image.load_img(img_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(display_img)
        plt.axis("off")
        plt.title(
            f"Status: {status}\nRipeness: {ripeness_pct:.1f}%\nEst. Days to Ripe: {days_to_ripe:.1f}",
            fontsize=14,
            color=title_color,
            fontweight="bold",
            pad=15,
        )
        plt.tight_layout()
        plt.show()

    return {"status": status, "ripeness_pct": ripeness_pct, "days_to_ripe": days_to_ripe}


def run_gui_loop(model) -> None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    print("=" * 40)
    print("Fruit Predictor — GUI mode")
    print("Click 'Cancel' in the file picker to exit.")
    print("=" * 40 + "\n")

    while True:
        path = filedialog.askopenfilename(
            title="Select a fruit image (Cancel to exit)",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
        )
        if path:
            predict_fruit(model, path)
        else:
            print("Exiting. Have a great day!")
            break


def main():
    parser = argparse.ArgumentParser(description="Predict fruit ripeness from an image.")
    parser.add_argument("--image", metavar="PATH", help="Image path for CLI prediction (omit for GUI mode)")
    parser.add_argument("--model", default="fruit_freshness_regression.keras", help="Trained model file")
    parser.add_argument("--no-plot", action="store_true", help="Suppress matplotlib output (CLI mode only)")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image:
        if not os.path.exists(args.image):
            sys.exit(f"Error: image not found at '{args.image}'")
        predict_fruit(model, args.image, show_plot=not args.no_plot)
    else:
        run_gui_loop(model)


if __name__ == "__main__":
    main()
