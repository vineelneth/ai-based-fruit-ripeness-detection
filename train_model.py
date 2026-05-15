#!/usr/bin/env python
"""
Train a MobileNetV2 regression model to predict fruit ripeness (%) and days to ripe.

Dataset layout (one subfolder per fruit, each with its own labels.csv):

    dataset/
    ├── apple/
    │   ├── apple_001.jpg
    │   ├── apple_002.jpg
    │   └── labels.csv          # columns: filename, ripeness_pct, days_to_ripe
    ├── banana/
    │   ├── banana_001.jpg
    │   └── labels.csv
    └── mango/
        └── ...

labels.csv columns:
    filename      - image filename relative to the fruit folder (e.g. apple_001.jpg)
    ripeness_pct  - ripeness percentage (0-100)
    days_to_ripe  - estimated days until peak ripeness (>= 0)

Usage:
    # Train on all fruits
    python train_model.py

    # Train on a single fruit only
    python train_model.py --fruit banana

    # Custom paths
    python train_model.py --dataset-dir /path/to/dataset --model-out mymodel.keras
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(dataset_dir: str, fruit_filter: str | None = None) -> pd.DataFrame:
    """
    Discover per-fruit subdirectories under dataset_dir and merge their labels.

    Each subdirectory must contain a labels.csv with columns:
        filename, ripeness_pct, days_to_ripe

    The returned DataFrame has a 'filename' column with paths relative to
    dataset_dir (e.g. 'apple/apple_001.jpg') so it can be used directly with
    flow_from_dataframe(directory=dataset_dir).
    """
    dataset_path = Path(dataset_dir).resolve()
    fruit_dirs = sorted(d for d in dataset_path.iterdir() if d.is_dir())

    if fruit_filter:
        fruit_dirs = [d for d in fruit_dirs if d.name == fruit_filter]
        if not fruit_dirs:
            raise ValueError(
                f"No fruit folder named '{fruit_filter}' found in {dataset_dir}. "
                f"Available: {[d.name for d in sorted(dataset_path.iterdir()) if d.is_dir()]}"
            )

    if not fruit_dirs:
        raise ValueError(
            f"No fruit subdirectories found in '{dataset_dir}'. "
            "Each fruit must have its own subfolder containing a labels.csv."
        )

    frames = []
    for fruit_dir in fruit_dirs:
        csv_path = fruit_dir / "labels.csv"
        if not csv_path.exists():
            print(f"  [WARN] {fruit_dir.name}/ has no labels.csv — skipping")
            continue
        df = pd.read_csv(csv_path)
        required = {"filename", "ripeness_pct", "days_to_ripe"}
        missing = required - set(df.columns)
        if missing:
            print(f"  [WARN] {fruit_dir.name}/labels.csv missing columns {missing} — skipping")
            continue
        # Prefix filename with fruit subfolder so flow_from_dataframe can resolve it
        df["filename"] = df["filename"].apply(lambda f: f"{fruit_dir.name}/{f}")
        df["fruit"] = fruit_dir.name
        frames.append(df)
        print(f"  Loaded {len(df):>5} images  [{fruit_dir.name}]")

    if not frames:
        raise ValueError("No valid fruit folders with labels.csv found.")

    merged = pd.concat(frames, ignore_index=True)
    print(f"  ─────────────────────────────────")
    print(f"  Total:  {len(merged):>5} images  [{len(frames)} fruit(s)]")
    return merged


def build_model() -> Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(192, 192, 3))
    for layer in base.layers[:-50]:
        layer.trainable = False
    for layer in base.layers[-50:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
    return model


def plot_history(history) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["mae"], label="Train MAE")
    axes[0].plot(history.history["val_mae"], label="Val MAE")
    axes[0].set_title("Mean Absolute Error (lower is better)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss (MSE)")
    axes[1].plot(history.history["val_loss"], label="Val Loss (MSE)")
    axes[1].set_title("Loss (MSE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train fruit ripeness regression model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir", default="dataset",
        help="Root dataset directory containing per-fruit subfolders (default: dataset/)",
    )
    parser.add_argument(
        "--fruit", default=None,
        help="Train on a single fruit subfolder only (e.g. --fruit banana). "
             "Omit to train on all fruits.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--model-out", default="fruit_freshness_regression.keras",
        help="Output model path (default: fruit_freshness_regression.keras)",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset_dir)
    model_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model_out)

    print(f"\nDataset root : {dataset_dir}")
    if args.fruit:
        print(f"Fruit filter : {args.fruit}")
    print()

    df = load_dataset(dataset_dir, fruit_filter=args.fruit)

    target_scaler = MinMaxScaler()
    df[["ripeness_pct", "days_to_ripe"]] = target_scaler.fit_transform(
        df[["ripeness_pct", "days_to_ripe"]]
    )
    joblib.dump(target_scaler, "target_scaler.save")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.2,
        height_shift_range=0.2,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    flow_kwargs = dict(
        directory=dataset_dir,
        x_col="filename",
        y_col=["ripeness_pct", "days_to_ripe"],
        target_size=(192, 192),
        batch_size=args.batch_size,
        class_mode="raw",
    )
    train_set = train_datagen.flow_from_dataframe(dataframe=train_df, **flow_kwargs)
    test_set = test_datagen.flow_from_dataframe(dataframe=test_df, **flow_kwargs)

    model = build_model()

    callbacks = [
        EarlyStopping(monitor="val_mae", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_out, monitor="val_mae", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_set,
        epochs=args.epochs,
        validation_data=test_set,
        callbacks=callbacks,
    )

    model.save(model_out)
    print(f"\nModel saved to {model_out}")

    plot_history(history)


if __name__ == "__main__":
    main()
