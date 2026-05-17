"""
Train a MobileNetV2 fruit classifier on the per-fruit dataset folders.

Usage:
  python train_fruit_classifier.py
  python train_fruit_classifier.py --epochs-head 10 --epochs-finetune 10 --batch-size 32

Outputs:
  fruit_classifier.keras  — saved model
  fruit_classes.json      — ordered list of class names (index = class id)
"""

import argparse
import json
import pathlib

import tensorflow as tf
from tensorflow import keras

DATASET_ROOT = "dataset"
IMG_SIZE = 224
MODEL_OUT = "fruit_classifier.keras"
CLASSES_OUT = "fruit_classes.json"
EXCLUDE = {"archive", "fruits-360"}
MIN_IMAGES = 50
AUTOTUNE = tf.data.AUTOTUNE


def get_class_names(root: pathlib.Path) -> list[str]:
    names = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in EXCLUDE:
            continue
        n_imgs = sum(1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if n_imgs >= MIN_IMAGES:
            names.append(d.name)
    return names


def make_dataset(root: str, class_names: list[str], batch_size: int, subset: str, seed: int = 42):
    ds = keras.utils.image_dataset_from_directory(
        root,
        class_names=class_names,
        validation_split=0.2,
        subset=subset,
        seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="int",
    )
    preprocess = keras.applications.mobilenet_v2.preprocess_input
    return ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def build_model(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = pathlib.Path(DATASET_ROOT)
    class_names = get_class_names(root)
    print(f"Classes ({len(class_names)}): {class_names}\n")

    train_ds = make_dataset(DATASET_ROOT, class_names, args.batch_size, "training")
    val_ds = make_dataset(DATASET_ROOT, class_names, args.batch_size, "validation")

    # Add augmentation to training set
    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.1),
    ])
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    model, base = build_model(len(class_names))
    model.summary(line_length=80)

    # Phase 1: train head with frozen base
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n--- Phase 1: training classification head (base frozen) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1)],
    )

    # Phase 2: fine-tune top 30 layers of base
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n--- Phase 2: fine-tuning top 30 base layers ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
        ],
    )

    model.save(MODEL_OUT)
    with open(CLASSES_OUT, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nSaved model  → {MODEL_OUT}")
    print(f"Saved classes → {CLASSES_OUT}")


if __name__ == "__main__":
    main()
