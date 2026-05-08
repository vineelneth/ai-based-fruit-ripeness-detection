"""
Train a MobileNetV2 regression model to predict fruit ripeness (%) and days to ripe.
Outputs a .keras model file and plots training curves on completion.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model() -> Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(192, 192, 3))
    for layer in base.layers[:-50]:
        layer.trainable = False
    for layer in base.layers[-50:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation="linear")(x)

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
    parser = argparse.ArgumentParser(description="Train fruit ripeness regression model.")
    parser.add_argument("--image-dir", default="dataset/all_images", help="Flattened image directory")
    parser.add_argument("--csv-path", default="fruit_labels.csv", help="Labels CSV file")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model-out", default="fruit_freshness_regression.keras", help="Output model path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
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
        directory=args.image_dir,
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
        ModelCheckpoint(args.model_out, monitor="val_mae", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_set,
        epochs=args.epochs,
        validation_data=test_set,
        callbacks=callbacks,
    )

    model.save(args.model_out)
    print(f"\nModel saved to {args.model_out}")

    plot_history(history)


if __name__ == "__main__":
    main()
