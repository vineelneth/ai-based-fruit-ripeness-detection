import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------------
# 1️⃣ Dataset Paths & Loading
# -----------------------------------
# Put ALL your images (train and test) in this single folder
image_directory = "dataset/all_images"
csv_path = "fruit_labels.csv"

# Load the labels from the CSV
df = pd.read_csv(csv_path)

# Split data: 80% for training, 20% for testing/validation
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# -----------------------------------
# 2️⃣ Data Preprocessing and Augmentation
# -----------------------------------
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

# NEW: Use flow_from_dataframe for regression tasks
train_set = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_directory,
    x_col="filename",
    y_col=["ripeness_pct", "days_to_ripe"],  # Predicting 2 values
    target_size=(192, 192),
    batch_size=32,
    class_mode="raw",  # 'raw' is required for regression arrays
)

test_set = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=image_directory,
    x_col="filename",
    y_col=["ripeness_pct", "days_to_ripe"],
    target_size=(192, 192),
    batch_size=32,
    class_mode="raw",
)

# -----------------------------------
# 3️⃣ Build Model (Transfer Learning for Regression)
# -----------------------------------
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(192, 192, 3)
)

# Freeze most layers, fine-tune last 50
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)

# NEW: Output layer is 2 nodes (Ripeness, Days) with 'linear' activation
output = Dense(2, activation="linear")(x)

model = Model(inputs=base_model.input, outputs=output)

# NEW: Compile with Mean Squared Error (loss) and Mean Absolute Error (metric)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])

# -----------------------------------
# 4️⃣ Train Model (With Auto-Stopping)
# -----------------------------------
# 1. Setup Early Stopping
early_stop = EarlyStopping(
    monitor="val_mae", patience=5, restore_best_weights=True, verbose=1
)

# 2. Setup Checkpoint
checkpoint = ModelCheckpoint(
    "fruit_freshness_regression.keras",
    monitor="val_mae",
    save_best_only=True,
    verbose=1,
)

# 3. Run the training with callbacks
history = model.fit(
    train_set, epochs=10, validation_data=test_set, callbacks=[early_stop, checkpoint]
)

# -----------------------------------
# 5️⃣ Save Model
# -----------------------------------
model.save("fruit_freshness_regression.keras")
print("Model trained and saved successfully as fruit_freshness_regression.keras")

# -----------------------------------
# 6️⃣ Plot Error (Accuracy equivalent for Regression)
# -----------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["mae"], label="Training MAE (Error)")
plt.plot(history.history["val_mae"], label="Validation MAE (Error)")
plt.title("Model Error (Lower is Better)")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss (MSE)")
plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()
