from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Load the new regression model
model = keras.models.load_model("fruit_freshness_regression.keras")
print("Regression Model loaded successfully.\n")


# Predict single image
def predict_fruit(img_path):
    # Load and preprocess image for the AI
    test_img = image.load_img(img_path, target_size=(192, 192))
    test_img_array = image.img_to_array(test_img)
    test_img_array = np.expand_dims(test_img_array, axis=0) / 255.0

    # Predict the raw numbers
    preds = model.predict(test_img_array, verbose=0)[0]

    # Extract the two values from the prediction array
    ripeness_pct = preds[0]
    days_to_ripe = preds[1]

    # Formatting constraints
    ripeness_pct = np.clip(ripeness_pct, 0, 100)
    days_to_ripe = max(0, days_to_ripe)

    # Determine Status
    filename = os.path.basename(img_path)
    if ripeness_pct >= 95:
        status = "Fully Ripe / Overripe"
        title_color = "red"
    else:
        status = "Unripe / Ripening"
        title_color = "green"

    # 1. Print the result to the Terminal
    print(f"--- {filename} ---")
    print(f"Status:       {status}")
    print(f"Ripeness:     {ripeness_pct:.1f}%")
    print(f"Days to Ripe: {days_to_ripe:.1f} days\n")

    # 2. Show the result visually on the Image
    display_img = image.load_img(img_path)  # Load original high-res image
    plt.figure(figsize=(6, 6))
    plt.imshow(display_img)
    plt.axis("off")  # Hide the graph lines and numbers

    # Create the text block for the top of the image
    title_text = f"Status: {status}\nRipeness: {ripeness_pct:.1f}%\nEst. Days to Ripe: {days_to_ripe:.1f}"

    # Display the window
    plt.title(title_text, fontsize=14, color=title_color, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.show()  # Code pauses here until you close the image window


# --- Visual File Picker Loop ---
def run_prediction_loop():
    # Set up the hidden tkinter window ONCE outside the loop
    root = tk.Tk()
    root.withdraw()

    print("=" * 40)
    print("Fruit Predictor Started!")
    print("Click 'Cancel' in the file picker to exit.")
    print("=" * 40)

    # Start the continuous loop
    while True:
        # Open the file explorer dialog
        file_path = filedialog.askopenfilename(
            title="Select a Fruit Image (Click 'Cancel' to Stop)",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
        )

        # If a file was selected, run the prediction
        if file_path:
            predict_fruit(file_path)
        # If no file was selected (User clicked Cancel), break the loop
        else:
            print("\nPrediction loop stopped. Have a great day!")
            break


# Run the application
run_prediction_loop()
