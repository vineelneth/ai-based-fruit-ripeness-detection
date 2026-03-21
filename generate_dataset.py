import os
import shutil
import csv
import random

# -----------------------------------
# 1️⃣ Setup Paths
# -----------------------------------
# This assumes the script is running in the folder containing 'dataset'
source_base = "dataset"
folders_to_scan = ["train", "test"]

# Where the new structure will go
dest_dir = "dataset/all_images"
csv_file = "fruit_labels.csv"

# Create the new destination folder if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# -----------------------------------
# 2️⃣ Prepare CSV Data
# -----------------------------------
csv_data = [["filename", "ripeness_pct", "days_to_ripe"]]


def generate_labels(folder_name):
    """Generates dummy data based on the old folder names."""
    name = folder_name.lower()

    if "unripe" in name:
        # Unripe fruit: low percentage, high days
        ripeness = round(random.uniform(10.0, 45.0), 1)
        days = round(random.uniform(5.0, 10.0), 1)
    elif "fresh" in name:
        # Fresh fruit: almost ripe or perfectly ripe
        ripeness = round(random.uniform(70.0, 95.0), 1)
        days = round(random.uniform(0.5, 3.0), 1)
    elif "rotten" in name:
        # Rotten fruit: overripe
        ripeness = 100.0
        days = 0.0
    else:
        # Fallback
        ripeness = round(random.uniform(40.0, 60.0), 1)
        days = round(random.uniform(4.0, 7.0), 1)

    return ripeness, days


# -----------------------------------
# 3️⃣ Process the Images
# -----------------------------------
print("Scanning folders and copying images... This might take a minute.")

for folder in folders_to_scan:
    scan_path = os.path.join(source_base, folder)

    if not os.path.exists(scan_path):
        print(f"Warning: Could not find path {scan_path}")
        continue

    for class_folder in os.listdir(scan_path):
        class_path = os.path.join(scan_path, class_folder)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                # Replace spaces in folder names (like "unripe apple") with underscores
                safe_class_name = class_folder.replace(" ", "_")
                new_filename = f"{folder}_{safe_class_name}_{file}"

                src_file = os.path.join(class_path, file)
                dst_file = os.path.join(dest_dir, new_filename)

                # Copy the image
                shutil.copy2(src_file, dst_file)

                # Generate numbers
                ripeness, days = generate_labels(class_folder)

                # Add to CSV
                csv_data.append([new_filename, ripeness, days])

# -----------------------------------
# 4️⃣ Save the CSV
# -----------------------------------
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print("\n✅ Done!")
print(f"Successfully copied images to: {dest_dir}")
print(f"Successfully generated {csv_file} with {len(csv_data)-1} image labels!")
