# Fruit Ripeness Detection

A computer vision system that estimates **fruit ripeness percentage** and **days until peak ripeness** from a single image, using transfer learning with MobileNetV2.

## Overview

Most fruit freshness models output a simple classification label ("fresh" or "rotten"). This project treats ripeness as a **continuous regression problem**, predicting two values simultaneously:

- **Ripeness (%)** — how ripe the fruit currently is (0–100%)
- **Days to Ripe** — estimated days until the fruit reaches peak ripeness

This enables more granular decisions useful for food suppliers, grocery stores, and consumers.

## Model Architecture

```
Input Image (192 × 192 × 3)
          │
  MobileNetV2 backbone
  (ImageNet pretrained)
  Last 50 layers unfrozen
          │
  GlobalAveragePooling2D
          │
   Dense(128, ReLU)
      Dropout(0.5)
          │
   Dense(2, Linear)
          │
  [ripeness_pct, days_to_ripe]
```

**Loss**: Mean Squared Error (MSE)  
**Metric**: Mean Absolute Error (MAE)  
**Optimizer**: Adam (lr=1e-4)

## Dataset

Expected raw dataset structure:

```
dataset/
├── train/
│   ├── freshapples/
│   ├── freshbananas/
│   ├── freshoranges/
│   ├── rottenapples/
│   ├── rottenbananas/
│   └── rottenoranges/
└── test/
    └── (same structure)
```

Labels are derived heuristically from folder names:

| Folder type | Ripeness (%) | Days to Ripe |
|-------------|-------------|--------------|
| `fresh*`    | 70 – 95     | 0.5 – 3      |
| `unripe*`   | 10 – 45     | 5 – 10       |
| `rotten*`   | 100         | 0            |

> Dataset source: [Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) on Kaggle.

## Setup

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
git clone https://github.com/vineelneth/ai-based-fruit-ripeness-detection.git
cd ai-based-fruit-ripeness-detection
pip install -r requirements.txt
```

### Download the dataset

**Option A — Kaggle CLI:**
```bash
kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification
unzip fruits-fresh-and-rotten-for-classification.zip -d dataset/
```

**Option B — Manual:**  
Download from the Kaggle page above and extract into `dataset/train/` and `dataset/test/`.

## Usage

### Step 1 — Flatten dataset and generate labels

Copies all images into a single directory and produces `fruit_labels.csv`:

```bash
python generate_dataset.py
```

### Step 2 — Train the model

```bash
python train_model.py
```

Key options:

```
--image-dir   Path to flattened image folder  (default: dataset/all_images)
--csv-path    Path to labels CSV              (default: fruit_labels.csv)
--epochs      Maximum training epochs         (default: 10)
--batch-size  Batch size                      (default: 32)
--model-out   Where to save the trained model (default: fruit_freshness_regression.keras)
```

Training uses early stopping (patience=5 on val MAE) and saves the best checkpoint automatically.

### Step 3 — Run predictions

**GUI mode** — opens a file picker dialog, loops until you click Cancel:

```bash
python predict_fruit.py
```

**CLI mode** — pass an image path directly:

```bash
python predict_fruit.py --image path/to/fruit.jpg
```

Example output:

```
--- apple.jpg ---
Status:       Unripe / Ripening
Ripeness:     42.3%
Days to Ripe: 6.8 days
```

## Project Structure

```
fruit_ripeness_detection/
├── dataset/                    # raw images (not tracked — download separately)
├── generate_dataset.py         # flatten dataset structure → fruit_labels.csv
├── train_model.py              # train MobileNetV2 regression model
├── predict_fruit.py            # GUI + CLI inference
├── fruit_labels.csv            # generated image labels
├── requirements.txt
└── README.md
```

## Demo

The model overlays ripeness predictions directly on the fruit image:

![Prediction example 1](Screenshot%202026-03-21%20144756.png)
![Prediction example 2](Screenshot%202026-03-21%20144809.png)

## Tech Stack

- **TensorFlow / Keras** — model training and inference
- **MobileNetV2** — pretrained ImageNet backbone
- **scikit-learn** — train/test split
- **Pandas** — CSV data loading
- **Matplotlib** — training curves and result visualization
- **Tkinter** — GUI file picker (built-in Python)

## Roadmap

- [ ] Webcam integration for real-time detection
- [ ] Expand to more fruit types (bananas, mangoes, citrus)
- [ ] Export to TensorFlow Lite for mobile deployment
- [ ] REST API wrapper for integration into inventory systems
