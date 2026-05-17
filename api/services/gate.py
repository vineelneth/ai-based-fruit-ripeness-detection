import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_FRUIT_SYNSETS = frozenset({
    "n07742313",  # Granny_Smith (apple)
    "n07745940",  # strawberry
    "n07747607",  # orange
    "n07749582",  # lemon
    "n07753113",  # fig
    "n07753275",  # pineapple
    "n07753592",  # banana
    "n07754684",  # jackfruit
    "n07760859",  # custard_apple
    "n07768694",  # pomegranate
    "n07716906",  # zucchini
    "n07717410",  # acorn_squash
    "n07717556",  # butternut_squash
    "n07718472",  # cucumber
    "n07718747",  # artichoke
    "n07720875",  # bell_pepper
    "n07714990",  # broccoli
    "n07715103",  # cauliflower
})

# User-friendly display names for each synset (overrides raw ImageNet label).
_SYNSET_LABELS = {
    "n07742313": "Apple",
    "n07745940": "Strawberry",
    "n07747607": "Orange",
    "n07749582": "Lemon",
    "n07753113": "Fig",
    "n07753275": "Pineapple",
    "n07753592": "Banana",
    "n07754684": "Jackfruit",
    "n07760859": "Custard Apple",
    "n07768694": "Pomegranate",
    "n07716906": "Zucchini",
    "n07717410": "Acorn Squash",
    "n07717556": "Butternut Squash",
    "n07718472": "Cucumber",
    "n07718747": "Artichoke",
    "n07720875": "Bell Pepper",
    "n07714990": "Broccoli",
    "n07715103": "Cauliflower",
}

_MIN_FRUIT_PROB = 0.05


class FruitGateService:
    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        """Load MobileNetV2 with imagenet classification head. Called once at startup."""
        from tensorflow.keras.applications import MobileNetV2

        logger.info("Loading ImageNet gate classifier (~14 MB weights on first run)...")
        self._model = MobileNetV2(weights="imagenet", include_top=True)
        logger.info("Gate classifier ready")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def check(self, image_bytes: bytes) -> tuple[bool, str, float]:
        """
        Returns (True, detected_label, confidence_pct) if the image has sufficient fruit probability.
        Returns (False, top_predicted_label, 0.0) otherwise.

        confidence_pct is the summed softmax probability across all fruit synsets (0–100).
        """
        if not self.is_loaded:
            raise RuntimeError("Gate model not loaded. Application may still be starting.")

        from tensorflow.keras.applications.mobilenet_v2 import (
            decode_predictions,
            preprocess_input,
        )

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        arr = preprocess_input(np.expand_dims(np.array(img, dtype=np.float32), 0))

        top10 = decode_predictions(self._model.predict(arr, verbose=0), top=10)[0]

        fruit_prob = sum(p for s, _, p in top10 if s in _FRUIT_SYNSETS)
        best_synset = next((s for s, _, _ in top10 if s in _FRUIT_SYNSETS), None)

        if fruit_prob >= _MIN_FRUIT_PROB:
            label = _SYNSET_LABELS.get(
                best_synset,
                (top10[0][1] if top10 else "fruit").replace("_", " ").title(),
            )
            return True, label, round(fruit_prob * 100, 1)

        top_label = top10[0][1].replace("_", " ").title() if top10 else "unknown"
        return False, top_label, 0.0


gate_service = FruitGateService()
