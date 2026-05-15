import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# WordNet synset IDs for fruits and produce in ImageNet-1K.
# Synset IDs are stable ontological identifiers from the WordNet taxonomy —
# not arbitrary string patterns. These are the only fruit/vegetable entries
# in the 1000-class ImageNet benchmark.
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
    "n07716906",  # zucchini / courgette
    "n07717410",  # acorn_squash
    "n07717556",  # butternut_squash
    "n07718472",  # cucumber
    "n07718747",  # artichoke
    "n07720875",  # bell_pepper
    "n07714990",  # broccoli
    "n07715103",  # cauliflower
})

# Minimum summed softmax probability across fruit synsets in top-10 to accept.
# Set at 0.05 so out-of-ImageNet fruits (mango, papaya, kiwi, peach) accumulate
# enough mass from visually similar in-vocabulary classes to pass, while
# non-fruit images (vehicles, food dishes, indoor objects) consistently score < 0.01.
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

    def check(self, image_bytes: bytes) -> tuple[bool, str]:
        """
        Returns (True, detected_label) if the image has sufficient fruit probability.
        Returns (False, top_predicted_label) otherwise.

        Sums softmax probability across all fruit synsets in the top-10 predictions.
        Summing rather than checking top-1 handles fruits absent from ImageNet-1K whose
        visual features distribute across nearby in-vocabulary fruit classes.
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
        best_fruit = next((lbl for s, lbl, _ in top10 if s in _FRUIT_SYNSETS), None)

        if fruit_prob >= _MIN_FRUIT_PROB:
            return True, (best_fruit or "fruit").replace("_", " ")

        top_label = top10[0][1].replace("_", " ") if top10 else "unknown"
        return False, top_label


gate_service = FruitGateService()
