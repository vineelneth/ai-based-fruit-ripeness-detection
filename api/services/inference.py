import io
import logging

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from api.core.config import settings

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self) -> None:
        self._model = None
        self._scaler = None

    def load(self) -> None:
        """Load model and scaler from disk. Called once at application startup."""
        import tensorflow as tf  # deferred import keeps startup traceback clean

        logger.info("Loading model from '%s'", settings.model_path)
        self._model = tf.keras.models.load_model(settings.model_path)

        logger.info("Loading scaler from '%s'", settings.scaler_path)
        self._scaler = joblib.load(settings.scaler_path)

        logger.info("Model and scaler ready")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._scaler is not None

    def predict(self, image_bytes: bytes) -> dict:
        """
        Run ripeness inference on raw image bytes.

        Raises:
            RuntimeError: model not loaded yet.
            ValueError:   image bytes cannot be decoded as a valid image.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Application may still be starting.")

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError("Cannot decode image. Ensure the upload is a valid JPEG or PNG.")

        img = img.resize((192, 192))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1, 192, 192, 3)

        raw_preds = self._model.predict(arr, verbose=0)  # (1, 2) normalized
        denormalized = self._scaler.inverse_transform(raw_preds.reshape(1, -1))[0]

        ripeness_pct = float(np.clip(denormalized[0], 0, 100))
        days_to_ripe = float(max(0.0, denormalized[1]))

        if ripeness_pct >= 95:
            status = "Fully Ripe / Overripe"
            days_to_ripe = 0.0  # already past peak — model outputs are independent, enforce consistency
        elif ripeness_pct >= 70:
            status = "Ripe"
        elif ripeness_pct >= 40:
            status = "Ripening"
        else:
            status = "Unripe"

        return {
            "ripeness_pct": round(ripeness_pct, 2),
            "days_to_ripe": round(days_to_ripe, 2),
            "status": status,
        }


# Module-level singleton — shared across all requests in the same worker
inference_service = InferenceService()
