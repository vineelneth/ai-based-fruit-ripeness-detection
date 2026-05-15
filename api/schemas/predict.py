from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    filename: str
    fruit_name: str = Field(..., description="Detected fruit name from the classifier")
    fruit_confidence: float = Field(..., ge=0, le=100, description="Classifier confidence (0–100)")
    ripeness_pct: float = Field(..., ge=0, le=100, description="Ripeness percentage (0–100)")
    days_to_ripe: float = Field(..., ge=0, description="Estimated days until peak ripeness")
    status: str = Field(..., description="Unripe | Ripening | Ripe | Fully Ripe / Overripe")


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded")
    model_loaded: bool
