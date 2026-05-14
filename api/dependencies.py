import logging

from fastapi import Header, HTTPException

from api.core.config import settings

logger = logging.getLogger(__name__)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Enforce API key when API_KEY env var is set. No-op if API_KEY is empty."""
    if settings.api_key and x_api_key != settings.api_key:
        logger.warning("Rejected request with invalid API key")
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
