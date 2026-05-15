import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from api.core.config import ALLOWED_CONTENT_TYPES, settings
from api.core.limiter import limiter
from api.dependencies import require_api_key
from api.schemas.predict import PredictResponse
from api.services.gate import gate_service
from api.services.inference import inference_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Single-worker executor: TF inference is not safe to parallelize on CPU
_executor = ThreadPoolExecutor(max_workers=1)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict fruit ripeness from an image",
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"description": "Invalid or missing API key"},
        413: {"description": "File exceeds size limit"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Image is not a recognized fruit, or cannot be decoded"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Model not ready"},
    },
)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Fruit image (JPEG or PNG)"),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{file.content_type}'. Accepted: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes) / 1_048_576:.1f} MB). Limit: {settings.max_file_size_mb} MB",
        )

    loop = asyncio.get_running_loop()

    # Gate: verify the image contains a fruit before running regression
    try:
        is_fruit, label = await loop.run_in_executor(
            _executor, gate_service.check, image_bytes
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Gate check failed for '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Image validation failed. Please try again.")

    if not is_fruit:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "UNSUPPORTED_IMAGE",
                "message": (
                    f"This looks like '{label}', not a fruit. "
                    "Please upload a clear photo of a fruit."
                ),
            },
        )

    try:
        result = await loop.run_in_executor(_executor, inference_service.predict, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Unhandled inference error for file '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Inference failed. Please try again.")

    return PredictResponse(filename=file.filename or "unknown", **result)
