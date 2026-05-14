import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.core.config import settings
from api.core.limiter import limiter
from api.core.logging_config import setup_logging
from api.routes.predict import router as predict_router
from api.schemas.predict import HealthResponse
from api.services.inference import inference_service

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading ML model (this may take ~10s)...")
    inference_service.load()
    logger.info("Startup complete — API ready")
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="Fruit Ripeness Detection API",
    description=(
        "Upload a fruit image and receive ripeness percentage, "
        "estimated days to peak ripeness, and a human-readable status label."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok" if inference_service.is_loaded else "degraded",
        model_loaded=inference_service.is_loaded,
    )
