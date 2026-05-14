from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),  # suppress conflict warning for model_path/model_config
    )

    model_path: str = "fruit_freshness_regression.keras"
    scaler_path: str = "target_scaler.save"
    log_level: str = "INFO"
    max_file_size_mb: int = 10

    # Leave empty to disable API key enforcement (public access, rate-limited only)
    api_key: str = ""

    # Comma-separated list of allowed CORS origins
    allowed_origins: str = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


# Validated at import time — bad env vars fail fast on startup
settings = Settings()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
