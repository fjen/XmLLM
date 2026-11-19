"""Application settings — single source of truth for configuration.

Auto-detects whether the app is running locally or on a Hugging Face Space
via the SPACE_ID environment variable.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.app.limits import DEFAULT_MAX_UPLOAD_BYTES


class AppMode(StrEnum):
    LOCAL = "local"
    SPACE = "space"


def _detect_mode() -> AppMode:
    """Detect execution mode from environment."""
    if os.environ.get("SPACE_ID"):
        return AppMode.SPACE
    return AppMode(os.environ.get("APP_MODE", "local"))


class Settings(BaseSettings):
    """Central settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Execution mode -------------------------------------------------------
    app_mode: AppMode = Field(default_factory=_detect_mode)

    # -- Storage --------------------------------------------------------------
    storage_root: Path = Field(default=Path("./data"))
    db_name: str = "app.db"

    # -- Server ---------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 7860
    log_level: str = "info"

    # -- Upload limits --------------------------------------------------------
    max_upload_size: int = 104_857_600  # 100 MB

    # POST /ocr limits. Defaults sized for archival scanning: 800 DPI A3
    # decodes to ~124 MP, so the pixel cap sits above that with margin.
    max_ocr_image_size: int = DEFAULT_MAX_UPLOAD_BYTES  # 100 MB compressed
    max_ocr_image_pixels: int = 150_000_000  # 150 MP decoded
    allowed_mime_types: str = "image/png,image/jpeg,image/tiff,image/webp"

    # -- ALTO export ----------------------------------------------------------
    # Omit ALTO lines below this engine confidence (0-1). JobService copies
    # the value into the document policy. The export view applies it. The
    # canonical document keeps every line. Unset keeps everything (default).
    alto_min_line_confidence: float | None = None

    # -- Provider defaults ----------------------------------------------------
    provider_timeout: int = 120
    api_provider_timeout: int = 60
    api_provider_max_retries: int = 2

    # -- OCR engine -----------------------------------------------------------
    # When set (e.g. http://gpu-box:8080), run_paddle_ocr posts to the
    # remote OCR service (scripts/ocr_service.py) instead of loading
    # PaddleOCR locally. The service returns the same word_box_json
    # payload in the same coordinate space, so nothing downstream changes.
    ocr_service_url: str | None = None
    # Paddle device passed to PaddleOCR when set (e.g. "gpu:0"). Unset,
    # PaddleOCR auto-selects: GPU if the paddle build supports CUDA, else
    # CPU. Note enable_mkldnn is CPU-only (oneDNN) and irrelevant on GPU.
    ocr_device: str | None = None
    # Recognition batch size passed to PaddleOCR
    # (text_recognition_batch_size). The pipeline default is 6; raising it
    # (16–32) improves throughput on GPU at the cost of recognition
    # memory. Unset = PaddleOCR default.
    ocr_recognition_batch_size: int | None = Field(default=None, ge=1)
    # Textline orientation classifier (per-line, runs ~once per detected
    # line). Unset = True (current behaviour). Setting false skips ~200
    # inferences per dense page — a large speedup on GPU — at the cost of
    # misreading lines that are rotated relative to the page. Only turn
    # off for scans known to be upright.
    ocr_textline_orientation: bool | None = None

    # -- Geometry -------------------------------------------------------------
    bbox_containment_tolerance: int = 5

    # -- HuggingFace ----------------------------------------------------------
    hf_home: Path | None = None
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # -- Derived properties ---------------------------------------------------

    @property
    def is_space(self) -> bool:
        return self.app_mode == AppMode.SPACE

    @property
    def db_path(self) -> Path:
        return self.storage_root / self.db_name

    @property
    def jobs_dir(self) -> Path:
        return self.storage_root / "jobs"

    @property
    def providers_dir(self) -> Path:
        return self.storage_root / "providers"

    @property
    def exports_dir(self) -> Path:
        return self.storage_root / "exports"

    @property
    def cache_dir(self) -> Path:
        return self.storage_root / "cache"

    @property
    def allowed_mime_types_set(self) -> set[str]:
        return {t.strip() for t in self.allowed_mime_types.split(",")}

    def ensure_directories(self) -> None:
        """Create all required storage directories."""
        for d in (
            self.storage_root,
            self.jobs_dir,
            self.providers_dir,
            self.exports_dir,
            self.cache_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def model_post_init(self, __context: object) -> None:
        """Apply Space-specific defaults after init."""
        # Propagate HF token to environment so huggingface_hub picks it up
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)

        if self.is_space:
            if self.storage_root == Path("./data"):
                object.__setattr__(self, "storage_root", Path("/data"))
            if self.hf_home is None:
                object.__setattr__(self, "hf_home", Path("/data/.huggingface"))
                os.environ.setdefault("HF_HOME", str(self.hf_home))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Factory for dependency injection (FastAPI Depends). Cached singleton."""
    return Settings()
