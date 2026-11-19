"""Image upload route — accepts an image, runs OCR, produces XML."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile

from src.app.api import get_job_service
from src.app.domain.models import RawProviderPayload
from src.app.ocr import (
    OcrPayloadTooLarge,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PADDLE_LANGS,
)
from src.app.settings import get_settings

router = APIRouter(tags=["ocr"])

logger = logging.getLogger(__name__)


@router.post("/ocr", status_code=201)
async def ocr_image(image: UploadFile, lang: str = "de") -> dict[str, Any]:
    """Upload an image, run PaddleOCR, and produce ALTO/PAGE XML.

    This is the main user-facing endpoint — no JSON payload needed.
    `lang` (default: "de") is the PaddleOCR language code (e.g. en, fr, de,
    ch); it also stamps the ISO 639-3 language onto the canonical document
    and the ALTO LANG attributes.
    """
    lang = lang.strip().lower()
    if lang not in SUPPORTED_PADDLE_LANGS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported lang '{lang}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_PADDLE_LANGS))}",
        )

    if not image.filename:
        raise HTTPException(status_code=422, detail="No filename provided")

    ext = Path(image.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format '{ext}'. Use: "
            f"{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}",
        )

    settings = get_settings()
    content = await image.read(settings.max_ocr_image_size + 1)
    if len(content) > settings.max_ocr_image_size:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {settings.max_ocr_image_size // (1024 * 1024)} MB)",
        )

    fd, tmp_name = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        tmp_path.write_bytes(content)

        from PIL import Image as PILImage
        try:
            with PILImage.open(tmp_path) as img:
                image_width, image_height = img.size
        except PILImage.DecompressionBombError as exc:
            # Pillow's own guard (~179 MP default) — treat as oversize
            raise HTTPException(
                status_code=413,
                detail="Image dimensions too large",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Unreadable image: {type(exc).__name__}",
            ) from exc
        if image_width * image_height > settings.max_ocr_image_pixels:
            raise HTTPException(
                status_code=413,
                detail=f"Image dimensions too large "
                f"(max {settings.max_ocr_image_pixels // 1_000_000} megapixels)",
            )

        from src.app.ocr import (
            paddle_lang_to_iso639_3,
            run_paddle_ocr,
        )
        ocr_started = time.monotonic()
        try:
            results = run_paddle_ocr(tmp_path, lang=lang)
        except (ValueError, AssertionError, SystemExit) as exc:
            # paddleocr signals an unusable lang/config: 3.x raises ValueError
            # ("No models are available for lang=..."); 2.x used a bare assert
            # in parse_lang (AssertionError, or SystemExit via sys.exit in
            # get_model_config). Log server-side, return no internals. The
            # allowlist gate above is the primary defense.
            logger.exception("PaddleOCR rejected lang=%s", lang)
            raise HTTPException(
                status_code=422,
                detail=f"PaddleOCR failed for lang '{lang}' "
                f"({type(exc).__name__})",
            ) from exc
        except OcrPayloadTooLarge as exc:
            # The request passed our gate but the service's limit is
            # lower. 413 with the service's words, not an opaque 500.
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception:
            # Engine/infra failure (network, disk, OOM) — server error,
            # full detail in logs only.
            logger.exception("PaddleOCR failed for lang=%s", lang)
            raise

        if not results:
            raise HTTPException(
                status_code=422,
                detail="PaddleOCR found no text in this image",
            )

        iso_lang = paddle_lang_to_iso639_3(lang)
        raw = RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=results,
            image_width=image_width,
            image_height=image_height,
            metadata={"language": iso_lang} if iso_lang else None,
        )

        svc = get_job_service()
        job = svc.create_job(
            provider_id="paddleocr",
            provider_family="word_box_json",
            source_filename=image.filename,
        )

        result = svc.run_job(
            job, raw,
            image_width=image_width,
            image_height=image_height,
            image_path=tmp_path,
        )

        # duration_ms covers only the canonical pipeline (run_job span); the
        # engine call happens before it and dominates real runs — report both.
        summary = result.to_summary()
        summary["ocr_ms"] = round((time.monotonic() - ocr_started) * 1000)
        return summary

    finally:
        tmp_path.unlink(missing_ok=True)
