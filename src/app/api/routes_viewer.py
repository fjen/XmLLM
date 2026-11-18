"""Viewer projection route."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.app.api import get_file_store
from src.app.ocr import SUPPORTED_IMAGE_EXTENSIONS

router = APIRouter(prefix="/jobs", tags=["viewer"])

# Media types for the persisted input image, keyed by file suffix. Keys must
# exactly match the shared ingest allowlist — the import-time assertion makes
# drift (new extension in ocr, stale map here) fail at startup, not at runtime.
_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
if set(_IMAGE_MEDIA_TYPES) != set(SUPPORTED_IMAGE_EXTENSIONS):
    raise RuntimeError(
        "_IMAGE_MEDIA_TYPES is out of sync with SUPPORTED_IMAGE_EXTENSIONS"
    )


@router.get("/{job_id}/viewer")
async def get_viewer_projection(job_id: str) -> dict[str, Any]:
    """Get the viewer projection JSON for a job.

    Returns a lightweight structure for the front-end viewer.
    Falls back to canonical document if viewer projection not yet built.
    """
    store = get_file_store()

    try:
        # Try viewer projection first
        viewer = store.load_viewer(job_id)
    except ValueError:
        # Malformed job id rejected by FileStore sanitization (path-safety).
        raise HTTPException(
            status_code=404,
            detail=f"No viewer data available for job '{job_id}'",
        ) from None
    if viewer is not None:
        return viewer

    # Fall back: return a minimal projection from canonical
    canonical = store.load_canonical(job_id)
    if canonical is None:
        raise HTTPException(
            status_code=404,
            detail=f"No viewer data available for job '{job_id}'",
        )

    # Build a minimal viewer-like response from canonical
    pages = canonical.get("pages", [])
    if not pages:
        raise HTTPException(status_code=404, detail="No pages in canonical document")

    page = pages[0]
    return {
        "image_ref": canonical.get("source", {}).get("filename", ""),
        "image_width": page.get("width", 0),
        "image_height": page.get("height", 0),
        "block_overlays": [],
        "line_overlays": [],
        "word_overlays": [],
        "non_text_overlays": [],
        "inspection_index": {},
        "validation_flags": [],
        "export_status": {
            "alto_export": "none",
            "page_export": "none",
            "viewer_render": "none",
        },
    }


@router.get("/{job_id}/image")
async def get_job_image(job_id: str) -> FileResponse:
    """Serve the persisted input image for a job.

    The viewer fetches this alongside the projection JSON to draw the
    overlay boxes on top of the original scan. Returns 404 when the job
    has no persisted input image (e.g. JSON-payload jobs, unknown ids).
    """
    try:
        path = get_file_store().get_input_image_path(job_id)
    except ValueError:
        # Malformed job id rejected by FileStore sanitization (path-safety).
        raise HTTPException(
            status_code=404,
            detail=f"No input image for job '{job_id}'",
        ) from None
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No input image for job '{job_id}'",
        )
    media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type)
