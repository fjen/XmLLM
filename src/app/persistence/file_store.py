"""File store — abstraction for persisting job artifacts on disk.

Storage layout:
    {STORAGE_ROOT}/
      jobs/
        {job_id}/
          input.jpg          — browser-renderable JPEG preview (UI jobs only)
          raw_payload.json.zst
          canonical.json.zst
          alto.xml.zst       — ALTO XML export (zstd; served decompressed)
          page.xml.zst       — PAGE XML export (zstd; served decompressed)
          viewer.json.zst
          events.json.zst

JSON artifacts are zstd-compressed and carry a `.json.zst` suffix. Loaders
also accept the legacy uncompressed `.json` names, so existing storage
roots keep working without migration.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import zstandard

_ZSTD_COMPRESSOR = zstandard.ZstdCompressor(level=3)

_logger = logging.getLogger(__name__)

# Formats a browser can draw in an <img> element. TIFF is accepted as
# input but cannot be rendered, so it never reaches the job folder as-is.
_RENDERABLE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def _write_zst(dest: Path, data: bytes) -> None:
    dest.write_bytes(_ZSTD_COMPRESSOR.compress(data))


def _read_zst(path: Path) -> bytes:
    """Decompress a zstd file; tolerates frames without content size."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f) as reader:
        return reader.read()


class FileStore:
    """Manages artifact persistence for jobs."""

    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root
        self._jobs_dir = storage_root / "jobs"
        self._providers_dir = storage_root / "providers"

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._providers_dir.mkdir(parents=True, exist_ok=True)

    # -- Job directory --------------------------------------------------------

    @staticmethod
    def _sanitize_id(value: str) -> str:
        """Reject path-traversal attempts in identifiers."""
        if not value or "/" in value or "\\" in value or ".." in value:
            raise ValueError(f"Invalid identifier (path traversal rejected): {value!r}")
        return value

    def job_dir(self, job_id: str) -> Path:
        d = self._jobs_dir / self._sanitize_id(job_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- Save artifacts -------------------------------------------------------

    def save_input_image(self, job_id: str, source_path: Path) -> Path:
        """Save a browser-renderable JPEG preview of the input image.

        The job directory keeps only the preview. Pipeline steps read the
        source file directly, and no export embeds the image, so the
        original is not duplicated into storage. A source that is already
        JPEG is copied verbatim. On conversion failure the original is
        copied as a fallback so the viewer route can still try to serve it.
        """
        job_dir = self.job_dir(job_id)
        dest = job_dir / "input.jpg"
        if source_path.suffix.lower() in {".jpg", ".jpeg"}:
            # copyfile, not copy2: the source may come from mkstemp (0600)
            # and the job folder should carry default permissions.
            shutil.copyfile(source_path, dest)
            return dest
        try:
            from PIL import Image

            with Image.open(source_path) as img:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.save(dest, format="JPEG", quality=85)
        except Exception:
            _logger.warning(
                "JPEG preview failed for %s (%s); copying the original",
                source_path.name, job_id, exc_info=True,
            )
            dest.unlink(missing_ok=True)
            fallback = job_dir / f"input{source_path.suffix}"
            shutil.copyfile(source_path, fallback)
            return fallback
        return dest

    def save_json(self, job_id: str, filename: str, data: Any) -> Path:
        """Save a JSON-serializable object, zstd-compressed (`.json.zst`)."""
        dest = self.job_dir(job_id) / self._sanitize_id(f"{filename}.zst")
        _write_zst(
            dest, json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        )
        return dest

    def save_raw_payload(self, job_id: str, data: dict) -> Path:
        return self.save_json(job_id, "raw_payload.json", data)

    def save_canonical(self, job_id: str, data: dict) -> Path:
        return self.save_json(job_id, "canonical.json", data)

    def save_alto(self, job_id: str, xml_bytes: bytes) -> Path:
        dest = self.job_dir(job_id) / "alto.xml.zst"
        _write_zst(dest, xml_bytes)
        return dest

    def save_page_xml(self, job_id: str, xml_bytes: bytes) -> Path:
        dest = self.job_dir(job_id) / "page.xml.zst"
        _write_zst(dest, xml_bytes)
        return dest

    def save_viewer(self, job_id: str, data: dict) -> Path:
        return self.save_json(job_id, "viewer.json", data)

    def save_events(self, job_id: str, events: list[dict]) -> Path:
        return self.save_json(job_id, "events.json", events)

    # -- Load artifacts -------------------------------------------------------

    def _resolve_artifact(self, job_id: str, filename: str) -> Path | None:
        """Single source of the dual-name rule: `<name>.zst` first, then
        the legacy uncompressed `<name>`. Used by every reader/checker."""
        d = self._jobs_dir / self._sanitize_id(job_id)
        zst = d / self._sanitize_id(f"{filename}.zst")
        if zst.exists():
            return zst
        legacy = d / self._sanitize_id(filename)
        if legacy.exists():
            return legacy
        return None

    def load_json(self, job_id: str, filename: str) -> Any:
        """Load a JSON artifact (.json.zst, or legacy .json). None if absent."""
        path = self._resolve_artifact(job_id, filename)
        if path is None:
            return None
        if path.name.endswith(".zst"):
            return json.loads(_read_zst(path))
        return json.loads(path.read_text(encoding="utf-8"))

    def load_xml(self, job_id: str, filename: str) -> bytes | None:
        """Load an XML artifact (.xml.zst, or legacy uncompressed .xml)."""
        path = self._resolve_artifact(job_id, filename)
        if path is None:
            return None
        if path.name.endswith(".zst"):
            return _read_zst(path)
        return path.read_bytes()

    def load_raw_payload(self, job_id: str) -> dict | None:
        return self.load_json(job_id, "raw_payload.json")

    def load_canonical(self, job_id: str) -> dict | None:
        return self.load_json(job_id, "canonical.json")

    def load_alto(self, job_id: str) -> bytes | None:
        return self.load_xml(job_id, "alto.xml")

    def load_page_xml(self, job_id: str) -> bytes | None:
        return self.load_xml(job_id, "page.xml")

    def load_viewer(self, job_id: str) -> dict | None:
        return self.load_json(job_id, "viewer.json")

    def load_events(self, job_id: str) -> list[dict] | None:
        return self.load_json(job_id, "events.json")

    # -- Input image ----------------------------------------------------------

    def get_input_image_path(self, job_id: str) -> Path | None:
        """Find the input image for a job, preferring renderable formats.

        Jobs written before the JPEG-preview change may still carry the
        original upload (`input.png`, `input.tif`, ...). Renderable files
        win; a non-renderable leftover is a last resort.
        """
        from src.app.ocr import SUPPORTED_IMAGE_EXTENSIONS

        d = self._jobs_dir / self._sanitize_id(job_id)
        if not d.exists():
            return None
        fallback: Path | None = None
        for f in sorted(d.iterdir()):
            if f.stem != "input" or f.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            if f.suffix.lower() in _RENDERABLE_SUFFIXES:
                return f
            if fallback is None:
                fallback = f
        return fallback

    # -- Provider profiles ----------------------------------------------------

    def save_provider(self, provider_id: str, data: dict) -> Path:
        dest = self._providers_dir / f"{self._sanitize_id(provider_id)}.json.zst"
        _write_zst(
            dest, json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        )
        return dest

    def load_provider(self, provider_id: str) -> dict | None:
        zst = self._providers_dir / f"{self._sanitize_id(provider_id)}.json.zst"
        if zst.exists():
            return json.loads(_read_zst(zst))
        legacy = self._providers_dir / f"{self._sanitize_id(provider_id)}.json"
        if legacy.exists():
            return json.loads(legacy.read_text(encoding="utf-8"))
        return None

    def list_providers(self) -> list[str]:
        if not self._providers_dir.exists():
            return []
        names: set[str] = set()
        for f in self._providers_dir.glob("*.json.zst"):
            names.add(f.name.removesuffix(".json.zst"))
        for f in self._providers_dir.glob("*.json"):  # legacy
            names.add(f.stem)
        return sorted(names)

    def delete_provider(self, provider_id: str) -> bool:
        removed = False
        for suffix in (".json.zst", ".json"):  # legacy fallback
            path = self._providers_dir / f"{self._sanitize_id(provider_id)}{suffix}"
            if path.exists():
                path.unlink()
                removed = True
        return removed

    # -- Listing --------------------------------------------------------------

    def list_jobs(self) -> list[str]:
        if not self._jobs_dir.exists():
            return []
        return sorted(
            [d.name for d in self._jobs_dir.iterdir() if d.is_dir()],
            reverse=True,
        )

    def job_exists(self, job_id: str) -> bool:
        return (self._jobs_dir / self._sanitize_id(job_id)).exists()

    def job_has_artifact(self, job_id: str, filename: str) -> bool:
        return self._resolve_artifact(job_id, filename) is not None
