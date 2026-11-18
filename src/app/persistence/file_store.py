"""File store — abstraction for persisting job artifacts on disk.

Storage layout:
    {STORAGE_ROOT}/
      jobs/
        {job_id}/
          input.{ext}        — original uploaded image (UI jobs only)
          raw_payload.json.zst
          canonical.json.zst
          alto.xml           — ALTO XML export (uncompressed)
          page.xml           — PAGE XML export (uncompressed)
          viewer.json.zst
          events.json.zst

JSON artifacts are zstd-compressed and carry a `.json.zst` suffix. Loaders
also accept the legacy uncompressed `.json` names, so existing storage
roots keep working without migration.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import zstandard

_ZSTD_COMPRESSOR = zstandard.ZstdCompressor(level=3)


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
        """Copy the input image into the job directory."""
        dest = self.job_dir(job_id) / f"input{source_path.suffix}"
        shutil.copy2(source_path, dest)
        return dest

    def save_json(self, job_id: str, filename: str, data: Any) -> Path:
        """Save a JSON-serializable object, zstd-compressed (`.json.zst`)."""
        dest = self.job_dir(job_id) / self._sanitize_id(f"{filename}.zst")
        _write_zst(
            dest, json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        )
        return dest

    def save_bytes(self, job_id: str, filename: str, data: bytes) -> Path:
        """Save raw bytes (e.g. XML)."""
        dest = self.job_dir(job_id) / self._sanitize_id(filename)
        dest.write_bytes(data)
        return dest

    def save_raw_payload(self, job_id: str, data: dict) -> Path:
        return self.save_json(job_id, "raw_payload.json", data)

    def save_canonical(self, job_id: str, data: dict) -> Path:
        return self.save_json(job_id, "canonical.json", data)

    def save_alto(self, job_id: str, xml_bytes: bytes) -> Path:
        return self.save_bytes(job_id, "alto.xml", xml_bytes)

    def save_page_xml(self, job_id: str, xml_bytes: bytes) -> Path:
        return self.save_bytes(job_id, "page.xml", xml_bytes)

    def save_viewer(self, job_id: str, data: dict) -> Path:
        return self.save_json(job_id, "viewer.json", data)

    def save_events(self, job_id: str, events: list[dict]) -> Path:
        return self.save_json(job_id, "events.json", events)

    # -- Load artifacts -------------------------------------------------------

    def load_json(self, job_id: str, filename: str) -> Any:
        """Load a JSON artifact. Reads `.json.zst`, falls back to the
        legacy uncompressed `.json`. Returns None if neither exists."""
        d = self._jobs_dir / self._sanitize_id(job_id)
        zst = d / self._sanitize_id(f"{filename}.zst")
        if zst.exists():
            return json.loads(_read_zst(zst))
        legacy = d / self._sanitize_id(filename)
        if legacy.exists():
            return json.loads(legacy.read_text(encoding="utf-8"))
        return None

    def load_bytes(self, job_id: str, filename: str) -> bytes | None:
        """Load raw bytes. Returns None if not found."""
        path = self._jobs_dir / self._sanitize_id(job_id) / self._sanitize_id(filename)
        if not path.exists():
            return None
        return path.read_bytes()

    def load_raw_payload(self, job_id: str) -> dict | None:
        return self.load_json(job_id, "raw_payload.json")

    def load_canonical(self, job_id: str) -> dict | None:
        return self.load_json(job_id, "canonical.json")

    def load_alto(self, job_id: str) -> bytes | None:
        return self.load_bytes(job_id, "alto.xml")

    def load_page_xml(self, job_id: str) -> bytes | None:
        return self.load_bytes(job_id, "page.xml")

    def load_viewer(self, job_id: str) -> dict | None:
        return self.load_json(job_id, "viewer.json")

    def load_events(self, job_id: str) -> list[dict] | None:
        return self.load_json(job_id, "events.json")

    # -- Input image ----------------------------------------------------------

    def get_input_image_path(self, job_id: str) -> Path | None:
        """Find the input image for a job (any extension)."""
        from src.app.ocr import SUPPORTED_IMAGE_EXTENSIONS

        d = self._jobs_dir / self._sanitize_id(job_id)
        if not d.exists():
            return None
        for f in d.iterdir():
            if f.stem == "input" and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                return f
        return None

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
        d = self._jobs_dir / self._sanitize_id(job_id)
        return (
            (d / self._sanitize_id(f"{filename}.zst")).exists()
            or (d / self._sanitize_id(filename)).exists()  # legacy
        )
