#!/usr/bin/env python3
"""batch_alto.py — batch ALTO XML generation for directories of scanned images.

Runs the full XmLLM pipeline in-process, no server:
    image (TIFF/JPEG/PNG/WebP/BMP) → PaddleOCR (in-process) →
    RawProviderPayload → canonical → enrichers → validation → ALTO XML

Each image becomes one job persisted under STORAGE_ROOT/jobs/<job_id>/
(alto.xml, canonical.json, events.json, ...) — the same artifacts the HTTP
API would produce, so the audit trail stays intact.

Usage:
    python scripts/batch_alto.py /path/to/book1 /path/to/book2
    python scripts/batch_alto.py /path/to/book --lang fr --out /path/to/alto
    STORAGE_ROOT=/data python scripts/batch_alto.py /path/to/book

Exit codes: 0 if every image produced ALTO; 1 if any file failed;
2 on usage errors (unsupported lang, no images, missing directory).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app.jobs.service import JobService
    from src.app.persistence.file_store import FileStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ALTO XML for every image (TIFF/JPEG/PNG/WebP/BMP)"
        " in the given directories.",
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Directories containing scanned images.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="PaddleOCR language code (default: en). E.g. en, fr, de, es.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional directory to copy each alto.xml into, named <image-stem>.xml.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=None,
        help="XmLLM storage root (default: STORAGE_ROOT env / .env, else ./data).",
    )
    return parser.parse_args(argv)


def collect_images(directories: list[Path]) -> list[Path]:
    from src.app.ocr import SUPPORTED_IMAGE_EXTENSIONS

    images: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            print(f"error: not a directory: {directory}", file=sys.stderr)
            sys.exit(2)
        images.extend(
            p
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
    return images


def process_image(
    tif: Path, svc: JobService, store: FileStore, lang: str
) -> tuple[str, Path | None]:
    """Run one image through the pipeline. Returns (status, alto_path)."""
    from PIL import Image as PILImage
    from src.app.domain.models import RawProviderPayload
    from src.app.ocr import (
        paddle_lang_to_iso639_3,
        paddle_result_to_payload,
        run_paddle_ocr,
    )

    with PILImage.open(tif) as img:
        image_width, image_height = img.size

    results = run_paddle_ocr(tif, lang=lang)
    if not results:
        return "no_text", None

    iso_lang = paddle_lang_to_iso639_3(lang)
    raw = RawProviderPayload(
        provider_id="paddleocr",
        adapter_id="adapter.word_box_json.v1",
        runtime_type="local",
        payload=paddle_result_to_payload(results),
        image_width=image_width,
        image_height=image_height,
        metadata={"language": iso_lang} if iso_lang else None,
    )

    job = svc.create_job(
        provider_id="paddleocr",
        provider_family="word_box_json",
        source_filename=tif.name,
    )
    result = svc.run_job(
        job, raw,
        image_width=image_width,
        image_height=image_height,
        image_path=tif,
    )

    if not result.has_alto:
        return str(result.status), None
    return str(result.status), store.job_dir(result.job_id) / "alto.xml"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    args.lang = args.lang.strip().lower()
    from src.app.ocr import SUPPORTED_PADDLE_LANGS
    if args.lang not in SUPPORTED_PADDLE_LANGS:
        print(
            f"error: unsupported lang '{args.lang}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_PADDLE_LANGS))}",
            file=sys.stderr,
        )
        return 2

    from src.app.persistence.db import Database
    from src.app.persistence.file_store import FileStore
    from src.app.settings import Settings

    if args.storage_root is not None:
        settings = Settings(storage_root=args.storage_root)
    else:
        settings = Settings()

    db = Database(settings.db_path)
    db.connect()
    try:
        store = FileStore(settings.storage_root)
        store.ensure_dirs()

        from src.app.jobs.service import JobService

        svc = JobService(db, store)

        images = collect_images(args.directories)
        if not images:
            print("error: no image files found in the given directories", file=sys.stderr)
            return 2
        print(f"{len(images)} image(s), lang={args.lang}, storage={settings.storage_root}")

        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            # Same stem from different formats/pages (e.g. page.tif +
            # page.jpg) must not abort the batch: disambiguate copy names.
            taken: set[str] = set()
            dest_names: dict[Path, str] = {}
            for img in images:
                name = f"{img.stem}.xml"
                if name in taken:
                    ext = img.suffix.lstrip(".").lower()
                    name = f"{img.stem}_{ext}.xml"
                    n = 1
                    while name in taken:
                        name = f"{img.stem}_{ext}_{n}.xml"
                        n += 1
                taken.add(name)
                dest_names[img] = name

        failures: list[tuple[str, str]] = []
        for tif in images:
            try:
                status, alto_path = process_image(tif, svc, store, args.lang)
            except Exception as exc:  # keep batch going on per-file errors
                failures.append((tif.name, f"{type(exc).__name__}: {exc}"))
                print(f"FAILED  {tif.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            if alto_path is None:
                failures.append((tif.name, f"no ALTO produced (status {status})"))
                print(f"FAILED  {tif.name}: no ALTO produced (status {status})")
                continue

            print(f"OK      {tif.name} -> {alto_path} (status {status})")
            if args.out is not None:
                dest = args.out / dest_names[tif]
                shutil.copy2(alto_path, dest)
                print(f"        copied to {dest}")
    finally:
        db.close()

    print(f"\ndone: {len(images) - len(failures)}/{len(images)} succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
