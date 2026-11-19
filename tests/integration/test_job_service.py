"""Integration test: full job lifecycle via JobService."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from src.app.domain.models import RawProviderPayload
from src.app.jobs.models import JobStatus
from src.app.jobs.service import JobService
from src.app.persistence.db import Database
from src.app.persistence.file_store import FileStore
from src.app.settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path


class TestJobService:
    def _setup(self, tmp_storage: Path) -> tuple[JobService, Database, FileStore]:
        db = Database(tmp_storage / "test.db")
        db.connect()
        store = FileStore(tmp_storage)
        store.ensure_dirs()
        return JobService(db, store), db, store

    def test_full_pipeline_succeeds(self, tmp_storage: Path, fixtures_dir: Path) -> None:
        svc, db, store = self._setup(tmp_storage)

        with open(fixtures_dir / "paddle_ocr_sample.json") as f:
            payload = json.load(f)

        raw = RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=payload,
            image_width=2480, image_height=3508,
        )

        # Create job
        job = svc.create_job(
            provider_id="paddleocr",
            provider_family="word_box_json",
            source_filename="test.png",
        )
        assert job.status == JobStatus.QUEUED

        # Run pipeline
        result = svc.run_job(job, raw, image_width=2480, image_height=3508)

        assert result.status == JobStatus.SUCCEEDED
        assert result.has_raw_payload
        assert result.has_canonical
        assert result.has_alto
        assert result.has_page_xml
        assert result.error is None
        assert result.duration_ms is not None
        assert result.duration_ms > 0

        # Verify artifacts on disk
        assert store.load_raw_payload(result.job_id) is not None
        assert store.load_canonical(result.job_id) is not None
        assert store.load_alto(result.job_id) is not None
        assert store.load_page_xml(result.job_id) is not None
        assert store.load_events(result.job_id) is not None

        # Verify in database
        db_job = svc.get_job(result.job_id)
        assert db_job is not None
        assert db_job.status == JobStatus.SUCCEEDED

        db.close()

    def test_job_appears_in_list(self, tmp_storage: Path, fixtures_dir: Path) -> None:
        svc, db, store = self._setup(tmp_storage)

        with open(fixtures_dir / "paddle_ocr_sample.json") as f:
            payload = json.load(f)

        raw = RawProviderPayload(
            provider_id="paddleocr", adapter_id="v1", runtime_type="local",
            payload=payload, image_width=2480, image_height=3508,
        )

        job = svc.create_job("paddleocr", "word_box_json")
        svc.run_job(job, raw, image_width=2480, image_height=3508)

        jobs = svc.list_jobs()
        assert len(jobs) >= 1
        assert any(j.job_id == job.job_id for j in jobs)
        db.close()

    def test_failed_job_captured(self, tmp_storage: Path) -> None:
        svc, db, store = self._setup(tmp_storage)

        raw = RawProviderPayload(
            provider_id="bad", adapter_id="v1", runtime_type="local",
            payload={"not": "a list"},  # will cause adapter to fail
        )

        job = svc.create_job("bad", "word_box_json")
        result = svc.run_job(job, raw, image_width=100, image_height=100)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert "list payload" in result.error

        # Should still be in DB
        db_job = svc.get_job(result.job_id)
        assert db_job is not None
        assert db_job.status == JobStatus.FAILED

        # Events should be saved even on failure
        events = store.load_events(result.job_id)
        assert events is not None
        db.close()

    def test_events_log_all_steps(self, tmp_storage: Path, fixtures_dir: Path) -> None:
        svc, db, store = self._setup(tmp_storage)

        with open(fixtures_dir / "paddle_ocr_sample.json") as f:
            payload = json.load(f)

        raw = RawProviderPayload(
            provider_id="paddleocr", adapter_id="v1", runtime_type="local",
            payload=payload, image_width=2480, image_height=3508,
        )

        job = svc.create_job("paddleocr", "word_box_json")
        svc.run_job(job, raw, image_width=2480, image_height=3508)

        events = store.load_events(job.job_id)
        assert events is not None
        steps = [e["step"] for e in events]
        assert "receive_file" in steps
        assert "save_raw" in steps
        assert "normalize" in steps
        assert "enrich" in steps
        assert "validate" in steps
        assert "compute_readiness" in steps
        assert "export_alto" in steps
        assert "export_page" in steps
        assert "persist" in steps
        db.close()

    def test_save_image_false_skips_input_copy(
        self, tmp_storage: Path, fixtures_dir: Path, tmp_path: Path
    ) -> None:
        from PIL import Image

        svc, db, store = self._setup(tmp_storage)
        with open(fixtures_dir / "paddle_ocr_sample.json") as f:
            payload = json.load(f)
        raw = RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=payload,
            image_width=2480, image_height=3508,
        )
        img = tmp_path / "page.png"
        Image.new("RGB", (10, 10), "white").save(img)

        job = svc.create_job(
            provider_id="paddleocr",
            provider_family="word_box_json",
            source_filename=img.name,
        )
        result = svc.run_job(
            job, raw, image_width=2480, image_height=3508,
            image_path=img, save_image=False,
        )

        assert result.status == JobStatus.SUCCEEDED
        assert result.has_alto
        # The source image is not duplicated into the job directory.
        assert store.get_input_image_path(result.job_id) is None
        # The RECEIVE_FILE step is recorded as skipped, not silently absent.
        steps = [(e["step"], e["status"]) for e in store.load_events(result.job_id)]
        assert ("receive_file", "skipped") in steps
        db.close()

    def test_default_saves_jpeg_preview(
        self, tmp_storage: Path, fixtures_dir: Path, tmp_path: Path
    ) -> None:
        from PIL import Image

        svc, db, store = self._setup(tmp_storage)
        with open(fixtures_dir / "paddle_ocr_sample.json") as f:
            payload = json.load(f)
        raw = RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=payload,
            image_width=2480, image_height=3508,
        )
        img = tmp_path / "page.png"
        Image.new("RGB", (10, 10), "white").save(img)

        job = svc.create_job(
            provider_id="paddleocr",
            provider_family="word_box_json",
            source_filename=img.name,
        )
        result = svc.run_job(
            job, raw, image_width=2480, image_height=3508, image_path=img
        )

        assert result.status == JobStatus.SUCCEEDED
        # The job folder keeps a JPEG preview, not the original PNG.
        copied = store.get_input_image_path(result.job_id)
        assert copied is not None
        assert copied.name == "input.jpg"
        assert not (tmp_storage / "jobs" / result.job_id / "input.png").exists()
        db.close()

    def test_canonical_json_on_disk_is_valid(self, tmp_storage: Path, fixtures_dir: Path) -> None:
        svc, db, store = self._setup(tmp_storage)

        with open(fixtures_dir / "paddle_ocr_sample.json") as f:
            payload = json.load(f)

        raw = RawProviderPayload(
            provider_id="paddleocr", adapter_id="v1", runtime_type="local",
            payload=payload, image_width=2480, image_height=3508,
        )

        job = svc.create_job("paddleocr", "word_box_json")
        svc.run_job(job, raw, image_width=2480, image_height=3508)

        # Load canonical from disk and validate it
        from src.app.domain.models import CanonicalDocument
        canon_data = store.load_canonical(job.job_id)
        doc = CanonicalDocument.model_validate(canon_data)
        assert doc.document_id == job.job_id
        assert len(doc.pages) == 1
        db.close()

    def test_alto_export_drops_low_confidence_lines(
        self, tmp_storage: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc, db, store = self._setup(tmp_storage)
        # Two engine lines: one ghost (0.40), one real (0.99).
        raw = RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=[
                [[[10, 10], [140, 10], [140, 40], [10, 40]],
                 ("dbaodan", 0.40, [["dbaodan", (10, 10, 140, 40)]])],
                [[[10, 50], [200, 50], [200, 80], [10, 80]],
                 ("Apotheke", 0.99, [["Apotheke", (10, 50, 200, 80)]])],
            ],
            image_width=2480, image_height=3508,
        )
        job = svc.create_job(
            provider_id="paddleocr",
            provider_family="word_box_json",
            source_filename="page.tif",
        )

        # Control run without the setting: both lines are exported.
        result = svc.run_job(job, raw, image_width=2480, image_height=3508)
        alto = store.load_alto(result.job_id)
        assert b"dbaodan" in alto and b"Apotheke" in alto

        # With the setting, the ghost line is dropped from the export only.
        monkeypatch.setenv("ALTO_MIN_LINE_CONFIDENCE", "0.75")
        # get_settings() is lru_cached; the control run above cached the
        # default instance. Clear so the new env var is picked up, and
        # clear again afterwards so nothing leaks into other tests.
        get_settings.cache_clear()
        job2 = svc.create_job(
            provider_id="paddleocr",
            provider_family="word_box_json",
            source_filename="page.tif",
        )
        result2 = svc.run_job(job2, raw, image_width=2480, image_height=3508)
        alto2 = store.load_alto(result2.job_id)
        assert b"dbaodan" not in alto2
        assert b"Apotheke" in alto2
        assert b"1 TextLine(s) omitted" in alto2
        # The canonical keeps everything.
        canon = store.load_canonical(result2.job_id)
        lines = [ln for r in canon["pages"][0]["text_regions"] for ln in r["lines"]]
        assert len(lines) == 2
        db.close()
