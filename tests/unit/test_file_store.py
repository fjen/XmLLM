"""Tests for the file store."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.app.persistence.file_store import FileStore

if TYPE_CHECKING:
    from pathlib import Path


class TestFileStore:
    def test_ensure_dirs(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        store.ensure_dirs()
        assert (tmp_storage / "jobs").exists()
        assert (tmp_storage / "providers").exists()

    def test_save_and_load_json(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        data = {"key": "value", "number": 42}
        store.save_json("job_001", "test.json", data)
        loaded = store.load_json("job_001", "test.json")
        assert loaded == data

    def test_load_nonexistent_returns_none(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        assert store.load_json("missing", "test.json") is None
        assert store.load_xml("missing", "test.xml") is None

    def test_save_raw_payload(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        store.save_raw_payload("job_001", {"payload": [1, 2, 3]})
        loaded = store.load_raw_payload("job_001")
        assert loaded == {"payload": [1, 2, 3]}

    def test_save_canonical(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        store.save_canonical("job_001", {"document_id": "doc1"})
        loaded = store.load_canonical("job_001")
        assert loaded["document_id"] == "doc1"

    def test_save_alto(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        xml = b"<?xml version='1.0'?><alto/>"
        store.save_alto("job_001", xml)
        assert store.load_alto("job_001") == xml

    def test_save_page_xml(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        xml = b"<?xml version='1.0'?><PcGts/>"
        store.save_page_xml("job_001", xml)
        assert store.load_page_xml("job_001") == xml

    def test_load_legacy_plain_xml(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        d = tmp_storage / "jobs" / "job_001"
        d.mkdir(parents=True)
        (d / "alto.xml").write_bytes(b"<alto>legacy</alto>")

        assert store.load_alto("job_001") == b"<alto>legacy</alto>"
        assert store.job_has_artifact("job_001", "alto.xml")

    def test_xml_zst_preferred_over_legacy(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        d = tmp_storage / "jobs" / "job_001"
        d.mkdir(parents=True)
        (d / "alto.xml").write_bytes(b"<alto>legacy</alto>")
        store.save_alto("job_001", b"<alto>zstd</alto>")

        assert store.load_alto("job_001") == b"<alto>zstd</alto>"

    def test_save_events(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        events = [{"step": "normalize", "status": "completed"}]
        store.save_events("job_001", events)
        loaded = store.load_events("job_001")
        assert loaded == events

    def test_list_jobs(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        store.save_json("job_aaa", "test.json", {})
        store.save_json("job_bbb", "test.json", {})
        jobs = store.list_jobs()
        assert "job_aaa" in jobs
        assert "job_bbb" in jobs

    def test_job_exists(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        assert not store.job_exists("job_001")
        store.save_json("job_001", "test.json", {})
        assert store.job_exists("job_001")

    def test_job_has_artifact(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        store.save_json("job_001", "canonical.json", {})
        assert store.job_has_artifact("job_001", "canonical.json")
        assert not store.job_has_artifact("job_001", "alto.xml")

    def test_provider_crud(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        store.ensure_dirs()
        store.save_provider("paddle", {"provider_id": "paddle", "family": "word_box_json"})
        assert store.load_provider("paddle") is not None
        assert "paddle" in store.list_providers()
        assert store.delete_provider("paddle")
        assert store.load_provider("paddle") is None

    def test_save_input_image_converts_to_jpeg_preview(
        self, tmp_storage: Path, tmp_path: Path
    ) -> None:
        from PIL import Image

        store = FileStore(tmp_storage)
        img = tmp_path / "test.png"
        Image.new("RGB", (10, 10), "white").save(img)
        result = store.save_input_image("job_001", img)
        assert result.name == "input.jpg"
        assert result.exists()
        # The original format is not duplicated into the job directory.
        assert not (tmp_storage / "jobs" / "job_001" / "input.png").exists()
        assert store.get_input_image_path("job_001") == result

    def test_save_input_image_tif_becomes_renderable_preview(
        self, tmp_storage: Path, tmp_path: Path
    ) -> None:
        from PIL import Image

        store = FileStore(tmp_storage)
        img = tmp_path / "scan.tif"
        Image.new("L", (10, 10), "white").save(img)
        result = store.save_input_image("job_002", img)
        assert result.name == "input.jpg"
        assert store.get_input_image_path("job_002") == result
        job_dir = tmp_storage / "jobs" / "job_002"
        # The browser cannot render TIFF; the job folder keeps no input.tif.
        assert [f.name for f in job_dir.iterdir()] == ["input.jpg"]

    def test_save_input_image_jpeg_copied_verbatim(
        self, tmp_storage: Path, tmp_path: Path
    ) -> None:
        from PIL import Image

        store = FileStore(tmp_storage)
        img = tmp_path / "photo.jpeg"
        Image.new("RGB", (10, 10), "white").save(img, format="JPEG")
        result = store.save_input_image("job_003", img)
        assert result.name == "input.jpg"
        assert result.read_bytes() == img.read_bytes()

    def test_get_input_image_path_prefers_renderable(
        self, tmp_storage: Path
    ) -> None:
        store = FileStore(tmp_storage)
        job_dir = tmp_storage / "jobs" / "job_legacy"
        job_dir.mkdir(parents=True)
        (job_dir / "input.tif").write_bytes(b"tiff")
        (job_dir / "input.png").write_bytes(b"png")
        # A renderable leftover from an old job wins over the TIFF.
        assert store.get_input_image_path("job_legacy").suffix == ".png"

    def test_get_input_image_path_falls_back_to_tif(
        self, tmp_storage: Path
    ) -> None:
        store = FileStore(tmp_storage)
        job_dir = tmp_storage / "jobs" / "job_tif_only"
        job_dir.mkdir(parents=True)
        (job_dir / "input.tif").write_bytes(b"tiff")
        # Last resort: serve the non-renderable original rather than nothing.
        assert store.get_input_image_path("job_tif_only").suffix == ".tif"


class TestZstdCompression:
    def test_save_json_writes_compressed_zst(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        # Repetitive content so the size win is measurable on a small file.
        data = {"text": "lorem ipsum " * 200, "n": 42}
        plain = json.dumps(data).encode()
        dest = store.save_json("job_001", "canonical.json", data)

        assert dest.name == "canonical.json.zst"
        raw = dest.read_bytes()
        assert raw[:4] == b"\x28\xb5\x2f\xfd"  # zstd frame magic
        assert len(raw) < len(plain)  # actually compressed
        assert store.load_json("job_001", "canonical.json") == data

    def test_load_legacy_plain_json(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        d = tmp_storage / "jobs" / "job_001"
        d.mkdir(parents=True)
        (d / "viewer.json").write_text('{"a": 1}', encoding="utf-8")

        assert store.load_json("job_001", "viewer.json") == {"a": 1}
        assert store.job_has_artifact("job_001", "viewer.json")

    def test_zst_preferred_over_legacy(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        d = tmp_storage / "jobs" / "job_001"
        d.mkdir(parents=True)
        (d / "canonical.json").write_text('{"v": "legacy"}', encoding="utf-8")
        store.save_json("job_001", "canonical.json", {"v": "zstd"})

        assert store.load_json("job_001", "canonical.json") == {"v": "zstd"}

    def test_provider_zst_and_legacy(self, tmp_storage: Path) -> None:
        store = FileStore(tmp_storage)
        store.ensure_dirs()
        store.save_provider("p1", {"provider_id": "p1"})
        assert (tmp_storage / "providers" / "p1.json.zst").exists()

        # A legacy plain provider file still loads and lists.
        (tmp_storage / "providers" / "p2.json").write_text(
            '{"provider_id": "p2"}', encoding="utf-8"
        )
        assert store.load_provider("p2") == {"provider_id": "p2"}
        assert store.list_providers() == ["p1", "p2"]

        assert store.delete_provider("p2") is True
        assert not (tmp_storage / "providers" / "p2.json").exists()
        assert store.delete_provider("p1") is True
        assert store.load_provider("p1") is None
