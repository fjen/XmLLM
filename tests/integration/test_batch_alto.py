"""Integration tests for scripts/batch_alto.py — batch images → ALTO, no server.

The PaddleOCR engine call is stubbed; everything downstream (payload
conversion, adapter, enrichers, validation, eligibility, ALTO serialization,
job persistence) runs for real.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from lxml import etree
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import batch_alto  # noqa: E402


def _make_image(path: Path) -> None:
    Image.new("RGB", (300, 200), "white").save(path)


@pytest.fixture
def stub_ocr(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Stub run_paddle_ocr; records (path, lang) calls."""
    calls: list[tuple[str, str]] = []

    def fake_run_paddle_ocr(image_path: Path, lang: str) -> list[Any]:
        calls.append((str(image_path), lang))
        return [
            [[[10, 10], [110, 10], [110, 40], [10, 40]], ("Bonjour", 0.98)],
            [[[10, 60], [210, 60], [210, 90], [10, 90]], ("monde", 0.97)],
        ]

    monkeypatch.setattr("src.app.ocr.run_paddle_ocr", fake_run_paddle_ocr)
    return calls


class TestCollectImages:
    def test_discovers_images_case_insensitive(self, tmp_path: Path) -> None:
        _make_image(tmp_path / "a.tiff")
        _make_image(tmp_path / "b.TIFF")
        _make_image(tmp_path / "c.Tif")
        _make_image(tmp_path / "d.jpg")
        _make_image(tmp_path / "e.JPG")
        _make_image(tmp_path / "f.jpeg")
        _make_image(tmp_path / "g.png")
        _make_image(tmp_path / "h.webp")
        _make_image(tmp_path / "i.bmp")
        (tmp_path / "notes.txt").write_text("not an image")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        _make_image(subdir / "j.tiff")  # not recursive

        found = batch_alto.collect_images([tmp_path])
        assert [p.name for p in found] == [
            "a.tiff", "b.TIFF", "c.Tif", "d.jpg", "e.JPG", "f.jpeg",
            "g.png", "h.webp", "i.bmp",
        ]

    def test_multiple_directories(self, tmp_path: Path) -> None:
        book1 = tmp_path / "book1"
        book2 = tmp_path / "book2"
        book1.mkdir()
        book2.mkdir()
        _make_image(book1 / "p1.tiff")
        _make_image(book2 / "p2.tiff")

        found = batch_alto.collect_images([book1, book2])
        assert [p.name for p in found] == ["p1.tiff", "p2.tiff"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert batch_alto.collect_images([empty]) == []

    def test_missing_directory_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            batch_alto.collect_images([tmp_path / "nope"])
        assert excinfo.value.code == 2


class TestBatchEndToEnd:
    def test_image_becomes_job_with_alto(
        self,
        stub_ocr: list[tuple[str, str]],
        tmp_path: Path,
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.tiff")
        storage = tmp_path / "storage"
        out = tmp_path / "out"

        rc = batch_alto.main(
            [str(book), "--lang", "fr", "--out", str(out),
             "--storage-root", str(storage)]
        )

        assert rc == 0
        # lang propagated to the (stubbed) engine
        assert stub_ocr == [(str(book / "page1.tiff"), "fr")]

        alto_path = out / "page1.xml"
        assert alto_path.exists()
        root = etree.parse(str(alto_path)).getroot()
        assert root.tag == "{http://www.loc.gov/standards/alto/ns-v4#}alto"
        strings = root.findall(".//{*}String")
        assert [s.get("CONTENT") for s in strings] == ["Bonjour", "monde"]
        # The --out copy is the plain decompressed bytes of the stored XML.
        from src.app.persistence.file_store import FileStore

        job_id = next((tmp_path / "storage" / "jobs").iterdir()).name
        assert alto_path.read_bytes() == FileStore(tmp_path / "storage").load_alto(job_id)

    def test_jpeg_becomes_job_with_alto(
        self,
        stub_ocr: list[tuple[str, str]],
        tmp_path: Path,
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.jpg")

        rc = batch_alto.main(
            [str(book), "--lang", "en", "--out", str(tmp_path / "out"),
             "--storage-root", str(tmp_path / "storage")]
        )

        assert rc == 0
        assert stub_ocr == [(str(book / "page1.jpg"), "en")]
        alto_path = tmp_path / "out" / "page1.xml"
        assert alto_path.exists()
        root = etree.parse(str(alto_path)).getroot()
        assert root.tag == "{http://www.loc.gov/standards/alto/ns-v4#}alto"
        strings = root.findall(".//{*}String")
        assert [s.get("CONTENT") for s in strings] == ["Bonjour", "monde"]

    def test_alto_carries_lang_attribute(
        self,
        stub_ocr: list[tuple[str, str]],
        tmp_path: Path,
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.tiff")

        rc = batch_alto.main(
            [str(book), "--lang", "fr", "--storage-root", str(tmp_path / "storage")]
        )

        assert rc == 0
        job_dirs = list((tmp_path / "storage" / "jobs").iterdir())
        assert len(job_dirs) == 1
        from io import BytesIO

        from src.app.persistence.file_store import FileStore

        alto_bytes = FileStore(tmp_path / "storage").load_alto(job_dirs[0].name)
        assert alto_bytes is not None
        root = etree.parse(BytesIO(alto_bytes)).getroot()
        for s in root.findall(".//{*}String"):
            assert s.get("LANG") == "fra"

    def test_unsupported_lang_rejected(
        self, stub_ocr: list[tuple[str, str]], tmp_path: Path
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.tiff")

        rc = batch_alto.main(
            [str(book), "--lang", "xx", "--storage-root", str(tmp_path / "storage")]
        )

        assert rc == 2  # rejected up front; no OCR, no jobs
        assert stub_ocr == []
        assert not (tmp_path / "storage" / "jobs").exists() or not list(
            (tmp_path / "storage" / "jobs").iterdir()
        )

    def test_no_images_exits_nonzero(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = batch_alto.main(
            [str(empty), "--storage-root", str(tmp_path / "storage")]
        )
        assert rc == 2

    def test_lang_normalized_case_insensitive(
        self, stub_ocr: list[tuple[str, str]], tmp_path: Path
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.tiff")

        rc = batch_alto.main(
            [str(book), "--lang", "FR", "--out", str(tmp_path / "out"),
             "--storage-root", str(tmp_path / "storage")]
        )

        assert rc == 0
        assert stub_ocr == [(str(book / "page1.tiff"), "fr")]  # normalized
        assert 'LANG="fra"' in (tmp_path / "out" / "page1.xml").read_text()

    def test_same_stem_formats_both_processed(
        self, stub_ocr: list[tuple[str, str]], tmp_path: Path
    ) -> None:
        """page.tif + page.jpg must not abort the batch — copies disambiguate."""
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page.jpg")
        _make_image(book / "page.tif")

        rc = batch_alto.main(
            [str(book), "--lang", "en", "--out", str(tmp_path / "out"),
             "--storage-root", str(tmp_path / "storage")]
        )

        assert rc == 0  # both processed, neither aborted the run
        assert stub_ocr == [
            (str(book / "page.jpg"), "en"),
            (str(book / "page.tif"), "en"),
        ]
        out = tmp_path / "out"
        # sorted order: page.jpg first → keeps page.xml; tif disambiguated
        assert (out / "page.xml").exists()
        assert (out / "page_tif.xml").exists()
        # both jobs persisted
        jobs = list((tmp_path / "storage" / "jobs").iterdir())
        assert len(jobs) == 2

    def test_failing_image_does_not_stop_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "good.tiff")
        _make_image(book / "bad.tiff")

        calls: list[str] = []

        def flaky_ocr(image_path: Path, lang: str) -> list[Any]:
            calls.append(image_path.name)
            if "bad" in image_path.name:
                raise RuntimeError("engine exploded")
            return [[[[10, 10], [110, 10], [110, 40], [10, 40]], ("ok", 0.9)]]

        monkeypatch.setattr("src.app.ocr.run_paddle_ocr", flaky_ocr)

        rc = batch_alto.main(
            [str(book), "--lang", "en", "--storage-root", str(tmp_path / "storage")]
        )

        assert rc == 1  # one failure
        assert sorted(calls) == ["bad.tiff", "good.tiff"]  # batch continued
        jobs_dir = tmp_path / "storage" / "jobs"
        assert len(list(jobs_dir.iterdir())) == 1  # only the good one produced a job


class TestSkipExisting:
    def test_rerun_skips_processed_images_with_out(
        self, stub_ocr: list[tuple[str, str]], tmp_path: Path
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.tiff")
        argv = [
            str(book), "--lang", "fr", "--out", str(tmp_path / "out"),
            "--storage-root", str(tmp_path / "storage"),
        ]

        assert batch_alto.main(argv) == 0
        assert len(stub_ocr) == 1
        stub_ocr.clear()

        # Rerun: dest XML exists → skipped, zero engine calls, still rc 0.
        assert batch_alto.main(argv) == 0
        assert stub_ocr == []
        assert (tmp_path / "out" / "page1.xml").exists()

    def test_no_skip_existing_reprocesses(
        self, stub_ocr: list[tuple[str, str]], tmp_path: Path
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.tiff")
        argv = [
            str(book), "--lang", "fr", "--out", str(tmp_path / "out"),
            "--storage-root", str(tmp_path / "storage"),
        ]

        assert batch_alto.main(argv) == 0
        stub_ocr.clear()

        assert batch_alto.main([*argv, "--no-skip-existing"]) == 0
        assert len(stub_ocr) == 1  # reprocessed despite existing output

    def test_rerun_skips_via_job_db_without_out(
        self, stub_ocr: list[tuple[str, str]], tmp_path: Path
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.tiff")
        argv = [str(book), "--lang", "fr", "--storage-root", str(tmp_path / "storage")]

        assert batch_alto.main(argv) == 0
        assert len(stub_ocr) == 1
        stub_ocr.clear()

        # No --out: a prior job for the same filename that produced ALTO
        # counts as done (looked up in the job database).
        assert batch_alto.main(argv) == 0
        assert stub_ocr == []


    def test_interrupted_copy_does_not_poison_skip(
        self, stub_ocr: list[tuple[str, str]], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A crash mid-copy must not leave a truncated dest that
        skip-existing would silently skip forever."""
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.tiff")
        out = tmp_path / "out"
        argv = [
            str(book), "--lang", "fr", "--out", str(out),
            "--storage-root", str(tmp_path / "storage"),
        ]

        class _SimulatedInterrupt(BaseException):
            """Like KeyboardInterrupt (escapes `except Exception`) without
            tripping pytest's own session-interrupt handling."""

        # os is shared with the module under test: capture the real
        # replace BEFORE patching, or the "restore" would restore the
        # crashing patch itself.
        real_replace = os.replace

        def crashing_replace(src: Path, dst: Path) -> None:
            # Simulate disk-full: temp file truncated, crash before the rename.
            Path(src).write_text("<alto>truncat", encoding="utf-8")
            raise _SimulatedInterrupt

        monkeypatch.setattr(batch_alto.os, "replace", crashing_replace)
        with pytest.raises(_SimulatedInterrupt):
            batch_alto.main(argv)

        # The dest was never completed; the .part marker exists.
        assert not (out / "page1.xml").exists()
        assert (out / "page1.xml.part").exists()

        # Rerun with the real replace: the partial state does not poison
        # skip-existing — the image is processed and written again.
        monkeypatch.setattr(batch_alto.os, "replace", real_replace)
        assert batch_alto.main(argv) == 0
        assert len(stub_ocr) == 2  # reprocessed
        assert (out / "page1.xml").exists()
        root = etree.parse(str(out / "page1.xml")).getroot()
        assert root.tag == "{http://www.loc.gov/standards/alto/ns-v4#}alto"
        assert not list(out.glob("*.part"))  # no stale marker left


class TestNoImageCopy:
    def test_batch_job_dir_has_no_input_image(
        self, stub_ocr: list[tuple[str, str]], tmp_path: Path
    ) -> None:
        book = tmp_path / "book"
        book.mkdir()
        _make_image(book / "page1.tiff")

        rc = batch_alto.main(
            [str(book), "--lang", "fr", "--storage-root", str(tmp_path / "storage")]
        )

        assert rc == 0
        job_dir = next((tmp_path / "storage" / "jobs").iterdir())
        artifacts = {p.name for p in job_dir.iterdir()}
        assert artifacts == {
            "raw_payload.json.zst", "canonical.json.zst", "alto.xml.zst",
            "page.xml.zst", "viewer.json.zst", "events.json.zst",
        }
        assert not any(p.stem == "input" for p in job_dir.iterdir())
