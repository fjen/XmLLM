"""Tests for the GPU OCR service script (scripts/ocr_service.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from scripts.ocr_service import MAX_UPLOAD_BYTES, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _small_upload_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.ocr_service.MAX_UPLOAD_BYTES", 64)


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []
    monkeypatch.setattr(
        "scripts.ocr_service.run_paddle_ocr_local",
        lambda path, lang: calls.append((path, lang))
        or [[[[1, 1], [2, 1], [2, 2], [1, 2]], ["x", 0.9]]],
    )
    return calls


class TestOcrService:
    def test_calls_local_engine_not_dispatcher(
        self, captured_calls: list
    ) -> None:
        r = client.post(
            "/ocr", params={"lang": "fr"},
            files={"image": ("page.png", b"png-bytes", "image/png")},
        )
        assert r.status_code == 200
        assert len(captured_calls) == 1  # dispatched once, no self-recursion

    def test_temp_file_uses_suffix_only(self, captured_calls: list) -> None:
        r = client.post(
            "/ocr", params={"lang": "fr"},
            files={"image": ("../../etc/cron.d/evil.png", b"png", "image/png")},
        )
        assert r.status_code == 200
        path, _lang = captured_calls[0]
        assert ".." not in str(path)
        assert path.suffix == ".png"
        # The caller-controlled name never appears in the path.
        assert "cron.d" not in str(path)
        assert "evil" not in str(path)

    def test_illegal_suffix_gets_bin(self, captured_calls: list) -> None:
        r = client.post(
            "/ocr", params={"lang": "fr"},
            files={"image": ("page.we!rd", b"bytes", "application/octet-stream")},
        )
        assert r.status_code == 200
        path, _lang = captured_calls[0]
        assert path.suffix == ".bin"

    def test_upload_cap_enforced(self, captured_calls: list) -> None:
        r = client.post(
            "/ocr", params={"lang": "fr"},
            files={"image": ("big.png", b"x" * (MAX_UPLOAD_BYTES + 1), "image/png")},
        )
        assert r.status_code == 413
        assert captured_calls == []

    def test_unsupported_lang_rejected(self, captured_calls: list) -> None:
        r = client.post(
            "/ocr", params={"lang": "klingon"},
            files={"image": ("page.png", b"png", "image/png")},
        )
        assert r.status_code == 400
        assert captured_calls == []
