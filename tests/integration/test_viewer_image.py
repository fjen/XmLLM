"""Integration tests for GET /jobs/{id}/image and the viewer JSON contract.

The viewer frontend consumes exactly two routes: the persisted
ViewerProjection (GET /jobs/{id}/viewer) and the input image
(GET /jobs/{id}/image). These tests guard both.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def paddle_payload_bytes(fixtures_dir: Path) -> bytes:
    with open(fixtures_dir / "paddle_ocr_sample.json", "rb") as f:
        return f.read()


@pytest.fixture
def stub_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the PaddleOCR engine call; everything downstream is real."""

    def fake_run_paddle_ocr(image_path: Path, lang: str) -> list:
        return [
            [[[10, 10], [140, 10], [140, 40], [10, 40]],
             ("Hello", 0.97, [["Hello", (10, 10, 140, 40)]])],
            [[[10, 50], [200, 50], [200, 80], [10, 80]],
             ("World", 0.93, [["World", (10, 50, 200, 80)]])],
        ]

    monkeypatch.setattr("src.app.ocr.run_paddle_ocr", fake_run_paddle_ocr)


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 150), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def completed_job(client: TestClient, stub_ocr: None) -> str:
    """A succeeded job created through POST /ocr (image persisted)."""
    r = client.post("/ocr", files={"image": ("page.png", _png_bytes(), "image/png")})
    assert r.status_code == 201
    return r.json()["job_id"]


class TestJobImage:
    """GET /jobs/{id}/image — serves the persisted input image."""

    def test_200_with_correct_content_type(self, client: TestClient, completed_job: str) -> None:
        r = client.get(f"/jobs/{completed_job}/image")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content == _png_bytes()

    def test_404_for_unknown_job(self, client: TestClient) -> None:
        r = client.get("/jobs/nonexistent/image")
        assert r.status_code == 404

    def test_404_for_job_without_input_image(
        self, client: TestClient, paddle_payload_bytes: bytes
    ) -> None:
        """Jobs created from a raw JSON payload persist no input image."""
        r = client.post(
            "/jobs",
            params={
                "provider_id": "paddleocr",
                "provider_family": "word_box_json",
                "image_width": 2480,
                "image_height": 3508,
            },
            files={
                "raw_payload_file": (
                    "p.json",
                    io.BytesIO(paddle_payload_bytes),
                    "application/json",
                ),
            },
        )
        assert r.status_code == 201
        job_id = r.json()["job_id"]

        r = client.get(f"/jobs/{job_id}/image")
        assert r.status_code == 404


class TestViewerProjectionContract:
    """Guards the pipeline → viewer contract the frontend relies on."""

    def test_word_overlays_nonempty_and_linked(
        self, client: TestClient, completed_job: str
    ) -> None:
        r = client.get(f"/jobs/{completed_job}/viewer")
        assert r.status_code == 200
        data = r.json()

        assert len(data["word_overlays"]) >= 2
        # Every overlay id must resolve in the inspection index — the UI's
        # click-for-details path depends on it. Checked for all four levels:
        # non_text overlays previously had no inspection entry, which made
        # clicking them a silent no-op in the viewer.
        for level in (
            "block_overlays",
            "line_overlays",
            "word_overlays",
            "non_text_overlays",
        ):
            for overlay in data[level]:
                assert overlay["id"] in data["inspection_index"], level
                assert len(overlay["bbox"]) == 4

    def test_export_status_and_flags_present(
        self, client: TestClient, completed_job: str
    ) -> None:
        r = client.get(f"/jobs/{completed_job}/viewer")
        assert r.status_code == 200
        data = r.json()
        assert set(data["export_status"]) == {"alto_export", "page_export", "viewer_render"}
        assert isinstance(data["validation_flags"], list)
