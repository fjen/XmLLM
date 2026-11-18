"""Tests for the remote OCR service dispatch (OCR_SERVICE_URL)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.app import ocr as ocr_module
from src.app.domain.models import RawProviderPayload
from src.app.domain.models.geometry import GeometryContext
from src.app.ocr import run_paddle_ocr
from src.app.providers.adapters.word_box_json import WordBoxJsonAdapter

# JSON form of the payload (lists, exactly what the service returns).
PAYLOAD = [
    [
        [[10, 10], [110, 10], [110, 40], [10, 40]],
        [
            "Hello world",
            0.9,
            [["Hello", [10, 10, 50, 40]], ["world", [60, 10, 110, 40]]],
        ],
    ],
]


class _StubOcrHandler(BaseHTTPRequestHandler):
    last_request: dict = {}

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        _StubOcrHandler.last_request = {"path": self.path, "body": body}
        payload = json.dumps(PAYLOAD).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:  # silence
        pass


class _FailingHandler(_StubOcrHandler):
    def do_POST(self) -> None:  # noqa: N802
        detail = json.dumps({"detail": "engine exploded"}).encode()
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(detail)))
        self.end_headers()
        self.wfile.write(detail)


@pytest.fixture
def stub_server():
    def start(handler: type[BaseHTTPRequestHandler]) -> str:
        server = HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    image = tmp_path / "page.png"
    image.write_bytes(b"stub-image-bytes")
    return image


def _use_service(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setattr(
        ocr_module,
        "get_settings",
        lambda: SimpleNamespace(ocr_service_url=url),
    )


class TestRemoteDispatch:
    def test_posts_image_and_returns_payload(
        self, stub_server, image_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = stub_server(_StubOcrHandler)
        _use_service(monkeypatch, url)

        payload = run_paddle_ocr(image_file, "fr")

        assert payload == PAYLOAD
        assert _StubOcrHandler.last_request["path"].startswith("/ocr")
        assert "lang=fr" in _StubOcrHandler.last_request["path"]
        assert b"stub-image-bytes" in _StubOcrHandler.last_request["body"]

    def test_json_payload_feeds_adapter(
        self, stub_server, image_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The service returns JSON lists; the adapter must accept them."""
        url = stub_server(_StubOcrHandler)
        _use_service(monkeypatch, url)

        payload = run_paddle_ocr(image_file, "fr")
        raw = RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=payload,
            image_width=200,
            image_height=120,
        )
        doc = WordBoxJsonAdapter().normalize(
            raw, GeometryContext(source_width=200, source_height=120),
            document_id="d",
        )
        words = doc.pages[0].text_regions[0].lines[0].words
        assert [w.text for w in words] == ["Hello", "world"]

    def test_http_error_raises_runtime_error(
        self, stub_server, image_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = stub_server(_FailingHandler)
        _use_service(monkeypatch, url)

        with pytest.raises(RuntimeError, match="HTTP 500.*engine exploded"):
            run_paddle_ocr(image_file, "fr")

    def test_unreachable_service_raises(
        self, image_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Port 1 is unassigned on the loopback; connection refused.
        _use_service(monkeypatch, "http://127.0.0.1:1")

        with pytest.raises(RuntimeError, match="unreachable"):
            run_paddle_ocr(image_file, "fr")

    def test_local_path_used_without_service_url(
        self, image_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_service(monkeypatch, "")
        with pytest.raises(Exception) as exc_info:
            run_paddle_ocr(image_file, "fr")
        # No HTTP call happened: the local engine path ran and failed on
        # the stub image bytes, proving the dispatch chose local.
        assert "unreachable" not in str(exc_info.value)
