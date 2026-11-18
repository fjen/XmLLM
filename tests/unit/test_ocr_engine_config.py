"""Tests for engine configuration passed to PaddleOCR (_get_ocr kwargs).

The engine constructor is faked via sys.modules (paddle/paddleocr are not
installed in CI); the assertions verify which kwargs reach PaddleOCR.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from src.app import ocr as ocr_module


@pytest.fixture(autouse=True)
def _fresh_instance_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the per-language instance cache."""
    monkeypatch.setattr(ocr_module, "_ocr_instances", {})


def _patch_engine(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
) -> None:
    class FakePaddleOCR:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "paddle",
        types.SimpleNamespace(is_compiled_with_cuda=lambda: False),
    )
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(PaddleOCR=FakePaddleOCR),
    )


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device: str | None,
    rec_batch_size: int | None,
    textline_orientation: bool | None = None,
) -> None:
    monkeypatch.setattr(
        ocr_module,
        "get_settings",
        lambda: types.SimpleNamespace(
            ocr_device=device,
            ocr_recognition_batch_size=rec_batch_size,
            ocr_textline_orientation=textline_orientation,
        ),
    )


class TestRecognitionBatchSize:
    def test_set_value_reaches_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        _patch_engine(monkeypatch, captured)
        _patch_settings(monkeypatch, device=None, rec_batch_size=16)

        ocr_module._get_ocr("en")

        assert captured["text_recognition_batch_size"] == 16

    def test_unset_uses_engine_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        _patch_engine(monkeypatch, captured)
        _patch_settings(monkeypatch, device=None, rec_batch_size=None)

        ocr_module._get_ocr("en")

        assert "text_recognition_batch_size" not in captured


class TestDevicePassthrough:
    def test_device_still_reaches_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        _patch_engine(monkeypatch, captured)
        _patch_settings(monkeypatch, device="gpu:0", rec_batch_size=None)

        ocr_module._get_ocr("en")

        assert captured["device"] == "gpu:0"


class TestTextlineOrientation:
    def test_false_reaches_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        _patch_engine(monkeypatch, captured)
        _patch_settings(
            monkeypatch, device=None, rec_batch_size=None,
            textline_orientation=False,
        )

        ocr_module._get_ocr("en")

        assert captured["use_textline_orientation"] is False

    def test_true_reaches_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        _patch_engine(monkeypatch, captured)
        _patch_settings(
            monkeypatch, device=None, rec_batch_size=None,
            textline_orientation=True,
        )

        ocr_module._get_ocr("en")

        assert captured["use_textline_orientation"] is True

    def test_unset_keeps_enabled_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        _patch_engine(monkeypatch, captured)
        _patch_settings(monkeypatch, device=None, rec_batch_size=None)

        ocr_module._get_ocr("en")

        assert captured["use_textline_orientation"] is True

    def test_instances_cached_per_lang(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        _patch_engine(monkeypatch, captured)
        _patch_settings(monkeypatch, device=None, rec_batch_size=None)

        first = ocr_module._get_ocr("en")
        second = ocr_module._get_ocr("en")

        assert first is second
