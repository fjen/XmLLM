"""Tests for PaddleOCR language parameterization and ISO 639-3 mapping."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import src.app.ocr as ocr_module
from src.app.ocr import paddle_lang_to_iso639_3, run_paddle_ocr


class _FakePaddleOCR:
    """Stub engine recording constructor kwargs, returning a fixed result."""

    instances: list[_FakePaddleOCR] = []

    def __init__(self, *, use_angle_cls: bool, lang: str, show_log: bool) -> None:
        self.lang = lang
        _FakePaddleOCR.instances.append(self)

    def ocr(self, image_path: str, cls: bool) -> list[Any]:
        return [[[[1, 1], [2, 1], [2, 2], [1, 2]], ("word", 0.9)]]


@pytest.fixture(autouse=True)
def fake_paddleocr(
    monkeypatch: pytest.MonkeyPatch,
) -> type[_FakePaddleOCR]:
    """Install a fake paddleocr module so the real _get_ocr path runs."""
    _FakePaddleOCR.instances = []
    module = types.ModuleType("paddleocr")
    module.PaddleOCR = _FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    monkeypatch.setattr(ocr_module, "_ocr_instances", {})
    return _FakePaddleOCR


class TestRunPaddleOcrLang:
    def test_lang_reaches_engine_constructor(self, tmp_path: Path) -> None:
        image = tmp_path / "page.tiff"
        image.write_bytes(b"fake")

        results = run_paddle_ocr(image, lang="fr")

        assert len(_FakePaddleOCR.instances) == 1
        assert _FakePaddleOCR.instances[0].lang == "fr"
        # Result shape passes through unchanged: [points, (text, conf)]
        assert results[0] == [[1, 1], [2, 1], [2, 2], [1, 2]]
        assert results[1] == ("word", 0.9)

    def test_instance_cached_per_language(self, tmp_path: Path) -> None:
        image = tmp_path / "page.tiff"
        image.write_bytes(b"fake")

        run_paddle_ocr(image, lang="fr")
        run_paddle_ocr(image, lang="en")
        run_paddle_ocr(image, lang="fr")  # reuses fr instance

        assert [e.lang for e in _FakePaddleOCR.instances] == ["fr", "en"]

    def test_cache_is_lru_bounded(self, tmp_path: Path) -> None:
        image = tmp_path / "page.tiff"
        image.write_bytes(b"fake")

        # 9 distinct allowlisted langs → 9 engine constructions, cache capped at 8
        nine = ["fr", "en", "de", "es", "it", "pt", "nl", "pl", "sv"]
        for lang in nine:
            run_paddle_ocr(image, lang=lang)
        assert len(_FakePaddleOCR.instances) == 9
        assert set(ocr_module._ocr_instances) == set(nine) - {"fr"}  # fr evicted

        # LRU: touching "en" moves it to the end, so a new insert evicts "de"
        run_paddle_ocr(image, lang="en")
        run_paddle_ocr(image, lang="ru")
        assert "de" not in ocr_module._ocr_instances
        assert "en" in ocr_module._ocr_instances
        assert len(ocr_module._ocr_instances) == 8
        assert len(_FakePaddleOCR.instances) == 10  # only ru was constructed


class TestSupportedLangs:
    def test_allowlist_matches_table(self) -> None:
        from src.app.ocr import SUPPORTED_PADDLE_LANGS

        assert "fr" in SUPPORTED_PADDLE_LANGS
        assert "en" in SUPPORTED_PADDLE_LANGS
        assert "japan" in SUPPORTED_PADDLE_LANGS
        assert "chinese_cht" in SUPPORTED_PADDLE_LANGS
        # Codes PaddleOCR does not accept are not allowlisted
        assert "ja" not in SUPPORTED_PADDLE_LANGS
        assert "xyz" not in SUPPORTED_PADDLE_LANGS


class TestPaddleLangToIso6393:
    @pytest.mark.parametrize(
        ("paddle_lang", "expected"),
        [
            ("fr", "fra"),
            ("en", "eng"),
            ("de", "deu"),
            ("FR", "fra"),  # case-insensitive
            (" fra ", "fra"),  # whitespace tolerated
            ("eng", "eng"),  # already ISO 639-3
            ("fra", "fra"),
            ("japan", "jpn"),  # PaddleOCR legacy alias
            ("korean", "kor"),
            ("ch", "zho"),
            ("ch_doc", "zho"),
            ("chinese_cht", "zho"),
            ("french", "fra"),
            ("german", "deu"),
        ],
    )
    def test_known_mappings(self, paddle_lang: str, expected: str) -> None:
        assert paddle_lang_to_iso639_3(paddle_lang) == expected

    @pytest.mark.parametrize(
        "unknown",
        ["xx", "fi", "ja", "ko", "zh", "el", "xyz", "zh_CN", "", "123", "fàr"],
    )
    def test_unknown_codes_return_none(self, unknown: str) -> None:
        assert paddle_lang_to_iso639_3(unknown) is None
