"""Tests for PaddleOCR language parameterization and ISO 639-3 mapping."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import src.app.ocr as ocr_module
from src.app.ocr import paddle_lang_to_iso639_3, run_paddle_ocr


class _FakePaddleOCR:
    """Stub engine recording constructor kwargs, returning a fixed result."""

    instances: list[_FakePaddleOCR] = []

    def __init__(
        self, *, lang: str, use_textline_orientation: bool,
        use_doc_orientation_classify: bool, use_doc_unwarping: bool,
        enable_mkldnn: bool, return_word_box: bool,
        device: str | None = None,
    ) -> None:
        self.lang = lang
        self.return_word_box = return_word_box
        self.device = device
        _FakePaddleOCR.instances.append(self)

    def predict(self, image_path: str) -> list[dict[str, Any]]:
        return [
            {
                "rec_polys": [[[1, 1], [2, 1], [2, 2], [1, 2]]],
                "rec_texts": ["word"],
                "rec_scores": [0.9],
            }
        ]


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
        # Word-level boxes are requested for real word segmentation
        assert _FakePaddleOCR.instances[0].return_word_box is True
        # Payload shape: [4-point polygon, (text, confidence)]
        assert results[0][0] == [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
        assert results[0][1] == ("word", 0.9)

    def test_device_reaches_engine_constructor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ocr_module,
            "get_settings",
            lambda: SimpleNamespace(ocr_device="gpu:0", ocr_service_url=None),
        )
        image = tmp_path / "page.tiff"
        image.write_bytes(b"fake")

        run_paddle_ocr(image, lang="fr")

        assert _FakePaddleOCR.instances[0].device == "gpu:0"

    def test_device_unset_leaves_auto_select(
        self, tmp_path: Path
    ) -> None:
        image = tmp_path / "page.tiff"
        image.write_bytes(b"fake")

        run_paddle_ocr(image, lang="fr")

        assert _FakePaddleOCR.instances[0].device is None

    def test_excluded_lang_skips_reconstruction(
        self, tmp_path: Path
    ) -> None:
        """Excluded languages keep the line-level transcription, flagged."""
        image = tmp_path / "page.tiff"
        image.write_bytes(b"fake")

        results = run_paddle_ocr(image, lang="ch")
        text, conf, words, reason = results[0][1]
        assert text == "word"
        assert conf == pytest.approx(0.9)
        assert words is None
        assert "zho" in reason
        assert "disabled" in reason

    def test_whitespace_script_keeps_plain_payload(
        self, tmp_path: Path
    ) -> None:
        """Engines without word data keep the two-element form for latin langs."""
        image = tmp_path / "page.tiff"
        image.write_bytes(b"fake")

        results = run_paddle_ocr(image, lang="en")
        assert results[0][1] == ("word", 0.9)

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
        assert "fi" in SUPPORTED_PADDLE_LANGS
        assert "japan" in SUPPORTED_PADDLE_LANGS
        assert "chinese_cht" in SUPPORTED_PADDLE_LANGS
        # Codes paddleocr 3.x does not accept are not allowlisted
        assert "ja" not in SUPPORTED_PADDLE_LANGS
        assert "ch_doc" not in SUPPORTED_PADDLE_LANGS
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
            ("ch", "zho"),
            ("chinese_cht", "zho"),
            ("french", "fra"),
            ("german", "deu"),
            ("fi", "fin"),  # LATIN in paddleocr 3.x
            ("th", "tha"),  # PP-OCRv5 script codes
            ("el", "ell"),
            ("te", "tel"),
            ("ta", "tam"),
            ("ka", "kat"),
        ],
    )
    def test_known_mappings(self, paddle_lang: str, expected: str) -> None:
        assert paddle_lang_to_iso639_3(paddle_lang) == expected

    @pytest.mark.parametrize(
        "unknown",
        ["xx", "ja", "ko", "zh", "ch_doc", "xyz", "zh_CN", "", "123", "fàr"],
    )
    def test_unknown_codes_return_none(self, unknown: str) -> None:
        assert paddle_lang_to_iso639_3(unknown) is None
