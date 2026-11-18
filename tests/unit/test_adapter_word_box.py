"""Tests for the word_box_json adapter (PaddleOCR format)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from src.app.domain.models import (
    EvidenceType,
    GeometryStatus,
    RawProviderPayload,
)
from src.app.domain.models.geometry import GeometryContext
from src.app.providers.adapters.word_box_json import WordBoxJsonAdapter

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def adapter() -> WordBoxJsonAdapter:
    return WordBoxJsonAdapter()


@pytest.fixture
def paddle_payload(fixtures_dir: Path) -> list:
    with open(fixtures_dir / "paddle_ocr_sample.json") as f:
        return json.load(f)


@pytest.fixture
def raw(paddle_payload: list) -> RawProviderPayload:
    return RawProviderPayload(
        provider_id="paddleocr",
        adapter_id="adapter.word_box_json.v1",
        runtime_type="local",
        payload=paddle_payload,
        image_width=2480,
        image_height=3508,
    )


@pytest.fixture
def geo_ctx() -> GeometryContext:
    return GeometryContext(source_width=2480, source_height=3508)


class TestWordBoxJsonAdapter:
    def test_family(self, adapter: WordBoxJsonAdapter) -> None:
        assert adapter.family == "word_box_json"

    def test_version(self, adapter: WordBoxJsonAdapter) -> None:
        assert "v1" in adapter.version

    def test_normalize_produces_valid_document(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(
            raw, geo_ctx, document_id="test_doc", source_filename="test.png"
        )
        assert doc.document_id == "test_doc"
        assert doc.source.filename == "test.png"
        assert len(doc.pages) == 1

    def test_page_dimensions(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        page = doc.pages[0]
        assert page.width == 2480
        assert page.height == 3508

    def test_correct_word_count(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        page = doc.pages[0]
        # Fixture items carry no word data → line-level fallback: no words,
        # the transcription lives on the line (ADR 0001).
        total_words = sum(
            len(line.words)
            for region in page.text_regions
            for line in region.lines
        )
        assert total_words == 0
        lines = [
            ln for region in page.text_regions for ln in region.lines
        ]
        assert [ln.text for ln in lines] == [
            "Bonjour", "le", "monde",
            "Ceci est une seconde ligne", "Troisième ligne de texte",
        ]

    def test_transcription_preserved(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        lines = [
            ln for r in doc.pages[0].text_regions for ln in r.lines
        ]
        assert lines[0].transcription == "Bonjour"
        assert lines[1].transcription == "le"
        assert lines[2].transcription == "monde"

    def test_confidence_preserved(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        first_line = doc.pages[0].text_regions[0].lines[0]
        assert first_line.confidence == pytest.approx(0.96)

    def test_geometry_is_exact(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        first_line = doc.pages[0].text_regions[0].lines[0]
        assert first_line.geometry.status == GeometryStatus.EXACT

    def test_polygon_preserved(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        first_line = doc.pages[0].text_regions[0].lines[0]
        assert first_line.geometry.polygon is not None
        assert len(first_line.geometry.polygon) == 4

    def test_bbox_correct(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        first_line = doc.pages[0].text_regions[0].lines[0]
        # From fixture: [[100,200],[300,200],[300,240],[100,240]]
        # → bbox = (100, 200, 200, 40)
        x, y, w, h = first_line.geometry.bbox
        assert x == pytest.approx(100)
        assert y == pytest.approx(200)
        assert w == pytest.approx(200)
        assert h == pytest.approx(40)

    def test_provenance_native(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        first_line = doc.pages[0].text_regions[0].lines[0]
        assert first_line.provenance.evidence_type == EvidenceType.PROVIDER_NATIVE
        assert first_line.provenance.provider == "paddleocr"
        assert "$[0]" in first_line.provenance.source_ref

    def test_block_is_inferred(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        region = doc.pages[0].text_regions[0]
        assert region.geometry.status == GeometryStatus.INFERRED
        assert region.provenance.evidence_type == EvidenceType.DERIVED

    def test_reading_order(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        assert doc.pages[0].reading_order == ["tb1"]

    def test_empty_payload_rejected(
        self,
        adapter: WordBoxJsonAdapter,
        geo_ctx: GeometryContext,
    ) -> None:
        raw = RawProviderPayload(
            provider_id="test",
            adapter_id="v1",
            runtime_type="local",
            payload=[],
        )
        with pytest.raises(ValueError, match="no items"):
            adapter.normalize(raw, geo_ctx, document_id="test_doc")

    def test_dict_payload_rejected(
        self,
        adapter: WordBoxJsonAdapter,
        geo_ctx: GeometryContext,
    ) -> None:
        raw = RawProviderPayload(
            provider_id="test",
            adapter_id="v1",
            runtime_type="local",
            payload={"not": "a list"},
        )
        with pytest.raises(ValueError, match="list payload"):
            adapter.normalize(raw, geo_ctx, document_id="test_doc")

    def test_json_roundtrip(
        self,
        adapter: WordBoxJsonAdapter,
        raw: RawProviderPayload,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = adapter.normalize(raw, geo_ctx, document_id="test_doc")
        data = doc.model_dump(mode="json")
        from src.app.domain.models import CanonicalDocument

        doc2 = CanonicalDocument.model_validate(data)
        assert doc2.document_id == doc.document_id


class TestLanguageStamping:
    """Caller-declared language (raw.metadata['language'], ISO 639-3)."""

    def _normalize(
        self,
        adapter: WordBoxJsonAdapter,
        paddle_payload: list,
        geo_ctx: GeometryContext,
        metadata: dict | None,
    ):
        raw = RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=paddle_payload,
            image_width=2480,
            image_height=3508,
            metadata=metadata,
        )
        return adapter.normalize(raw, geo_ctx, document_id="test_doc")

    def test_lang_stamped_on_all_levels(
        self,
        adapter: WordBoxJsonAdapter,
        paddle_payload: list,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = self._normalize(adapter, paddle_payload, geo_ctx, {"language": "fra"})
        for region in doc.pages[0].text_regions:
            assert region.lang == "fra"
            for line in region.lines:
                assert line.lang == "fra"
                for word in line.words:
                    assert word.lang == "fra"

    def test_no_metadata_keeps_lang_none(
        self,
        adapter: WordBoxJsonAdapter,
        paddle_payload: list,
        geo_ctx: GeometryContext,
    ) -> None:
        doc = self._normalize(adapter, paddle_payload, geo_ctx, None)
        for region in doc.pages[0].text_regions:
            assert region.lang is None
            for line in region.lines:
                for word in line.words:
                    assert word.lang is None

    @pytest.mark.parametrize(
        "bad_lang",
        ["fraa", "FR", "fr1", "fr", "fàr", "éèê", 123, None, ["fra"]],
    )
    def test_invalid_lang_ignored(
        self,
        adapter: WordBoxJsonAdapter,
        paddle_payload: list,
        geo_ctx: GeometryContext,
        bad_lang: object,
    ) -> None:
        doc = self._normalize(
            adapter, paddle_payload, geo_ctx, {"language": bad_lang}
        )
        for region in doc.pages[0].text_regions:
            assert region.lang is None


class TestWordLevelPayload:
    """Per-line word boxes — the optional third payload tuple element."""

    @staticmethod
    def _raw() -> RawProviderPayload:
        payload = [
            # Line with word boxes → one word per box
            [
                [[10, 10], [110, 10], [110, 40], [10, 40]],
                (
                    "Hello world",
                    0.9,
                    [["Hello", (10, 10, 50, 40)], ["world", (60, 10, 110, 40)]],
                ),
            ],
            # Line without word data → fallback single word
            [[[10, 60], [110, 60], [110, 90], [10, 90]], ("Second", 0.8)],
        ]
        return RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=payload,
            image_width=200,
            image_height=120,
        )

    def test_word_boxes_become_words(
        self, adapter: WordBoxJsonAdapter, geo_ctx: GeometryContext
    ) -> None:
        doc = adapter.normalize(self._raw(), geo_ctx, document_id="d")
        lines = doc.pages[0].text_regions[0].lines
        assert [w.text for w in lines[0].words] == ["Hello", "world"]
        # Line without word data → line-level fallback: no words, the
        # transcription lives on the line (ADR 0001)
        assert lines[1].words == []
        assert lines[1].transcription == "Second"

    def test_word_boxes_converted_to_xywh(
        self, adapter: WordBoxJsonAdapter, geo_ctx: GeometryContext
    ) -> None:
        doc = adapter.normalize(self._raw(), geo_ctx, document_id="d")
        words = doc.pages[0].text_regions[0].lines[0].words
        assert words[0].geometry.bbox == (10.0, 10.0, 40.0, 30.0)
        assert words[1].geometry.bbox == (60.0, 10.0, 50.0, 30.0)

    def test_word_provenance_is_derived(
        self, adapter: WordBoxJsonAdapter, geo_ctx: GeometryContext
    ) -> None:
        doc = adapter.normalize(self._raw(), geo_ctx, document_id="d")
        word = doc.pages[0].text_regions[0].lines[0].words[0]
        assert word.provenance.evidence_type is EvidenceType.DERIVED
        assert word.provenance.source_ref == "$[0].text_word"

    def test_word_ids_are_unique(
        self, adapter: WordBoxJsonAdapter, geo_ctx: GeometryContext
    ) -> None:
        doc = adapter.normalize(self._raw(), geo_ctx, document_id="d")
        ids = [
            w.id
            for r in doc.pages[0].text_regions
            for ln in r.lines
            for w in ln.words
        ]
        assert ids == ["w1_1", "w1_2"]

    def test_malformed_word_entry_raises(
        self, adapter: WordBoxJsonAdapter, geo_ctx: GeometryContext
    ) -> None:
        payload = [
            [
                [[10, 10], [110, 10], [110, 40], [10, 40]],
                ("Hello", 0.9, [["Hello", (1, 2, 3)]]),
            ]
        ]
        raw = RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=payload,
            image_width=200,
            image_height=120,
        )
        with pytest.raises(ValueError, match="word 0"):
            adapter.normalize(raw, geo_ctx, document_id="d")

    def test_ambiguous_line_falls_back_and_warns(
        self, adapter: WordBoxJsonAdapter, geo_ctx: GeometryContext
    ) -> None:
        payload = [
            # words=None + reason → line-level word retained, document flagged
            [
                [[10, 10], [110, 10], [110, 40], [10, 40]],
                ("a b", 0.9, None, "word fragments do not align"),
            ],
        ]
        raw = RawProviderPayload(
            provider_id="paddleocr",
            adapter_id="adapter.word_box_json.v1",
            runtime_type="local",
            payload=payload,
            image_width=200,
            image_height=120,
        )
        doc = adapter.normalize(raw, geo_ctx, document_id="d")
        line = doc.pages[0].text_regions[0].lines[0]
        # Line-level fallback: no word geometry, transcription retained
        assert line.words == []
        assert line.transcription == "a b"
        assert doc.pages[0].warnings == [
            "1 of 1 lines: word fragments do not align"
        ]
