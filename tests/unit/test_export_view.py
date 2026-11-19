"""Tests for policy-driven export views."""

from __future__ import annotations

from src.app.domain.models import (
    AltoReadiness,
    CanonicalDocument,
    EvidenceType,
    Geometry,
    GeometryStatus,
    Hyphenation,
    Page,
    PageXmlReadiness,
    Provenance,
    ReadinessLevel,
    Source,
    TextLine,
    TextRegion,
    Word,
)
from src.app.domain.models.status import InputType
from src.app.policies.document_policy import DocumentPolicy
from src.app.policies.export_view import build_alto_export_view


def _prov() -> Provenance:
    return Provenance(
        provider="test", adapter="test.v1", source_ref="$",
        evidence_type=EvidenceType.PROVIDER_NATIVE,
    )


def _geo(y: float = 220) -> Geometry:
    return Geometry(bbox=(110, y, 1100, 42), status=GeometryStatus.EXACT)


def _line(line_id: str, y: float, confidence: float | None) -> TextLine:
    return TextLine(
        id=line_id, geometry=_geo(y), provenance=_prov(),
        confidence=confidence,
        words=[Word(
            id=f"w_{line_id}", text="word", geometry=_geo(y),
            confidence=confidence, provenance=_prov(),
        )],
    )


def _hyphenated_line(
    line_id: str, y: float, confidence: float, part: int, text: str
) -> TextLine:
    word = Word(
        id=f"w_{line_id}", text=text, geometry=_geo(y),
        confidence=confidence, provenance=_prov(),
        hyphenation=Hyphenation(
            is_hyphenated=True, part=part, full_form="Silberling"
        ),
    )
    return TextLine(
        id=line_id, geometry=_geo(y), provenance=_prov(),
        confidence=confidence, words=[word],
    )


def _doc(lines: list[TextLine], reading_order: list[str] | None = None) -> CanonicalDocument:
    return CanonicalDocument(
        document_id="doc_export_view",
        source=Source(input_type=InputType.IMAGE, filename="test.png"),
        pages=[Page(
            id="p1", page_index=0, width=2480, height=3508,
            alto_readiness=AltoReadiness(level=ReadinessLevel.FULL),
            page_readiness=PageXmlReadiness(level=ReadinessLevel.FULL),
            reading_order=reading_order or ["tb1"],
            text_regions=[TextRegion(
                id="tb1", geometry=_geo(200), provenance=_prov(), lines=lines,
            )],
        )],
    )


class TestAltoExportView:
    def test_no_threshold_keeps_document_unchanged(self) -> None:
        doc = _doc([_line("tl_low", 220, 0.40)])

        view, omitted = build_alto_export_view(doc, DocumentPolicy())

        assert view is doc
        assert omitted == 0

    def test_drops_line_below_threshold(self) -> None:
        doc = _doc([
            _line("tl_low", 220, 0.40),
            _line("tl_high", 300, 0.99),
            _line("tl_none", 380, None),
        ])
        policy = DocumentPolicy(alto_min_line_confidence=0.75)

        view, omitted = build_alto_export_view(doc, policy)

        ids = [ln.id for ln in view.pages[0].text_regions[0].lines]
        assert ids == ["tl_high", "tl_none"]
        assert omitted == 1
        # The canonical document keeps every line.
        assert len(doc.pages[0].text_regions[0].lines) == 3

    def test_removes_block_that_loses_every_line(self) -> None:
        doc = _doc([_line("tl_low", 220, 0.40)])
        policy = DocumentPolicy(alto_min_line_confidence=0.75)

        view, omitted = build_alto_export_view(doc, policy)

        assert view.pages[0].text_regions == []
        assert view.pages[0].reading_order == []
        assert omitted == 1

    def test_keeps_reading_order_of_kept_blocks(self) -> None:
        doc = _doc([_line("tl_high", 220, 0.99)])

        view, omitted = build_alto_export_view(
            doc, DocumentPolicy(alto_min_line_confidence=0.75)
        )

        assert view is doc
        assert view.pages[0].reading_order == ["tb1"]
        assert omitted == 0

    def test_keeps_low_confidence_line_that_holds_a_hyphenation_part(self) -> None:
        doc = _doc([
            _hyphenated_line("tl_a", 220, 0.40, 1, "Sil-"),
            _hyphenated_line("tl_b", 260, 0.99, 2, "berling"),
        ])

        view, omitted = build_alto_export_view(
            doc, DocumentPolicy(alto_min_line_confidence=0.75)
        )

        # The pair stays together: the consumer joins part 1 and part 2.
        assert view is doc
        assert omitted == 0
