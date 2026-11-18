"""Tests for the readiness validator."""

from __future__ import annotations

from lxml import etree
from src.app.domain.models import (
    AltoReadiness,
    CanonicalDocument,
    EvidenceType,
    Geometry,
    GeometryStatus,
    Page,
    PageXmlReadiness,
    Provenance,
    ReadinessLevel,
    Source,
    TextLine,
    TextRegion,
    Word,
)
from src.app.domain.models.status import InputType, MissingCapability
from src.app.normalization.canonical_builder import CanonicalBuilder
from src.app.policies.export_policy import check_alto_export
from src.app.serializers.alto_xml import serialize_alto
from src.app.validators.export_eligibility_validator import (
    compute_export_eligibility,
)
from src.app.validators.readiness_validator import (
    compute_document_readiness,
    compute_page_alto_readiness,
    compute_page_pagexml_readiness,
)


def _prov() -> Provenance:
    return Provenance(
        provider="test", adapter="v1", source_ref="$",
        evidence_type=EvidenceType.PROVIDER_NATIVE,
    )


def _geo(status: GeometryStatus = GeometryStatus.EXACT) -> Geometry:
    return Geometry(bbox=(10, 10, 100, 30), status=status)


def _complete_page() -> Page:
    """A fully complete page with all data."""
    return Page(
        id="p1", page_index=0, width=2480, height=3508,
        alto_readiness=AltoReadiness(level=ReadinessLevel.FULL),
        page_readiness=PageXmlReadiness(level=ReadinessLevel.FULL),
        reading_order=["tb1"],
        text_regions=[
            TextRegion(
                id="tb1", geometry=_geo(), provenance=_prov(), lang="fra",
                lines=[
                    TextLine(
                        id="tl1", geometry=_geo(), provenance=_prov(),
                        words=[
                            Word(id="w1", text="Hello", geometry=_geo(),
                                 provenance=_prov(), confidence=0.95),
                        ],
                    ),
                ],
            ),
        ],
    )


class TestAltoReadiness:
    def test_complete_page_is_full(self) -> None:
        page = _complete_page()
        r = compute_page_alto_readiness(page)
        assert r.level == ReadinessLevel.FULL
        assert r.missing == []

    def test_missing_word_geometry_is_partial(self) -> None:
        """Word-level capabilities degrade to PARTIAL, not NONE (ADR 0001):
        the compatibility <String> still yields valid ALTO with full text."""
        page = Page(
            id="p1", page_index=0, width=2480, height=3508,
            text_regions=[
                TextRegion(
                    id="tb1", geometry=_geo(), provenance=_prov(),
                    lines=[TextLine(
                        id="tl1", geometry=_geo(), provenance=_prov(),
                        words=[Word(
                            id="w1", text="Hello",
                            geometry=_geo(GeometryStatus.UNKNOWN),
                            provenance=_prov(),
                        )],
                    )],
                ),
            ],
        )
        r = compute_page_alto_readiness(page)
        assert r.level == ReadinessLevel.PARTIAL
        assert MissingCapability.WORD_GEOMETRY in r.missing

    def test_wordless_lines_degrade_to_partial(self) -> None:
        """A page whose lines carry only transcriptions is PARTIAL with
        WORD_TEXT/WORD_GEOMETRY missing — exportable, not word-complete."""
        page = Page(
            id="p1", page_index=0, width=2480, height=3508,
            text_regions=[
                TextRegion(
                    id="tb1", geometry=_geo(), provenance=_prov(),
                    lines=[TextLine(
                        id="tl1", geometry=_geo(), provenance=_prov(),
                        transcription="Hello",
                        words=[],
                    )],
                ),
            ],
        )
        r = compute_page_alto_readiness(page)
        assert r.level == ReadinessLevel.PARTIAL
        assert MissingCapability.WORD_TEXT in r.missing
        assert MissingCapability.WORD_GEOMETRY in r.missing

    def test_blank_line_does_not_degrade_readiness(self) -> None:
        """A whitespace-only transcription carries nothing to locate and
        must not downgrade an otherwise word-complete page."""
        page = Page(
            id="p1", page_index=0, width=2480, height=3508,
            reading_order=["tb1"],
            text_regions=[
                TextRegion(
                    id="tb1", geometry=_geo(), provenance=_prov(),
                    lines=[
                        TextLine(
                            id="tl1", geometry=_geo(), provenance=_prov(),
                            words=[Word(
                                id="w1", text="Hello",
                                geometry=_geo(), provenance=_prov(),
                                confidence=0.9,
                            )],
                        ),
                        TextLine(
                            id="tl2", geometry=_geo(), provenance=_prov(),
                            transcription="   ",
                            words=[],
                        ),
                    ],
                ),
            ],
        )
        r = compute_page_alto_readiness(page)
        assert r.level == ReadinessLevel.FULL
        assert r.missing == []

    def test_fallback_page_exports_with_zero_canonical_words(self) -> None:
        """A fallback line survives ALTO export as a compatibility <String>
        while carrying zero canonical words — export must not make a
        PARTIAL page appear word-ready (ADR 0001)."""
        builder = CanonicalBuilder("doc1", InputType.IMAGE)
        page = builder.add_page("p1", 0, 2480, 3508)
        region = page.add_text_region("tb1", _geo(), _prov())
        region.add_line(
            "tl1", _geo(), _prov(), transcription="Hello world"
        )
        doc = builder.build()

        line = doc.pages[0].text_regions[0].lines[0]
        assert line.words == []  # zero canonical words

        # Readiness: word-level capabilities missing, page PARTIAL
        r = compute_page_alto_readiness(doc.pages[0])
        assert r.level == ReadinessLevel.PARTIAL
        assert MissingCapability.WORD_TEXT in r.missing
        assert MissingCapability.WORD_GEOMETRY in r.missing

        # Export: allowed under the default policy, via the compatibility
        # <String> — but the exported ALTO must not claim word-level data.
        decision = check_alto_export(compute_export_eligibility(doc))
        assert decision.allowed

        root = etree.fromstring(serialize_alto(doc))
        ns = "{http://www.loc.gov/standards/alto/ns-v4#}"
        strings = root.findall(f".//{ns}String")
        assert len(strings) == 1
        assert strings[0].get("CONTENT") == "Hello world"
        assert strings[0].get("ID") == "tl1_str"  # compatibility, not w*

    def test_missing_confidence_is_partial(self) -> None:
        page = Page(
            id="p1", page_index=0, width=2480, height=3508,
            reading_order=["tb1"],
            text_regions=[
                TextRegion(
                    id="tb1", geometry=_geo(), provenance=_prov(),
                    lines=[TextLine(
                        id="tl1", geometry=_geo(), provenance=_prov(),
                        words=[Word(
                            id="w1", text="Hello", geometry=_geo(),
                            provenance=_prov(), confidence=None,
                        )],
                    )],
                ),
            ],
        )
        r = compute_page_alto_readiness(page)
        assert r.level == ReadinessLevel.PARTIAL
        assert MissingCapability.CONFIDENCE in r.missing

    def test_no_reading_order_is_partial(self) -> None:
        page = Page(
            id="p1", page_index=0, width=2480, height=3508,
            reading_order=[],
            text_regions=[
                TextRegion(
                    id="tb1", geometry=_geo(), provenance=_prov(),
                    lines=[TextLine(
                        id="tl1", geometry=_geo(), provenance=_prov(),
                        words=[Word(
                            id="w1", text="Hello", geometry=_geo(),
                            provenance=_prov(), confidence=0.9,
                        )],
                    )],
                ),
            ],
        )
        r = compute_page_alto_readiness(page)
        assert r.level == ReadinessLevel.PARTIAL
        assert MissingCapability.READING_ORDER in r.missing

    def test_empty_page_is_none(self) -> None:
        page = Page(id="p1", page_index=0, width=2480, height=3508)
        r = compute_page_alto_readiness(page)
        assert r.level == ReadinessLevel.NONE


class TestPageXmlReadiness:
    def test_complete_page_is_full(self) -> None:
        page = _complete_page()
        r = compute_page_pagexml_readiness(page)
        assert r.level == ReadinessLevel.FULL

    def test_no_regions_is_none(self) -> None:
        page = Page(id="p1", page_index=0, width=2480, height=3508)
        r = compute_page_pagexml_readiness(page)
        assert r.level == ReadinessLevel.NONE

    def test_regions_without_word_geo_still_ok(self) -> None:
        """PAGE XML is more lenient — word geometry is not critical."""
        page = Page(
            id="p1", page_index=0, width=2480, height=3508,
            reading_order=["tb1"],
            text_regions=[
                TextRegion(
                    id="tb1", geometry=_geo(), provenance=_prov(),
                    lines=[TextLine(
                        id="tl1", geometry=_geo(), provenance=_prov(),
                        words=[Word(
                            id="w1", text="Hello",
                            geometry=_geo(GeometryStatus.UNKNOWN),
                            provenance=_prov(),
                        )],
                    )],
                ),
            ],
        )
        r = compute_page_pagexml_readiness(page)
        # PAGE doesn't require word geometry — should still be achievable
        assert r.level in (ReadinessLevel.FULL, ReadinessLevel.PARTIAL)


class TestDocumentReadiness:
    def test_single_full_page(self) -> None:
        doc = CanonicalDocument(
            document_id="test",
            source=Source(input_type=InputType.IMAGE),
            pages=[_complete_page()],
        )
        dr = compute_document_readiness(doc)
        assert dr.level == ReadinessLevel.FULL

    def test_mixed_pages(self) -> None:
        full_page = _complete_page()
        empty_page = Page(id="p2", page_index=1, width=2480, height=3508)
        doc = CanonicalDocument(
            document_id="test",
            source=Source(input_type=InputType.IMAGE),
            pages=[full_page, empty_page],
        )
        dr = compute_document_readiness(doc)
        assert dr.level == ReadinessLevel.DEGRADED
        assert len(dr.page_readiness) == 2
