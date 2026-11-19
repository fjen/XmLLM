"""Tests for the ALTO XML serializer."""

from __future__ import annotations

from lxml import etree
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
from src.app.domain.models.status import BlockRole, InputType
from src.app.serializers.alto_xml import ALTO_NS, serialize_alto, serialize_alto_to_string


def _prov(ref: str = "$.test") -> Provenance:
    return Provenance(
        provider="test",
        adapter="test.v1",
        source_ref=ref,
        evidence_type=EvidenceType.PROVIDER_NATIVE,
    )


def _geo(x: float = 100, y: float = 200, w: float = 300, h: float = 50) -> Geometry:
    return Geometry(bbox=(x, y, w, h), status=GeometryStatus.EXACT)


def _simple_doc() -> CanonicalDocument:
    """A simple one-page document with one block, one line, two words."""
    return CanonicalDocument(
        document_id="doc_alto_test",
        source=Source(input_type=InputType.IMAGE, filename="test.png"),
        pages=[
            Page(
                id="p1",
                page_index=0,
                width=2480,
                height=3508,
                alto_readiness=AltoReadiness(level=ReadinessLevel.FULL),
                page_readiness=PageXmlReadiness(level=ReadinessLevel.FULL),
                reading_order=["tb1"],
                text_regions=[
                    TextRegion(
                        id="tb1",
                        role=BlockRole.BODY,
                        geometry=_geo(100, 200, 1200, 900),
                        lang="fra",
                        provenance=_prov(),
                        lines=[
                            TextLine(
                                id="tl1",
                                geometry=_geo(110, 220, 1100, 42),
                                lang="fra",
                                provenance=_prov(),
                                words=[
                                    Word(
                                        id="w1",
                                        text="Bonjour",
                                        geometry=_geo(110, 220, 90, 40),
                                        lang="fra",
                                        confidence=0.96,
                                        provenance=_prov(),
                                    ),
                                    Word(
                                        id="w2",
                                        text="monde",
                                        geometry=_geo(220, 220, 80, 40),
                                        lang="fra",
                                        confidence=0.94,
                                        provenance=_prov(),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


class TestAltoSerialization:
    def test_produces_valid_xml(self) -> None:
        doc = _simple_doc()
        xml_bytes = serialize_alto(doc)
        # Should parse without error
        root = etree.fromstring(xml_bytes)
        assert root.tag == f"{{{ALTO_NS}}}alto"

    def test_has_description(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        desc = root.find(f"{{{ALTO_NS}}}Description")
        assert desc is not None

    def test_has_layout(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        layout = root.find(f"{{{ALTO_NS}}}Layout")
        assert layout is not None

    def test_page_attributes(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        page = root.find(f".//{{{ALTO_NS}}}Page")
        assert page is not None
        assert page.get("ID") == "p1"
        assert page.get("WIDTH") == "2480"
        assert page.get("HEIGHT") == "3508"

    def test_text_block_exists(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        tb = root.find(f".//{{{ALTO_NS}}}TextBlock")
        assert tb is not None
        assert tb.get("ID") == "tb1"

    def test_text_block_bbox(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        tb = root.find(f".//{{{ALTO_NS}}}TextBlock")
        assert tb.get("HPOS") == "100"
        assert tb.get("VPOS") == "200"
        assert tb.get("WIDTH") == "1200"
        assert tb.get("HEIGHT") == "900"

    def test_text_line_exists(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        tl = root.find(f".//{{{ALTO_NS}}}TextLine")
        assert tl is not None
        assert tl.get("ID") == "tl1"

    def test_strings_exist(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        strings = root.findall(f".//{{{ALTO_NS}}}String")
        assert len(strings) == 2

    def test_string_content(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        strings = root.findall(f".//{{{ALTO_NS}}}String")
        assert strings[0].get("CONTENT") == "Bonjour"
        assert strings[1].get("CONTENT") == "monde"

    def test_string_bbox(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        s = root.findall(f".//{{{ALTO_NS}}}String")[0]
        assert s.get("HPOS") == "110"
        assert s.get("VPOS") == "220"
        assert s.get("WIDTH") == "90"
        assert s.get("HEIGHT") == "40"

    def test_string_confidence(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        s = root.findall(f".//{{{ALTO_NS}}}String")[0]
        assert s.get("WC") == "0.96"

    def test_string_lang(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        tb = root.find(f".//{{{ALTO_NS}}}TextBlock")
        assert tb.get("LANG") == "fra"

    def test_sp_between_words(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        tl = root.find(f".//{{{ALTO_NS}}}TextLine")
        children = list(tl)
        # Should be: String, SP, String
        tags = [c.tag.split("}")[-1] for c in children]
        assert tags == ["String", "SP", "String"]

    def test_measurement_unit(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        mu = root.find(f".//{{{ALTO_NS}}}MeasurementUnit")
        assert mu is not None
        assert mu.text == "pixel"

    def test_filename_in_description(self) -> None:
        doc = _simple_doc()
        root = etree.fromstring(serialize_alto(doc))
        fn = root.find(f".//{{{ALTO_NS}}}fileName")
        assert fn is not None
        assert fn.text == "test.png"

    def test_string_output(self) -> None:
        doc = _simple_doc()
        xml_str = serialize_alto_to_string(doc)
        assert '<?xml version=' in xml_str
        assert "Bonjour" in xml_str
        assert "monde" in xml_str


class TestAltoHyphenation:
    def test_hyphenated_words(self) -> None:
        doc = CanonicalDocument(
            document_id="doc_hyph",
            source=Source(input_type=InputType.IMAGE),
            pages=[
                Page(
                    id="p1",
                    page_index=0,
                    width=1000,
                    height=1000,
                    alto_readiness=AltoReadiness(level=ReadinessLevel.FULL),
                    page_readiness=PageXmlReadiness(level=ReadinessLevel.FULL),
                    text_regions=[
                        TextRegion(
                            id="tb1",
                            geometry=_geo(0, 0, 1000, 500),
                            provenance=_prov(),
                            lines=[
                                TextLine(
                                    id="tl1",
                                    geometry=_geo(0, 0, 500, 40),
                                    provenance=_prov(),
                                    words=[
                                        Word(
                                            id="w1",
                                            text="patri-",
                                            geometry=_geo(0, 0, 80, 30),
                                            provenance=_prov(),
                                            hyphenation=Hyphenation(
                                                is_hyphenated=True,
                                                part=1,
                                                full_form="patrimoine",
                                            ),
                                        ),
                                    ],
                                ),
                                TextLine(
                                    id="tl2",
                                    geometry=_geo(0, 50, 500, 40),
                                    provenance=_prov(),
                                    words=[
                                        Word(
                                            id="w2",
                                            text="moine",
                                            geometry=_geo(0, 50, 70, 30),
                                            provenance=_prov(),
                                            hyphenation=Hyphenation(
                                                is_hyphenated=True,
                                                part=2,
                                                full_form="patrimoine",
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )

        root = etree.fromstring(serialize_alto(doc))
        strings = root.findall(f".//{{{ALTO_NS}}}String")
        assert len(strings) == 2

        s1 = strings[0]
        assert s1.get("CONTENT") == "patri-"
        assert s1.get("SUBS_TYPE") == "HypPart1"
        assert s1.get("SUBS_CONTENT") == "patrimoine"

        s2 = strings[1]
        assert s2.get("CONTENT") == "moine"
        assert s2.get("SUBS_TYPE") == "HypPart2"
        assert s2.get("SUBS_CONTENT") == "patrimoine"


def _wordless_line_doc() -> CanonicalDocument:
    """One line with a transcription but no reconstructed words (ADR 0001)."""
    return CanonicalDocument(
        document_id="doc_alto_wordless",
        source=Source(input_type=InputType.IMAGE, filename="test.png"),
        pages=[
            Page(
                id="p1",
                page_index=0,
                width=2480,
                height=3508,
                alto_readiness=AltoReadiness(level=ReadinessLevel.PARTIAL),
                page_readiness=PageXmlReadiness(level=ReadinessLevel.PARTIAL),
                reading_order=["tb1"],
                text_regions=[
                    TextRegion(
                        id="tb1",
                        role=BlockRole.BODY,
                        geometry=_geo(100, 200, 1200, 900),
                        lang="fra",
                        provenance=_prov(),
                        lines=[
                            TextLine(
                                id="tl1",
                                geometry=_geo(110, 220, 1100, 42),
                                lang="fra",
                                confidence=0.91,
                                provenance=_prov(),
                                transcription="Bonjour le monde",
                                words=[],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


class TestCompatibilityString:
    """Line-level fallback export (ADR 0001) — full text, line box."""

    def test_wordless_line_exports_single_string(self) -> None:
        root = etree.fromstring(serialize_alto(_wordless_line_doc()))
        strings = root.findall(f".//{{{ALTO_NS}}}String")
        assert len(strings) == 1
        s = strings[0]
        assert s.get("CONTENT") == "Bonjour le monde"
        assert s.get("ID") == "tl1_str"
        assert float(s.get("HPOS")) == 110.0
        assert float(s.get("VPOS")) == 220.0
        assert float(s.get("WIDTH")) == 1100.0
        assert float(s.get("HEIGHT")) == 42.0
        assert s.get("WC") == "0.91"
        assert s.get("LANG") == "fra"

    def test_wordless_line_has_no_sp(self) -> None:
        root = etree.fromstring(serialize_alto(_wordless_line_doc()))
        assert not root.findall(f".//{{{ALTO_NS}}}SP")


class TestLowConfidenceExportFilter:
    """ALTO_MIN_LINE_CONFIDENCE — drop suspect lines from the export only."""

    def _two_line_doc(self) -> CanonicalDocument:
        low = TextLine(
            id="tl_low", geometry=_geo(110, 220, 1100, 42),
            provenance=_prov(), confidence=0.52,
            words=[Word(
                id="w1", text="dbaodan", geometry=_geo(110, 220, 90, 40),
                confidence=0.52, provenance=_prov(),
            )],
        )
        high = TextLine(
            id="tl_high", geometry=_geo(110, 300, 1100, 42),
            provenance=_prov(), confidence=0.99,
            words=[Word(
                id="w2", text="Apotheke", geometry=_geo(110, 300, 90, 40),
                confidence=0.99, provenance=_prov(),
            )],
        )
        return CanonicalDocument(
            document_id="doc_filter_test",
            source=Source(input_type=InputType.IMAGE, filename="test.png"),
            pages=[Page(
                id="p1", page_index=0, width=2480, height=3508,
                alto_readiness=AltoReadiness(level=ReadinessLevel.FULL),
                page_readiness=PageXmlReadiness(level=ReadinessLevel.FULL),
                reading_order=["tb1"],
                text_regions=[TextRegion(
                    id="tb1", geometry=_geo(100, 200, 1200, 900),
                    provenance=_prov(), lines=[low, high],
                )],
            )],
        )

    def test_default_keeps_every_line(self) -> None:
        xml = serialize_alto_to_string(self._two_line_doc())
        assert "dbaodan" in xml and "Apotheke" in xml
        assert "omitted" not in xml

    def test_drops_lines_below_threshold(self) -> None:
        xml = serialize_alto_to_string(
            self._two_line_doc(), min_line_confidence=0.75
        )
        assert "dbaodan" not in xml
        assert "Apotheke" in xml
        assert "1 TextLine(s) omitted" in xml
        assert "0.75" in xml

    def test_drops_block_emptied_by_filter(self) -> None:
        doc = self._two_line_doc()
        region = doc.pages[0].text_regions[0].model_copy(
            update={"lines": [doc.pages[0].text_regions[0].lines[0]]}
        )
        page = doc.pages[0].model_copy(update={"text_regions": [region]})
        doc = doc.model_copy(update={"pages": [page]})

        xml = serialize_alto_to_string(doc, min_line_confidence=0.75)

        assert "TextBlock" not in xml
        assert "1 TextLine(s) omitted" in xml

    def test_keeps_lines_without_confidence(self) -> None:
        doc = self._two_line_doc()
        fallback = TextLine(
            id="tl_fb", geometry=_geo(110, 380, 1100, 42),
            provenance=_prov(), confidence=None,
            transcription="unlesbare Zeile", words=[],
        )
        region = doc.pages[0].text_regions[0]
        region = region.model_copy(update={"lines": [*region.lines, fallback]})
        page = doc.pages[0].model_copy(update={"text_regions": [region]})
        doc = doc.model_copy(update={"pages": [page]})

        xml = serialize_alto_to_string(doc, min_line_confidence=0.75)

        assert "unlesbare Zeile" in xml
        assert "dbaodan" not in xml
