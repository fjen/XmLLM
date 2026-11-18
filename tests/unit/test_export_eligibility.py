"""Tests for export eligibility and export policy."""

from __future__ import annotations

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
from src.app.policies.document_policy import DocumentPolicy, strict_policy
from src.app.policies.export_policy import check_alto_export, check_page_export
from src.app.validators.export_eligibility_validator import compute_export_eligibility


def _prov() -> Provenance:
    return Provenance(
        provider="test", adapter="v1", source_ref="$",
        evidence_type=EvidenceType.PROVIDER_NATIVE,
    )


def _geo() -> Geometry:
    return Geometry(bbox=(10, 10, 100, 30), status=GeometryStatus.EXACT)


def _complete_doc() -> CanonicalDocument:
    return CanonicalDocument(
        document_id="test",
        source=Source(input_type=InputType.IMAGE),
        pages=[Page(
            id="p1", page_index=0, width=2480, height=3508,
            alto_readiness=AltoReadiness(level=ReadinessLevel.FULL),
            page_readiness=PageXmlReadiness(level=ReadinessLevel.FULL),
            reading_order=["tb1"],
            text_regions=[
                TextRegion(id="tb1", geometry=_geo(), provenance=_prov(),
                    lines=[TextLine(id="tl1", geometry=_geo(), provenance=_prov(),
                        words=[Word(id="w1", text="Hello", geometry=_geo(),
                            provenance=_prov(), confidence=0.95)])])],
        )],
    )


def _empty_doc() -> CanonicalDocument:
    return CanonicalDocument(
        document_id="test",
        source=Source(input_type=InputType.IMAGE),
        pages=[Page(id="p1", page_index=0, width=2480, height=3508)],
    )


class TestExportEligibility:
    def test_complete_doc_full_eligible(self) -> None:
        doc = _complete_doc()
        elig = compute_export_eligibility(doc)
        assert elig.alto_export == ReadinessLevel.FULL
        assert elig.page_export == ReadinessLevel.FULL
        assert elig.viewer_render == ReadinessLevel.FULL

    def test_empty_doc_none(self) -> None:
        doc = _empty_doc()
        elig = compute_export_eligibility(doc)
        assert elig.alto_export == ReadinessLevel.NONE
        assert elig.page_export == ReadinessLevel.NONE
        assert elig.viewer_render == ReadinessLevel.NONE

    def test_strict_policy_downgrades_partial(self) -> None:
        # A doc with missing confidence → partial
        doc = CanonicalDocument(
            document_id="test",
            source=Source(input_type=InputType.IMAGE),
            pages=[Page(
                id="p1", page_index=0, width=2480, height=3508,
                reading_order=["tb1"],
                text_regions=[
                    TextRegion(id="tb1", geometry=_geo(), provenance=_prov(),
                        lines=[TextLine(id="tl1", geometry=_geo(), provenance=_prov(),
                            words=[Word(id="w1", text="Hello", geometry=_geo(),
                                provenance=_prov(), confidence=None)])])],
            )],
        )
        policy = strict_policy()
        elig = compute_export_eligibility(doc, policy)
        # Strict mode downgrades partial to none
        assert elig.alto_export == ReadinessLevel.NONE

    def test_viewer_degraded_for_regions_without_exports(self) -> None:
        # Doc with unknown word geometry — ALTO none, but viewer shows something
        doc = CanonicalDocument(
            document_id="test",
            source=Source(input_type=InputType.IMAGE),
            pages=[Page(
                id="p1", page_index=0, width=2480, height=3508,
                text_regions=[
                    TextRegion(id="tb1", geometry=_geo(), provenance=_prov(),
                        lines=[TextLine(id="tl1", geometry=_geo(), provenance=_prov(),
                            words=[Word(id="w1", text="Hello",
                                geometry=Geometry(bbox=(10, 10, 100, 30),
                                    status=GeometryStatus.UNKNOWN),
                                provenance=_prov())])])],
            )],
        )
        elig = compute_export_eligibility(doc)
        # ALTO is none (missing word geo), but viewer can show regions
        assert elig.viewer_render in (ReadinessLevel.FULL, ReadinessLevel.DEGRADED)


class TestExportPolicy:
    def test_alto_allowed_full(self) -> None:
        doc = _complete_doc()
        elig = compute_export_eligibility(doc)
        decision = check_alto_export(elig)
        assert decision.allowed is True
        assert decision.reason == "OK"

    def test_alto_refused_none(self) -> None:
        doc = _empty_doc()
        elig = compute_export_eligibility(doc)
        decision = check_alto_export(elig)
        assert decision.allowed is False
        assert "not possible" in decision.reason

    def test_alto_partial_default_allowed(self) -> None:
        doc = CanonicalDocument(
            document_id="test",
            source=Source(input_type=InputType.IMAGE),
            pages=[Page(
                id="p1", page_index=0, width=2480, height=3508,
                reading_order=["tb1"],
                text_regions=[
                    TextRegion(id="tb1", geometry=_geo(), provenance=_prov(),
                        lines=[TextLine(id="tl1", geometry=_geo(), provenance=_prov(),
                            words=[Word(id="w1", text="Hello", geometry=_geo(),
                                provenance=_prov(), confidence=None)])])],
            )],
        )
        elig = compute_export_eligibility(doc)
        decision = check_alto_export(elig)
        assert decision.allowed is True

    def test_alto_partial_strict_refused(self) -> None:
        doc = CanonicalDocument(
            document_id="test",
            source=Source(input_type=InputType.IMAGE),
            pages=[Page(
                id="p1", page_index=0, width=2480, height=3508,
                reading_order=["tb1"],
                text_regions=[
                    TextRegion(id="tb1", geometry=_geo(), provenance=_prov(),
                        lines=[TextLine(id="tl1", geometry=_geo(), provenance=_prov(),
                            words=[Word(id="w1", text="Hello", geometry=_geo(),
                                provenance=_prov(), confidence=None)])])],
            )],
        )
        policy = strict_policy()
        elig = compute_export_eligibility(doc, policy)
        decision = check_alto_export(elig, policy)
        assert decision.allowed is False

    def test_page_allowed_full(self) -> None:
        doc = _complete_doc()
        elig = compute_export_eligibility(doc)
        decision = check_page_export(elig)
        assert decision.allowed is True

    def test_page_refused_none(self) -> None:
        doc = _empty_doc()
        elig = compute_export_eligibility(doc)
        decision = check_page_export(elig)
        assert decision.allowed is False


class TestRequireWordsForAlto:
    """Interaction of require_words_for_alto with allow_partial_alto
    on a line-level-fallback document (ADR 0001)."""

    @staticmethod
    def _fallback_doc() -> CanonicalDocument:
        return CanonicalDocument(
            document_id="test",
            source=Source(input_type=InputType.IMAGE),
            pages=[Page(
                id="p1", page_index=0, width=2480, height=3508,
                reading_order=["tb1"],
                text_regions=[
                    TextRegion(id="tb1", geometry=_geo(), provenance=_prov(),
                        lines=[TextLine(
                            id="tl1", geometry=_geo(), provenance=_prov(),
                            transcription="Hello world",
                            words=[],
                        )]),
                ],
            )],
        )

    def test_default_policy_allows_fallback(self) -> None:
        """Default: no require_words_for_alto, partial allowed → export."""
        elig = compute_export_eligibility(self._fallback_doc())
        assert MissingCapability.WORD_TEXT in elig.alto_missing
        decision = check_alto_export(elig)
        assert decision.allowed is True

    def test_require_words_refuses_fallback(self) -> None:
        policy = DocumentPolicy(require_words_for_alto=True)
        elig = compute_export_eligibility(self._fallback_doc(), policy)
        decision = check_alto_export(elig, policy)
        assert decision.allowed is False
        assert "require_words_for_alto" in decision.reason

    def test_allow_partial_alto_false_refuses_fallback(self) -> None:
        policy = DocumentPolicy(allow_partial_alto=False)
        elig = compute_export_eligibility(self._fallback_doc(), policy)
        decision = check_alto_export(elig, policy)
        assert decision.allowed is False
        assert "partial" in decision.reason

    def test_require_words_ignores_confidence_only_partial(self) -> None:
        """PARTIAL from a missing word confidence (word data present) is
        not a word-level-data refusal."""
        doc = CanonicalDocument(
            document_id="test",
            source=Source(input_type=InputType.IMAGE),
            pages=[Page(
                id="p1", page_index=0, width=2480, height=3508,
                reading_order=["tb1"],
                text_regions=[
                    TextRegion(id="tb1", geometry=_geo(), provenance=_prov(),
                        lines=[TextLine(id="tl1", geometry=_geo(),
                            provenance=_prov(),
                            words=[Word(id="w1", text="Hello",
                                geometry=_geo(), provenance=_prov(),
                                confidence=None)])]),
                ],
            )],
        )
        policy = DocumentPolicy(require_words_for_alto=True)
        elig = compute_export_eligibility(doc, policy)
        assert MissingCapability.WORD_TEXT not in elig.alto_missing
        assert MissingCapability.WORD_GEOMETRY not in elig.alto_missing
        decision = check_alto_export(elig, policy)
        assert decision.allowed is True
