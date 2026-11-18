"""Export policy — decides whether a specific export should proceed.

Uses the document policy and export eligibility to make a final go/no-go
decision for each export format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.app.domain.models.status import MissingCapability, ReadinessLevel
from src.app.policies.document_policy import DocumentPolicy

if TYPE_CHECKING:
    from src.app.domain.models.readiness import ExportEligibility


@dataclass(frozen=True)
class ExportDecision:
    """Result of an export policy check."""

    allowed: bool
    level: ReadinessLevel
    reason: str


def check_alto_export(
    eligibility: ExportEligibility,
    policy: DocumentPolicy | None = None,
) -> ExportDecision:
    """Check if ALTO export should proceed."""
    if policy is None:
        policy = DocumentPolicy()

    level = eligibility.alto_export

    if level == ReadinessLevel.NONE:
        return ExportDecision(
            allowed=False,
            level=level,
            reason=(
                "ALTO export not possible: missing required data"
                " (word text/geometry or line geometry)"
            ),
        )

    # Fine-grained requirements (ADR 0001): word-level data is no longer
    # critical for readiness (line-level fallback exports a compatibility
    # <String>), so require_words_for_alto is enforced here, at the policy
    # boundary. require_lines_for_alto is normally already enforced by
    # readiness (LINE_GEOMETRY is critical → NONE) and is checked for
    # completeness.
    if policy.require_words_for_alto and (
        MissingCapability.WORD_TEXT in eligibility.alto_missing
        or MissingCapability.WORD_GEOMETRY in eligibility.alto_missing
    ):
        return ExportDecision(
            allowed=False,
            level=eligibility.alto_export,
            reason=(
                "ALTO export refused: policy requires word-level text and"
                " geometry (require_words_for_alto) but some lines only"
                " carry a line-level transcription"
            ),
        )
    if policy.require_lines_for_alto and (
        MissingCapability.LINE_GEOMETRY in eligibility.alto_missing
    ):
        return ExportDecision(
            allowed=False,
            level=eligibility.alto_export,
            reason=(
                "ALTO export refused: policy requires line-level geometry"
                " (require_lines_for_alto)"
            ),
        )

    if level == ReadinessLevel.PARTIAL and not policy.allow_partial_alto:
        return ExportDecision(
            allowed=False,
            level=level,
            reason="ALTO export is partial but policy does not allow partial exports",
        )

    if level == ReadinessLevel.DEGRADED:
        return ExportDecision(
            allowed=False,
            level=level,
            reason="ALTO export is degraded: too much data missing",
        )

    return ExportDecision(allowed=True, level=level, reason="OK")


def check_page_export(
    eligibility: ExportEligibility,
    policy: DocumentPolicy | None = None,
) -> ExportDecision:
    """Check if PAGE XML export should proceed."""
    if policy is None:
        policy = DocumentPolicy()

    level = eligibility.page_export

    if level == ReadinessLevel.NONE:
        return ExportDecision(
            allowed=False,
            level=level,
            reason="PAGE export not possible: missing required data",
        )

    if level == ReadinessLevel.PARTIAL and not policy.allow_partial_page:
        return ExportDecision(
            allowed=False,
            level=level,
            reason="PAGE export is partial but policy does not allow partial exports",
        )

    if level == ReadinessLevel.DEGRADED:
        return ExportDecision(
            allowed=False,
            level=level,
            reason="PAGE export is degraded: too much data missing",
        )

    return ExportDecision(allowed=True, level=level, reason="OK")
