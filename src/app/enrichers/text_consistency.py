"""text_consistency enricher — checks that word texts aggregate correctly.

Adds warnings to pages where the concatenation of words in a line
doesn't produce sensible text (e.g. empty words, suspicious patterns).
This enricher does NOT modify text — it only adds warnings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.app.enrichers import BaseEnricher

if TYPE_CHECKING:
    from src.app.domain.models import CanonicalDocument, TextLine
    from src.app.policies.document_policy import DocumentPolicy


class TextConsistencyEnricher(BaseEnricher):
    @property
    def name(self) -> str:
        return "text_consistency"

    def enrich(
        self, doc: CanonicalDocument, policy: DocumentPolicy
    ) -> CanonicalDocument:
        new_pages = []
        changed = False

        for page in doc.pages:
            warnings = list(page.warnings)

            for region in page.text_regions:
                for line in region.lines:
                    line_warnings = self._check_line(region.id, line.id, line)
                    if line_warnings:
                        warnings.extend(line_warnings)
                        changed = True
                    low_conf = self._check_confidence(region.id, line.id, line)
                    if low_conf:
                        warnings.append(low_conf)
                        changed = True

            new_pages.append(page.model_copy(update={"warnings": warnings}))

        if changed:
            return doc.model_copy(update={"pages": new_pages})
        return doc

    # Engine confidence below this marks a line as suspect. Measured on
    # bleed-through scans: ghost lines (mirrored backside text misread as
    # garbage) all score below 0.75, real text sits at 0.97 and above.
    SUSPECT_CONFIDENCE_THRESHOLD = 0.75

    @staticmethod
    def _check_confidence(region_id: str, line_id: str, line: TextLine) -> str | None:
        """Flag lines whose engine confidence marks them as suspect.

        The engine scores text lines, not words, so this is a per-line
        signal. Warnings only — the text is never dropped or changed.
        """
        confidence = line.confidence
        if confidence is None or confidence >= TextConsistencyEnricher.SUSPECT_CONFIDENCE_THRESHOLD:
            return None
        return (
            f"{region_id}/{line_id}: low engine confidence "
            f"({confidence:.2f}) — possible ghost text or unreadable passage"
        )

    @staticmethod
    def _check_line(region_id: str, line_id: str, line: TextLine) -> list[str]:
        """Check text consistency within a line."""
        warnings: list[str] = []

        for word in line.words:
            text = word.text
            if not text.strip():
                warnings.append(
                    f"{region_id}/{line_id}: word {word.id} has blank text"
                )

            # Check for suspiciously long "words" (likely unsplit lines)
            if len(text) > 100:
                warnings.append(
                    f"{region_id}/{line_id}: word {word.id} "
                    f"is suspiciously long ({len(text)} chars) — may be an unsplit line"
                )

        return warnings
