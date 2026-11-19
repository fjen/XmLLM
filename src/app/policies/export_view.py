"""Export views — policy-driven pruning before serialization.

A serializer must not decide which nodes to export. This module applies
the export rules of the document policy to a canonical document. It
returns the view that the serializer receives, plus the number of omitted
lines. The canonical document keeps every node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app.domain.models import (
        CanonicalDocument,
        Page,
        TextLine,
        TextRegion,
    )
    from src.app.policies.document_policy import DocumentPolicy


def build_alto_export_view(
    doc: CanonicalDocument, policy: DocumentPolicy
) -> tuple[CanonicalDocument, int]:
    """Remove ALTO lines below the policy confidence threshold.

    Lines without a confidence stay. A line that carries a hyphenation part
    stays too, because the consumer joins the two parts of a pair. A text
    block that loses every line is removed, and the page reading order no
    longer names it. If no line is removed, the input document comes back
    unchanged.

    Returns:
        The pruned document and the number of omitted lines.
    """
    threshold = policy.alto_min_line_confidence
    if threshold is None:
        return doc, 0

    pages: list[Page] = []
    omitted = 0
    for page in doc.pages:
        regions: list[TextRegion] = []
        for region in page.text_regions:
            lines = [ln for ln in region.lines if _keep_line(ln, threshold)]
            omitted += len(region.lines) - len(lines)
            if lines:
                regions.append(region.model_copy(update={"lines": lines}))
        kept_ids = {region.id for region in regions}
        pages.append(page.model_copy(update={
            "text_regions": regions,
            "reading_order": [
                region_id
                for region_id in page.reading_order
                if region_id in kept_ids
            ],
        }))

    if omitted == 0:
        return doc, 0
    return doc.model_copy(update={"pages": pages}), omitted


def _keep_line(line: TextLine, threshold: float) -> bool:
    """A line stays when it meets the limit, has no score, or carries a
    hyphenation part. The exporter keeps hyphenation pairs together."""
    if line.confidence is None or line.confidence >= threshold:
        return True
    return any(word.hyphenation is not None for word in line.words)
