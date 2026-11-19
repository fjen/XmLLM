"""block_split enricher — splits the synthesized single text region into
band/column regions using only existing line boxes.

The word_box_json adapter wraps every line of a page into one region,
because PP-OCR supplies text lines without layout analysis. On multi-
column pages that mixes unrelated columns into one block and interleaves
them in reading order. This enricher regroups the lines geometrically:

1. Lines spanning most of the text width separate horizontal bands.
   Consecutive spanning lines with a small vertical gap share one band
   (a full-width paragraph stays together; a heading separates bands).
2. Within a band, lines cluster into columns by x-range overlap. A line
   that bridges the gutter merges the columns — the split degrades to
   one region instead of guessing.
3. Reading order follows the structure: bands top-to-bottom, columns
   left-to-right, lines top-to-bottom within a column.

The enricher never invents coordinates: every region bbox is the union
of its own lines, and no line is split, moved, or dropped. When no
confident split exists, the page is returned unchanged.
"""

from __future__ import annotations

from statistics import median
from typing import TYPE_CHECKING

from src.app.domain.models import Geometry, Provenance, TextRegion
from src.app.domain.models.status import EvidenceType, GeometryStatus
from src.app.enrichers import BaseEnricher
from src.app.geometry.bbox import union_all

if TYPE_CHECKING:
    from src.app.domain.models import CanonicalDocument, Page, TextLine
    from src.app.policies.document_policy import DocumentPolicy

# A line wider than this fraction of the text width spans the page and
# separates horizontal bands (title, full-width paragraphs).
_SPAN_FRACTION = 0.8
# Consecutive spanning lines closer than this multiple of the median line
# height share one band. A full-width paragraph keeps together; a heading
# with real space around it separates bands.
_SPAN_GAP_FACTOR = 1.2


class BlockSplitEnricher(BaseEnricher):
    @property
    def name(self) -> str:
        return "block_split"

    def enrich(
        self, doc: CanonicalDocument, policy: DocumentPolicy
    ) -> CanonicalDocument:
        if not policy.allow_reading_order_inference:
            return doc

        new_pages = []
        changed = False
        for page in doc.pages:
            new_page = self._split_page(page)
            if new_page is not page:
                changed = True
            new_pages.append(new_page)
        if changed:
            return doc.model_copy(update={"pages": new_pages})
        return doc

    def _split_page(self, page: Page) -> Page:
        # Only the synthesized single-region case. Pages with several
        # regions already carry layout from their provider. The builder
        # seeds reading_order with the single region id, so that alone
        # is not a signal that someone else owns the structure.
        if len(page.text_regions) != 1:
            return page
        region = page.text_regions[0]
        lines = sorted(
            region.lines,
            key=lambda ln: (ln.geometry.bbox[1], ln.geometry.bbox[0]),
        )
        groups = self._group_lines(lines)
        if len(groups) <= 1 or sum(len(g) for g in groups) != len(lines):
            return page

        new_regions: list[TextRegion] = []
        for i, group in enumerate(groups, start=1):
            group = sorted(
                group, key=lambda ln: (ln.geometry.bbox[1], ln.geometry.bbox[0])
            )
            new_regions.append(
                TextRegion(
                    id=f"tb{i}",
                    geometry=self._geometry(group),
                    provenance=self._provenance(region, group),
                    lang=region.lang,
                    lines=group,
                )
            )
        return page.model_copy(update={
            "text_regions": new_regions,
            "reading_order": [r.id for r in new_regions],
        })

    def _group_lines(self, lines: list[TextLine]) -> list[list[TextLine]]:
        """Partition lines into reading-order groups (bands and columns)."""
        median_h = median(ln.geometry.bbox[3] for ln in lines)
        min_x = min(ln.geometry.bbox[0] for ln in lines)
        max_x = max(ln.geometry.bbox[0] + ln.geometry.bbox[2] for ln in lines)
        text_w = max_x - min_x

        # Bands: maximal runs of same-kind lines, where a spanning band
        # closes when the vertical gap to the next spanning line is real.
        bands: list[tuple[bool, list[TextLine]]] = []
        for line in lines:
            bbox = line.geometry.bbox
            spanning = bbox[2] >= _SPAN_FRACTION * text_w
            if bands:
                is_spanning, current = bands[-1]
                gap = bbox[1] - max(
                    ln.geometry.bbox[1] + ln.geometry.bbox[3] for ln in current
                )
                if is_spanning == spanning and not (
                    spanning and gap > _SPAN_GAP_FACTOR * median_h
                ):
                    current.append(line)
                    continue
            bands.append((spanning, [line]))

        # Column bands split by x-range overlap; spanning bands stay whole.
        groups: list[list[TextLine]] = []
        for is_spanning, band_lines in bands:
            if is_spanning:
                groups.append(band_lines)
            else:
                groups.extend(self._cluster_columns(band_lines))
        return groups

    @staticmethod
    def _cluster_columns(band_lines: list[TextLine]) -> list[list[TextLine]]:
        """Cluster a band's lines into columns by x-range overlap."""
        clusters: list[list[TextLine]] = []
        cluster_max_x = float("-inf")
        for line in sorted(band_lines, key=lambda ln: ln.geometry.bbox[0]):
            x0, _, width, _ = line.geometry.bbox
            right = x0 + width
            if not clusters or x0 >= cluster_max_x:
                clusters.append([line])
                cluster_max_x = right
            else:
                clusters[-1].append(line)
                cluster_max_x = max(cluster_max_x, right)
        return clusters

    @staticmethod
    def _geometry(group: list[TextLine]) -> Geometry:
        return Geometry(
            bbox=union_all([ln.geometry.bbox for ln in group]),
            status=GeometryStatus.INFERRED,
        )

    @staticmethod
    def _provenance(
        region: TextRegion, group: list[TextLine]
    ) -> Provenance:
        return Provenance(
            provider=region.provenance.provider,
            adapter="enricher.block_split.v1",
            source_ref=region.provenance.source_ref,
            evidence_type=EvidenceType.DERIVED,
            derived_from=[ln.id for ln in group],
        )
