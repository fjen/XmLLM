"""word_box_json adapter — handles PaddleOCR and similar providers.

PaddleOCR output format (standard):
    [
        [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ("text", confidence)],
        ...
    ]

Each item is a detected text line with a 4-point polygon and
a (text, confidence[, words]) tuple.  Each item becomes one line; when the
engine supplies word-level boxes (the optional third tuple element) the
line gets one word per word box, otherwise it gets a single word spanning
the line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.app.domain.models import (
    CanonicalDocument,
    Geometry,
    Provenance,
    RawProviderPayload,
)
from src.app.domain.models.lang import normalize_iso_lang
from src.app.domain.models.status import (
    EvidenceType,
    GeometryStatus,
    InputType,
)
from src.app.geometry.bbox import union_all
from src.app.geometry.normalization import four_point_to_polygon, four_point_to_xywh
from src.app.normalization.canonical_builder import CanonicalBuilder
from src.app.providers.adapters.base import BaseAdapter

if TYPE_CHECKING:
    from src.app.domain.models.geometry import GeometryContext


class WordBoxJsonAdapter(BaseAdapter):
    """Adapter for the word_box_json family (PaddleOCR, etc.)."""

    @property
    def family(self) -> str:
        return "word_box_json"

    @property
    def version(self) -> str:
        return "adapter.word_box_json.v1"

    def normalize(
        self,
        raw: RawProviderPayload,
        geometry_context: GeometryContext,
        *,
        document_id: str,
        source_filename: str | None = None,
    ) -> CanonicalDocument:
        payload = raw.payload
        if not isinstance(payload, list):
            raise ValueError(
                f"word_box_json expects a list payload, got {type(payload).__name__}"
            )

        builder = CanonicalBuilder(
            document_id=document_id,
            input_type=InputType.IMAGE,
            filename=source_filename,
        )

        page = builder.add_page(
            page_id="p1",
            page_index=0,
            width=geometry_context.source_width,
            height=geometry_context.source_height,
        )

        # Collect all word bboxes for building a block-level bbox
        word_bboxes: list[tuple[float, float, float, float]] = []
        word_data: list[dict] = []

        for idx, item in enumerate(payload):
            points, text_conf = self._parse_item(item, idx)
            text, confidence, words, reason = self._parse_text_conf(text_conf, idx)

            bbox = four_point_to_xywh(points)
            polygon = four_point_to_polygon(points)

            # Apply resize factor if present
            if geometry_context.resize_factor and geometry_context.resize_factor != 1.0:
                factor = 1.0 / geometry_context.resize_factor
                bbox = (bbox[0] * factor, bbox[1] * factor, bbox[2] * factor, bbox[3] * factor)
                polygon = [(x * factor, y * factor) for x, y in polygon]

            # Clamp negative coordinates (real OCR data can produce these)
            bbox = (max(0.0, bbox[0]), max(0.0, bbox[1]), bbox[2], bbox[3])
            polygon = [(max(0.0, x), max(0.0, y)) for x, y in polygon]

            # Per-line word boxes ([l, t, r, b] → [x, y, w, h]) get the same
            # resize + clamp treatment as the line itself. Lines without word
            # data fall back to one word spanning the line (see build loop).
            line_words: list[dict] = []
            if words:
                for wtext, (left, top, right, bottom) in words:
                    if not wtext.strip():
                        continue
                    wb = (left, top, right - left, bottom - top)
                    if geometry_context.resize_factor and geometry_context.resize_factor != 1.0:
                        factor = 1.0 / geometry_context.resize_factor
                        wb = (wb[0] * factor, wb[1] * factor, wb[2] * factor, wb[3] * factor)
                    wb = (max(0.0, wb[0]), max(0.0, wb[1]), wb[2], wb[3])
                    line_words.append({"text": wtext.strip(), "bbox": wb})

            word_bboxes.append(bbox)
            word_data.append({
                "bbox": bbox,
                "polygon": polygon,
                "text": text,
                "confidence": confidence,
                "source_idx": idx,
                "words": line_words,
                "ambiguous_reason": reason if not line_words else None,
            })

        if not word_data:
            raise ValueError("word_box_json payload contains no items")

        # Optional caller-declared language (NOT provider-detected) via
        # payload metadata (ISO 639-3, e.g. from paddle_lang_to_iso639_3).
        # Anything invalid is silently ignored.
        lang = normalize_iso_lang(
            raw.metadata.get("language") if raw.metadata else None
        )

        # Build a single inferred text block: each item becomes one line,
        # with real word boxes when the engine supplies them (ADR 0001).
        block_bbox = union_all(word_bboxes)
        block_prov = Provenance(
            provider=raw.provider_id,
            adapter=self.version,
            source_ref="$",
            evidence_type=EvidenceType.DERIVED,
            derived_from=[f"w{i+1}" for i in range(len(word_data))],
        )

        region = page.add_text_region(
            region_id="tb1",
            geometry=Geometry(
                bbox=block_bbox,
                status=GeometryStatus.INFERRED,
            ),
            provenance=block_prov,
            lang=lang,
        )

        for i, wd in enumerate(word_data):
            line_id = f"tl{i + 1}"

            prov = Provenance(
                provider=raw.provider_id,
                adapter=self.version,
                source_ref=f"$[{wd['source_idx']}]",
                evidence_type=EvidenceType.PROVIDER_NATIVE,
            )

            geo = Geometry(
                bbox=wd["bbox"],
                polygon=wd["polygon"],
                status=GeometryStatus.EXACT,
            )

            line = region.add_line(
                line_id,
                geometry=geo,
                provenance=prov,
                confidence=wd["confidence"],
                lang=lang,
                transcription=wd["text"],
            )

            if wd["words"]:
                # Real word boxes from the engine. Each box is the smallest
                # axis-aligned rectangle enclosing that word's fragment
                # boxes (measured provider output), so geometry stays
                # EXACT; the transcription-anchored grouping that produced
                # the words is recorded as derived evidence.
                word_prov = Provenance(
                    provider=raw.provider_id,
                    adapter=self.version,
                    source_ref=f"$[{wd['source_idx']}].text_word",
                    evidence_type=EvidenceType.DERIVED,
                    derived_from=[],
                )
                for j, w in enumerate(wd["words"]):
                    line.add_word(
                        f"w{i + 1}_{j + 1}",
                        text=w["text"],
                        geometry=Geometry(
                            bbox=w["bbox"],
                            polygon=None,
                            status=GeometryStatus.EXACT,
                        ),
                        provenance=word_prov,
                        confidence=wd["confidence"],
                        lang=lang,
                    )
            # No word data (or ambiguous alignment): line-level fallback
            # (ADR 0001). The line keeps its transcription and line polygon
            # with NO word geometry — no fake word spanning the line. The
            # ALTO serializer emits a compatibility <String> with the line
            # box so full text survives export; that fallback does not
            # meet the DFG Viewer's word-coordinate requirement.

        # Lines whose word fragments could not be mapped to the transcription
        # keep the line-level word (original data retained); flag the page
        # once instead of once per line.
        ambiguous = [wd for wd in word_data if wd["ambiguous_reason"]]
        if ambiguous:
            page.add_warning(
                f"{len(ambiguous)} of {len(word_data)} lines: "
                f"{ambiguous[0]['ambiguous_reason']}"
            )

        return builder.build()

    # -- Private helpers ---------------------------------------------------------

    @staticmethod
    def _parse_item(item: object, idx: int) -> tuple[list, object]:
        """Extract (points, text_conf) from a PaddleOCR result item."""
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"Item {idx}: expected [points, (text, conf)], got {type(item).__name__}"
            )
        points = item[0]
        text_conf = item[1]
        if not isinstance(points, list) or len(points) != 4:
            got = len(points) if isinstance(points, list) else type(points).__name__
            raise ValueError(
                f"Item {idx}: expected 4 polygon points, got {got}"
            )
        return points, text_conf

    @staticmethod
    def _parse_text_conf(
        text_conf: object, idx: int
    ) -> tuple[
        str,
        float | None,
        list[tuple[str, tuple[float, float, float, float]]] | None,
        str | None,
    ]:
        """Extract (text, confidence, words, ambiguous_reason).

        `words` is the optional per-line word list carried by engines that
        provide word-level boxes (PaddleOCR with return_word_box): a list of
        [word_text, [left, top, right, bottom]]. `ambiguous_reason` is set
        when the engine had word data but it could not be mapped to the line
        transcription (words is None then) — the caller keeps the line-level
        transcription with no word geometry and flags the line. Two-element
        tuples and bare strings have neither (None, None) — same fallback.
        """
        if isinstance(text_conf, (list, tuple)) and len(text_conf) in (2, 3, 4):
            text = str(text_conf[0])
            try:
                confidence = float(text_conf[1])
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = None
            words: list[tuple[str, tuple[float, float, float, float]]] | None = None
            if len(text_conf) >= 3 and text_conf[2] is not None:
                raw_words = text_conf[2]
                if not isinstance(raw_words, (list, tuple)):
                    raise ValueError(
                        f"Item {idx}: expected word list, got {type(raw_words).__name__}"
                    )
                words = []
                for j, entry in enumerate(raw_words):
                    if (
                        not isinstance(entry, (list, tuple))
                        or len(entry) != 2
                        or not isinstance(entry[1], (list, tuple))
                        or len(entry[1]) != 4
                    ):
                        raise ValueError(
                            f"Item {idx} word {j}: expected [text, [l, t, r, b]]"
                        )
                    try:
                        box = tuple(float(v) for v in entry[1])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Item {idx} word {j}: non-numeric box"
                        ) from exc
                    words.append((str(entry[0]), box))
                if not words:
                    words = None
            reason: str | None = None
            if len(text_conf) == 4 and text_conf[3] is not None:
                reason = str(text_conf[3])
            return text, confidence, words, reason
        if isinstance(text_conf, str):
            return text_conf, None, None, None
        raise ValueError(
            f"Item {idx}: expected (text, confidence[, words[, reason]]) or "
            f"text string, got {type(text_conf).__name__}"
        )
