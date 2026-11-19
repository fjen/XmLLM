"""PaddleOCR integration — runs OCR on an image and returns word_box_json payload."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.app.settings import get_settings

logger = logging.getLogger(__name__)


class OcrPayloadTooLarge(RuntimeError):
    """The remote OCR service rejected the upload as too large.

    Subclasses RuntimeError so existing engine-error handlers keep
    working; the /ocr route maps it to an honest 413 instead of 500.
    """

_ocr_instances: dict[str, Any] = {}

# Bound on cached PaddleOCR instances. Each holds a full model set, so an
# unbounded dict keyed by user-supplied lang values is a memory DoS in a
# long-lived server. LRU: hits are moved to the end so cycling many langs
# keeps the most recently used instead of reloading models every request.
_OCR_CACHE_LIMIT = 8


def _get_ocr(lang: str) -> Any:
    """Lazy-load PaddleOCR 3.x, one instance per language.

    Coordinate contract: rec_polys and word fragment boxes are treated as
    coordinates in the image passed to PaddleOCR; the adapter maps both
    into the pixel space of the image served by `/jobs/{id}/image` (via
    the pipeline's `resize_factor`). To keep that mapping exact, the
    coordinate-altering preprocessors are disabled:
    `use_doc_unwarping=False` (UVDoc would return polys in dewarped space)
    and `use_doc_orientation_classify=False` (a rotated/upside-down upload
    would otherwise return polys in rotated space while the viewer still
    draws the unrotated bytes — silently wrong overlays). Disabling these
    does not by itself make the OCR input identical to the presented image
    if the adapter resized or cropped the input — the resize_factor
    mapping accounts for exactly those transforms. Rotated input must be
    corrected before upload; OCR quality on uncorrected rotated input
    degrades, which is honest and visible. Both flags also skip two model
    loads per instance.

    Scope of the guarantee: this only removes Paddle's own orientation/
    unwarping transforms. The contract is verified by
    tests/e2e/test_coordinate_contract.py, which overlays both line
    polygons and word boxes on the exact image used for ALTO presentation
    (ink-overlap check on a locally supplied image + payload pair).

    `enable_mkldnn=False` works around a paddle 3.3 oneDNN/PIR bug
    (ConvertPirAttribute2RuntimeAttribute) that crashes PP-OCRv6 inference.
    """
    if lang in _ocr_instances:
        instance = _ocr_instances.pop(lang)
        _ocr_instances[lang] = instance  # move to end → LRU
        return instance
    if len(_ocr_instances) >= _OCR_CACHE_LIMIT:
        _ocr_instances.pop(next(iter(_ocr_instances)))
    import paddle
    from paddleocr import PaddleOCR

    settings = get_settings()
    device = settings.ocr_device
    kwargs: dict[str, Any] = {
        "lang": lang,
        "use_textline_orientation": True,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "enable_mkldnn": False,
        "return_word_box": True,
    }
    if device:
        kwargs["device"] = device
    if settings.ocr_recognition_batch_size is not None:
        kwargs["text_recognition_batch_size"] = settings.ocr_recognition_batch_size
    if settings.ocr_textline_orientation is not None:
        kwargs["use_textline_orientation"] = settings.ocr_textline_orientation
    _ocr_instances[lang] = PaddleOCR(**kwargs)
    logger.info(
        "PaddleOCR[%s]: CUDA build=%s, device=%s",
        lang, paddle.is_compiled_with_cuda(), device or "auto",
    )
    return _ocr_instances[lang]


def merge_paddle_words(
    line_text: str,
    fragments: list[tuple[str, list[Any]]],
) -> list[tuple[str, tuple[float, float, float, float]]] | None:
    """Reconstruct whitespace-delimited words from PaddleOCR fragments.

    With ``return_word_box=True``, PaddleOCR can return positioned text
    fragments whose boundaries do not necessarily match printed word
    boundaries ('Universitätsbibliothek' may arrive as
    'Universit'|'ä'|'tsbibliothek'). The line transcription is the
    reference: fragments are matched to character spans in it, in order,
    so repeated fragments bind to the right occurrence; word boundaries
    come from the transcription's whitespace, never from geometric gaps.

    Matching rules: leading and trailing whitespace is trimmed from each
    fragment before matching (a fragment ``" ä "`` maps to ``ä``, it does
    not force spaces into ``Universitätsbibliothek``); internal whitespace
    is preserved and makes the fragment span multiple tokens (a fragment
    ``"Die Universität"`` covers two tokens → fallback). Invariant: every
    non-whitespace character in the reference line must be covered exactly
    once by the ordered fragment matches, and every matched fragment must
    contribute to at most one token's box. If either condition fails, the
    caller retains the line transcription and polygon, reports the reason,
    and emits no canonical words.

    Returns one (word_text, (left, top, right, bottom)) per whitespace-
    delimited token. The word's text is the transcription slice itself and
    its box is the smallest axis-aligned rectangle enclosing the member
    fragment boxes — or None when the mapping is ambiguous:

    - a fragment's text does not occur at the expected position in the
      transcription (recognition drift, reordering, dropped characters),
    - a fragment with non-whitespace content spans more than one word (its
      box cannot be subdivided into measured per-word boxes),
    - the fragments leave non-whitespace transcription text uncovered
      (before the first fragment, between fragments, or as suffix).
    On None the caller must keep the line-level data and flag the line —
    no estimated word boundaries or coordinates. Line-end hyphenation
    ('Kran-') stays its own token; fragments are never merged across
    lines.
    """
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", line_text)]
    if not word_spans:
        return None

    # Sequential matching with a forward cursor: each fragment must occur at
    # or after the previous fragment's end, so repeated fragments map to the
    # correct occurrence. Only transcription whitespace may be skipped
    # between matches — reaching a fragment across non-whitespace text means
    # the mapping is broken (e.g. ["aaa"] must not bind to the second "aaa"
    # of "aaa aaa" and quietly leave the first one uncovered).
    frag_spans: list[tuple[int, int, list[Any]]] = []
    cursor = 0
    for text, box in fragments:
        frag = text.strip()
        if not frag:
            continue  # whitespace-only fragment: no ink, belongs to no word
        start = line_text.find(frag, cursor)
        if start < 0:
            return None
        if line_text[cursor:start].strip():
            return None  # match skips non-whitespace transcription text
        end = start + len(frag)
        cursor = end
        frag_spans.append((start, end, box))

    if line_text[cursor:].strip():
        return None  # fragments leave non-whitespace transcription text

    # A fragment shared by two words cannot yield measured per-word boxes.
    # A fragment covering only whitespace belongs to no word (it has no
    # ink) and is skipped — that is not an error. A non-whitespace fragment
    # outside every word span, however, means the mapping is broken.
    for f_start, f_end, _ in frag_spans:
        overlapping = sum(1 for w_start, w_end in word_spans
                          if f_start < w_end and f_end > w_start)
        if overlapping > 1:
            return None
        if overlapping == 0 and line_text[f_start:f_end].strip():
            return None

    words: list[tuple[str, tuple[float, float, float, float]]] = []
    for w_start, w_end in word_spans:
        members = [box for f_start, f_end, box in frag_spans
                   if f_start < w_end and f_end > w_start]
        if not members:
            return None
        left = min(float(b[0]) for b in members)
        top = min(float(b[1]) for b in members)
        right = max(float(b[2]) for b in members)
        bottom = max(float(b[3]) for b in members)
        words.append((line_text[w_start:w_end], (left, top, right, bottom)))
    return words


def run_paddle_ocr(image_path: Path, lang: str) -> list[Any]:
    """Run PaddleOCR and return the word_box_json payload.

    Dispatches to the remote OCR service (`OCR_SERVICE_URL`, served by
    scripts/ocr_service.py) when configured, and runs local inference
    otherwise. Both paths return the same payload in the same coordinate
    space:
        [[4-point polygon, (text, confidence[, words[, ambiguous_reason]])], ...]

    With `return_word_box=True` each item's third element carries words
    reconstructed from the line transcription (see merge_paddle_words) in
    the same original-image pixel space as the line polygon. When a line's
    fragments cannot be mapped to the transcription, the item carries
    words=None plus an `ambiguous_reason` instead; the adapter keeps the
    line-level word and adds a document warning. Items without word data
    at all keep the two-element form (adapter falls back to one word per
    line).
    """
    service_url = get_settings().ocr_service_url
    if service_url:
        return _run_remote_ocr(image_path, lang, service_url)
    return run_paddle_ocr_local(image_path, lang)


def _run_remote_ocr(image_path: Path, lang: str, service_url: str) -> list[Any]:
    """Post the image to the remote OCR service and return its payload.

    The service computes line polygons and word boxes in the uploaded
    image's pixel space (same bytes, same coordinates), so the viewer
    contract holds. Engine failures raise RuntimeError, which the job
    pipeline maps to 422 like any other engine error.
    """
    import httpx

    url = service_url.rstrip("/") + "/ocr"
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f)}
        try:
            response = httpx.post(
                url, files=files, params={"lang": lang}, timeout=600.0
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OCR service unreachable at {url}: {exc}") from exc
    if response.status_code == 413:
        # The request passed our own gate but the service's limit is
        # lower: surface it as 413 instead of an opaque server error.
        raise OcrPayloadTooLarge(
            f"OCR service rejected the upload as too large "
            f"({response.text[:200] or 'limit reached'})"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"OCR service {url} failed: HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"OCR service {url} returned non-list payload")
    return payload


def run_paddle_ocr_local(image_path: Path, lang: str) -> list[Any]:
    """Run local PaddleOCR (3.x `predict`) inference."""
    ocr = _get_ocr(lang)
    results = ocr.predict(str(image_path))
    iso3 = paddle_lang_to_iso639_3(lang)
    payload: list[Any] = []
    for res in results:
        polys = res["rec_polys"]
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        words = res.get("text_word")
        word_boxes = res.get("text_word_boxes")
        have_words = (
            words is not None
            and word_boxes is not None
            and len(words) == len(polys)
            and len(word_boxes) == len(polys)
        )
        for i, (poly, text, score) in enumerate(
            zip(polys, texts, scores, strict=True)
        ):
            points = [[float(x), float(y)] for x, y in poly]
            line_text = str(text)
            if not line_text.strip():
                # No text to align: keep the plain two-element form. The
                # adapter drops this detection; a fragmentation reason
                # would misdescribe it.
                payload.append([points, (line_text, float(score))])
                continue
            if iso3 in WORD_RECONSTRUCTION_DISABLED_LANGS:
                # Exclusion policy (ADR 0001): no reconstruction, the line
                # keeps its transcription, the page is flagged.
                payload.append([
                    points,
                    (
                        line_text,
                        float(score),
                        None,
                        f"word reconstruction disabled for language "
                        f"'{iso3}' (exclusion policy, ADR 0001)",
                    ),
                ])
                continue
            if have_words:
                fragments = [
                    (str(tok), box)
                    for tok, box in zip(words[i], word_boxes[i], strict=True)
                ]
                merged = merge_paddle_words(line_text, fragments)
                if merged is not None:
                    payload.append([points, (line_text, float(score), merged)])
                    continue
                # Ambiguous mapping: retain the line-level data, flag it.
                payload.append([
                    points,
                    (
                        line_text,
                        float(score),
                        None,
                        "word fragments do not align with the line "
                        "transcription; word boundaries not derived",
                    ),
                ])
                continue
            payload.append([points, (line_text, float(score))])
    return payload


# PaddleOCR language codes → ISO 639-3 (the canonical model requires
# ^[a-z]{3}$, and ALTO LANG expects ISO 639-2/3). Keys verified against
# paddleocr 3.7 model selection (_pipelines/ocr.py): PP-OCRv6 langs
# ({ch, chinese_cht, en, japan} ∪ LATIN_LANGS) plus the PP-OCRv5/v3
# script codes (korean, th, el, te, ta, ka, and the script groups).
# ch_doc existed only in 2.x model dicts and is deliberately absent.
# Unmapped codes return None — lang stays unset and ALTO omits LANG
# rather than guessing.
_PADDLE_LANG_TO_ISO3 = {
    "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita",
    "pt": "por", "nl": "nld", "pl": "pol", "sv": "swe", "ru": "rus",
    "uk": "ukr", "bg": "bul", "cs": "ces", "da": "dan", "fi": "fin",
    "hu": "hun", "ro": "ron", "tr": "tur", "ar": "ara", "hi": "hin",
    "ch": "zho", "chinese_cht": "zho", "japan": "jpn", "korean": "kor",
    "french": "fra", "german": "deu", "th": "tha", "el": "ell",
    "te": "tel", "ta": "tam", "ka": "kat",
}


_ISO3_CODES = frozenset(_PADDLE_LANG_TO_ISO3.values())

# Exclusion policy (ADR 0001): caller-declared languages for which
# whitespace-based word reconstruction is disabled — a deliberate
# conservative choice, not a claim about the scripts (Korean normally
# uses spaces between written units). Whitespace splitting would merge
# such a line into one line-sized token without ever triggering the
# ambiguity checks. The decision applies to the whole line as declared,
# including mixed-script lines.
WORD_RECONSTRUCTION_DISABLED_LANGS = frozenset({"zho", "jpn", "kor", "tha"})

# Allowlist of language codes accepted by POST /ocr and scripts/batch_alto.py.
# Bounds the code space before it reaches the engine or the instance cache.
SUPPORTED_PADDLE_LANGS: frozenset[str] = frozenset(_PADDLE_LANG_TO_ISO3)

# Image file extensions accepted as OCR input — shared by POST /ocr,
# scripts/batch_alto.py, and FileStore.get_input_image_path so every
# surface accepts the same formats.
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
)


def paddle_lang_to_iso639_3(lang: str) -> str | None:
    """Map a PaddleOCR language code to ISO 639-3 for canonical `lang` fields.

    Accepts the mapped table keys (2-letter codes and PaddleOCR legacy
    aliases) and ISO 639-3 codes the table already produces (e.g. 'fra',
    'eng') — anything else returns None rather than guessing.
    """
    code = lang.strip().lower()
    if code in _ISO3_CODES:
        return code
    return _PADDLE_LANG_TO_ISO3.get(code)
