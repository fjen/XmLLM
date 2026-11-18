"""PaddleOCR integration — runs OCR on an image and returns word_box_json payload."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ocr_instances: dict[str, Any] = {}

# Bound on cached PaddleOCR instances. Each holds a full model set, so an
# unbounded dict keyed by user-supplied lang values is a memory DoS in a
# long-lived server. LRU: hits are moved to the end so cycling many langs
# keeps the most recently used instead of reloading models every request.
_OCR_CACHE_LIMIT = 8


def _get_ocr(lang: str) -> Any:
    """Lazy-load PaddleOCR 3.x, one instance per language.

    Coordinate contract: rec_polys are in the ORIGINAL image's pixel space —
    the same space the viewer's `/jobs/{id}/image` serves. To keep that
    guarantee, the coordinate-altering preprocessors are disabled:
    `use_doc_unwarping=False` (UVDoc would return polys in dewarped space)
    and `use_doc_orientation_classify=False` (a rotated/upside-down upload
    would otherwise return polys in rotated space while the viewer still
    draws the unrotated bytes — silently wrong overlays). Rotated input
    must be corrected before upload; OCR quality on uncorrected rotated
    input degrades, which is honest and visible. Both flags also skip two
    model loads per instance.

    `enable_mkldnn=False` works around a paddle 3.3 oneDNN/PIR bug
    (ConvertPirAttribute2RuntimeAttribute) that crashes PP-OCRv6 inference.
    """
    if lang in _ocr_instances:
        instance = _ocr_instances.pop(lang)
        _ocr_instances[lang] = instance  # move to end → LRU
        return instance
    if len(_ocr_instances) >= _OCR_CACHE_LIMIT:
        _ocr_instances.pop(next(iter(_ocr_instances)))
    from paddleocr import PaddleOCR
    _ocr_instances[lang] = PaddleOCR(
        lang=lang,
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        enable_mkldnn=False,
    )
    return _ocr_instances[lang]


def run_paddle_ocr(image_path: Path, lang: str) -> list[Any]:
    """Run PaddleOCR (3.x `predict`) and return the word_box_json payload:
        [[4-point polygon, (text, confidence)], ...]
    """
    ocr = _get_ocr(lang)
    results = ocr.predict(str(image_path))
    payload: list[Any] = []
    for res in results:
        polys = res["rec_polys"]
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        for poly, text, score in zip(polys, texts, scores, strict=True):
            points = [[float(x), float(y)] for x, y in poly]
            payload.append([points, (str(text), float(score))])
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
