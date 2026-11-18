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
    """Lazy-load PaddleOCR (heavy import, ~2s first time), one instance per language."""
    if lang in _ocr_instances:
        instance = _ocr_instances.pop(lang)
        _ocr_instances[lang] = instance  # move to end → LRU
        return instance
    if len(_ocr_instances) >= _OCR_CACHE_LIMIT:
        _ocr_instances.pop(next(iter(_ocr_instances)))
    from paddleocr import PaddleOCR
    _ocr_instances[lang] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _ocr_instances[lang]


def run_paddle_ocr(image_path: Path, lang: str) -> list[Any]:
    """Run PaddleOCR on an image file.

    Args:
        image_path: image to process.
        lang: PaddleOCR language code (e.g. 'en', 'fr', 'de').

    Returns the raw PaddleOCR result in word_box_json format:
        [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ("text", confidence)]
    """
    ocr = _get_ocr(lang)
    results = ocr.ocr(str(image_path), cls=True)
    if not results or not results[0]:
        return []
    return results[0]


def paddle_result_to_payload(results: list[Any]) -> list[Any]:
    """Convert PaddleOCR results to the word_box_json payload format."""
    payload = []
    for item in results:
        points = item[0]
        text, conf = item[1]
        payload.append([points, [text, conf]])
    return payload


# PaddleOCR language codes → ISO 639-3 (the canonical model requires
# ^[a-z]{3}$, and ALTO LANG expects ISO 639-2/3). Keys verified against
# paddleocr 2.8–2.10 `parse_lang` aliasing (2-letter codes map to script
# groups; plus direct keys and legacy aliases). Codes PaddleOCR 2.x does
# not accept (e.g. fi, ja, ko, zh) are deliberately absent. Unmapped
# codes return None — lang stays unset and ALTO omits LANG rather than
# guessing.
_PADDLE_LANG_TO_ISO3 = {
    "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita",
    "pt": "por", "nl": "nld", "pl": "pol", "sv": "swe", "ru": "rus",
    "uk": "ukr", "bg": "bul", "cs": "ces", "da": "dan",
    "hu": "hun", "ro": "ron", "tr": "tur", "ar": "ara", "hi": "hin",
    "ch": "zho", "ch_doc": "zho", "chinese_cht": "zho",
    "japan": "jpn", "korean": "kor", "french": "fra", "german": "deu",
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
