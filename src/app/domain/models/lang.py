"""ISO 639-3 language code handling — shared contract for caller-declared langs."""

from __future__ import annotations

import re

# The canonical model validates lang against ^[a-z]{3}$ (ISO 639-3).
ISO_LANG_PATTERN = re.compile(r"^[a-z]{3}$")


def normalize_iso_lang(value: object) -> str | None:
    """Return `value` if it is a valid ISO 639-3 code, else None.

    Used for caller-declared languages (e.g. RawProviderPayload
    metadata["language"]). Anything invalid or non-string becomes None —
    the canonical model then keeps lang=None and serializers omit the
    LANG attribute rather than guessing.
    """
    if isinstance(value, str) and ISO_LANG_PATTERN.fullmatch(value):
        return value
    return None
