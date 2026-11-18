"""Shared test helpers."""

from __future__ import annotations

from typing import Any


def with_word_boxes(payload: list[Any]) -> list[Any]:
    """Add synthetic word boxes to a line-level PaddleOCR payload.

    Splits each line's text on whitespace and subdivides the line's
    bounding box horizontally, proportional to word length. The boxes are
    obviously synthetic test data, but they let feature tests exercise the
    word-level code paths (adapter word branch, one ALTO <String> per
    word, hyphenation detection, viewer word overlays) that line-level-
    only payloads skip under the line-level fallback (ADR 0001).
    """
    out: list[Any] = []
    for points, text_conf in payload:
        text, conf = text_conf[0], text_conf[1]
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        words = [w for w in str(text).split(" ") if w]
        total = sum(len(w) for w in words) or 1
        span = (right - left) / total
        x = left
        word_data = []
        for w in words:
            word_data.append([w, (x, top, x + span * len(w), bottom)])
            x += span * len(w)
        out.append([points, (text, conf, word_data)])
    return out
