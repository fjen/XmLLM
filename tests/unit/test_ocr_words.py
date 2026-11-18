"""Tests for transcription-anchored PaddleOCR fragment merging."""

from __future__ import annotations

import pytest
from src.app.ocr import merge_paddle_words


def box(left: float, t: float, r: float, b: float) -> list[float]:
    return [left, t, r, b]


class TestMergePaddleWords:
    def test_spec_example(self) -> None:
        """Line transcription is the reference; fragments merge per token."""
        words = merge_paddle_words(
            "Die Universitätsbibliothek öffnet.",
            [
                ("Die", box(10, 10, 30, 30)),
                ("Universit", box(35, 10, 80, 30)),
                ("ä", box(80, 10, 88, 30)),
                ("tsbibliothek", box(88, 10, 160, 30)),
                ("öffnet.", box(165, 10, 200, 30)),
            ],
        )
        assert [w for w, _ in words] == [
            "Die", "Universitätsbibliothek", "öffnet.",
        ]
        # Union of the three fragment boxes of the long word
        assert words[1][1] == (35.0, 10.0, 160.0, 30.0)
        # Punctuation stays attached to its token
        assert words[2][1] == (165.0, 10.0, 200.0, 30.0)

    def test_reconstruction_matches_reference(self) -> None:
        ref = "Thüringen. Arzneitaxe 15, Kran-"
        tokens = ["Th", "ü", "ringen", ". ", "Arzneitaxe", "1", "5", ", ", "Kran-"]
        boxes = [box(i * 10, 0, i * 10 + 9, 10) for i in range(len(tokens))]
        words = merge_paddle_words(ref, list(zip(tokens, boxes, strict=True)))
        assert words is not None
        assert " ".join(w for w, _ in words) == " ".join(ref.split())
        # Line-end hyphenation stays its own token
        assert words[-1][0] == "Kran-"

    def test_repeated_fragments_bind_by_position(self) -> None:
        words = merge_paddle_words(
            "a a", [("a", box(0, 0, 5, 5)), ("a", box(10, 0, 15, 5))]
        )
        assert words == [("a", (0.0, 0.0, 5.0, 5.0)), ("a", (10.0, 0.0, 15.0, 5.0))]

    def test_unknown_fragment_is_ambiguous(self) -> None:
        assert merge_paddle_words("abc", [("ax", box(0, 0, 9, 9))]) is None

    def test_reordered_fragments_are_ambiguous(self) -> None:
        assert merge_paddle_words(
            "ab c", [("c", box(20, 0, 25, 9)), ("ab", box(0, 0, 15, 9))]
        ) is None

    def test_match_may_not_skip_non_whitespace_text(self) -> None:
        # ["aaa"] must not bind to the second "aaa" of "aaa aaa" and
        # quietly leave the first occurrence uncovered.
        assert merge_paddle_words("aaa aaa", [("aaa", box(0, 0, 9, 9))]) is None

    def test_whitespace_gap_between_matches_is_allowed(self) -> None:
        words = merge_paddle_words(
            "aaa aaa",
            [("aaa", box(0, 0, 9, 9)), ("aaa", box(12, 0, 21, 9))],
        )
        assert words == [
            ("aaa", (0.0, 0.0, 9.0, 9.0)),
            ("aaa", (12.0, 0.0, 21.0, 9.0)),
        ]

    def test_fragment_whitespace_is_trimmed_before_matching(self) -> None:
        # A fragment " ä " maps to ä — no spaces forced into the word.
        words = merge_paddle_words(
            "Universität",
            [
                ("Universit", box(0, 0, 40, 9)),
                (" ä ", box(40, 0, 48, 9)),
                ("t", box(48, 0, 52, 9)),
            ],
        )
        assert words == [("Universität", (0.0, 0.0, 52.0, 9.0))]

    def test_fragment_with_internal_space_spans_two_tokens(self) -> None:
        # "Die Universität" covers two tokens — measured per-word boxes
        # are impossible, so the line is ambiguous.
        assert merge_paddle_words(
            "Die Universität",
            [("Die Universität", box(0, 0, 60, 9))],
        ) is None

    def test_fragment_spanning_two_words_is_ambiguous(self) -> None:
        """No estimated subdivisions — flag instead."""
        assert merge_paddle_words("a b", [("a b", box(0, 0, 20, 9))]) is None

    def test_incomplete_tiling_is_ambiguous(self) -> None:
        assert merge_paddle_words("ab c", [("ab", box(0, 0, 15, 9))]) is None

    def test_whitespace_only_fragment_between_words_is_skipped(self) -> None:
        words = merge_paddle_words(
            "a b",
            [("a", box(0, 0, 4, 9)), (" ", box(4, 0, 7, 9)), ("b", box(8, 0, 12, 9))],
        )
        assert words == [("a", (0.0, 0.0, 4.0, 9.0)), ("b", (8.0, 0.0, 12.0, 9.0))]

    def test_uncovered_trailing_space_is_not_ambiguous(self) -> None:
        words = merge_paddle_words("hi ", [("hi", box(0, 0, 8, 9))])
        assert words == [("hi", (0.0, 0.0, 8.0, 9.0))]

    def test_empty_fragment_skipped(self) -> None:
        words = merge_paddle_words("a", [("", box(0, 0, 0, 0)), ("a", box(0, 0, 5, 5))])
        assert words == [("a", (0.0, 0.0, 5.0, 5.0))]

    def test_empty_line_is_ambiguous(self) -> None:
        assert merge_paddle_words("", []) is None

    def test_trailing_space_tiled(self) -> None:
        """A fragment carrying the trailing space still tiles the line."""
        words = merge_paddle_words(
            "hi ", [("hi ", box(0, 0, 10, 9))]
        )
        assert words == [("hi", (0.0, 0.0, 10.0, 9.0))]

    def test_non_numeric_box_raises(self) -> None:
        with pytest.raises(ValueError):
            merge_paddle_words("a", [("a", ["x", 0, 1, 2])])
