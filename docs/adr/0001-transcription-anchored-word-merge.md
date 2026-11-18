# Transcription-anchored word reconstruction for PaddleOCR lines

PaddleOCR's `return_word_box=True` emits one box per recognition column —
sub-word fragments (`Universitätsbibliothek` → `Universit`|`ä`|`tsbibliothek`),
not linguistic words. We reconstruct one word per whitespace-delimited token,
using the **line transcription as the reference**: fragments are matched
sequentially to character spans in the transcription (so repeated fragments
bind to the right occurrence), word boundaries come from the transcription's
whitespace — never from geometric gaps — and a word's box is the union of its
member fragment boxes (`left = min(x)`, `top = min(y)`, `right = max(x)`,
`bottom = max(y)`), which the ALTO serializer emits as
`HPOS`/`VPOS`/`WIDTH`/`HEIGHT`. One `<String>` per reconstructed word, `<SP>`
between them. The alternative of anchoring on fragment boundaries was
rejected because fragments carry the separating spaces themselves and can
drift from the transcription; the transcription is the only trusted text.

**Ambiguity policy (deliberate, do not "fix"):** when a line's fragments
leave non-whitespace transcription text uncovered, cannot be matched in
order, or a single fragment spans more than one word, the per-word boxes
are *not* derived — the line keeps its line-level data (one word spanning
the line, original transcription preserved) and the page gets an
aggregated warning. Whitespace-only fragments belong to no word (no ink)
and are skipped. No estimated
subdivisions are ever presented as measured coordinates; the same rule
forbids whitespace-proportional splitting of a line polygon.

## Considered options

- **Raw Paddle tokens as words** — fully provider-native, but "words" are
  sub-word char clusters; wrong `<String>` semantics for downstream consumers.
- **Whitespace-proportional subdivision of the line box** — no engine change,
  but invents coordinates without measured basis; rejected on the
  no-invented-geometry invariant.
- **Keep one word per line (previous behaviour)** — zero risk, but words
  equalled lines and ALTO/PAGE export was semantically wrong.

## Consequences

- `word_box_json` payload items may carry `(text, confidence, words)` or, for
  ambiguous lines, `(text, confidence, None, reason)`; lines without any word
  data keep the plain `(text, confidence)` form and fall back to one word per
  line (other providers in the family).
- CJK lines (per-character fragments, no whitespace) may often be ambiguous →
  line-level fallback with warning; that is honest, not a bug.
- Line-end hyphenation (`Kran-`) stays its own token; fragments are never
  merged across lines.
- Word boxes are in the same original-image pixel space as the line polygons;
  the doc-orientation/unwarping preprocessors stay disabled to preserve that
  (see `_get_ocr` coordinate contract).
