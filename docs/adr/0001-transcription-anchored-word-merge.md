# Transcription-anchored word reconstruction for PaddleOCR lines

PaddleOCR can return positioned text fragments (`return_word_box=True`). The fragment boundaries do not always match printed word boundaries. `Universitätsbibliothek` can arrive as three fragments: `Universit`, `ä`, `tsbibliothek`.

We reconstruct one word per whitespace-delimited token. The line transcription is the reference. We match the fragments to character spans in the transcription, in order, with a forward cursor. A repeated fragment binds to the correct occurrence, because the cursor only moves forward. Between two matches, you can skip only transcription whitespace.

Word boundaries come from the whitespace in the transcription. Geometry never decides a word boundary.

Each word gets its text from the transcription slice. Its box is the smallest axis-aligned rectangle that encloses the boxes of its fragments. The ALTO serializer writes this box as `HPOS`, `VPOS`, `WIDTH` and `HEIGHT`. It writes one `<String>` per word and `<SP>` between words.

We rejected matching on fragment boundaries. Fragments carry the separating spaces themselves. They can also drift from the transcription. The transcription is the only text we trust.

## How we match fragments

We trim leading and trailing whitespace from each fragment before we match it. The fragment ` ä ` maps to `ä`. It does not put spaces into `Universitätsbibliothek`. We keep internal whitespace.

A fragment with internal whitespace spans two tokens. Example: `Die Universität` covers two tokens. This is a fallback case.

The matching invariant: the ordered fragment matches cover every non-whitespace character in the reference line exactly once. Every matched fragment contributes to at most one token box. A violation is a fallback, never a guess. These cases violate the invariant:

- non-whitespace text before the first fragment, between fragments, or after the last fragment
- a fragment that spans two tokens

## Fallback

When the invariant fails, we do not derive word geometry. The line keeps its transcription and its line polygon. The canonical model has no word for this line. We do not put a fake word across the line.

The ALTO serializer writes a compatibility `<String>` with the line box. Full text survives the export. This `<String>` does not meet the word-coordinate requirement of the DFG Viewer.

We never estimate word coordinates. We never split a line polygon in proportion to whitespace.

Whitespace-only fragments have no ink. They belong to no word, and we skip them.

## Exclusion policy

`WORD_RECONSTRUCTION_DISABLED_LANGS` lists caller-declared languages with reconstruction off: `zho`, `jpn`, `kor`, `tha`. This is a conservative choice. It is not a claim about these scripts. Korean normally uses spaces between written units.

A line from one of these languages with no whitespace could otherwise become one line-sized token. The ambiguity checks would not catch this. The policy applies to the whole line as declared. This includes mixed-script lines. The page gets a warning like any other fallback.

## Considered options

- Raw Paddle fragments as words. Fully provider-native. But the "words" are sub-word character clusters. The `<String>` semantics are wrong downstream.
- Whitespace-proportional subdivision of the line box. No engine change. But it estimates word coordinates without fragment-level evidence. Rejected.
- One word per line (previous behaviour). No coordinate fabrication. But the word granularity misleads. Words equalled lines. ALTO and PAGE export was semantically wrong.

## Consequences

- The `word_box_json` payload item can carry `(text, confidence, words)` or `(text, confidence, None, reason)`. A flagged line carries `words=None` plus a reason. A line without any word data keeps the form `(text, confidence)` and takes the same line-level fallback.
- `TextLine` has a `transcription` field. It holds the provider's line transcription. The `text` property prefers it over the word join. Words can be empty only when a transcription is present.
- Readiness comes from the canonical data, not from the serialized XML. A compatibility `<String>` with the line box is not evidence of `WORD_TEXT` or `WORD_GEOMETRY`. Aggregation rule: a line with non-whitespace transcription text and no word-level data makes the page `PARTIAL`. Blank or whitespace-only lines do not change the page level.
- Word-level capabilities are not critical. The default policy can export such a page. `require_words_for_alto` (default off) blocks compatibility exports at the policy boundary. `allow_partial_alto` stays the coarse knob.
- Word boxes carry only a bbox. `polygon` is `None`. The polygon stays on the line. Word provenance is `derived` with source ref `$[i].text_word`.
- A document with a wrong excluded language loses word reconstruction. The policy trusts the caller-declared language.
- Line-end hyphenation (`Kran-`) stays its own token. We never merge fragments across lines.
- Coordinate spaces: `rec_polys` and fragment boxes are coordinates in the image that we pass to PaddleOCR. The adapter maps both into the pixel space of the image from `/jobs/{id}/image`. It uses the pipeline `resize_factor`.
- If you disable orientation correction and unwarping, the Paddle transforms disappear. But the adapter can resize or crop the input. Then the OCR input differs from the presented image. The `resize_factor` mapping covers exactly those transforms.
- `tests/e2e/test_coordinate_contract.py` tests this contract. It draws line polygons and word boxes on the exact image used for the ALTO presentation and checks ink. You supply a local image and payload pair with `scripts/generate_contract_fixture.py`. The repository contains no real scans. The test skips when you give no pair.
