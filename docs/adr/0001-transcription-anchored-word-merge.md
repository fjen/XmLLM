# Transcription-anchored word reconstruction for PaddleOCR lines

With `return_word_box=True`, PaddleOCR can return positioned text fragments
whose boundaries do not necessarily match printed word boundaries
(`Universitätsbibliothek` may arrive as `Universit`|`ä`|`tsbibliothek`).
We reconstruct one word per whitespace-delimited token, using the **line
transcription as the reference**: fragments are matched sequentially to
character spans in the transcription with a forward cursor; word boundaries
come from the transcription's whitespace — never from geometric gaps.

**Matching rules.** Leading and trailing whitespace is trimmed from each
fragment before matching (`" ä "` maps to `ä`; it does not force spaces
into `Universitätsbibliothek`); internal whitespace is preserved and makes
a fragment span multiple tokens (`"Die Universität"` covers two tokens →
fallback). Invariant: **every non-whitespace character in the reference
line must be covered exactly once by the ordered fragment matches, and
every matched fragment must contribute to at most one token's box.** A
violation — before the first fragment, between fragments (only
transcription whitespace may be skipped), after the last fragment, or a
fragment spanning two tokens — is a fallback, never a guess.

Each word's text is the transcription slice itself; its box is the
smallest axis-aligned rectangle enclosing its member fragment boxes,
emitted by the ALTO serializer as `HPOS`/`VPOS`/`WIDTH`/`HEIGHT`. One
`<String>` per reconstructed word, `<SP>` between them. Anchoring on
fragment boundaries instead was rejected because fragments carry the
separating spaces themselves and can drift from the transcription; the
transcription is the only trusted text.

**Fallback is a line-level transcription, not a word.** When the invariant
fails, no word geometry is derived. The line keeps its transcription and
line polygon in the canonical model — there is no fake word spanning the
line. The ALTO serializer emits a compatibility `<String>` with the line
box so full text survives export; that is a compatibility fallback and
does **not** meet the DFG Viewer's word-coordinate requirement. No
estimated subdivisions are ever presented as measured coordinates; the
same rule forbids whitespace-proportional splitting of a line polygon.
Whitespace-only fragments belong to no word (no ink) and are skipped.

**Exclusion policy, not a script classification.**
`WORD_RECONSTRUCTION_DISABLED_LANGS` (`zho`, `jpn`, `kor`, `tha`) lists
caller-declared languages for which whitespace-based reconstruction is
disabled — a deliberate conservative choice, not a claim about the scripts
(Korean normally uses spaces between written units). A line from one of
these languages **with no whitespace** could otherwise be treated as one
line-sized token without ever triggering the ambiguity checks. The
decision applies to the whole line as declared, including mixed-script
lines, and the page is flagged like any other fallback.

## Considered options

- **Raw Paddle fragments as words** — fully provider-native, but "words"
  are sub-word char clusters; wrong `<String>` semantics downstream.
- **Whitespace-proportional subdivision of the line box** — no engine
  change, but estimates word coordinates without fragment-level evidence;
  rejected.
- **One word per line (previous behaviour)** — no coordinate fabrication,
  but misleading word granularity; words equalled lines and ALTO/PAGE
  export was semantically wrong.

## Consequences

- `word_box_json` payload items may carry `(text, confidence, words)` or,
  for flagged lines, `(text, confidence, None, reason)`; lines without any
  word data keep the plain `(text, confidence)` form and take the same
  line-level fallback (other providers in the family).
- `TextLine` gained a `transcription` field (the provider's line-level
  transcription; `text` prefers it over the word join). Words may be
  empty only when a transcription is present.
- **Readiness is computed from the canonical data, not the serialized
  XML**: a compatibility `<String>` with the line box is evidence of
  neither `WORD_TEXT` nor `WORD_GEOMETRY`. Aggregation rule: any line
  containing **non-whitespace** transcription text that lacks word-level
  data makes the page PARTIAL; blank or whitespace-only lines do not.
  Word-level capabilities are not critical, so such a page remains
  exportable under the default policy. `DocumentPolicy.require_words_for_alto`
  (default off) is enforced at the policy boundary for operators who want
  to refuse compatibility exports; `allow_partial_alto` remains the
  coarse knob.
- Word boxes carry bbox-only geometry (`polygon=None`); polygons stay on
  the line. Word provenance is `derived` (`$[i].text_word`).
- A document mislabeled with an excluded language loses reconstruction —
  the policy trusts the caller-declared language.
- Line-end hyphenation (`Kran-`) stays its own token; fragments are never
  merged across lines.
- **Coordinate spaces, named explicitly**: `rec_polys` and fragment boxes
  are treated as coordinates in the image passed to PaddleOCR. The adapter
  maps both into the pixel space of the image served by
  `/jobs/{id}/image` (via the pipeline's `resize_factor`). Disabling
  Paddle's orientation correction and unwarping removes those transforms,
  but does not by itself make the OCR input identical to the presented
  image if the adapter resized or cropped the input — the mapping must
  account for exactly those transforms. The contract is tested in-repo by
  `tests/e2e/test_coordinate_contract.py`, which overlays both line
  polygons and word boxes on the exact image used for ALTO presentation
  (ink-overlap check). It runs against a locally supplied image + payload
  pair (`scripts/generate_contract_fixture.py`); real scans are never
  committed to the repository, so the test skips when no pair is
  provided.
