# ALTO export targets the DFG-Viewer consumption path

We verified how the target consumer reads our ALTO. An agent built a mixed-page export from a real OCR payload (434 lines with reconstructed words, 7 line-level fallback lines) and checked two things: validation against the official ALTO v4 schema, and the source code of the DFG-Viewer (github.com/slub/dfg-viewer). We decided to keep the export shape and record the consumer contract here, because it constrains every future change to `alto_xml.py`.

## What the consumer does (verified in source)

The DFG-Viewer contains almost no ALTO logic. It delegates to Kitodo Presentation (`composer.json` requires `kitodo/presentation: ^7.1`). The parser is `Classes/Format/Alto.php` in that repository. The facts below cite that file.

- The parser registers namespaces ns-v2, ns-v3 and ns-v4 explicitly (`Alto.php:153-167`). Our `ns-v4` URI is supported.
- Both consumption paths (raw fulltext and the MiniOCR word list) select `Layout/Page/PrintSpace/TextBlock/TextLine/String` (`Alto.php:44` and `Alto.php:79`). Our structure matches.
- Raw fulltext joins each `String` `CONTENT` with a space (`Alto.php:50-62`). `SP` elements are ignored. Spacing is synthesized, so `<SP>` matters for the schema, not for this consumer.
- The parser never reads `WC`. Its attribute access covers `CONTENT`, `SUBS_TYPE`, `SUBS_CONTENT` (fulltext and hyphenation) and `HPOS`, `VPOS`, `WIDTH`, `HEIGHT` (MiniOCR coordinates) — nothing else. A quality flag carried only in `WC` therefore does not reach this consumer.
- Our `WC` value is the engine's line-level score, copied onto every `String` of that line (`word_box_json.py` has no per-word scores; the raw word entries carry text and bbox only). Consumers filtering on `WC` are filtering lines.
- With the policy rule `alto_min_line_confidence` set (value from `ALTO_MIN_LINE_CONFIDENCE`), the export view omits `TextLine` elements whose confidence falls below the value, and `TextBlock` elements that lose every line. Omitted lines never reach the consumer's fulltext or word list. The export carries an XML comment naming the count and the threshold. The default keeps every line; the canonical document is never filtered. The setting is read once per process, and the API serves the stored file without re-serializing: exports already on disk are unchanged. Reprocess (for example `batch_alto.py --no-skip-existing`) to regenerate them under a new threshold.
- Coordinates are copied verbatim: `getCoordinates` returns `HPOS`, `VPOS`, `WIDTH`, `HEIGHT` unchanged (`Alto.php:140-144`). Our `MeasurementUnit="pixel"` matches the page-pixel space we export.
- Element IDs are never read. The `_str` suffix on compatibility Strings is invisible to the consumer.
- Hyphenation: `getRawText` assumes a `HypPart1` String is immediately followed by its `HypPart2` in document order, and joins them (`Alto.php:52-62`). The serializer emits `SUBS_TYPE`/`SUBS_CONTENT` when the `HyphenationBasicEnricher` marks a pair. Marked pairs stay adjacent in export order: pairs come from consecutive lines, and words export in line order. The export view never removes a line that carries a hyphenation part, so a low score cannot orphan a pair. Verified on a real page: 12 pairs marked, 24 `SUBS_TYPE` attributes in the export, so the Kitodo join fires in practice.
- Raw fulltext hyphenation is therefore a best-effort join, with two error types observed on that page. Unmarked: line-end hyphens whose continuation starts uppercase stay space-separated — `Kran-` / `Branntwein`, also `Trichi-`, `Be-`, `Homöo-`, `Hy-`. Marked but wrong: adjacency plus a lowercase-start heuristic cannot prove a word break. Observed mis-joins, re-confirmed from the enrichment run: `Ge-` + `im` joins to "Geim" (a function word is not a continuation), `Min.-Wasser-` + `und` joins across an abbreviation period (a list hyphen, not a word break), `Schwei-` + `synthet.` joins with a part 2 that is itself an abbreviated token, and `Phosphor-` + `säuren,flüchtigeinranzig.Butter261.` joins into an OCR-mashed run-on. Expect both error types in fulltext on dense directory text.
- Improvement directions, deliberately not done here: never mark a pair whose hyphen follows an abbreviation period (`Min.-` is the clear case, `Schwei-` is ambiguous but `Min.-` is not). And check part 2 for plausibility (length, no embedded punctuation, case rules) before the serializer emits `SUBS_TYPE`.

## Export format details

The consumer facts above constrain these mappings.

### ALTO XML v4

- Namespace: `http://www.loc.gov/standards/alto/ns-v4#`
- Mapping: `Page` / `TextBlock` / `TextLine` / `String`
- Attributes: `HPOS`, `VPOS`, `WIDTH`, `HEIGHT` (integers), `CONTENT`, `WC`, `LANG`
- Hyphenation: `SUBS_TYPE` (`HypPart1` / `HypPart2`), `SUBS_CONTENT`
- `<Description>` carries the measurement unit, the filename and the processing software

### PAGE XML 2019

- Namespace: `http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15`
- Mapping: `TextRegion` / `TextLine` / `Word`
- Coordinates: `<Coords points="x1,y1 x2,y2 ...">` preserves polygons when available
- `<TextEquiv><Unicode>` at region, line and word levels
- `<ReadingOrder>` / `<OrderedGroup>` / `<RegionRefIndexed>`
- Region `@type` maps from the block role (paragraph, heading, footnote, ...)

### Key difference

ALTO uses axis-aligned bounding boxes with integer coordinates. PAGE XML uses polygons and preserves the original quadrilateral geometry from providers like PaddleOCR.

## Schema validation

The mixed-page export validates against the official `alto-4-2.xsd` (fetched from www.loc.gov/standards/alto/v4/alto-4-2.xsd) with zero errors. Note: the GitHub mirror `altoxml/schema` does not carry alto-4-2.xsd. LOC is the authoritative host.

Schema validation is not wired into CI. It needs a network fetch of the XSD. Re-run the validation by hand when the serializer changes.

## Consequences

- Compatibility Strings from the line-level fallback (ADR 0001) are safe for this consumer. Each renders as one word spanning the line box. A click or a highlight on a "word" in such a line covers the whole line. Word-precision interaction is lost on those lines only.
- Every `TextLine` must contain at least one `String`. The canonical invariant (words or transcription, never neither) guarantees this. Keep it.
- `SP` stays in the export. The XSD content model expects it between Strings, and other consumers do read it.
- Do not rely on element IDs in ALTO for anything consumer-facing. They are our bookkeeping.
