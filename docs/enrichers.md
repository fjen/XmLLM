# Enrichers

Enrichers run after normalization, before validation. Each one takes a `CanonicalDocument` and produces a new immutable document. The document policy controls what an enricher may do.

| Enricher | What it does | Provenance |
|---|---|---|
| `polygon_to_bbox` | Derives a bbox from the polygon when the geometry is `unknown` | `inferred` |
| `bbox_repair_light` | Clips bboxes that overflow the page boundaries | `repaired` |
| `lang_propagation` | Propagates the language from region and line to child nodes | unchanged |
| `reading_order_simple` | Infers the reading order from spatial position, top to bottom and left to right | `inferred` |
| `block_split` | Splits the single synthesized region into band/column regions from the line boxes, and writes the reading order from that tree | `derived` |
| `hyphenation_basic` | Marks a word that ends with `-` at a line end when the next line starts lowercase | `inferred` |
| `text_consistency` | Warns on blank or over-long words, and flags lines with engine confidence below 0.75 as possible ghost text | warnings only |

## Known limits

`block_split` works on line boxes only, so it knows no semantic roles. Column clusters split on disjoint x-ranges. A margin number that sits clear of the text column therefore becomes its own block. A line that bridges a column gutter merges the columns instead of guessing. A short paragraph-ending line can become its own block. The reading order stays correct in all these cases. The blocks follow geometry, not layout semantics.

A future layout-model pass (PP-DocLayout) would slot in as a region proposer feeding the same assignment. A note for that path: jobs persist only the JPEG preview of the input. A model re-run over an existing job works from a quality-85 JPEG, not the archival original. Its blocks must not claim provider-native evidence measured on the source scan.

`hyphenation_basic` uses a basic adjacency heuristic. It misses continuations that start uppercase, and it can mis-pair list hyphens. ADR 0002 records the observed cases and the consumer-side effect.

`text_consistency` flags lines whose engine confidence falls below 0.75. The engine scores text lines, not words, so this is a per-line signal. It is warnings only: the enricher never drops or changes text. The warnings surface in the viewer as validation flags. They do not reach the ALTO or PAGE export. The DFG-Viewer consumer reads only the attributes listed in ADR 0002, and `WC` is not among them. The companion setting `ALTO_MIN_LINE_CONFIDENCE` sets a policy rule. The export view removes such lines before serialization. The canonical document keeps everything.

Enrichers never invent text and never invent coordinates without a geometric basis. See `AGENTS.md` for the full rule set.
