# Enrichers

Enrichers run after normalization, before validation. Each one takes a `CanonicalDocument` and produces a new immutable document. The document policy controls what an enricher may do.

| Enricher | What it does | Provenance |
|---|---|---|
| `polygon_to_bbox` | Derives a bbox from the polygon when the geometry is `unknown` | `inferred` |
| `bbox_repair_light` | Clips bboxes that overflow the page boundaries | `repaired` |
| `lang_propagation` | Propagates the language from region and line to child nodes | unchanged |
| `reading_order_simple` | Infers the reading order from spatial position, top to bottom and left to right | `inferred` |
| `hyphenation_basic` | Marks a word that ends with `-` at a line end when the next line starts lowercase | `inferred` |
| `text_consistency` | Warns on blank words or words longer than 100 characters | warnings only |

## Known limits

`hyphenation_basic` uses a basic adjacency heuristic. It misses continuations that start uppercase, and it can mis-pair list hyphens. ADR 0002 records the observed cases and the consumer-side effect.

Enrichers never invent text and never invent coordinates without a geometric basis. See `AGENTS.md` for the full rule set.
