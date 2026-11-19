# The canonical document

The `CanonicalDocument` is the central model of XmLLM. It represents what the system knows about a page, not what a specific model produced. Adapters fill it, enrichers refine it, validators check it, and the serializers derive ALTO XML and PAGE XML from it.

## Hierarchy

```
CanonicalDocument
  └── Page[]
       ├── TextRegion[] (blocks)
       │    └── TextLine[]
       │         └── Word[]
       └── NonTextRegion[] (illustrations, tables, separators)
```

## Every node carries

- **geometry**: `bbox` as `(x, y, width, height)`, an optional `polygon`, and a `status` (`exact`, `inferred`, `repaired`, `unknown`)
- **provenance**: `provider`, `adapter`, `source_ref`, `evidence_type` (`provider_native`, `derived`, `repaired`, `manual`), and `derived_from`
- **metadata**: an extensible `dict` for future fields without schema changes

## Geometry conventions

| Convention | Value |
|---|---|
| bbox format | `(x, y, width, height)` |
| Coordinate origin | `top_left` |
| Unit | `px` |
| Polygon | `list[tuple[float, float]]` or `None` |

The adapter converts providers that return `(x1, y1, x2, y2)`. No serializer performs implicit geometry conversion.

## Provenance rules

Every node states how the system obtained its data. The evidence types:

| Type | Meaning |
|---|---|
| `provider_native` | Measured provider output, unchanged |
| `derived` | Computed from provider output, for example the transcription-anchored word boxes (ADR 0001) |
| `repaired` | Reserved. Repairs today surface in the geometry `status`, not the evidence type |
| `manual` | Human-entered |

The `derived_from` list names the canonical IDs that produced the node. It stays empty for native nodes. It must only name IDs that exist in the document.

## Validation

Three validators guard each pipeline run after the enrichers:

| Validator | Checks |
|---|---|
| Structural | ID uniqueness, reading order references, bbox containment within a configurable tolerance |
| Readiness | Per-page ALTO and PAGE capability levels (`full`, `partial`, `degraded`, `none`) |
| Export eligibility | Go/no-go per export format, under the document policy |

A fourth validator checks canonical JSON against the Pydantic schema. It is not part of the job pipeline today.

The document policy has three modes:

| Mode | Inference | Partial exports | Tolerance |
|---|---|---|---|
| `strict` | No polygon-to-bbox, no language propagation, no reading-order inference, no block splitting | Refused | 5 px |
| `standard` (default) | Polygon-to-bbox, language propagation, reading order, block splitting, hyphenation | Allowed | 5 px |
| `permissive` | All enrichments | Allowed | 10 px |

All modes enforce two rules: no invented text, no invented bounding boxes.
