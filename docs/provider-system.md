# The provider system

The provider system separates three concerns. Each layer answers one question:

| Layer | Question | Examples |
|---|---|---|
| Runtime | How do I execute it? | `local`, `hub`, `api` |
| Family | What shape is the output? | `word_box_json`, `line_box_json`, `text_only` |
| Profile | What is this specific instance? | PaddleOCR local at `/models/paddle`, a Qwen API endpoint |

## Adapter families

| Family | Output shape | Geometry | ALTO export |
|---|---|---|---|
| `word_box_json` | Lines with 4-point polygons; words reconstructed from them (PaddleOCR, ADR 0001) | Exact | Full |
| `line_box_json` | Lines with bboxes, no word segmentation | Exact at line level | Full (1 word per line) |
| `text_only` | Structured text, no coordinates (mLLM) | Unknown | Refused |

The `text_only` family produces an honest refusal: ALTO needs coordinates, and the family has none. The system does not guess.

Under ADR 0001, `word_box_json` lines carry reconstructed words when the engine can map fragments to the transcription. Lines without word data keep their transcription and take the line-level fallback.

## Capability matrix

Each provider profile declares a `CapabilityMatrix`:

```
block_geometry, line_geometry, word_geometry, polygon_geometry,
baseline, reading_order, text_confidence, language,
non_text_regions, tables, rotation_info
```

No runtime decision reads the matrix today. Readiness and export eligibility derive from the canonical data (ADR 0001). The matrix documents what each provider can deliver.
