# XmLLM

XmLLM converts scanned pages into validated **ALTO XML** and **PAGE XML**. It runs OCR (PaddleOCR, or your own provider output), normalizes the result into an internal canonical model with full provenance, and exports standards-compliant XML from that model. A web UI shows the result as an overlay on the page image.

The same code runs locally and in Docker.

## How it works

```
image or raw provider JSON
  → provider runtime (local / hub / api)
  → raw payload (stored for audit)
  → adapter (provider output → canonical model)
  → CanonicalDocument
  → enrichers (bbox repair, language, reading order, hyphenation)
  → validators (structure, readiness, export eligibility)
  → policy (strict / standard / permissive)
  → ALTO XML + PAGE XML + viewer projection
```

Three internal objects keep the responsibilities apart:

| Object | Role | Never used for |
|---|---|---|
| `RawProviderPayload` | Source truth | Export, rendering |
| `CanonicalDocument` | Business truth | Direct UI rendering |
| `ViewerProjection` | Rendering truth | Validation, export decisions |

The domain layer depends on nothing. FastAPI, persistence and the frontend sit on top of it. Every node in the canonical model carries geometry (`bbox` as `(x, y, width, height)`, origin top-left, unit px) and provenance (`provider`, `adapter`, `source_ref`, `evidence_type`). Details: [docs/canonical-document.md](docs/canonical-document.md), [docs/provider-system.md](docs/provider-system.md), [docs/enrichers.md](docs/enrichers.md) and [docs/adr/0002-alto-export-consumer-contract.md](docs/adr/0002-alto-export-consumer-contract.md).

## Quick start

### Local

```bash
git clone https://github.com/fjen/XmLLM.git
cd XmLLM
pip install -e ".[dev]"
cp .env.example .env
uvicorn src.app.main:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860 for the web UI and http://localhost:7860/docs for the API documentation.

### Docker

```bash
docker compose up --build
```

## Batch processing

Export ALTO XML for a directory of scans without the server:

```bash
python scripts/batch_alto.py /path/to/book --lang de
python scripts/batch_alto.py /path/to/book --lang de --out /path/to/alto
```

- `--lang` is a PaddleOCR language code (`de`, `fr`, `en`, `ch`, `japan`, ...). The batch maps it to ISO 639-3 and stamps it into the canonical nodes, so ALTO carries `LANG` attributes.
- The batch skips images whose ALTO output already exists. Without `--out`, an image counts as done when a prior job for the same filename produced ALTO. Files with the same name in different directories count as the same image. Use `--no-skip-existing` to reprocess.
- `--out` copies each `alto.xml` as `<image-stem>.xml`. `--storage-root` overrides `STORAGE_ROOT`.
- Batch jobs keep the source image in place. The web UI shows no page image for them.
- Exit code 0 only if every image produced ALTO.

## Remote OCR (GPU offload)

Local CPU inference is slow. Run PaddleOCR on a GPU box and point XmLLM at it:

```
OCR_SERVICE_URL=http://gpu-box:8080
```

The service returns the same payload as local inference, so nothing downstream changes. Measured on an A40: about 2.7 s per archival page, versus 60 to 230 s on CPU.

Setup, security, version pinning and benchmarks: [docs/ocr-service.md](docs/ocr-service.md).

One trap fails silently there: the repo pins the CPU paddle wheel, and `uv run` / `uv sync` restore it from `uv.lock`. On the GPU box, install the GPU wheel last and never re-sync. The startup log and `GET /version` catch it.

## Configuration

Copy `.env.example` to `.env`. The OCR settings:

| Variable | Default | Meaning |
|---|---|---|
| `OCR_SERVICE_URL` | unset | Offload OCR to `scripts/ocr_service.py`. Unset = local |
| `OCR_DEVICE` | unset | Paddle device, e.g. `gpu:0`. Unset = auto-select |
| `OCR_RECOGNITION_BATCH_SIZE` | unset | Recognition batch size. Unset = Paddle default |
| `OCR_TEXTLINE_ORIENTATION` | unset | Set `false` to skip the per-line orientation classifier. Unset = on |

`.env.example` documents the other variables (`STORAGE_ROOT`, `MAX_UPLOAD_SIZE`, `BBOX_CONTAINMENT_TOLERANCE`, ...).

## API

Start the server and open `/docs` for the full OpenAPI reference. Main endpoints:

- `POST /ocr`: upload an image, run OCR, get ALTO/PAGE XML. `?lang=` selects the language
- `POST /jobs` and `GET /jobs/{id}`: run and inspect a job from raw provider JSON
- `GET /jobs/{id}/alto`, `/jobs/{id}/pagexml`, `/jobs/{id}/canonical`, `/jobs/{id}/raw`: downloads
- `GET /jobs/{id}/viewer`: viewer projection for the web UI
- `GET /providers`: provider profile management
- `GET /health`: status

## Validation and policy

Export decisions pass through validators and a document policy. When the engine cannot map word positions, the line keeps its transcription and the system flags the page instead of guessing coordinates. The validator set, the policy modes and the no-invention rules: [docs/canonical-document.md](docs/canonical-document.md). The word reconstruction contract: [docs/adr/0001-transcription-anchored-word-merge.md](docs/adr/0001-transcription-anchored-word-merge.md). The export consumer contract: [docs/adr/0002-alto-export-consumer-contract.md](docs/adr/0002-alto-export-consumer-contract.md).

## Testing

```bash
pytest                      # everything
pytest tests/unit/          # unit tests
pytest tests/integration/   # integration tests
```

More than 650 tests cover domain models, geometry, adapters, serializers, validators, enrichers, persistence, API routes and end-to-end fixtures.

## License

Apache 2.0. See [LICENSE](LICENSE).
