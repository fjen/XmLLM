# OCR service deployment

`scripts/ocr_service.py` runs PaddleOCR inference on a separate machine and serves the same payload the UI would compute locally. Use it when local CPU inference is too slow and a GPU server is available. One page that takes 60–230 s on CPU typically runs several times faster on GPU; measure your own numbers (see Benchmarking).

## How it fits

```
Browser → XmLLM UI (port 7860) → OCR service (GPU box, port 8080) → PaddleOCR engine
```

Set `OCR_SERVICE_URL` on the XmLLM machine and every OCR job (UI, batch CLI) goes to the service instead of loading PaddleOCR locally. The service returns the same `word_box_json` payload in the uploaded image's pixel space, so coordinates, the viewer contract, ALTO/PAGE export and the word reconstruction all behave exactly as before.

Unset `OCR_SERVICE_URL` means local inference. Nothing else changes.

## Requirements on the GPU box

1. **A checkout of this repository.** The service imports `src.app.ocr`; pip installs alone are not enough.
2. **Python 3.11+**.
3. **The CUDA-matched Paddle wheel, installed sync-free.** The repo pins the CPU wheel (`paddlepaddle`), and `uv.lock` reinstalls it on every `uv run` or `uv sync`. Build the environment once, then never let uv touch it again:

   ```bash
   uv venv
   uv pip install -e ".[dev]"          # fastapi, uvicorn, python-multipart included
   uv pip uninstall paddlepaddle       # CPU wheel out
   uv pip install "paddlepaddle-gpu>=3.3,<4" \
       -i https://www.paddlepaddle.org.cn/packages/stable/cu126/   # GPU wheel in, LAST
   ```

   Pick the index URL that matches the box's CUDA version (see the installation matrix at paddlepaddle.org). An NVIDIA driver new enough for that CUDA version must be installed.

   **Do not run `uv run` or `uv sync` in this checkout after the swap.** Either would re-sync from `uv.lock`, reinstall the CPU wheel, and the service would silently run on CPU. If you need `uv run`, pass `--no-sync` and launch the interpreter directly as shown below.

Verify the CUDA build before starting:

```bash
.venv/bin/python -c "import paddle; print(paddle.is_compiled_with_cuda()); paddle.utils.run_check()"
```

The first line must print `True`; `run_check()` must not raise. If it prints `False`, uv re-synced the CPU wheel — repeat the swap.

## Run the service

Launch the interpreter directly, never through `uv run` (it re-syncs the lock and undoes the GPU wheel swap):

```bash
.venv/bin/python scripts/ocr_service.py --host 0.0.0.0 --port 8080 --device gpu:0
```

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `0.0.0.0` / `8080` | Bind address |
| `--device` | auto | `gpu:0` pins the GPU. Unset, PaddleOCR auto-selects: GPU if the paddle build supports CUDA, else CPU |
| `--max-upload-mb` | 50 | Upload size cap |

On startup, the first request per language downloads models to `~/.paddlex` (about 2 minutes per language). Later requests are fast. Each configured language holds one model set in memory (at most 8, LRU).

Check the log line on the first request: `PaddleOCR[fr]: CUDA build=True, device=gpu:0`. If it says `CUDA build=False`, the box has the CPU wheel and will run at CPU speed.

## Wire it into XmLLM

On the XmLLM machine:

```bash
echo 'OCR_SERVICE_URL=http://gpu-box:8080' >> .env
# restart the UI so the setting is read
uv run uvicorn src.app.main:app --host 0.0.0.0 --port 7860
```

## Version pinning (important)

The service applies the transcription-anchored word merge and the language exclusion policy inside its own process. Its build **must be the same commit** as the XmLLM instance calling it. A mismatch changes payload semantics while the recorded provenance still claims the caller's adapter version.

Verify after every deploy of either side:

```bash
curl -s http://gpu-box:8080/version        # {"service": "xmllm-ocr", "commit": "abc1234"}
git -C /path/to/XmLLM rev-parse --short HEAD
```

Both values must match. `GET /version` reports `unknown` when the box has no git metadata; fix that (fresh clone or `git fetch`) before trusting the deployment.

## Security

The endpoint is **unauthenticated**. Do not expose it beyond a trusted network. Mitigations built in: uploads are capped (`--max-upload-mb`), files land in temp files named from the file suffix only (no path traversal), and unsupported language codes are rejected. Restrict access with a firewall or a reverse proxy with authentication if the network is shared.

## Benchmarking

Time one page both ways, same image, second runs only (the first request per language includes model loading):

```bash
# remote
time curl -s -X POST "http://gpu-box:8080/ocr?lang=fr" -F "image=@00000038.jpg" > /dev/null

# local (on the XmLLM machine, OCR_SERVICE_URL unset)
time curl -s -X POST "http://localhost:7860/ocr" -F "image=@00000038.jpg" > /dev/null
```

The win is `GPU speedup − network cost`: the whole image ships per job, so a slow link eats into the gain.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CUDA build=False` in the log | The CPU wheel is back — usually a `uv run`/`uv sync` after the swap re-synced it from `uv.lock`. Redo the swap (see Requirements) and launch with `.venv/bin/python`, not `uv run`. |
| `CUDA build=True` but slow | `--device` unset and auto-selection picked CPU, or the GPU is busy. Pin with `--device gpu:0` and check `nvidia-smi`. |
| Connection errors from the UI | Service down, wrong port, or a firewall. `curl http://gpu-box:8080/version` from the XmLLM machine. |
| HTTP 413 | Upload above `--max-upload-mb`. |
| HTTP 400 `unsupported lang` | Language not in the allowlist (`SUPPORTED_PADDLE_LANGS` in `src/app/ocr`). |
| Job fails with `OCR service ... failed` | Engine error on the GPU box. The detail is in the service log; the UI shows the first 500 characters. |

The service has no hot reload. Restart it after pulling a new commit, and re-check `/version`.
