#!/usr/bin/env python3
"""Remote OCR service for a GPU server (run this on the GPU box).

Serves the exact word_box_json payload that local PaddleOCR inference
produces, so the XmLLM UI can offload inference:

    # on the GPU server (CUDA-matched install required):
    #   pip install "paddlepaddle-gpu>=3.3,<4" "paddleocr>=3.7,<4" fastapi uvicorn python-multipart
    #   python scripts/ocr_service.py --host 0.0.0.0 --port 8080

    # on the XmLLM machine:
    #   OCR_SERVICE_URL=http://gpu-box:8080 uv run uvicorn src.app.main:app --port 7860

Endpoint: POST /ocr?lang=fr with multipart field "image". Returns the
payload JSON; coordinates are in the uploaded image's pixel space, so the
viewer coordinate contract holds (the preprocessors that would alter the
coordinate space are disabled here, same as locally).

Deployment requirements:
- This script needs a CHECKOUT OF THIS REPO on the GPU box (it imports
  src.app.ocr), not just the pip installs.
- The service build MUST be the same commit as the XmLLM instance calling
  it: the transcription-anchored word merge and the exclusion policy run
  in this process, so a version mismatch changes payload semantics while
  the recorded provenance still claims the caller's adapter version.
  GET /version reports the commit; check it after every deploy.

Hardening notes:
- The handler calls the LOCAL engine runner directly, never the
  dispatching wrapper: if a .env with OCR_SERVICE_URL exists on this box,
  the service must not POST to itself and recurse.
- Uploads land in a NamedTemporaryFile whose name uses only the file
  suffix, never the caller-controlled filename (no path traversal).
- Uploads are capped (--max-upload-mb, default 50); this endpoint is
  unauthenticated, so do not expose it beyond trusted networks.

Note: PaddleOCR's GPU build needs the paddlepaddle-gpu wheel matching the
server's CUDA version (see paddlepaddle.org installation matrix). CPU-only
servers work too but gain little over local inference.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile  # noqa: E402
from src.app.ocr import SUPPORTED_PADDLE_LANGS, run_paddle_ocr_local  # noqa: E402

app = FastAPI(title="XmLLM OCR service")


@app.get("/version")
def version() -> dict:
    """Report the build this service runs.

    The service applies the transcription-anchored word merge and the
    exclusion policy inside its own process, so its build MUST match the
    commit of the XmLLM instance calling it (ADR 0001's guarantees and the
    recorded provenance are only as good as the build that produced the
    payload). Operators should verify this after every deploy of either
    side.
    """
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
            cwd=Path(__file__).parent.parent,
        ).stdout.strip()
    except Exception:
        commit = "unknown (no git metadata)"
    return {"service": "xmllm-ocr", "commit": commit}

# Kept in sync with settings.max_ocr_image_size (the client's own upload
# gate); --max-upload-mb overrides for a deployment with other needs.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]{1,10}")


@app.post("/ocr")
def ocr(image: UploadFile, lang: str = "en") -> list:
    """Sync handler: Starlette runs it in its threadpool, so the blocking
    predict() call holds one worker thread, not the event loop."""
    if lang not in SUPPORTED_PADDLE_LANGS:
        raise HTTPException(status_code=400, detail=f"unsupported lang: {lang}")

    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")

    suffix = Path(image.filename or "").suffix
    if not _SUFFIX_RE.fullmatch(suffix):
        suffix = ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = Path(tmp.name)
    try:
        return run_paddle_ocr_local(path, lang)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-upload-mb", type=int, default=50)
    parser.add_argument(
        "--device", default=None,
        help="Paddle device (e.g. gpu:0). Unset: PaddleOCR auto-selects"
             " GPU when the paddlepaddle-gpu build supports CUDA.",
    )
    args = parser.parse_args()
    if args.device:
        os.environ["OCR_DEVICE"] = args.device
    global MAX_UPLOAD_BYTES
    MAX_UPLOAD_BYTES = args.max_upload_mb * 1024 * 1024
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
