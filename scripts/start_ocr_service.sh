#!/bin/bash
# Start the XmLLM OCR service on an inference box (see docs/ocr-service.md).
#
# Sync-free runtime: never run "uv run"/"uv sync" in the checkout after the
# paddlepaddle-gpu swap, or the pinned CPU wheel comes back.
#
# Site note (ocr-d): cuDNN 9 lives in the system python3.10 nvidia package
# and is not on the default loader path; the LD_LIBRARY_PATH below picks it
# up. The check keeps the script generic for boxes with system cuDNN.
cd "$(dirname "$0")/.."   # repo root

CUDNN_DIR=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
if [ -d "$CUDNN_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDNN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

HOST="${OCR_SERVICE_HOST:-127.0.0.1}"
PORT="${OCR_SERVICE_PORT:-8080}"
DEVICE="${OCR_DEVICE:-}"
ARGS=(--host "$HOST" --port "$PORT")
if [ -n "$DEVICE" ]; then
    ARGS+=(--device "$DEVICE")
fi
exec .venv/bin/python scripts/ocr_service.py "${ARGS[@]}"
