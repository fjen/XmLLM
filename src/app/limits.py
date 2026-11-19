"""Shared limits.

The client upload gate and the remote OCR service read one default. An
environment override does not reach the remote service.
"""

# 100 MB.
DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
