#!/usr/bin/env python3
"""Generate the coordinate-contract payload fixture for a local image.

Runs the real PaddleOCR engine on an image you supply and writes the
word_box_json payload (with reconstructed words) as JSON. No real scans
are committed to the repository — point the coordinate-contract test at
the generated files via environment variables:

    XMLLM_CONTRACT_IMAGE=/path/to/scan.png \
    XMLLM_CONTRACT_PAYLOAD=/path/to/payload.json \
    .venv/bin/python -m pytest tests/e2e/test_coordinate_contract.py

Usage:
    .venv/bin/python scripts/generate_contract_fixture.py IMAGE OUTPUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.ocr import run_paddle_ocr  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    image_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])

    payload = run_paddle_ocr(image_path, lang="en")

    def conv(item: list) -> list:
        points, tc = item
        if len(tc) == 3:
            return [points, [tc[0], tc[1], [[w, list(b)] for w, b in tc[2]]]]
        if len(tc) == 4:
            return [points, [tc[0], tc[1], None, tc[3]]]
        return [points, [tc[0], tc[1]]]

    out_path.write_text(
        json.dumps([conv(it) for it in payload], ensure_ascii=False)
    )
    print(f"wrote {out_path}: {len(payload)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
