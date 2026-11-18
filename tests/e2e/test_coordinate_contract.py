"""Coordinate-contract e2e test (ADR 0001).

Verifies that the line polygons and reconstructed word boxes produced from
a real PaddleOCR run land on ink in the exact image that would be served
by ``GET /jobs/{id}/image`` — i.e. that the adapter's mapping from the
image passed to PaddleOCR into the presented pixel space is correct for
both line polygons and fragment-derived word boxes.

No real scans are committed to the repository. Provide a local pair via
environment variables (generate the payload with
``scripts/generate_contract_fixture.py``):

    XMLLM_CONTRACT_IMAGE=/path/to/scan.png \\
    XMLLM_CONTRACT_PAYLOAD=/path/to/payload.json \\
    .venv/bin/python -m pytest tests/e2e/test_coordinate_contract.py

The test skips when either variable is unset or the file is missing.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest
from PIL import Image
from src.app.domain.models import RawProviderPayload
from src.app.domain.models.geometry import GeometryContext
from src.app.enrichers import EnricherPipeline
from src.app.enrichers.bbox_repair_light import BboxRepairLightEnricher
from src.app.enrichers.lang_propagation import LangPropagationEnricher
from src.app.enrichers.reading_order_simple import ReadingOrderSimpleEnricher
from src.app.normalization.pipeline import normalize
from src.app.policies.document_policy import DocumentPolicy
from src.app.viewer.projection_builder import build_projection

IMAGE_PATH = os.environ.get("XMLLM_CONTRACT_IMAGE", "")
PAYLOAD_PATH = os.environ.get("XMLLM_CONTRACT_PAYLOAD", "")

pytestmark = pytest.mark.skipif(
    not (IMAGE_PATH and PAYLOAD_PATH
         and Path(IMAGE_PATH).exists() and Path(PAYLOAD_PATH).exists()),
    reason="set XMLLM_CONTRACT_IMAGE and XMLLM_CONTRACT_PAYLOAD to a local "
           "scan + payload (see scripts/generate_contract_fixture.py)",
)


@pytest.fixture(scope="module")
def viewer() -> dict:
    with open(PAYLOAD_PATH) as f:
        payload = json.load(f)
    img = Image.open(IMAGE_PATH)
    raw = RawProviderPayload(
        provider_id="paddleocr",
        adapter_id="adapter.word_box_json.v1",
        runtime_type="local",
        payload=payload,
        image_width=img.width,
        image_height=img.height,
    )
    doc = normalize(
        raw,
        family="word_box_json",
        geometry_context=GeometryContext(
            source_width=img.width, source_height=img.height
        ),
        document_id="coord_contract",
    )
    doc = EnricherPipeline([
        BboxRepairLightEnricher(),
        LangPropagationEnricher(),
        ReadingOrderSimpleEnricher(),
    ]).run(doc, DocumentPolicy())
    return build_projection(doc).model_dump(mode="json")


def _ink_ratio(img: Image.Image, box: tuple[float, float, float, float]) -> float:
    """Fraction of dark pixels in the box region of the exact image."""
    x, y, w, h = box
    if w < 2 or h < 2 or x < 0 or y < 0 or x + w > img.width or y + h > img.height:
        return 0.0
    crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
    dark = sum(1 for px in crop.convert("L").getdata() if px < 128)
    return dark / (crop.width * crop.height)


def _on_ink(img: Image.Image, box: tuple[float, float, float, float]) -> bool:
    return _ink_ratio(img, box) > 0.02


class TestCoordinateContract:
    def test_line_polygons_map_to_presented_image(self, viewer: dict) -> None:
        img = Image.open(IMAGE_PATH)
        assert viewer["line_overlays"], "no line overlays in projection"
        misses = [
            o["id"] for o in viewer["line_overlays"] if not _on_ink(img, o["bbox"])
        ]
        # Tolerance: a few degenerate boxes (e.g. punctuation-only lines)
        # may dip below the threshold; the contract fails on gross misses.
        assert len(misses) <= len(viewer["line_overlays"]) * 0.05, misses[:10]

    def test_word_boxes_map_to_presented_image(self, viewer: dict) -> None:
        img = Image.open(IMAGE_PATH)
        assert len(viewer["word_overlays"]) >= 100
        random.seed(7)
        sample = random.sample(viewer["word_overlays"], min(300, len(viewer["word_overlays"])))
        misses = [o["id"] for o in sample if not _on_ink(img, o["bbox"])]
        assert len(misses) <= len(sample) * 0.02, misses[:10]

    def test_word_boxes_carry_no_polygon(self, viewer: dict) -> None:
        # Word geometry is bbox-only (enclosing rectangle); the polygon
        # stays on the line (ADR 0001).
        assert all(o.get("polygon") is None for o in viewer["word_overlays"])
