import io

import numpy as np
import pytest
from PIL import Image

from image_utils import (
    ImageValidationError,
    draw_overlays,
    fit_on_black_canvas,
    image_to_png_bytes,
    normalize_image,
    parse_yolo_boxes,
    stable_viewer_html,
)


def image_bytes(array: np.ndarray, mode: str | None = None) -> bytes:
    image = Image.fromarray(array, mode=mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "array",
    [
        np.full((16, 12), 127, dtype=np.uint8),
        np.full((16, 12, 3), (10, 20, 30), dtype=np.uint8),
    ],
)
def test_normalize_image_supports_grayscale_and_rgb(array):
    normalized = normalize_image(image_bytes(array))

    assert normalized.mode == "RGB"
    assert normalized.size == (12, 16)


def test_normalize_image_supports_16_bit_png():
    array = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16)

    normalized = normalize_image(image_bytes(array, mode="I;16"))

    assert normalized.mode == "RGB"
    assert normalized.getpixel((1, 0)) == (255, 255, 255)


def test_normalize_image_handles_constant_non_uint8_array():
    array = np.full((8, 8), 400, dtype=np.uint16)

    normalized = normalize_image(image_bytes(array, mode="I;16"))

    assert normalized.mode == "RGB"
    assert normalized.size == (8, 8)


def test_normalize_image_rejects_corrupt_data():
    with pytest.raises(ImageValidationError):
        normalize_image(b"not-an-image")


def test_parse_yolo_boxes_accepts_multiple_lines_and_optional_class():
    boxes = parse_yolo_boxes(
        "0 0.5 0.5 0.2 0.4\n"
        "0.25 0.25 0.1 0.1"
    )

    assert boxes == pytest.approx(
        [
            (0.4, 0.3, 0.6, 0.7),
            (0.2, 0.2, 0.3, 0.3),
        ]
    )


@pytest.mark.parametrize(
    "text",
    [
        "0 1.2 0.5 0.2 0.2",
        "0 0.5 0.5 0 0.2",
        "0 0.05 0.5 0.2 0.2",
        "not valid",
    ],
)
def test_parse_yolo_boxes_rejects_invalid_input(text):
    with pytest.raises(ValueError):
        parse_yolo_boxes(text)


def test_draw_overlays_and_export_png():
    image = Image.new("RGB", (200, 100), "black")
    detections = [{"confidence": 0.9, "xyxy": [[20, 10, 80, 70]]}]
    ground_truth = [(0.5, 0.2, 0.9, 0.8)]

    annotated = draw_overlays(image, detections, ground_truth)
    canvas = fit_on_black_canvas(annotated, (300, 300))
    exported = image_to_png_bytes(annotated)

    assert annotated.getbbox() is not None
    assert canvas.size == (300, 300)
    assert exported.startswith(b"\x89PNG")


def test_stable_viewer_html_has_fixed_aspect_ratio_and_escaped_alt_text():
    image = Image.new("RGB", (32, 16), "black")

    markup = stable_viewer_html(image, 'Radiografía "original"')

    assert 'data-testid="stable-xray-viewer"' in markup
    assert "aspect-ratio:1/1" in markup
    assert 'width="960" height="960"' in markup
    assert 'alt="Radiografía &quot;original&quot;"' in markup
    assert "data:image/png;base64," in markup
