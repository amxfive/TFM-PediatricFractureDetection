import base64
import html
import io
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps, UnidentifiedImageError


class ImageValidationError(ValueError):
    """Raised when uploaded image bytes cannot be prepared for display."""


def normalize_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            array = np.asarray(source)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "No se pudo abrir la imagen. Utilice un archivo JPG o PNG válido."
        ) from exc

    if array.size == 0:
        raise ImageValidationError("La imagen no contiene píxeles.")

    if array.dtype == np.uint8:
        normalized = array
    elif array.dtype == np.uint16:
        normalized = np.rint(array.astype(np.float32) / 257.0).astype(np.uint8)
    else:
        finite_values = array[np.isfinite(array)]
        if finite_values.size == 0:
            raise ImageValidationError("La imagen no contiene valores válidos.")
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
        if maximum == minimum:
            normalized = np.zeros(array.shape, dtype=np.uint8)
        else:
            scaled = (array.astype(np.float32) - minimum) / (maximum - minimum)
            normalized = np.clip(np.rint(scaled * 255), 0, 255).astype(np.uint8)

    try:
        return Image.fromarray(normalized).convert("RGB")
    except (TypeError, ValueError) as exc:
        raise ImageValidationError(
            "La imagen utiliza un formato de píxel no compatible."
        ) from exc


def apply_view_adjustments(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> Image.Image:
    adjusted = image.copy()
    if contrast != 1.0:
        adjusted = ImageEnhance.Contrast(adjusted).enhance(contrast)
    if brightness != 1.0:
        adjusted = ImageEnhance.Brightness(adjusted).enhance(brightness)
    return adjusted


def fit_on_black_canvas(
    image: Image.Image,
    canvas_size: tuple[int, int] = (960, 960),
) -> Image.Image:
    contained = ImageOps.contain(image.convert("RGB"), canvas_size)
    canvas = Image.new("RGB", canvas_size, "black")
    offset = (
        (canvas_size[0] - contained.width) // 2,
        (canvas_size[1] - contained.height) // 2,
    )
    canvas.paste(contained, offset)
    return canvas


def parse_yolo_boxes(text: str) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.replace(",", " ").split()
        if len(parts) == 5:
            parts = parts[1:]
        elif len(parts) != 4:
            raise ValueError(
                f"Línea {line_number}: use 'clase x y ancho alto' o 'x y ancho alto'."
            )

        try:
            center_x, center_y, width, height = map(float, parts)
        except ValueError as exc:
            raise ValueError(
                f"Línea {line_number}: todas las coordenadas deben ser numéricas."
            ) from exc

        if not all(0 <= value <= 1 for value in (center_x, center_y, width, height)):
            raise ValueError(
                f"Línea {line_number}: las coordenadas deben estar entre 0 y 1."
            )
        if width == 0 or height == 0:
            raise ValueError(
                f"Línea {line_number}: el ancho y el alto deben ser mayores que cero."
            )

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        if min(x1, y1) < 0 or max(x2, y2) > 1:
            raise ValueError(
                f"Línea {line_number}: la caja queda fuera de los límites de la imagen."
            )
        boxes.append((x1, y1, x2, y2))

    return boxes


def _font(size: int) -> ImageFont.ImageFont:
    font_paths = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    )
    for path in font_paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    padding = 4
    draw.rectangle(
        (left - padding, top - padding, right + padding, bottom + padding),
        fill=color,
    )
    draw.text(xy, text, fill="white", font=font)


def draw_overlays(
    image: Image.Image,
    detections: Iterable[dict] = (),
    ground_truth_boxes: Iterable[tuple[float, float, float, float]] = (),
) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    line_width = max(3, round(min(annotated.size) / 220))
    font = _font(max(14, round(min(annotated.size) / 45)))

    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"][0]
        coordinates = (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )
        draw.rectangle(coordinates, outline="#1787D4", width=line_width)
        label_y = max(2, coordinates[1] - font.size - 10)
        _draw_label(
            draw,
            (coordinates[0] + 3, label_y),
            f"IA {detection['confidence']:.0%}",
            "#126AA3",
            font,
        )

    for box_number, (x1, y1, x2, y2) in enumerate(ground_truth_boxes, start=1):
        coordinates = (
            int(round(x1 * annotated.width)),
            int(round(y1 * annotated.height)),
            int(round(x2 * annotated.width)),
            int(round(y2 * annotated.height)),
        )
        draw.rectangle(coordinates, outline="#1B8A5A", width=line_width)
        label_y = max(2, coordinates[1] - font.size - 10)
        _draw_label(
            draw,
            (coordinates[0] + 3, label_y),
            f"Ref. {box_number}",
            "#176C49",
            font,
        )

    return annotated


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def stable_viewer_html(image: Image.Image, alt_text: str) -> str:
    encoded_image = base64.b64encode(image_to_png_bytes(image)).decode("ascii")
    escaped_alt_text = html.escape(alt_text, quote=True)
    return (
        '<div data-testid="stable-xray-viewer" '
        'style="width:100%;aspect-ratio:1/1;overflow:hidden;background:#000;'
        'border-radius:0.5rem;line-height:0;contain:layout paint size;">'
        f'<img src="data:image/png;base64,{encoded_image}" '
        f'alt="{escaped_alt_text}" width="960" height="960" '
        'style="display:block;width:100%;height:100%;object-fit:contain;'
        'background:#000;" loading="eager" decoding="sync">'
        "</div>"
    )
