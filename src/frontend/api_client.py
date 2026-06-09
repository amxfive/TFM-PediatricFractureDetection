import time

import requests


class BackendError(RuntimeError):
    """Base exception for inference backend failures."""


class BackendTimeoutError(BackendError):
    """The inference request exceeded its timeout."""


class BackendUnavailableError(BackendError):
    """The inference service could not be reached."""


class BackendResponseError(BackendError):
    """The inference service returned an invalid response."""


def _validate_response(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise BackendResponseError("La respuesta no es un objeto JSON.")

    num_detections = payload.get("num_detections")
    detections = payload.get("detections")
    if not isinstance(num_detections, int) or num_detections < 0:
        raise BackendResponseError("El número de detecciones no es válido.")
    if not isinstance(detections, list) or len(detections) != num_detections:
        raise BackendResponseError("La lista de detecciones no es válida.")

    normalized_detections = []
    for index, detection in enumerate(detections, start=1):
        if not isinstance(detection, dict):
            raise BackendResponseError(f"La detección {index} no es válida.")
        confidence = detection.get("confidence")
        xyxy = detection.get("xyxy")
        class_name = detection.get("class_name", "fracture")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise BackendResponseError(
                f"La confianza de la detección {index} no es válida."
            )
        if (
            not isinstance(xyxy, list)
            or len(xyxy) != 1
            or not isinstance(xyxy[0], list)
            or len(xyxy[0]) != 4
            or not all(isinstance(value, (int, float)) for value in xyxy[0])
        ):
            raise BackendResponseError(
                f"Las coordenadas de la detección {index} no son válidas."
            )
        normalized_detections.append(
            {
                "confidence": float(confidence),
                "xyxy": [[float(value) for value in xyxy[0]]],
                "class_name": str(class_name),
            }
        )

    return {
        "num_detections": num_detections,
        "detections": normalized_detections,
    }


def predict_fractures(
    backend_url: str,
    image_bytes: bytes,
    filename: str,
    confidence: float,
    timeout: float = 120,
) -> tuple[dict, float]:
    started_at = time.perf_counter()
    extension = filename.rsplit(".", maxsplit=1)[-1].lower()
    media_type = "image/png" if extension == "png" else "image/jpeg"

    try:
        response = requests.post(
            backend_url,
            files={"file": (filename, image_bytes, media_type)},
            data={"confidence": confidence},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise BackendTimeoutError(
            f"El backend no respondió en {timeout:.0f} segundos."
        ) from exc
    except requests.ConnectionError as exc:
        raise BackendUnavailableError(
            "No se pudo establecer conexión con el backend."
        ) from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "?"
        raise BackendResponseError(
            f"El backend respondió con el código HTTP {status_code}."
        ) from exc
    except requests.RequestException as exc:
        raise BackendUnavailableError(str(exc)) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise BackendResponseError("El backend no devolvió JSON válido.") from exc

    return _validate_response(payload), time.perf_counter() - started_at
