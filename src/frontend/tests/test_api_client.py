import pytest
import requests

import api_client


class FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=None):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


def test_predict_fractures_returns_validated_payload(monkeypatch):
    response = FakeResponse(
        {
            "num_detections": 1,
            "detections": [
                {
                    "confidence": 0.82,
                    "xyxy": [[10, 20, 100, 120]],
                    "class_name": "fracture",
                }
            ],
        }
    )
    monkeypatch.setattr(api_client.requests, "post", lambda *args, **kwargs: response)

    payload, duration = api_client.predict_fractures(
        "http://backend/predict",
        b"image",
        "case.png",
        0.3,
    )

    assert payload["num_detections"] == 1
    assert payload["detections"][0]["confidence"] == pytest.approx(0.82)
    assert duration >= 0


def test_predict_fractures_maps_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(api_client.requests, "post", raise_timeout)

    with pytest.raises(api_client.BackendTimeoutError):
        api_client.predict_fractures(
            "http://backend/predict",
            b"image",
            "case.jpg",
            0.3,
        )


def test_predict_fractures_rejects_invalid_payload(monkeypatch):
    response = FakeResponse({"num_detections": 1, "detections": []})
    monkeypatch.setattr(api_client.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(api_client.BackendResponseError):
        api_client.predict_fractures(
            "http://backend/predict",
            b"image",
            "case.jpg",
            0.3,
        )
