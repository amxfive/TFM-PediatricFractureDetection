from pathlib import Path

from streamlit.testing.v1 import AppTest

import api_client


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "SHF_001.jpg"


def app_test() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=15)


def test_app_opens_with_default_demo_case():
    app = app_test().run()

    assert not app.exception
    assert app.session_state.source_mode == "Caso demo"
    assert app.session_state.selected_case_id == "caso_04"
    assert app.session_state.analysis_result is None
    assert len(app.get("imgs")) == 2


def test_changing_demo_case_clears_previous_result():
    app = app_test().run()
    app.session_state.analysis_result = {"num_detections": 0, "detections": []}
    app.session_state.analysis_signature = "stale"

    app.selectbox(key="selected_case_id").select("caso_05").run()

    assert not app.exception
    assert app.session_state.selected_case_id == "caso_05"
    assert app.session_state.analysis_result is None


def test_uploaded_image_replaces_demo_case():
    app = app_test().run()
    image_bytes = EXAMPLE_PATH.read_bytes()

    app.segmented_control(key="source_mode").set_value("Subir radiografía").run()
    app.file_uploader(key="uploaded_xray").upload(
        "radiografia.jpg",
        image_bytes,
        "image/jpeg",
    ).run()

    assert not app.exception
    assert app.session_state.source_mode == "Subir radiografía"
    assert app.session_state.uploaded_xray.name == "radiografia.jpg"
    assert len(app.get("imgs")) == 2


def test_successful_analysis_displays_detection(monkeypatch):
    def fake_predict(*args, **kwargs):
        return (
            {
                "num_detections": 1,
                "detections": [
                    {
                        "confidence": 0.76,
                        "xyxy": [[100, 120, 320, 400]],
                        "class_name": "fracture",
                    }
                ],
            },
            0.42,
        )

    monkeypatch.setattr(api_client, "predict_fractures", fake_predict)
    app = app_test().run()
    app.button(key="analyze_case").click().run()

    assert not app.exception
    assert app.session_state.analysis_result["num_detections"] == 1
    assert app.session_state.analysis_duration == 0.42
    assert any("Se identificaron 1 detección" in item.value for item in app.warning)


def test_analysis_without_detections_uses_cautious_language(monkeypatch):
    monkeypatch.setattr(
        api_client,
        "predict_fractures",
        lambda *args, **kwargs: (
            {"num_detections": 0, "detections": []},
            0.21,
        ),
    )
    app = app_test().run()
    app.button(key="analyze_case").click().run()

    assert not app.exception
    assert any(
        "No se identificaron detecciones por encima del umbral" in item.value
        for item in app.info
    )


def test_backend_failure_keeps_case_ready_for_retry(monkeypatch):
    def fail(*args, **kwargs):
        raise api_client.BackendUnavailableError("backend unavailable")

    monkeypatch.setattr(api_client, "predict_fractures", fail)
    app = app_test().run()
    app.button(key="analyze_case").click().run()

    assert not app.exception
    assert app.session_state.analysis_result is None
    assert app.session_state.analysis_error[0] == "unavailable"
    assert app.button(key="retry_analysis")
