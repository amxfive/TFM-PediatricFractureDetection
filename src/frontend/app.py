import hashlib
import io
import os
from pathlib import Path

import streamlit as st

import api_client
from case_catalog import CASES_BY_ID, DEFAULT_CASE_ID, DEMO_CASES
from image_utils import (
    ImageValidationError,
    apply_view_adjustments,
    draw_overlays,
    fit_on_black_canvas,
    image_to_png_bytes,
    normalize_image,
    parse_yolo_boxes,
    stable_viewer_html,
)


st.set_page_config(
    page_title="Sistema de Detección de Fracturas Pediátricas",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000/predict")
EXAMPLES_DIR = Path(__file__).parent / "examples"
CANVAS_SIZE = (960, 960)
CONTROL_DEFAULTS = {
    "confidence_threshold": 0.30,
    "viewer_brightness": 1.0,
    "viewer_contrast": 1.0,
}


def clear_analysis() -> None:
    st.session_state.analysis_result = None
    st.session_state.analysis_signature = None
    st.session_state.analysis_duration = None
    st.session_state.analysis_error = None


def control_key(name: str) -> str:
    prefix = "demo" if st.session_state.source_mode == "Caso demo" else "upload"
    return f"{prefix}_{name}"


def active_control_values() -> tuple[float, float, float]:
    return (
        st.session_state.get(
            control_key("confidence_threshold"),
            CONTROL_DEFAULTS["confidence_threshold"],
        ),
        st.session_state.get(
            control_key("viewer_brightness"),
            CONTROL_DEFAULTS["viewer_brightness"],
        ),
        st.session_state.get(
            control_key("viewer_contrast"),
            CONTROL_DEFAULTS["viewer_contrast"],
        ),
    )


def reset_upload_defaults() -> None:
    for name in CONTROL_DEFAULTS:
        st.session_state.pop(f"upload_{name}", None)


def handle_source_change() -> None:
    clear_analysis()
    if st.session_state.source_mode == "Subir radiografía":
        reset_upload_defaults()


def handle_upload_change() -> None:
    clear_analysis()
    reset_upload_defaults()


def initialize_state() -> None:
    defaults = {
        "source_mode": "Caso demo",
        "selected_case_id": DEFAULT_CASE_ID,
        "analysis_result": None,
        "analysis_signature": None,
        "analysis_duration": None,
        "analysis_error": None,
        "evaluation_mode": False,
        "ground_truth_text": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_data(show_spinner=False)
def load_demo_bytes(filename: str) -> bytes:
    return (EXAMPLES_DIR / filename).read_bytes()


def current_case() -> tuple[bytes | None, str | None, object | None]:
    if st.session_state.source_mode == "Caso demo":
        case = CASES_BY_ID[st.session_state.selected_case_id]
        return load_demo_bytes(case.filename), case.filename, case

    uploaded_file = st.session_state.get("uploaded_xray")
    if uploaded_file is None:
        return None, None, None
    uploaded_file.seek(0)
    return uploaded_file.read(), uploaded_file.name, None


def analysis_signature(image_bytes: bytes, threshold: float) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    return f"{digest}:{threshold:.2f}"


def run_analysis(image_bytes: bytes, filename: str, signature: str) -> None:
    clear_analysis()
    with st.status(
        "Analizando la radiografía...",
        expanded=True,
        state="running",
    ) as status:
        st.write("Aplicando el preprocesamiento y ejecutando el detector.")
        try:
            result, duration = api_client.predict_fractures(
                BACKEND_URL,
                image_bytes,
                filename,
            active_control_values()[0],
            )
        except api_client.BackendTimeoutError as exc:
            st.session_state.analysis_error = ("timeout", str(exc))
            status.update(label="El análisis agotó el tiempo de espera", state="error")
            return
        except api_client.BackendUnavailableError as exc:
            st.session_state.analysis_error = ("unavailable", str(exc))
            status.update(label="No se pudo contactar con el servicio de IA", state="error")
            return
        except api_client.BackendResponseError as exc:
            st.session_state.analysis_error = ("response", str(exc))
            status.update(label="El servicio devolvió una respuesta no válida", state="error")
            return

        st.session_state.analysis_result = result
        st.session_state.analysis_duration = duration
        st.session_state.analysis_signature = signature
        status.update(label="Análisis completado", state="complete", expanded=False)


def render_header() -> None:
    title_col, badge_col = st.columns([5, 1], vertical_alignment="center")
    with title_col:
        st.title("Sistema de Detección de Fracturas Pediátricas")
        st.caption(
            "Lectura asistida de radiografías pediátricas de miembro superior."
        )
    with badge_col:
        with st.container(horizontal_alignment="right"):
            st.badge(
                "Demo de investigación",
                icon=":material/science:",
                color="blue",
            )


def render_case_selector() -> None:
    with st.container(border=True):
        st.subheader("1. Seleccione una radiografía", anchor=False)
        st.segmented_control(
            "Origen de la imagen",
            ["Caso demo", "Subir radiografía"],
            key="source_mode",
            on_change=handle_source_change,
            width="stretch",
        )

        if st.session_state.source_mode == "Caso demo":
            st.selectbox(
                "Caso de demostración",
                options=[case.case_id for case in DEMO_CASES],
                key="selected_case_id",
                format_func=lambda case_id: CASES_BY_ID[case_id].selector_label,
                on_change=clear_analysis,
            )
            case = CASES_BY_ID[st.session_state.selected_case_id]
            st.badge(
                case.clinical_reference,
                icon=(
                    ":material/personal_injury:"
                    if case.expected_fracture
                    else ":material/check_circle:"
                ),
                color="orange" if case.expected_fracture else "green",
            )
            st.caption(
                f"Región: {case.anatomy} · Proyección: {case.projection}."
            )
        else:
            st.file_uploader(
                "Suba una radiografía en formato JPG o PNG",
                type=["jpg", "jpeg", "png"],
                key="uploaded_xray",
                max_upload_size=20,
                on_change=handle_upload_change,
                help="Tamaño máximo: 20 MB.",
            )
            st.warning(
                "Utilice solo imágenes anonimizadas y sin información identificable.",
                icon=":material/privacy_tip:",
            )


def render_advanced_controls() -> list[tuple[float, float, float, float]]:
    parsed_ground_truth: list[tuple[float, float, float, float]] = []

    with st.popover(
        "Opciones avanzadas",
        icon=":material/tune:",
        width="stretch",
    ):
        st.slider(
            "Umbral de confianza",
            min_value=0.10,
            max_value=0.90,
            value=CONTROL_DEFAULTS["confidence_threshold"],
            step=0.05,
            format="%.2f",
            key=control_key("confidence_threshold"),
            on_change=clear_analysis,
            help="Las detecciones con una confianza inferior no se mostrarán.",
        )
        st.slider(
            "Brillo del visor",
            min_value=0.5,
            max_value=3.0,
            value=CONTROL_DEFAULTS["viewer_brightness"],
            step=0.1,
            format="%.1f",
            key=control_key("viewer_brightness"),
        )
        st.slider(
            "Contraste del visor",
            min_value=0.5,
            max_value=3.0,
            value=CONTROL_DEFAULTS["viewer_contrast"],
            step=0.1,
            format="%.1f",
            key=control_key("viewer_contrast"),
        )
        st.caption(
            "El brillo y el contraste solo modifican la visualización. "
            "No alteran la imagen enviada al modelo."
        )

        st.toggle(
            "Modo de evaluación",
            key="evaluation_mode",
            help="Permite superponer cajas de referencia en formato YOLO normalizado.",
        )
        if st.session_state.evaluation_mode:
            st.text_area(
                "Cajas de referencia",
                key="ground_truth_text",
                placeholder=(
                    "Una caja por línea:\n"
                    "0 0.370996 0.795670 0.314947 0.262517"
                ),
                help="Formato: clase x_centro y_centro ancho alto, con valores entre 0 y 1.",
            )
            if st.session_state.ground_truth_text.strip():
                try:
                    parsed_ground_truth = parse_yolo_boxes(
                        st.session_state.ground_truth_text
                    )
                    st.caption(
                        f"{len(parsed_ground_truth)} caja(s) de referencia válida(s)."
                    )
                except ValueError as exc:
                    st.error(str(exc), icon=":material/error:")

    return parsed_ground_truth


def render_result_summary(result: dict, duration: float, demo_case: object | None) -> None:
    detections = result["detections"]
    num_detections = result["num_detections"]
    max_confidence = max(
        (detection["confidence"] for detection in detections),
        default=None,
    )

    with st.container(horizontal=True):
        st.metric("Hallazgos", num_detections, border=True)
        st.metric(
            "Confianza máxima",
            f"{max_confidence:.1%}" if max_confidence is not None else "—",
            border=True,
        )
        st.metric(
            "Umbral aplicado",
            f"{active_control_values()[0]:.0%}",
            border=True,
        )
        st.metric("Tiempo total", f"{duration:.2f} s", border=True)

    if num_detections:
        st.warning(
            f"Se identificaron {num_detections} detección(es) por encima del "
            "umbral configurado. Revise las regiones señaladas.",
            icon=":material/warning:",
        )
    else:
        st.info(
            "No se identificaron detecciones por encima del umbral configurado.",
            icon=":material/info:",
        )

    if demo_case is not None:
        expected_positive = demo_case.expected_fracture
        predicted_positive = num_detections > 0
        matches_reference = expected_positive == predicted_positive
        with st.container(border=True):
            st.markdown("**Referencia clínica del caso demo**")
            st.badge(
                demo_case.clinical_reference,
                icon=(
                    ":material/personal_injury:"
                    if expected_positive
                    else ":material/check_circle:"
                ),
                color="orange" if expected_positive else "green",
            )
            if matches_reference:
                st.success(
                    "La clasificación binaria del resultado coincide con la referencia.",
                    icon=":material/check_circle:",
                )
            else:
                st.warning(
                    "La clasificación binaria del resultado no coincide con la referencia.",
                    icon=":material/compare_arrows:",
                )


def render_error() -> bool:
    if st.session_state.analysis_error is None:
        return False

    error_type, detail = st.session_state.analysis_error
    messages = {
        "timeout": (
            "El análisis tardó más de lo esperado.",
            "Compruebe la conexión y vuelva a intentarlo.",
        ),
        "unavailable": (
            "El servicio de IA no está disponible.",
            "El Space puede estar iniciándose. Espere unos segundos y reintente.",
        ),
        "response": (
            "No se pudo interpretar la respuesta del servicio.",
            "Vuelva a intentarlo. Si el problema continúa, revise los logs del backend.",
        ),
    }
    title, guidance = messages.get(
        error_type,
        ("No se pudo completar el análisis.", "Vuelva a intentarlo."),
    )
    st.error(f"**{title}** {guidance}", icon=":material/error:")
    with st.expander("Detalle técnico"):
        st.code(detail)
    return True


initialize_state()
render_header()
render_case_selector()

image_bytes, filename, demo_case = current_case()

if image_bytes is None or filename is None:
    st.info(
        "Suba una radiografía anonimizada para preparar el análisis.",
        icon=":material/upload_file:",
    )
    st.stop()

try:
    base_image = normalize_image(image_bytes)
except ImageValidationError as exc:
    st.error(str(exc), icon=":material/broken_image:")
    st.caption("Seleccione otro archivo sin abandonar la sesión.")
    st.stop()

current_signature = analysis_signature(
    image_bytes,
    active_control_values()[0],
)
if (
    st.session_state.analysis_signature is not None
    and st.session_state.analysis_signature != current_signature
):
    clear_analysis()

with st.container(border=True):
    info_col, controls_col = st.columns([3, 1], vertical_alignment="center")
    with info_col:
        st.subheader("2. Revise y analice", anchor=False)
        source_label = "Caso demo" if demo_case is not None else "Imagen subida"
        st.caption(
            f"{source_label} · {filename} · "
            f"{base_image.width} × {base_image.height} px"
        )
    with controls_col:
        ground_truth_boxes = render_advanced_controls()

    analyze_requested = st.button(
        "Analizar con IA",
        type="primary",
        icon=":material/search_insights:",
        width="stretch",
        key="analyze_case",
    )

if analyze_requested:
    run_analysis(image_bytes, filename, current_signature)

_, viewer_brightness, viewer_contrast = active_control_values()
view_image = apply_view_adjustments(
    base_image,
    brightness=viewer_brightness,
    contrast=viewer_contrast,
)
result = st.session_state.analysis_result
detections = result["detections"] if result else []
annotated_view = draw_overlays(view_image, detections, ground_truth_boxes)

viewer_left, viewer_right = st.columns(2, gap="large")
with viewer_left:
    with st.container(border=True):
        st.markdown("**Radiografía original**")
        st.caption("Visualización ajustable. La imagen original no se modifica.")
        st.html(
            stable_viewer_html(
                fit_on_black_canvas(view_image, CANVAS_SIZE),
                "Radiografía original",
            ),
        )

with viewer_right:
    with st.container(border=True):
        st.markdown("**Resultado del análisis**")
        if result is None:
            st.caption("El resultado aparecerá aquí después de ejecutar la IA.")
        else:
            st.caption("IA en azul · Referencia de evaluación en verde.")
        st.html(
            stable_viewer_html(
                fit_on_black_canvas(annotated_view, CANVAS_SIZE),
                "Resultado del análisis de la radiografía",
            ),
        )

if render_error():
    if st.button(
        "Reintentar análisis",
        icon=":material/refresh:",
        type="secondary",
        key="retry_analysis",
    ):
        run_analysis(image_bytes, filename, current_signature)
        st.rerun()

if result is not None:
    st.subheader("3. Resumen del resultado", anchor=False)
    render_result_summary(
        result,
        st.session_state.analysis_duration,
        demo_case,
    )

    export_image = draw_overlays(base_image, detections, ground_truth_boxes)
    st.download_button(
        "Descargar imagen anotada",
        data=image_to_png_bytes(export_image),
        file_name=f"{Path(filename).stem}_anotada.png",
        mime="image/png",
        icon=":material/download:",
        on_click="ignore",
    )

st.space("medium")
st.caption(
    "Herramienta experimental desarrollada para un TFM. No constituye un "
    "diagnóstico médico ni sustituye la valoración de profesionales sanitarios. "
    "Las imágenes se procesan en memoria y no se almacenan de forma persistente."
)
