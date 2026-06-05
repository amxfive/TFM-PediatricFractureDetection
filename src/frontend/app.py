import streamlit as st
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import numpy as np
import requests
import io
from pathlib import Path

if "res_data" not in st.session_state:
    st.session_state.res_data = {}

if "analizado" not in st.session_state:
    st.session_state.analizado = False

if "ejemplo_activo" not in st.session_state:
    st.session_state.ejemplo_activo = None

st.set_page_config(
    page_title="TFM - Detección de Fracturas Pediátricas",
    page_icon="\U0001fa79",
    layout="wide",
)

st.markdown("""
<style>
:root {
    --medical-blue: #005A9C;
    --medical-light: #E6F0F9;
}
div.stButton > button {
    background-color: var(--medical-blue);
    color: white;
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    transition: all 0.3s;
}
div.stButton > button:hover {
    background-color: #003F70;
    box-shadow: 0 4px 8px rgba(0,90,156,0.3);
}
div[data-testid="metric-container"] {
    background-color: var(--medical-light);
    border-radius: 8px;
    padding: 15px;
    border-left: 5px solid var(--medical-blue);
}
[data-testid="stImage"] img {
    max-height: 70vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    border: 2px solid #eaebf0;
    margin: auto;
    display: block;
    background-color: #000;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

EJEMPLOS_DIR = Path(__file__).parent / "examples"

EJEMPLOS_META = [
    ("09678e1b-distalUR_334257201609240543_front.png", "Fractura", "Distal de radio/cúbito"),
    ("0e91fd4d-midshaftUR_295711201510240018_side.png", "Fractura", "Diáfisis de radio/cúbito"),
    ("310c070b-proximalUR_608317202211280374_side.png", "Fractura", "Proximal de radio/cúbito"),
    ("SHF_001.jpg", "Fractura", "Supracondílea de húmero"),
    ("UR_001.jpg", "Fractura", "Radio/cúbito"),
    ("WRI_001.png", "Fractura", "Muñeca"),
    ("12667489-proximalUR_389715201712230734_side.png", "Sano", "Radio/cúbito proximal"),
    ("3796cf71-distalUR_605664202211070678_side.png", "Sano", "Radio/cúbito distal"),
    ("431064ca-proximalUR_492231202001110685_front.png", "Sano", "Radio/cúbito proximal"),
    ("NoF_UR_001.jpg", "Sano", "Radio/cúbito"),
    ("NoF_UR_002.jpg", "Sano", "Radio/cúbito"),
    ("NoF_UR_003.jpg", "Sano", "Radio/cúbito"),
]


def procesar_imagen(img_bytes):
    raw = Image.open(io.BytesIO(img_bytes))
    arr = np.array(raw)
    if arr.dtype == np.uint16:
        arr = (arr / 256).astype("uint8")
    elif arr.dtype != np.uint8:
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype("uint8")
    img = Image.fromarray(arr).convert("RGB")
    if contraste != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contraste)
    if brillo != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brillo)
    return img


def mostrar_visor(img_bytes, nombre):
    img_visual = procesar_imagen(img_bytes)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### Radiografía Original")
        st.image(img_visual, width="stretch")
        if st.button("Analizar Imagen con IA"):
            st.session_state.res_data = {}
            st.session_state.analizado = True
            st.rerun()

    with col2:
        st.markdown("#### Resultado del Análisis")
        if st.session_state.analizado:
            if not st.session_state.res_data:
                with st.spinner("IA analizando radiografía..."):
                    try:
                        ext = "png" if nombre.lower().endswith(".png") else "jpeg"
                        files = {"file": (nombre, img_bytes, f"image/{ext}")}
                        data_form = {"confidence": conf_threshold}
                        resp = requests.post("http://backend:8000/predict", files=files, data=data_form)
                        resp.raise_for_status()
                        st.session_state.res_data = resp.json()
                    except Exception as e:
                        st.error(f"Error de conexión con el Backend: {e}")
                        st.stop()

            img_dibujo = img_visual.copy()
            draw = ImageDraw.Draw(img_dibujo)
            for det in st.session_state.res_data["detections"]:
                x1, y1, x2, y2 = det["xyxy"][0]
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=5)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((x1, int(y1) - 25), f"{det['confidence']:.0%}", fill="blue", font=font)

            st.image(img_dibujo, width="stretch")
            st.markdown("---")
            num_frac = st.session_state.res_data["num_detections"]
            if num_frac > 0:
                st.error(f"\u26a0\ufe0f **ALERTA:** Se han detectado {num_frac} posible(s) fractura(s).")
                cols_met = st.columns(min(num_frac, 3))
                for i, det in enumerate(st.session_state.res_data["detections"]):
                    if i < 3:
                        with cols_met[i]:
                            st.metric(label=f"Hallazgo {i+1}", value=f"{det['confidence']:.1%}")
            else:
                st.success("\u2705 **NORMAL:** No se han detectado fracturas significativas.")
        else:
            st.info("\U0001f448 Haga clic en 'Analizar Imagen con IA' para obtener los resultados.")


with st.sidebar:
    st.markdown("### \u2699\ufe0f Sensibilidad de IA")
    conf_threshold = st.slider(
        "Umbral de Confianza", min_value=0.10, max_value=0.90, value=0.30, step=0.05
    )

    st.markdown("---")
    st.markdown("### \U0001f39b\ufe0f Visor Radiológico")
    st.caption("Ajuste de Windowing (Brillo/Contraste)")
    contraste = st.slider("Contraste (Hueso/Tejido)", 0.5, 3.0, 1.0, 0.1)
    brillo = st.slider("Brillo (Exposición)", 0.5, 3.0, 1.0, 0.1)

    st.markdown("---")
    with st.expander("Ground Truth"):
        st.caption("Superpone coordenadas reales del dataset para validación.")
        st.text_input(
            "Coordenadas YOLO",
            placeholder="Ej: 0.370996 0.79567 0.314947 0.262517",
        )


st.markdown("# \U0001fa79 Sistema de Detección de Fracturas Pediátricas")
st.markdown("##### Traumatología Pediátrica - Apoyo al Diagnóstico por IA")

uploaded_file = st.file_uploader(
    "Arrastre o seleccione la radiografía del paciente (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    on_change=lambda: (
        setattr(st.session_state, "analizado", False),
        setattr(st.session_state, "ejemplo_activo", None),
    ),
)

tab1, tab2 = st.tabs(["\U0001f4c1 Visor Radiológico", "\U0001f4f7 Ejemplos"])

with tab1:
    if uploaded_file is not None:
        st.session_state.ejemplo_activo = None
        uploaded_file.seek(0)
        mostrar_visor(uploaded_file.read(), uploaded_file.name)
    elif st.session_state.ejemplo_activo is not None:
        mostrar_visor(
            st.session_state.ejemplo_activo["bytes"],
            st.session_state.ejemplo_activo["name"],
        )
    else:
        st.markdown("""
        <div style='text-align: center; color: #555; padding: 40px;'>
            <h3>Suba una radiografía pediátrica para comenzar</h3>
            <p>Use el selector de archivos de arriba para cargar una imagen JPG o PNG,
            o seleccione un ejemplo en la pestaña de al lado.</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    if st.session_state.ejemplo_activo is not None:
        st.success("\u2705 Imagen cargada correctamente. Cambie a la pesta\u00f1a 'Visor Radiológico' para analizarla con IA.")

    st.markdown("### Im\u00e1genes de ejemplo")
    st.caption("Haga clic en 'Cargar' para probar el sistema con una imagen precargada")

    for fname, label, tipo in EJEMPLOS_META:
        ruta = EJEMPLOS_DIR / fname
        icono = "\U0001f9b4" if label == "Fractura" else "\u2705"
        color = "#C62828" if label == "Fractura" else "#2E7D32"
        col_a, col_b = st.columns([6, 2])
        with col_a:
            st.markdown(
                f"<span style='color:{color}; font-weight:700;'>{icono} {label}</span>"
                f" &mdash; <code>{fname}</code>"
                f" &nbsp;&nbsp;<small style='color:#888;'>{tipo}</small>",
                unsafe_allow_html=True,
            )
        with col_b:
            if st.button("Cargar", key=f"ej_{fname}", use_container_width=True):
                with open(ruta, "rb") as f:
                    st.session_state.ejemplo_activo = {"bytes": f.read(), "name": fname}
                st.session_state.analizado = False
                st.session_state.res_data = {}
                st.rerun()


st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption(
    "\u00a9 2026 TFM - Detección de Fracturas Pediátricas. "
    "**Aviso Legal:** Este software es una herramienta de cribado en fase experimental. "
    "Los resultados generados por el modelo YOLOv8 no constituyen un diagnóstico médico "
    "definitivo y deben ser validados por un radiólogo titulado."
)
