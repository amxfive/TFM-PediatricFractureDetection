import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import numpy as np

# --- 1. CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(
    page_title="Viamed IA - Traumatología",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar variables de estado (Memoria de la app)
if 'analizado' not in st.session_state:
    st.session_state.analizado = False

# Función para reiniciar el análisis si se sube una imagen nueva
def reset_analisis():
    st.session_state.analizado = False

st.markdown("""
    <style>
    :root {
        --viamed-blue: #005A9C;
        --viamed-light: #E6F0F9;
    }
    
    /* 1. Estilo para las radiografías (Negatoscopio) */
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
    
    /* 2. EXCEPCIÓN: Estilo para el logo en la barra lateral */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        background-color: transparent !important; /* Quita el fondo negro */
        border: none !important;                  /* Quita el borde */
        box-shadow: none !important;              /* Quita la sombra */
        max-height: none !important;              /* Quita el límite de altura */
    }
    
    /* Estilos del botón y tarjetas (Igual que antes) */
    div.stButton > button {
        background-color: var(--viamed-blue);
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
        background-color: var(--viamed-light);
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid var(--viamed-blue);
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA DEL MODELO ---
@st.cache_resource 
def load_model():
    return YOLO("/home/allohi2002/Repositories/TFM-PediatricFractureDetection/weights/yoloV26_grazpedwri/best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error crítico al cargar el modelo de IA: {e}")
    st.stop()

# --- 3. SIDEBAR (PANEL LATERAL Y WINDOWING) ---
with st.sidebar:
    st.image("/home/allohi2002/Repositories/TFM-PediatricFractureDetection/src/interface/logo_VIAMED.png", use_container_width=True)
    st.markdown("---")
    
    st.markdown("### ⚙️ Sensibilidad de IA")
    conf_threshold = st.slider(
        "Umbral de Confianza", 
        min_value=0.10, max_value=0.90, value=0.25, step=0.05
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Visor Radiológico")
    st.caption("Ajuste de Windowing (Brillo/Contraste)")
    
    contraste = st.slider("Contraste (Hueso/Tejido)", 0.5, 3.0, 1.0, 0.1)
    brillo = st.slider("Brillo (Exposición)", 0.5, 3.0, 1.0, 0.1)

# --- 4. ZONA PRINCIPAL ---
st.title("🏥 vIAmed - Sistema de Apoyo al Diagnóstico Radiológico")
st.subheader("Traumatología Pediátrica - Detección de Fracturas por IA")

# OJO AQUÍ: Usamos on_change para resetear la app si suben otra foto
uploaded_file = st.file_uploader("Arrastre o seleccione la radiografía del paciente (JPG/PNG)", type=["jpg", "jpeg", "png"], on_change=reset_analisis)

if uploaded_file:
    # 1. Lectura y normalización base
    raw_img = Image.open(uploaded_file)
    img_array = np.array(raw_img)
    
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype('uint8')
    elif img_array.dtype != np.uint8:
        img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype('uint8')

    img = Image.fromarray(img_array).convert("RGB")
    
    # 2. Aplicar Windowing (Brillo y Contraste)
    if contraste != 1.0:
        enhancer_c = ImageEnhance.Contrast(img)
        img = enhancer_c.enhance(contraste)
        
    if brillo != 1.0:
        enhancer_b = ImageEnhance.Brightness(img)
        img = enhancer_b.enhance(brillo)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🖼️ Radiografía Original")
        st.image(img, use_container_width=True)
        
        # El botón ahora actualiza el session_state
        if st.button("🔍 Analizar Imagen con IA"):
            st.session_state.analizado = True

    with col2:
        st.markdown("#### 🎯 Resultado del Análisis")
        
        # En lugar de comprobar el botón, comprobamos la memoria
        if st.session_state.analizado:
            # Quitamos el st.spinner para que la actualización al mover el slider sea inmediata y sin parpadeos molestos
            results = model.predict(img, conf=conf_threshold, verbose=False) # verbose=False para no saturar tu terminal
            
            res_plotted = results[0].plot(line_width=2)
            res_plotted = res_plotted.astype('uint8')
            res_img = Image.fromarray(res_plotted[:, :, ::-1])
            
            st.image(res_img, use_container_width=True)
            
            st.markdown("---")
            num_fracturas = len(results[0].boxes)
            
            if num_fracturas > 0:
                st.error(f"⚠️ **ALERTA:** Se han detectado {num_fracturas} posible(s) fractura(s).")
                metric_cols = st.columns(min(num_fracturas, 3))
                for i, box in enumerate(results[0].boxes):
                    if i < 3:
                        with metric_cols[i]:
                            conf = float(box.conf[0])
                            st.metric(label=f"Hallazgo {i+1}", value=f"{conf:.1%}")
            else:
                st.success("✅ **NORMAL:** No se han detectado fracturas significativas.")
        else:
            st.info("El sistema está listo. Presione el botón 'Analizar Imagen' para iniciar.")

else:
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 50px;'>
        <h2>Bienvenido al Sistema de Triaje de IA</h2>
        <p>Por favor, cargue una imagen en la sección superior para comenzar el análisis.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("© 2026 Hospital Viamed Santa Ángela de la Cruz. **Aviso Legal:** Este software es una herramienta de cribado en fase experimental. Los resultados generados por el modelo YOLOv8 no constituyen un diagnóstico médico definitivo y deben ser validados por un radiólogo titulado.")