import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import requests
from PIL import ImageDraw
import io

# --- INICIALIZACIÓN DEL ESTADO (Haz esto siempre al principio) ---
if "res_data" not in st.session_state:
    st.session_state.res_data = {} # Lo creamos como un diccionario vacío

if "analizado" not in st.session_state:
    st.session_state.analizado = False

# --- 1. CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="Viamed IA - Traumatología", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    :root {
        --viamed-blue: #005A9C;
        --viamed-light: #E6F0F9;
    }
    
    /* Estilos del botón general */
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
    
    /* Estilos de las métricas (tarjetas de IA) */
    div[data-testid="metric-container"] {
        background-color: var(--viamed-light);
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid var(--viamed-blue);
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. SISTEMA DE LOGIN ---
# Inicializar el estado de la sesión si no existe
if 'logeado' not in st.session_state:
    st.session_state.logeado = False

# Pantalla de Login (Si no está logeado, mostramos esto y DETENEMOS la ejecución)
if not st.session_state.logeado:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Usamos columnas para centrar el cuadro de login
    _, col_login, _ = st.columns([1, 1, 1])
    
    with col_login:
        st.image("logo_VIAMED.png", width='content')
        st.markdown("<h3 style='text-align: center;'>Acceso Médico</h3>", unsafe_allow_html=True)
        
        # Formulario de credenciales
        with st.form("login_form"):
            usuario = st.text_input("Usuario (ID Médico)")
            password = st.text_input("Contraseña", type="password") # type="password" oculta los caracteres
            submit = st.form_submit_button("Iniciar Sesión", type="primary")
            
            if submit:
                # Credenciales de prueba para tu PoC (Puedes cambiarlas)
                if usuario == "viamed2026" and password == "viamed2026":
                    st.session_state.logeado = True
                    st.rerun() # Recarga la página instantáneamente para entrar a la app
                else:
                    st.error("Credenciales incorrectas. Acceso denegado.")
    
    # Detenemos el script aquí para que no cargue el modelo ni la app si no hay login
    st.stop()

# Inicializar variables de estado (Memoria de la app)
if 'analizado' not in st.session_state:
    st.session_state.analizado = False

# Función para reiniciar el análisis si se sube una imagen nueva
def reset_analisis():
    st.session_state.analizado = False

st.markdown("""
    <style>
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
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        max-height: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR (PANEL LATERAL Y WINDOWING) ---
with st.sidebar:
    st.image("logo_VIAMED.png", width="content")
    st.markdown("---")
    
    st.markdown("### ⚙️ Sensibilidad de IA")
    conf_threshold = st.slider(
        "Umbral de Confianza", 
        min_value=0.10, max_value=0.90, value=0.40, step=0.05
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Visor Radiológico")
    st.caption("Ajuste de Windowing (Brillo/Contraste)")

    contraste = st.slider("Contraste (Hueso/Tejido)", 0.5, 3.0, 1.0, 0.1)
    brillo = st.slider("Brillo (Exposición)", 0.5, 3.0, 1.0, 0.1)

    st.markdown("---")
    with st.expander("Ground Truth"):   
        st.caption("Superpone las coordenadas reales del dataset para validar la precisión de la IA.")
        
        gt_input = st.text_input(
            "Coordenadas YOLO", 
            placeholder="Ej: 0.370996 0.79567 0.314947 0.262517",
            help="Pega aquí los 4 valores (x_center y_center width height) o la línea completa del archivo .txt"
        )
    st.markdown("---")  
    st.markdown("### 🔒 Sesión")
    st.write("Usuario: **medico01**")
    if st.button("Cerrar Sesión"):
        st.session_state.logeado = False
        st.rerun()

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

    # CORRECCIÓN IMPORTANTE: Separamos los flujos
    img_visual = Image.fromarray(img_array).convert("RGB") # Esta es INTOCABLE para el modelo

    # 2. Aplicar Windowing SOLO a la imagen visual
    if contraste != 1.0:
        enhancer_c = ImageEnhance.Contrast(img_visual)
        img_visual = enhancer_c.enhance(contraste)
        
    if brillo != 1.0:
        enhancer_b = ImageEnhance.Brightness(img_visual)
        img_visual = enhancer_b.enhance(brillo)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### Radiografía Original")
        # Mostramos la imagen ajustada por el médico
        st.image(img_visual, width='content')
        
        if st.button("Analizar Imagen con IA"):
            st.session_state.analizado = True

    with col2:
        st.markdown("#### Resultado del Análisis")

        if st.session_state.analizado:
            # 1. Preparar la imagen para enviarla al Backend
            # Convertimos la imagen de Streamlit a bytes para la API
            buf = io.BytesIO()
            img_visual.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # 2. Llamada al Backend (Sustituye 'localhost' por la IP de tu servidor si es necesario)
            if  not(st.session_state.res_data):
                with st.spinner('IA analizando radiografía...'):
                    try:
                        files = {"file": ("imagen.png", byte_im, "image/png")}
                        data_form = {"confidence": conf_threshold, "gt_input": gt_input if gt_input else ""}
                        
                        response = requests.post("http://backend:8000/predict", files=files, data=data_form)
                        response.raise_for_status() # Lanza error si la API falla
                        st.session_state.res_data = response.json()
                    except Exception as e:
                        st.error(f"Error de conexión con el Backend: {e}")
                        st.stop()

            # 3. Dibujar sobre la imagen visual (con brillo ajustado) usando Pillow
            # Creamos una copia para no sobreescribir la original
            img_dibujo = img_visual.copy()
            draw = ImageDraw.Draw(img_dibujo)
            
            # Dibujar PREDICCIONES de la IA (Rojo)
            for det in st.session_state.res_data["detections"]:
                x1, y1, x2, y2 = det["xyxy"][0]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                # Opcional: Escribir la confianza
                draw.text((x1, y1 - 15), f"IA: {det['confidence']:.2%}", fill="red")

            # Dibujar GROUND TRUTH (Verde) si existe en la respuesta
            if st.session_state.res_data.get("ground_truth"):
                gt = st.session_state.res_data["ground_truth"]
                draw.rectangle([gt["x1"], gt["y1"], gt["x2"], gt["y2"]], outline="#00FF00", width=4)
                draw.text((gt["x1"], gt["y1"] - 15), "Ground Truth", fill="#00FF00")

            # 4. Mostrar la imagen final resultante
            st.image(img_dibujo, use_container_width=True)

            # 5. Métricas de hallazgos (usando el JSON del backend)
            st.markdown("---")
            num_fracturas = st.session_state.res_data["num_detections"]
            
            if num_fracturas > 0:
                st.error(f"⚠️ **ALERTA:** Se han detectado {num_fracturas} posible(s) fractura(s).")
                cols = st.columns(min(num_fracturas, 3))
                for i, det in enumerate(st.session_state.res_data["detections"]):
                    if i < 3:
                        with cols[i]:
                            st.metric(label=f"Hallazgo {i+1}", value=f"{det['confidence']:.1%}")
            else:
                st.success("✅ **NORMAL:** No se han detectado fracturas significativas.")
        else:
            # Mensaje mientras no haya análisis
            st.info("👈 Haga clic en 'Analizar Imagen con IA' para revelar los resultados.")
            
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