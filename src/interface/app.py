import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import numpy as np
import cv2
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
        st.image("/home/allohi2002/Repositories/TFM-PediatricFractureDetection/src/interface/logo_VIAMED.png", width='content')
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

# --- 2. CARGA DEL MODELO ---
@st.cache_resource 
def load_model():
    return YOLO("src/models/E6_test.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error crítico al cargar el modelo de IA: {e}")
    st.stop()

# --- 3. SIDEBAR (PANEL LATERAL Y WINDOWING) ---
with st.sidebar:
    st.image("/home/allohi2002/Repositories/TFM-PediatricFractureDetection/src/interface/logo_VIAMED.png", width="content")
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
    img_ai = Image.fromarray(img_array).convert("RGB") # Esta es INTOCABLE para el modelo
    img_visual = img_ai.copy()                         # Esta es para los ojos del médico
    
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
            # 1. IA predice sobre la original limpia
            results = model.predict(img_ai, conf=conf_threshold, verbose=False)
            
            # 2. IA dibuja su caja sobre la imagen con el brillo ajustado
            results[0].orig_img = np.array(img_visual) 
            res_plotted = results[0].plot(line_width=2)
            
            # --- NUEVO: DIBUJAR CAJA GROUND TRUTH (VISUAL) ---
           # --- NUEVO: DIBUJAR CAJA GROUND TRUTH (FORMATO YOLO) ---
            if gt_input:
                try:
                    # 1. Limpieza de texto: cambiamos comas por puntos y dividimos por espacios
                    clean_input = gt_input.replace(',', '.')
                    coords = [float(c.strip()) for c in clean_input.split()]
                    
                    # Truco: Si copias toda la línea del txt (ej: "0 0.37..."), ignoramos el primer número (la clase)
                    if len(coords) == 5:
                        coords = coords[1:]
                        
                    if len(coords) == 4:
                        x_c, y_c, w, h = coords
                        
                        # 2. Obtener las dimensiones reales de la imagen en píxeles
                        img_h, img_w, _ = res_plotted.shape
                        
                        # 3. Desnormalizar: Convertir porcentajes YOLO a píxeles absolutos (x1, y1, x2, y2)
                        x1 = int((x_c - w / 2) * img_w)
                        y1 = int((y_c - h / 2) * img_h)
                        x2 = int((x_c + w / 2) * img_w)
                        y2 = int((y_c + h / 2) * img_h)
                        
                        # 4. Dibujar rectángulo VERDE puro
                        cv2.rectangle(res_plotted, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Poner la etiqueta
                        cv2.putText(res_plotted, "Ground Truth", (max(0, x1), max(20, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    st.error("Formato inválido. Pega los números separados por espacios.")
            # --------------------------------------------------------
            # ------------------------------------------------
            
            # 3. Mostrar imagen final
            res_img = Image.fromarray(res_plotted[:, :, ::-1].astype('uint8'))
            st.image(res_img, width='content')
            
            # 4. Métricas de hallazgos
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