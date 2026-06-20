# Deteccion de fracturas pediatricas de miembro superior

Sistema experimental de deteccion y localizacion de fracturas en radiografias
pediatricas de miembro superior mediante modelos YOLO.

Este repositorio contiene el codigo desarrollado para un Trabajo Fin de Master
(TFM) centrado en comparar dos estrategias:

- **Arquitectura generalista:** un unico detector procesa radiografias de
  muneca, radio/cubito y humero.
- **Arquitectura especialista:** un clasificador anatomico enruta cada imagen
  hacia un detector entrenado especificamente para su region.

Los modelos generan cajas delimitadoras para una unica clase, `fracture`. La
ausencia de fractura se representa mediante la ausencia de detecciones.

> [!IMPORTANT]
> Este proyecto es una herramienta experimental de investigacion. No constituye
> un producto sanitario, no emite diagnosticos y no sustituye la valoracion de
> profesionales sanitarios.

## Contenido

- [Objetivos](#objetivos)
- [Arquitecturas evaluadas](#arquitecturas-evaluadas)
- [Resultados principales](#resultados-principales)
- [Aplicacion de demostracion](#aplicacion-de-demostracion)
- [Instalacion](#instalacion)
- [Ejecucion con Docker](#ejecucion-con-docker)
- [Ejecucion local](#ejecucion-local)
- [API](#api)
- [Evaluacion por agentes](#evaluacion-por-agentes)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Datos y modelos](#datos-y-modelos)
- [Reproducibilidad](#reproducibilidad)
- [Limitaciones](#limitaciones)
- [Licencia](#licencia)

## Objetivos

El proyecto estudia si un detector unico puede generalizar entre diferentes
regiones del miembro superior pediatrico o si resulta mas conveniente dividir
el problema entre modelos especialistas.

Los objetivos principales son:

1. Combinar conjuntos de datos de distintas regiones anatomicas.
2. Entrenar y comparar diferentes familias y tamanos de YOLO.
3. Evaluar una arquitectura generalista frente a una arquitectura con
   clasificador y detectores especialistas.
4. Comparar las localizaciones de la IA con anotaciones de perfiles humanos con
   diferente experiencia radiologica.
5. Construir una demostracion web reproducible mediante FastAPI, Streamlit y
   Docker.

## Arquitecturas evaluadas

### Modelo generalista

```text
Radiografia
    |
    v
Preprocesamiento
    |
    v
Detector YOLO generalista
    |
    v
Cajas de fractura
```

El detector se entrena con imagenes de muneca, radio/cubito y humero. En la
demostracion actual se utiliza **E6 (YOLOv8m)**, configurable mediante la
variable de entorno `MODEL_PATH`.

### Modelo especialista

```text
Radiografia
    |
    v
Clasificador anatomico
    |
    +----> Muneca ------> Detector WRI
    |
    +----> Radio/cubito -> Detector UR
    |
    +----> Humero ------> Detector SHF
                              |
                              v
                       Cajas de fractura
```

Las clases esperadas por el clasificador son:

- `wrist`
- `ulna_radius`
- `supracondylar`

La evaluacion final combina el clasificador `C12.0` con los detectores
seleccionados para cada region.

## Preprocesamiento

El backend aplica el mismo flujo antes de la inferencia:

1. Decodificacion en escala de grises conservando la profundidad original.
2. Normalizacion a 8 bits cuando la imagen utiliza otra profundidad.
3. Filtro bilateral para reducir ruido preservando bordes.
4. CLAHE para mejorar el contraste local.
5. Conversion a tres canales RGB para la entrada de YOLO.

El procesamiento se realiza en memoria. La API no guarda las radiografias de
forma persistente.

## Resultados principales

El indice de seleccion definido en el trabajo situo a los siguientes modelos
generalistas en las primeras posiciones:

| Posicion | Experimento | Modelo | Indice |
|---:|---|---|---:|
| 1 | E7 | YOLO11n | 0.821 |
| 2 | E6 | YOLOv8m | 0.817 |
| 3 | E5 | YOLOv8s | 0.807 |

En la evaluacion por agentes, E6 fue el modelo con mayor concordancia global
frente al radiologo experto:

| Comparacion con el radiologo experto | E5 | E6 | E7 | Especialista |
|---|---:|---:|---:|---:|
| IoU promedio | 0.246 | **0.309** | 0.285 | 0.297 |
| Coincidencia con IoU > 0.2 | 62.11 % | **76.23 %** | 72.39 % | 72.20 % |
| Coincidencia con IoU > epsilon | 77.02 % | **86.63 %** | 82.33 % | 82.35 % |

Estos valores miden similitud espacial entre anotaciones. No deben
interpretarse como equivalencia diagnostica ni como validacion clinica del
sistema.

Las figuras y tablas generadas se encuentran en:

```text
src/evaluation/results/
|-- images/
`-- tables/
```

## Aplicacion de demostracion

La demostracion sigue una arquitectura de dos servicios:

- **Frontend:** aplicacion Streamlit para seleccionar o cargar una radiografia,
  ajustar el umbral de confianza y visualizar las detecciones.
- **Backend:** API FastAPI que ejecuta el preprocesamiento y la inferencia.

La interfaz incluye:

- Casos de demostracion.
- Carga de JPG, JPEG y PNG anonimizados.
- Ajustes de brillo y contraste solo para la visualizacion.
- Configuracion del umbral de confianza.
- Superposicion opcional de etiquetas YOLO de referencia.
- Descarga de la imagen anotada.
- Avisos de privacidad y de uso exclusivamente experimental.

## Requisitos

- Python 3.10 o 3.11.
- Docker y Docker Compose para la ejecucion recomendada.
- Git.
- GPU compatible con PyTorch opcional. La inferencia tambien puede ejecutarse
  en CPU.

## Instalacion

### Con `uv`

El repositorio incluye `pyproject.toml` y `uv.lock`:

```bash
git clone https://github.com/amxfive/TFM-PediatricFractureDetection.git
cd TFM-PediatricFractureDetection
uv sync
```

### Con `venv` y `pip`

```bash
git clone https://github.com/amxfive/TFM-PediatricFractureDetection.git
cd TFM-PediatricFractureDetection

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

En Windows PowerShell, la activacion del entorno es:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Ejecucion con Docker

La forma recomendada de iniciar la aplicacion completa es:

```bash
docker compose up --build
```

Servicios disponibles:

- Frontend: <http://localhost:8501>
- Backend: <http://localhost:8000>
- Documentacion OpenAPI: <http://localhost:8000/docs>
- Estado del backend: <http://localhost:8000/health>

El archivo `docker-compose.yml` monta los pesos generalistas en modo lectura.
En el estado actual del repositorio, el peso disponible para E6 es:

```text
models_weights/generalist_architectures/E6_yoloV8m.pt
```

Antes de iniciar Docker, compruebe que la variable `MODEL_PATH` de
`docker-compose.yml` apunte a `/app/models/E6_yoloV8m.pt`. La configuracion
historica puede conservar el nombre `E6_yoloV8m_optA.pt`, que no coincide con el
archivo incluido actualmente.

Para detener los servicios:

```bash
docker compose down
```

## Ejecucion local

### Backend

Desde la raiz del repositorio:

```bash
export MODEL_PATH="models_weights/generalist_architectures/E6_yoloV8m.pt"
python3 -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

En PowerShell:

```powershell
$env:MODEL_PATH="models_weights/generalist_architectures/E6_yoloV8m.pt"
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend

En otra terminal:

```bash
export BACKEND_URL="http://localhost:8000/predict"
streamlit run src/frontend/app.py
```

En PowerShell:

```powershell
$env:BACKEND_URL="http://localhost:8000/predict"
streamlit run src/frontend/app.py
```

## API

### Comprobar el estado

```bash
curl http://localhost:8000/health
```

Ejemplo de respuesta:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "E6_yoloV8m.pt",
  "preprocessing_version": "grayscale-anydepth_norm8_bilateral-5-50-50_clahe-2.0-8x8_rgb_v1"
}
```

### Ejecutar una prediccion

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@radiografia.png" \
  -F "confidence=0.40"
```

La respuesta contiene las cajas en coordenadas `xyxy` expresadas en pixeles:

```json
{
  "num_detections": 1,
  "detections": [
    {
      "confidence": 0.87,
      "xyxy": [[245.1, 318.7, 376.4, 449.2]],
      "class_name": "fracture"
    }
  ]
}
```

## Evaluacion por agentes

La evaluacion compara:

- E5, E6 y E7.
- Arquitectura especialista.
- Usuario de control.
- Residente R1 de Radiologia.
- Radiologo experto.

Los JSON utilizan una estructura compatible con las exportaciones de Label
Studio. Los scripts comunes se encuentran en:

```text
src/utils/evaluation_compare_results/
```

### Generar predicciones del modelo especialista

Se necesita un clasificador YOLO de tipo `*-cls.pt` y los tres detectores:

> [!NOTE]
> El repositorio no incluye actualmente el peso definitivo del clasificador en
> `models_weights/classifier_models/`. El ejemplo utiliza el nombre
> `router.pt`; debe copiarse ahi el clasificador entrenado o indicarse su ruta
> real mediante `--router-model`.

```bash
python3 src/utils/evaluation_compare_results/eval_agentIA_specialists.py \
  --router-model models_weights/classifier_models/router.pt \
  --images data/processed_2/EvalDatasetProperID \
  --output src/evaluation/annotation_json/IA_Evaluation_specialist_agents.json \
  --conf 0.30 \
  --classifier-imgsz 224 \
  --detector-imgsz 1024
```

Los especialistas pueden sobrescribirse desde la linea de comandos:

```bash
python3 src/utils/evaluation_compare_results/eval_agentIA_specialists.py \
  --router-model models_weights/classifier_models/router.pt \
  --specialist wrist=models_weights/especialist_architectures/esp_grazpedwri.pt \
  --specialist ulna_radius=models_weights/especialist_architectures/esp_pediURF.pt \
  --specialist supracondylar=models_weights/especialist_architectures/esp_pediaSHF.pt
```

### Generar matrices y graficas

Desde la raiz del repositorio:

```bash
python3 src/utils/evaluation_compare_results/calculate_matrix/calcular_concordancia.py
python3 src/utils/evaluation_compare_results/calculate_matrix/match_rate_matrix.py
python3 src/utils/evaluation_compare_results/iou_by_zone.py
python3 src/utils/evaluation_compare_results/metrics_efficiency.py
```

Las salidas incluyen:

- Matriz de IoU promedio.
- Matriz de IoU promedio por region anatomica.
- Tasa de coincidencia con IoU superior a 0.2.
- Tasa de coincidencia con cualquier solapamiento positivo.
- Grafica y tabla de eficiencia operativa.

### Visualizar anotaciones por imagen

```bash
python3 src/utils/evaluation_compare_results/visualize_and_iou.py
```

El script genera una imagen por caso con las cajas de los agentes y los IoU por
parejas:

```text
src/evaluation/results/images/iou_human_pairs/
```

Tambien genera:

```text
src/evaluation/results/tables/iou_per_image_human_pairs.csv
```

## Estructura del repositorio

```text
.
|-- data/
|   |-- colab_yaml/               # Configuraciones YAML para entrenamiento
|   `-- processed_2/              # Datos procesados locales
|-- models_weights/
|   |-- generalist_architectures/
|   |-- especialist_architectures/
|   `-- classifier_models/
|-- notebooks/                    # Experimentacion y entrenamiento
|-- src/
|   |-- backend/                  # API FastAPI e inferencia
|   |-- frontend/                 # Aplicacion Streamlit actual
|   |-- interface/                # Prototipo Streamlit antiguo
|   |-- evaluation/               # JSON, matrices y resultados
|   `-- utils/                    # Datos, evaluacion y utilidades
|-- docker-compose.yml
|-- pyproject.toml
|-- requirements.txt
`-- uv.lock
```

`src/interface/app.py` corresponde a una prueba de concepto anterior. La
aplicacion mantenida para la demostracion es `src/frontend/app.py`.

## Datos y modelos

El trabajo utiliza conjuntos publicos de radiografias pediatricas, entre ellos:

- **GRAZPEDWRI-DX**, centrado en muneca.
- **PediaSHF**, centrado en fracturas supracondileas de humero.
- **PediURF**, centrado en radio y cubito.

Las imagenes originales no se redistribuyen mediante este repositorio. Deben
obtenerse desde sus fuentes oficiales y utilizarse conforme a sus respectivas
licencias y condiciones de acceso.

Aunque el caso de uso se planteo inicialmente junto al Hospital Viamed Santa
Angela de la Cruz, finalmente no se utilizaron imagenes proporcionadas por el
hospital. Profesionales vinculados al Hospital Universitario Virgen de Valme
colaboraron en el etiquetado del conjunto empleado en la evaluacion por
agentes.

Los pesos presentes en `models_weights/` son artefactos de investigacion. Antes
de redistribuirlos o utilizarlos fuera de este trabajo deben revisarse las
licencias de Ultralytics y de los conjuntos de datos empleados.

## Reproducibilidad

Durante la experimentacion:

- Los entrenamientos se ejecutaron principalmente en Google Colab con GPU
  NVIDIA A100.
- Se utilizo transferencia de aprendizaje desde pesos YOLO preentrenados.
- Los experimentos y metricas se registraron mediante Weights & Biases.
- Se fijaron semillas en las librerias y procesos que lo permitian.
- Los modelos se evaluaron por separado en los conjuntos de test de cada
  region.

La reproducibilidad completa requiere descargar y preparar los datasets
originales. El repositorio facilita el codigo, las configuraciones y las
estructuras utilizadas, pero no incluye todos los datos medicos de entrada.

## Limitaciones

- El sistema solo ha sido estudiado en radiografias pediatricas de miembro
  superior.
- La evaluacion por agentes utiliza un numero limitado de casos y un
  participante por perfil.
- La concordancia de cajas no equivale a exactitud diagnostica.
- No se ha realizado una validacion clinica prospectiva.
- El sistema no controla actualmente todas las posibles imagenes fuera de
  distribucion.
- La aplicacion solo acepta JPG, JPEG y PNG; no procesa DICOM directamente.

## Citacion

Si este repositorio resulta util para otro trabajo, puede citarse
provisionalmente como:

```bibtex
@mastersthesis{pediatric_fracture_detection_tfm,
  title  = {Deteccion de fracturas pediatricas de miembro superior mediante arquitecturas generalistas y especialistas},
  author = {Autor del TFM},
  school = {Universidad Loyola Andalucia},
  year   = {2026}
}
```

Sustituya `Autor del TFM` por el nombre definitivo del autor antes de utilizar
la referencia.

## Licencia

Este repositorio no incluye actualmente un archivo de licencia. En ausencia de
una licencia explicita, el codigo, los pesos y el resto de artefactos no deben
considerarse automaticamente autorizados para su copia, modificacion o
redistribucion.

Las licencias de los datasets y de las dependencias utilizadas se aplican de
forma independiente.

## Agradecimientos

Se agradece la colaboracion de los profesionales que participaron en el
etiquetado y la evaluacion de las radiografias, especialmente la aportacion
realizada desde el Hospital Universitario Virgen de Valme. Tambien se agradece
al Hospital Viamed Santa Angela de la Cruz su participacion en la
contextualizacion inicial del problema clinico.
