# Pediatric Upper-Limb Fracture Detection

Experimental system for the detection and localization of fractures in pediatric upper-limb X-rays using YOLO models.

This repository contains the code developed for a Master's Thesis (TFM) focused on comparing two strategies:

- **Generalist architecture:** a single detector processes wrist, radius/ulna, and humerus X-rays.
- **Specialist architecture:** an anatomical classifier routes each image to a detector specifically trained for its region.

The models generate bounding boxes for a single class, `fracture`. The absence of a fracture is represented by the absence of detections.

> [!IMPORTANT]
> This project is an experimental research tool. It does not constitute a medical device, it does not issue diagnoses, and it does not replace the assessment of healthcare professionals.

## Table of Contents

- [Objectives](#objectives)
- [Evaluated Architectures](#evaluated-architectures)
- [Main Results](#main-results)
- [Demonstration Application](#demonstration-application)
- [Installation](#installation)
- [Running with Docker](#running-with-docker)
- [Local Execution](#local-execution)
- [API](#api)
- [Agent-Based Evaluation](#agent-based-evaluation)
- [Repository Structure](#repository-structure)
- [Data and Models](#data-and-models)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [License](#license)

## Objectives

The project studies whether a single detector can generalize across different pediatric upper-limb regions or if it is more suitable to divide the problem among specialist models.

The main objectives are:

1. Combine datasets from different anatomical regions.
2. Train and compare different YOLO families and sizes.
3. Evaluate a generalist architecture against an architecture with a classifier and specialist detectors.
4. Compare AI localizations with annotations from human profiles with different radiological expertise.
5. Build a reproducible web demonstration using FastAPI, Streamlit, and Docker.

## Evaluated Architectures

### Generalist Model

```text
X-ray
    |
    v
Preprocessing
    |
    v
Generalist YOLO Detector
    |
    v
Fracture Bounding Boxes
```

The detector is trained with images of the wrist, radius/ulna, and humerus. The current demonstration uses **E6 (YOLOv8m)**, which is configurable via the `MODEL_PATH` environment variable.

### Specialist Model

```text
X-ray
    |
    v
Anatomical Classifier
    |
    +----> Wrist ---------> WRI Detector
    |
    +----> Radius/ulna ---> UR Detector
    |
    +----> Humerus -------> SHF Detector
                              |
                              v
                      Fracture Bounding Boxes
```

The classes expected by the classifier are:

- `wrist`
- `ulna_radius`
- `supracondylar`

The final evaluation combines the `C12.0` classifier with the detectors selected for each region.

## Preprocessing

The backend applies the same pipeline before inference:

1. Grayscale decoding preserving the original bit depth.
2. Normalization to 8 bits when the image uses a different depth.
3. Bilateral filter to reduce noise while preserving edges.
4. CLAHE to improve local contrast.
5. Conversion to three-channel RGB for YOLO input.

Processing is done in memory. The API does not persistently save the X-rays.

## Main Results

The selection index defined in the study placed the following generalist models in the top positions:

| Position | Experiment | Model | Index |
|---:|---|---|---:|
| 1 | E7 | YOLO11n | 0.821 |
| 2 | E6 | YOLOv8m | 0.817 |
| 3 | E5 | YOLOv8s | 0.807 |

In the agent-based evaluation, E6 was the model with the highest overall agreement compared to the expert radiologist:

| Comparison with Expert Radiologist | E5 | E6 | E7 | Specialist |
|---|---:|---:|---:|---:|
| Average IoU | 0.246 | **0.309** | 0.285 | 0.297 |
| Match rate with IoU > 0.2 | 62.11 % | **76.23 %** | 72.39 % | 72.20 % |
| Match rate with IoU > epsilon | 77.02 % | **86.63 %** | 82.33 % | 82.35 % |

These values measure spatial similarity between annotations. They should not be interpreted as diagnostic equivalence or as a clinical validation of the system.

The generated figures and tables can be found in:

```text
src/evaluation/results/
|-- images/
`-- tables/
```

## Demonstration Application

The demonstration follows a two-service architecture:

- **Frontend:** Streamlit application to select or upload an X-ray, adjust the confidence threshold, and visualize detections.
- **Backend:** FastAPI API that executes preprocessing and inference.

The interface includes:

- Demonstration cases when images are provided locally.
- Upload of anonymized JPG, JPEG, and PNG files.
- Brightness and contrast adjustments for visualization purposes only.
- Confidence threshold configuration.
- Optional overlay of reference YOLO labels.
- Download of the annotated image.
- Privacy and purely experimental use warnings.

## Requirements

- Python 3.10 or 3.11.
- Docker and Docker Compose for the recommended execution.
- Git.
- PyTorch-compatible GPU (optional). Inference can also run on CPU.

## Installation

### With `uv`

The repository includes `pyproject.toml` and `uv.lock`:

```bash
git clone [https://github.com/amxfive/TFM-PediatricFractureDetection.git](https://github.com/amxfive/TFM-PediatricFractureDetection.git)
cd TFM-PediatricFractureDetection
uv sync
```

### With `venv` and `pip`

```bash
git clone [https://github.com/amxfive/TFM-PediatricFractureDetection.git](https://github.com/amxfive/TFM-PediatricFractureDetection.git)
cd TFM-PediatricFractureDetection

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

In Windows PowerShell, the environment activation is:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Running with Docker

The recommended way to start the full application is:

```bash
docker compose up --build
```

Available services:

- Frontend: <http://localhost:8501>
- Backend: <http://localhost:8000>
- OpenAPI Documentation: <http://localhost:8000/docs>
- Backend Status: <http://localhost:8000/health>

The `docker-compose.yml` file mounts the generalist weights in read-only mode. In the current state of the repository, the available weight for E6 is:

```text
models_weights/generalist_architectures/E6_yoloV8m.pt
```

To stop the services:

```bash
docker compose down
```

## Local Execution

### Backend

From the repository root:

```bash
export MODEL_PATH="models_weights/generalist_architectures/E6_yoloV8m.pt"
python3 -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

In PowerShell:

```powershell
$env:MODEL_PATH="models_weights/generalist_architectures/E6_yoloV8m.pt"
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend

In another terminal:

```bash
export BACKEND_URL="http://localhost:8000/predict"
streamlit run src/frontend/app.py
```

In PowerShell:

```powershell
$env:BACKEND_URL="http://localhost:8000/predict"
streamlit run src/frontend/app.py
```

## API

### Check Status

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "E6_yoloV8m.pt",
  "preprocessing_version": "grayscale-anydepth_norm8_bilateral-5-50-50_clahe-2.0-8x8_rgb_v1"
}
```

### Run a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@radiografia.png" \
  -F "confidence=0.40"
```

The response contains the bounding boxes in `xyxy` coordinates expressed in pixels:

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

## Agent-Based Evaluation

The evaluation compares:

- E5, E6, and E7.
- Specialist architecture.
- Control user.
- Radiology R1 resident.
- Expert radiologist.

The JSON files use a structure compatible with Label Studio exports. Common scripts are located in:

```text
src/utils/evaluation_compare_results/
```

### Generate Specialist Model Predictions

A `*-cls.pt` type YOLO classifier and the three detectors are required:

> [!NOTE]
> The repository currently does not include the final classifier weight in `models_weights/classifier_models/`. The example uses the name `router.pt`; the trained classifier must be copied there, or its actual path must be indicated via `--router-model`.

```bash
python3 src/utils/evaluation_compare_results/eval_agentIA_specialists.py \
  --router-model models_weights/classifier_models/router.pt \
  --images data/processed_2/EvalDatasetProperID \
  --output src/evaluation/annotation_json/IA_Evaluation_specialist_agents.json \
  --conf 0.30 \
  --classifier-imgsz 224 \
  --detector-imgsz 1024
```

The specialists can be overridden from the command line:

```bash
python3 src/utils/evaluation_compare_results/eval_agentIA_specialists.py \
  --router-model models_weights/classifier_models/router.pt \
  --specialist wrist=models_weights/especialist_architectures/esp_grazpedwri.pt \
  --specialist ulna_radius=models_weights/especialist_architectures/esp_pediURF.pt \
  --specialist supracondylar=models_weights/especialist_architectures/esp_pediaSHF.pt
```

### Generate Matrices and Graphs

From the repository root:

```bash
python3 src/utils/evaluation_compare_results/calculate_matrix/calcular_concordancia.py
python3 src/utils/evaluation_compare_results/calculate_matrix/match_rate_matrix.py
python3 src/utils/evaluation_compare_results/iou_by_zone.py
python3 src/utils/evaluation_compare_results/metrics_efficiency.py
```

The outputs include:

- Average IoU matrix.
- Average IoU matrix by anatomical region.
- Match rate with IoU greater than 0.2.
- Match rate with any positive overlap.
- Operational efficiency graph and table.

### Visualize Annotations per Image

```bash
python3 src/utils/evaluation_compare_results/visualize_and_iou.py
```

The script generates one image per case with the agents' boxes and pairwise IoUs:

```text
src/evaluation/results/images/iou_human_pairs/
```

It also generates:

```text
src/evaluation/results/tables/iou_per_image_human_pairs.csv
```

## Repository Structure

```text
.
|-- data/
|   |-- colab_yaml/               # YAML configurations for training
|   `-- processed_2/              # Local processed data
|-- models_weights/
|   |-- generalist_architectures/
|   |-- especialist_architectures/
|   `-- classifier_models/
|-- notebooks/                    # Experimentation and training
|-- src/
|   |-- backend/                  # FastAPI API and inference
|   |-- frontend/                 # Current Streamlit application
|   |-- evaluation/               # JSONs, matrices, and results
|   `-- utils/                    # Data, evaluation, and utilities
|-- docker-compose.yml
|-- pyproject.toml
|-- requirements.txt
`-- uv.lock
```

The application maintained for the demonstration is `src/frontend/app.py`.

## Data and Models

The project uses public pediatric X-ray datasets, including:

- **GRAZPEDWRI-DX**, focused on the wrist.
- **PediaSHF**, focused on supracondylar humerus fractures.
- **PediURF**, focused on the radius and ulna.

The original images are not redistributed through this repository. They must be obtained from their official sources and used in accordance with their respective licenses and access conditions.

Although the use case was initially proposed alongside the Hospital Viamed Santa Angela de la Cruz, ultimately, no images provided by the hospital were used. Professionals linked to the Hospital Universitario Virgen de Valme collaborated in labeling the dataset used in the agent-based evaluation.

The weights present in `models_weights/` are research artifacts. Before redistributing or using them outside of this project, the licenses for Ultralytics and the datasets used must be reviewed.

## Reproducibility

During experimentation:

- Trainings were primarily executed on Google Colab with NVIDIA A100 GPUs.
- Transfer learning from pre-trained YOLO weights was used.
- Experiments and metrics were logged using Weights & Biases.
- Seeds were set in libraries and processes that allowed it.
- Models were evaluated separately on the test sets of each region.

Full reproducibility requires downloading and preparing the original datasets. The repository provides the code, configurations, and structures used, but it does not include all the input medical data.

## Limitations

- The system has only been studied on pediatric upper-limb X-rays.
- The agent-based evaluation uses a limited number of cases and one participant per profile.
- Bounding box agreement does not equate to diagnostic accuracy.
- No prospective clinical validation has been performed.
- The system currently does not handle all possible out-of-distribution images.
- The application only accepts JPG, JPEG, and PNG; it does not process DICOM directly.

## Citation

If this repository is useful for another project, it can be provisionally cited as:

```bibtex
@mastersthesis{pediatric_fracture_detection_tfm,
  title  = {Detección de fracturas pediátricas de miembro superior mediante arquitecturas generalistas y especialistas},
  author = {Álvaro Lorenzo Hidalgo and José María Manzano Crespo},
  school = {Universidad Loyola Andalucía},
  year   = {2026}
}
```

## License

This repository currently does not include a license file. In the absence of an explicit license, the code, weights, and other artifacts should not be considered automatically authorized for copying, modification, or redistribution.

The licenses of the datasets and dependencies used apply independently.

## Acknowledgments

We thank the professionals who participated in the labeling and evaluation of the X-rays, especially the contribution made from the Hospital Universitario Virgen de Valme. We also thank the Hospital Viamed Santa Angela de la Cruz for its participation in the initial contextualization of the clinical problem.
