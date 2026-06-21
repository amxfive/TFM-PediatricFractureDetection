# AGENTS.md

## Project Overview

This repository supports a TFM project for pediatric upper-limb fracture detection in X-ray images using YOLO object detection. The main research direction is comparing:

- A generalist architecture: one detector model for all upper-limb fracture regions.
- A specialist architecture: an anatomical classifier/router selects a region-specific detector.

The expected detection output is bounding boxes for one class: `fracture`.

## Main Architecture

- `src/backend/main.py`: FastAPI inference backend. It currently loads a single YOLO detector (`E6_yoloV8m.pt`) and exposes:
  - `GET /health`
  - `POST /predict`
- `src/frontend/app.py`: Streamlit frontend intended for Docker usage. It uploads JPG/PNG images, calls the backend, and draws detections.
- `docker-compose.yml`: starts backend and frontend containers.

The backend preprocessing path decodes image bytes, normalizes non-8-bit images, applies bilateral denoising, applies CLAHE, converts to RGB, then runs YOLO.

## Models And Data

Generalist model weights:

- `models_weights/generalist_architectures/E3_yoloV8n_optA.pt`
- `models_weights/generalist_architectures/E5_yoloV8s.pt`
- `models_weights/generalist_architectures/E6_yoloV8m.pt`
- `models_weights/generalist_architectures/E7_yoloV11n.pt`

Specialist model weights:

- `models_weights/especialist_architectures/esp_pediURF.pt`
- `models_weights/especialist_architectures/esp_grazpedwri.pt`
- `models_weights/especialist_architectures/esp_pediaSHF.pt`

Classifier/router data:

- `data/processed_2/ExpDataset_classification`
- Classes observed: `supracondylar`, `wrist`, `ulna_radius`
- `models_weights/classifier_models` contains pretrained starting points, but a
  trained router model is still needed before specialist inference can run end
  to end.

Dataset YAMLs for Colab/Kaggle training are in `data/colab_yaml/`.

## Evaluation

Evaluation utilities live under `src/utils/evaluation_compare_results/` and produce or consume Label Studio-style JSON files in `src/evaluation/annotation_json/`.

Useful evaluation concepts already present:

- Generate IA evaluation JSON from YOLO predictions.
- Compare pairwise IoU between human and IA annotations.
- Compute IoU by anatomical zone (`UR`, `WRI`, `SHF`).
- Compute inference/annotation efficiency metrics using `lead_time`.

Existing result tables are under `src/evaluation/results/tables/`.

## Development Notes

- Prefer `rg`/`rg --files` for repo search.
- Use Python 3.10/3.11 compatible code.
- Use existing dependencies: FastAPI, Streamlit, Ultralytics, OpenCV, NumPy, Pillow.
- Keep preprocessing consistent between training/evaluation/backend when changing inference behavior.
- Be careful with large/sensitive artifacts. `data/raw`, `data/processed`, `data/processed_2`, image files, DICOMs, and many generated artifacts are ignored or sensitive.
- Do not move or delete model weights unless explicitly asked.
- Some scripts under `src/utils/dataset_manipulation/` and `src/utils/evaluation_compare_results/` write files when executed. Read them before running.
- PowerShell may not have `git` on PATH in this workspace; `wsl git ...` works from the repository.

## Known Friction Points

- Docker Compose mounts the generalist model directory read-only at
  `/app/models` and selects the active model through `MODEL_PATH`.
- Text encoding appears correct in files, but PowerShell output may show mojibake for Spanish accents.
- The specialist architecture needs a router model and backend routing logic before it is complete.
