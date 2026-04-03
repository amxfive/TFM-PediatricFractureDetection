import wandb

# 1. Iniciar sesión y proyecto
run = wandb.init(project="TFM_Fracturas", job_type="dataset-upload")

# 2. Crear el objeto 'Artifact'
# Ponle un nombre descriptivo y el tipo 'dataset'
artifact = wandb.Artifact('FracAtlas', type='dataset')

# 3. Añadir la carpeta completa (imágenes y etiquetas)
artifact.add_dir('data/processed_2/FracAtlas_YOLO')

# 4. Subir y cerrar
run.log_artifact(artifact)
run.finish()