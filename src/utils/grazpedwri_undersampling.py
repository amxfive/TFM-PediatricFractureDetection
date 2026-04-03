import os
import random
import shutil
from tqdm import tqdm

# 1. CONFIGURACIÓN DE RUTAS ORIGEN (Ajusta estas)
SRC_BASE = 'data/raw/GRAZPEDWRI-DX'
# 2. CONFIGURACIÓN DE DESTINO
DEST_BASE = 'data/raw/GRAZ_BALANCED'

# Definición de repartos (Imágenes finales deseadas)
# Para TRAIN aplicamos el equilibrio que hablamos
CONFIG = {
    'train': {'target_pos': 1600, 'target_neg': 200}, # Total 1800
    'valid':   {'target_pos': 200,  'target_neg': 50},  # Val pequeño y rápido
    'test':  {'target_pos': None,  'target_neg': None}  # Test equilibrado
}

def build_yolo_struct(split):
    src_img_dir = os.path.join(SRC_BASE, split, 'images')
    src_lbl_dir = os.path.join(SRC_BASE, split, 'labels')
    
    dest_img_dir = os.path.join(DEST_BASE, split, 'images')
    dest_lbl_dir = os.path.join(DEST_BASE, split, 'labels')
    
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)

    positives, negatives = [], []
    
    # Clasificar archivos del split actual
    imgs = [f for f in os.listdir(src_img_dir) if f.lower().endswith('.png')]
    for img in imgs:
        lbl = img.replace('.png', '.txt')
        lbl_path = os.path.join(src_lbl_dir, lbl)
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            positives.append(img)
        else:
            negatives.append(img)

    # Lógica de selección
    target_pos = CONFIG[split]['target_pos']
    target_neg = CONFIG[split]['target_neg']

    if target_pos is None: # Copiar todo (caso Test)
        selected = positives + negatives
    else:
        sel_pos = random.sample(positives, min(len(positives), target_pos))
        sel_neg = random.sample(negatives, min(len(negatives), target_neg))
        selected = sel_pos + sel_neg

    print(f"📦 Procesando {split}: {len(selected)} imágenes...")
    for img_name in tqdm(selected):
        shutil.copy2(os.path.join(src_img_dir, img_name), os.path.join(dest_img_dir, img_name))
        lbl_name = img_name.replace('.png', '.txt')
        shutil.copy2(os.path.join(src_lbl_dir, lbl_name), os.path.join(dest_lbl_dir, lbl_name))

# Ejecutar para los tres bloques
for s in ['train', 'valid', 'test']:
    build_yolo_struct(s)

print(f"\n✅ Dataset listo en: {DEST_BASE}")