import os
import random
import shutil
from tqdm import tqdm

def split_dataset(base_path, output_path, train_size=0.8, val_size=0.1, test_size=0.1):
    # Rutas de origen
    img_dir = os.path.join(base_path, 'images')
    lbl_dir = os.path.join(base_path, 'labels')
    
    # Obtener lista de nombres de archivos (sin extensión)
    nombres = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(nombres)
    
    # Calcular índices
    total = len(nombres)
    num_train = int(total * train_size)
    num_val = int(total * val_size)
    
    splits = {
        'train': nombres[:num_train],
        'valid': nombres[num_train:num_train+num_val],
        'test': nombres[num_train+num_val:]
    }
    
    for split, lista_nombres in splits.items():
        # Crear carpetas de destino
        dest_img = os.path.join(output_path, split, 'images')
        dest_lbl = os.path.join(output_path, split, 'labels')
        os.makedirs(dest_img, exist_ok=True)
        os.makedirs(dest_lbl, exist_ok=True)
        
        for name in tqdm(lista_nombres, desc=f"Creando {split}"):
            # Encontrar la extensión original de la imagen
            for ext in ['.jpg', '.png', '.jpeg']:
                if os.path.exists(os.path.join(img_dir, name + ext)):
                    shutil.copy2(os.path.join(img_dir, name + ext), os.path.join(dest_img, name + ext))
                    break
            
            # Copiar el label (.txt)
            if os.path.exists(os.path.join(lbl_dir, name + '.txt')):
                shutil.copy2(os.path.join(lbl_dir, name + '.txt'), os.path.join(dest_lbl, name + '.txt'))

# Uso:
split_dataset('data/raw/PediaSHF-DX-yolo/train', 'data/raw/PediaSHF-DX_split')