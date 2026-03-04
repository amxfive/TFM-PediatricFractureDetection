import os
import glob
import pandas as pd

# 1. Configuración de rutas
DRIVE_ROOT = "/content/drive/MyDrive/TFM_Fracturas"
EXPERIMENTS_DIR = f"{DRIVE_ROOT}/experiments"
EXCEL_OUTPUT = f"{DRIVE_ROOT}/Resultados_TFM_Anexo.xlsx"

def generar_excel_resultados(ruta_experimentos, archivo_salida):
    print("🔍 Escaneando experimentos en Drive...")
    
    # Buscar todas las carpetas de experimentos
    carpetas_exp = [f.path for f in os.scandir(ruta_experimentos) if f.is_dir()]
    
    lista_resultados = []
    
    for exp in carpetas_exp:
        nombre_exp = os.path.basename(exp)
        ruta_csv = os.path.join(exp, "results.csv")
        
        # Si el experimento terminó y tiene su CSV
        if os.path.exists(ruta_csv):
            # Leer el CSV de Ultralytics
            df = pd.read_csv(ruta_csv)
            
            # Ultralytics mete espacios molestos en los nombres de las columnas, los limpiamos
            df.columns = df.columns.str.strip()
            
            # Coger la MEJOR época (la que tenga mayor mAP50)
            mejor_epoca = df.loc[df['metrics/mAP50(B)'].idxmax()]
            
            # Extraer métricas clave
            diccionario_exp = {
                "ID Experimento": nombre_exp,
                "Mejor Época": int(mejor_epoca['epoch']),
                "Sensibilidad (Recall) %": round(mejor_epoca['metrics/recall(B)'] * 100, 2),
                "Precisión (VPP) %": round(mejor_epoca['metrics/precision(B)'] * 100, 2),
                "mAP@50 %": round(mejor_epoca['metrics/mAP50(B)'] * 100, 2),
                "mAP@50-95 %": round(mejor_epoca['metrics/mAP50-95(B)'] * 100, 2),
                "Loss Entrenamiento (Box)": round(mejor_epoca['train/box_loss'], 4),
                "Loss Validación (Box)": round(mejor_epoca['val/box_loss'], 4)
            }
            lista_resultados.append(diccionario_exp)
            print(f"✅ Añadido: {nombre_exp}")
        else:
            print(f"⚠️ Ignorado (Sin CSV): {nombre_exp}")
            
    # 2. Convertir a DataFrame y ordenar por mAP (de mejor a peor)
    df_final = pd.DataFrame(lista_resultados)
    
    if not df_final.empty:
        df_final = df_final.sort_values(by="mAP@50 %", ascending=False)
        
        # 3. Guardar en Excel
        df_final.to_excel(archivo_salida, index=False)
        print("\n" + "="*50)
        print(f"🎉 EXCEL GENERADO CON ÉXITO: {archivo_salida}")
        print("="*50)
        return df_final
    else:
        print("❌ No se encontraron resultados para exportar.")
        return None

# Ejecutar el script
df_reporte = generar_excel_resultados(EXPERIMENTS_DIR, EXCEL_OUTPUT)

# Mostrar una previsualización en el notebook
if df_reporte is not None:
    display(df_reporte)