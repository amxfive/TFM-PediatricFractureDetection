import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

# 1. Cargar datos
input_file = 'data/raw/DatosPacienteVIAMED/metadatos_pacientes_viamed.csv'
df = pd.read_csv(input_file)

# 2. Limpieza de Edad
def clean_age(age_str):
    if pd.isna(age_str) or not isinstance(age_str, str) or age_str == "N/A":
        return np.nan
    try:
        val = int(age_str[:-1])
        unit = age_str[-1].upper()
        if unit == 'Y': return val
        if unit == 'M': return val / 12.0
        if unit == 'D': return val / 365.0
        return float(val)
    except:
        return np.nan

df['Edad_Anos'] = df['Paciente_Edad'].apply(clean_age)
df_unique = df.drop_duplicates(subset=['Paciente_ID']).copy()

# 3. Cálculos Estadísticos
total_img = len(df)
total_pacientes = len(df_unique)
pediatricos = len(df_unique[df_unique['Edad_Anos'] < 18])
edad_media = df_unique['Edad_Anos'].mean()
rango_min = df_unique['Edad_Anos'].min()
rango_max = df_unique['Edad_Anos'].max()

# 4. CONFIGURACIÓN DEL LAYOUT (La clave para que no se corte)
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(10, 14)) # Más alto para que quepa todo holgado

# Dividimos la figura en 3 filas con proporciones: 
# La fila de texto ocupa "1 parte", las gráficas ocupan "2.5 partes" cada una.
gs = gridspec.GridSpec(3, 1, height_ratios=[0.8, 2.5, 2.5], figure=fig)

# --- ZONA SUPERIOR: RESUMEN DE TEXTO ---
ax_text = fig.add_subplot(gs[0])
ax_text.axis('off') # Quitamos ejes para que parezca una hoja en blanco

texto_resumen = (
    f"RESUMEN ESTADÍSTICO - DATASET VIAMED\n"
    f"──────────────────────────────────────────\n"
    f"• Total Imágenes Procesadas:   {total_img}\n"
    f"• Pacientes Únicos Reales:     {total_pacientes}\n"
    f"• Pacientes Pediátricos (<18): {pediatricos}  ({(pediatricos/total_pacientes)*100:.1f}%)\n"
    f"• Edad Media (Cohorte):        {edad_media:.1f} años\n"
    f"• Rango de Edad:               {rango_min:.0f} - {rango_max:.0f} años"
)

# Ponemos el texto centrado en su propia caja
ax_text.text(0.5, 0.5, texto_resumen, 
             ha='center', va='center', fontsize=14, family='monospace',
             bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="gray", alpha=0.9))

# --- ZONA MEDIA: GRÁFICO 1 (ANATOMÍA) ---
ax1 = fig.add_subplot(gs[1])
body_parts = df['Parte_Cuerpo'].value_counts()
sns.barplot(x=body_parts.values, y=body_parts.index, hue=body_parts.index, palette='viridis', ax=ax1, legend=False)
ax1.set_title('Volumen Total de Imágenes por Región Anatómica', fontsize=14, fontweight='bold')
ax1.set_xlabel('Número de Imágenes')
ax1.set_ylabel('')

# --- ZONA INFERIOR: GRÁFICO 2 (EDAD) ---
ax2 = fig.add_subplot(gs[2])
sns.histplot(df_unique['Edad_Anos'].dropna(), bins=15, kde=True, color='salmon', ax=ax2)
ax2.axvline(18, color='red', linestyle='--', linewidth=2, label='Límite Pediátrico (18 años)')
ax2.legend(loc='upper right')
ax2.set_title('Distribución de Edad (Pacientes Únicos)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Edad (Años)')
ax2.set_ylabel('Frecuencia (Nº Pacientes)')

# 5. Ajuste final y Guardado
plt.tight_layout()
output_path = 'data/raw/DatosPacienteVIAMED/dashboard_viamed_final.png'
plt.savefig(output_path, dpi=300)

print(f"✅ Dashboard generado perfectamente en: {output_path}")
print(f"Datos clave: {pediatricos} pacientes pediátricos de {total_pacientes} totales.")