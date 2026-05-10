#!/usr/bin/env python3
"""Genera heatmap de la matriz de Kappa."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    matriz = np.load('src/evaluation/matrix/kappa_matrix.npy')
    
    nombres = ['yoloV8n_optA', 'yoloV8m_optA', 'yoloV11n_optA', 'Usuario_Control']
    n = len(nombres)
    
    half_matrix = np.full((n, n), np.nan)
    
    for i in range(n):
        for j in range(i, n):
            half_matrix[i, j] = matriz[i, j]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(
        half_matrix,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        xticklabels=nombres,
        yticklabels=nombres,
        vmin=-0.5,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={'label': "Kappa de Cohen"},
        ax=ax,
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 12}
    )
    
    ax.set_title('Matriz de Kappa de Cohen (Binario)', fontsize=14)
    ax.set_xlabel('Evaluador', fontsize=12)
    ax.set_ylabel('Evaluador', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('src/evaluation/results/kappa_halfmatrix.png', dpi=500, bbox_inches='tight')
    print("Heatmap guardado en src/evaluation/results/kappa_halfmatrix.png")

if __name__ == '__main__':
    main()