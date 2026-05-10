#!/usr/bin/env python3
"""Genera half-matrix de concordancia con diagonal."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    matriz = np.load('src/evaluation/matrix/concordance_matrix.npy')
    
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
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'IoU Concordance'},
        ax=ax,
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 12}
    )
    
    ax.set_title('Matriz de Concordancia (Half-Matrix)', fontsize=14)
    ax.set_xlabel('Evaluador', fontsize=12)
    ax.set_ylabel('Evaluador', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('src/evaluation/results/concordance_halfmatrix.png', dpi=500, bbox_inches='tight')
    print("Half-matrix guardada en src/evaluation/results/concordance_halfmatrix.png")

if __name__ == '__main__':
    main()