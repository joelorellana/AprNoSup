#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación práctica del Análisis de Componentes Principales (PCA)

Este script demuestra la aplicación del PCA en un conjunto de datos
multidimensional, incluyendo la preparación de datos, aplicación del algoritmo,
visualización de resultados y análisis de la varianza explicada.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Configuración para visualizaciones más estéticas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def cargar_y_preparar_datos():
    """
    Carga el conjunto de datos Wine y prepara los datos para PCA.
    
    Returns:
        tuple: (X_std, y, feature_names) donde X_std son los datos estandarizados,
               y son las etiquetas de clase, y feature_names son los nombres de las características.
    """
    print("1. Cargando y preparando los datos...")
    
    # Cargar el conjunto de datos Wine
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    
    # Crear un DataFrame para mejor manipulación
    df = pd.DataFrame(X, columns=feature_names)
    
    print(f"   - Dimensiones del conjunto de datos: {df.shape}")
    print(f"   - Características: {', '.join(feature_names)}")
    print(f"   - Número de clases: {len(np.unique(y))}")
    
    # Estandarizar los datos (media=0, desviación estándar=1)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    print("   - Datos estandarizados correctamente")
    
    return X_std, y, feature_names

def aplicar_pca(X_std, n_components=None):
    """
    Aplica PCA a los datos estandarizados.
    
    Args:
        X_std (numpy.ndarray): Datos estandarizados
        n_components (int, optional): Número de componentes a retener. 
                                     Si es None, se retienen todos los componentes.
    
    Returns:
        tuple: (pca, X_pca) donde pca es el modelo PCA ajustado y X_pca son los datos transformados.
    """
    print("\n2. Aplicando PCA...")
    
    # Crear y ajustar el modelo PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    
    print(f"   - Forma de los datos transformados: {X_pca.shape}")
    
    return pca, X_pca

def analizar_varianza_explicada(pca, feature_names):
    """
    Analiza y visualiza la varianza explicada por cada componente principal.
    
    Args:
        pca (sklearn.decomposition.PCA): Modelo PCA ajustado
        feature_names (list): Nombres de las características originales
    """
    print("\n3. Analizando la varianza explicada...")
    
    # Varianza explicada por cada componente
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print("   - Varianza explicada por componente:")
    for i, var in enumerate(explained_variance_ratio):
        print(f"     PC{i+1}: {var:.4f} ({cumulative_variance_ratio[i]:.4f} acumulado)")
    
    # Visualización de la varianza explicada
    plt.figure(figsize=(12, 5))
    
    # Gráfico de barras de varianza explicada
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8, color='skyblue')
    plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', color='red', marker='o')
    plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.7)
    plt.title('Varianza Explicada por Componente')
    plt.xlabel('Componente Principal')
    plt.ylabel('Proporción de Varianza Explicada')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(True, alpha=0.3)
    
    # Scree plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Componente Principal')
    plt.ylabel('Valor Propio')
    plt.xticks(range(1, len(pca.explained_variance_) + 1))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_varianza_explicada.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Análisis de los componentes principales
    print("\n   - Análisis de los componentes principales:")
    components_df = pd.DataFrame(pca.components_, columns=feature_names)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(components_df, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title('Contribución de las Características a cada Componente Principal')
    plt.xlabel('Características Originales')
    plt.ylabel('Componentes Principales')
    plt.tight_layout()
    plt.savefig('pca_componentes.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualizar_datos_pca(X_pca, y):
    """
    Visualiza los datos transformados por PCA en 2D y 3D.
    
    Args:
        X_pca (numpy.ndarray): Datos transformados por PCA
        y (numpy.ndarray): Etiquetas de clase
    """
    print("\n4. Visualizando los datos transformados...")
    
    # Visualización 2D (primeros dos componentes principales)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                         edgecolor='k', s=100, alpha=0.8)
    plt.colorbar(scatter, label='Clase de Vino')
    plt.title('Proyección PCA 2D del Conjunto de Datos Wine')
    plt.xlabel('Primer Componente Principal')
    plt.ylabel('Segundo Componente Principal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualización 3D (primeros tres componentes principales)
    if X_pca.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                           c=y, cmap='viridis', s=100, alpha=0.8)
        plt.colorbar(scatter, label='Clase de Vino', ax=ax)
        ax.set_title('Proyección PCA 3D del Conjunto de Datos Wine')
        ax.set_xlabel('Primer Componente Principal')
        ax.set_ylabel('Segundo Componente Principal')
        ax.set_zlabel('Tercer Componente Principal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pca_3d.png', dpi=300, bbox_inches='tight')
        plt.show()

def reconstruir_datos(pca, X_pca, X_std):
    """
    Reconstruye los datos originales a partir de los componentes principales
    y calcula el error de reconstrucción.
    
    Args:
        pca (sklearn.decomposition.PCA): Modelo PCA ajustado
        X_pca (numpy.ndarray): Datos transformados por PCA
        X_std (numpy.ndarray): Datos originales estandarizados
    """
    print("\n5. Reconstruyendo datos y calculando error...")
    
    # Reconstruir los datos
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Calcular el error de reconstrucción
    reconstruction_error = np.mean((X_std - X_reconstructed) ** 2)
    print(f"   - Error de reconstrucción medio: {reconstruction_error:.6f}")
    
    # Visualizar la reconstrucción para algunas características
    n_features_to_plot = min(4, X_std.shape[1])
    
    plt.figure(figsize=(15, 3 * n_features_to_plot))
    for i in range(n_features_to_plot):
        plt.subplot(n_features_to_plot, 1, i+1)
        plt.plot(X_std[:, i], 'b-', label='Original', alpha=0.7)
        plt.plot(X_reconstructed[:, i], 'r--', label='Reconstruido', alpha=0.7)
        plt.title(f'Característica {i+1}: Original vs. Reconstruida')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_reconstruccion.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Función principal que ejecuta el flujo completo del análisis PCA."""
    print("=== Análisis de Componentes Principales (PCA) ===\n")
    
    # Cargar y preparar los datos
    X_std, y, feature_names = cargar_y_preparar_datos()
    
    # Aplicar PCA manteniendo todos los componentes
    pca, X_pca = aplicar_pca(X_std)
    
    # Analizar la varianza explicada
    analizar_varianza_explicada(pca, feature_names)
    
    # Visualizar los datos transformados
    visualizar_datos_pca(X_pca, y)
    
    # Reconstruir los datos y calcular el error
    reconstruir_datos(pca, X_pca, X_std)
    
    print("\n=== Análisis PCA Completado ===")

if __name__ == "__main__":
    main()
