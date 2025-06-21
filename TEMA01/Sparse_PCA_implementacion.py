#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación práctica del Análisis de Componentes Principales Disperso (Sparse PCA)

Este script demuestra la aplicación del Sparse PCA en un conjunto de datos
multidimensional, incluyendo la comparación con PCA tradicional, visualización
de la dispersión en los componentes y análisis de la varianza explicada.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA
from sklearn.datasets import fetch_california_housing
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suprimir advertencias de convergencia
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Suprimir advertencia de obsolescencia del dataset Boston
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuración para visualizaciones más estéticas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def cargar_y_preparar_datos():
    """
    Carga el conjunto de datos California Housing y prepara los datos para Sparse PCA.
    
    Returns:
        tuple: (X_std, feature_names, df) donde X_std son los datos estandarizados,
               feature_names son los nombres de las características, y df es el DataFrame original.
    """
    print("1. Cargando y preparando los datos...")
    
    # Cargar el conjunto de datos California Housing
    california = fetch_california_housing()
    X = california.data
    feature_names = california.feature_names
    
    # Crear un DataFrame para mejor manipulación
    df = pd.DataFrame(X, columns=feature_names)
    
    print(f"   - Dimensiones del conjunto de datos: {df.shape}")
    print(f"   - Características: {', '.join(feature_names)}")
    
    # Mostrar estadísticas descriptivas
    print("\n   - Estadísticas descriptivas:")
    print(df.describe().round(2))
    
    # Estandarizar los datos (media=0, desviación estándar=1)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    print("\n   - Datos estandarizados correctamente")
    
    return X_std, feature_names, df

def aplicar_pca_y_sparse_pca(X_std, feature_names, n_components=5):
    """
    Aplica PCA tradicional y Sparse PCA a los datos estandarizados.
    
    Args:
        X_std (numpy.ndarray): Datos estandarizados
        feature_names (list): Nombres de las características
        n_components (int): Número de componentes a retener
    
    Returns:
        tuple: (pca, spcas, X_pca, X_spcas, alphas) donde pca y spcas son los modelos ajustados,
               X_pca y X_spcas son los datos transformados, y alphas son los valores de regularización.
    """
    print(f"\n2. Aplicando PCA tradicional y Sparse PCA con {n_components} componentes...")
    
    # Aplicar PCA tradicional
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    
    # Aplicar Sparse PCA con diferentes niveles de alpha (parámetro de regularización)
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0]
    spcas = []
    X_spcas = []
    
    for alpha in alphas:
        spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42, max_iter=1000)
        X_spca = spca.fit_transform(X_std)
        spcas.append(spca)
        X_spcas.append(X_spca)
        
        # Calcular la dispersión (porcentaje de coeficientes exactamente cero)
        n_zeros = np.sum(spca.components_ == 0)
        total_elements = spca.components_.size
        sparsity = n_zeros / total_elements * 100
        
        print(f"   - Sparse PCA (alpha={alpha}): {sparsity:.2f}% de coeficientes son cero")
    
    return pca, spcas, X_pca, X_spcas, alphas

def comparar_componentes(pca, spcas, feature_names, alphas):
    """
    Compara los componentes principales entre PCA tradicional y Sparse PCA.
    
    Args:
        pca (sklearn.decomposition.PCA): Modelo PCA tradicional ajustado
        spcas (list): Lista de modelos Sparse PCA ajustados con diferentes alphas
        feature_names (list): Nombres de las características
        alphas (list): Valores de alpha utilizados para los modelos Sparse PCA
    """
    print("\n3. Comparando componentes principales...")
    
    n_components = pca.components_.shape[0]
    n_features = len(feature_names)
    
    # Crear un DataFrame para los componentes de PCA tradicional
    pca_df = pd.DataFrame(pca.components_, 
                         columns=feature_names,
                         index=[f'PC{i+1}' for i in range(n_components)])
    
    # Visualizar los componentes de PCA tradicional
    plt.figure(figsize=(14, 4))
    sns.heatmap(pca_df, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title('Componentes Principales - PCA Tradicional')
    plt.tight_layout()
    plt.savefig('sparse_pca_tradicional_componentes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualizar y comparar los componentes para diferentes valores de alpha
    for i, (spca, alpha) in enumerate(zip(spcas, alphas)):
        spca_df = pd.DataFrame(spca.components_, 
                              columns=feature_names,
                              index=[f'SPC{i+1}' for i in range(n_components)])
        
        plt.figure(figsize=(14, 4))
        sns.heatmap(spca_df, cmap='coolwarm', annot=True, fmt=".2f")
        plt.title(f'Componentes Principales - Sparse PCA (alpha={alpha})')
        plt.tight_layout()
        plt.savefig(f'sparse_pca_alpha_{alpha}_componentes.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Visualizar la evolución de la dispersión para un componente específico
    component_to_show = 0  # Primer componente
    
    plt.figure(figsize=(12, 6))
    
    # Añadir el componente de PCA tradicional como referencia
    plt.bar(range(n_features), pca.components_[component_to_show], alpha=0.3, color='gray', label='PCA Tradicional')
    
    # Añadir los componentes de Sparse PCA para diferentes alphas
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    for i, (spca, alpha, color) in enumerate(zip(spcas, alphas, colors)):
        plt.bar(range(n_features), spca.components_[component_to_show], alpha=0.7, color=color, label=f'Sparse PCA (alpha={alpha})')
        plt.pause(0.5)  # Pausa para efecto visual
    
    plt.xticks(range(n_features), feature_names, rotation=90)
    plt.title(f'Evolución de la Dispersión en el Primer Componente Principal')
    plt.xlabel('Características')
    plt.ylabel('Coeficientes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sparse_pca_evolucion_dispersion.png', dpi=300, bbox_inches='tight')
    plt.show()

def analizar_dispersion(spcas, alphas, feature_names):
    """
    Analiza la dispersión en los componentes principales para diferentes valores de alpha.
    
    Args:
        spcas (list): Lista de modelos Sparse PCA ajustados con diferentes alphas
        alphas (list): Valores de alpha utilizados para los modelos Sparse PCA
        feature_names (list): Nombres de las características
    """
    print("\n4. Analizando la dispersión en los componentes...")
    
    # Calcular la dispersión para cada valor de alpha
    sparsities = []
    for spca in spcas:
        n_zeros = np.sum(spca.components_ == 0)
        total_elements = spca.components_.size
        sparsity = n_zeros / total_elements * 100
        sparsities.append(sparsity)
    
    # Visualizar la relación entre alpha y dispersión
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, sparsities, 'o-', linewidth=2, markersize=10)
    plt.title('Relación entre Alpha y Dispersión')
    plt.xlabel('Valor de Alpha (Parámetro de Regularización)')
    plt.ylabel('Dispersión (% de Coeficientes = 0)')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sparse_pca_alpha_vs_dispersion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analizar qué características son seleccionadas más frecuentemente
    feature_importance = np.zeros(len(feature_names))
    
    for spca in spcas:
        # Contar cuántas veces cada característica tiene un coeficiente no nulo
        non_zero_mask = (spca.components_ != 0)
        feature_counts = np.sum(non_zero_mask, axis=0)
        feature_importance += feature_counts
    
    # Normalizar
    feature_importance = feature_importance / len(spcas)
    
    # Visualizar la importancia de las características
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_names)), feature_importance, color='teal')
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.title('Frecuencia de Selección de Características en Sparse PCA')
    plt.xlabel('Características')
    plt.ylabel('Frecuencia de Selección (Promedio)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sparse_pca_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualizar_datos_transformados(X_pca, X_spcas, alphas):
    """
    Visualiza los datos transformados por PCA tradicional y Sparse PCA.
    
    Args:
        X_pca (numpy.ndarray): Datos transformados por PCA tradicional
        X_spcas (list): Lista de datos transformados por Sparse PCA con diferentes alphas
        alphas (list): Valores de alpha utilizados para los modelos Sparse PCA
    """
    print("\n5. Visualizando los datos transformados...")
    
    # Visualizar los datos transformados en 2D
    plt.figure(figsize=(15, 10))
    
    # PCA tradicional
    plt.subplot(2, 3, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=30)
    plt.title('PCA Tradicional')
    plt.xlabel('Primer Componente')
    plt.ylabel('Segundo Componente')
    plt.grid(True, alpha=0.3)
    
    # Sparse PCA para diferentes valores de alpha
    for i, (X_spca, alpha) in enumerate(zip(X_spcas, alphas)):
        plt.subplot(2, 3, i+2)
        plt.scatter(X_spca[:, 0], X_spca[:, 1], alpha=0.7, s=30)
        plt.title(f'Sparse PCA (alpha={alpha})')
        plt.xlabel('Primer Componente')
        plt.ylabel('Segundo Componente')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sparse_pca_datos_transformados.png', dpi=300, bbox_inches='tight')
    plt.show()

def reconstruir_y_comparar(X_std, pca, spcas, alphas):
    """
    Reconstruye los datos originales a partir de PCA tradicional y Sparse PCA,
    y compara los errores de reconstrucción.
    
    Args:
        X_std (numpy.ndarray): Datos estandarizados originales
        pca (sklearn.decomposition.PCA): Modelo PCA tradicional ajustado
        spcas (list): Lista de modelos Sparse PCA ajustados con diferentes alphas
        alphas (list): Valores de alpha utilizados para los modelos Sparse PCA
    """
    print("\n6. Reconstruyendo datos y comparando errores...")
    
    # Transformar y reconstruir con PCA tradicional
    X_pca = pca.transform(X_std)
    X_pca_reconstructed = pca.inverse_transform(X_pca)
    pca_error = np.mean((X_std - X_pca_reconstructed) ** 2)
    
    print(f"   - Error de reconstrucción con PCA tradicional: {pca_error:.6f}")
    
    # Transformar y reconstruir con Sparse PCA para diferentes alphas
    spca_errors = []
    
    for i, (spca, alpha) in enumerate(zip(spcas, alphas)):
        # Transformar los datos con Sparse PCA
        X_spca = spca.transform(X_std)
        # Reconstrucción con Sparse PCA
        X_spca_reconstructed = np.dot(X_spca, spca.components_)
        spca_error = np.mean((X_std - X_spca_reconstructed) ** 2)
        spca_errors.append(spca_error)
        
        print(f"   - Error de reconstrucción con Sparse PCA (alpha={alpha}): {spca_error:.6f}")
    
    # Visualizar la comparación de errores
    plt.figure(figsize=(10, 6))
    plt.axhline(y=pca_error, color='r', linestyle='-', label='PCA Tradicional')
    plt.plot(alphas, spca_errors, 'o-', linewidth=2, markersize=10, label='Sparse PCA')
    plt.title('Comparación de Errores de Reconstrucción')
    plt.xlabel('Valor de Alpha (Parámetro de Regularización)')
    plt.ylabel('Error de Reconstrucción Medio')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sparse_pca_errores_reconstruccion.png', dpi=300, bbox_inches='tight')
    plt.show()

def caso_practico_interpretacion(spcas, alphas, feature_names, df):
    """
    Presenta un caso práctico de interpretación de los componentes dispersos.
    
    Args:
        spcas (list): Lista de modelos Sparse PCA ajustados con diferentes alphas
        alphas (list): Valores de alpha utilizados para los modelos Sparse PCA
        feature_names (list): Nombres de las características
        df (pandas.DataFrame): DataFrame original con los datos
    """
    print("\n7. Caso práctico: Interpretación de componentes dispersos...")
    
    # Seleccionar un modelo Sparse PCA con un nivel de dispersión razonable
    selected_idx = 2  # Índice correspondiente a alpha=0.5 (ajustar según sea necesario)
    selected_spca = spcas[selected_idx]
    selected_alpha = alphas[selected_idx]
    
    print(f"   - Analizando Sparse PCA con alpha={selected_alpha}")
    
    # Analizar cada componente principal
    for i, component in enumerate(selected_spca.components_):
        # Identificar características con coeficientes no nulos
        non_zero_indices = np.where(component != 0)[0]
        non_zero_features = [feature_names[idx] for idx in non_zero_indices]
        non_zero_values = component[non_zero_indices]
        
        # Ordenar por magnitud
        sorted_indices = np.argsort(np.abs(non_zero_values))[::-1]
        sorted_features = [non_zero_features[idx] for idx in sorted_indices]
        sorted_values = non_zero_values[sorted_indices]
        
        print(f"\n   - Componente {i+1}:")
        print(f"     * Número de características seleccionadas: {len(non_zero_features)} de {len(feature_names)}")
        print(f"     * Características principales (ordenadas por importancia):")
        
        for feat, val in zip(sorted_features, sorted_values):
            print(f"       - {feat}: {val:.4f}")
        
        # Visualizar este componente
        plt.figure(figsize=(12, 5))
        
        # Mostrar todos los coeficientes
        plt.subplot(1, 2, 1)
        plt.bar(range(len(feature_names)), component, color='skyblue')
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.title(f'Componente {i+1} - Todos los Coeficientes')
        plt.grid(True, alpha=0.3)
        
        # Mostrar solo coeficientes no nulos
        plt.subplot(1, 2, 2)
        plt.bar(range(len(sorted_features)), sorted_values, color='teal')
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
        plt.title(f'Componente {i+1} - Solo Coeficientes No Nulos')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'sparse_pca_componente_{i+1}_interpretacion.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análisis de correlación entre las características seleccionadas
        if len(sorted_features) > 1:
            print(f"     * Matriz de correlación entre las características seleccionadas:")
            corr_matrix = df[sorted_features].corr()
            print(corr_matrix.round(2))
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f'Correlación entre Características del Componente {i+1}')
            plt.tight_layout()
            plt.savefig(f'sparse_pca_componente_{i+1}_correlacion.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """Función principal que ejecuta el flujo completo del análisis Sparse PCA."""
    print("=== Análisis de Componentes Principales Disperso (Sparse PCA) ===\n")
    
    # Cargar y preparar los datos
    X_std, feature_names, df = cargar_y_preparar_datos()
    
    # Aplicar PCA tradicional y Sparse PCA
    pca, spcas, X_pca, X_spcas, alphas = aplicar_pca_y_sparse_pca(X_std, feature_names)
    
    # Comparar componentes
    comparar_componentes(pca, spcas, feature_names, alphas)
    
    # Analizar dispersión
    analizar_dispersion(spcas, alphas, feature_names)
    
    # Visualizar datos transformados
    visualizar_datos_transformados(X_pca, X_spcas, alphas)
    
    # Reconstruir y comparar errores
    reconstruir_y_comparar(X_std, pca, spcas, alphas)
    
    # Caso práctico de interpretación
    caso_practico_interpretacion(spcas, alphas, feature_names, df)
    
    print("\n=== Análisis Sparse PCA Completado ===")

if __name__ == "__main__":
    main()
