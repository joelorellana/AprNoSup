#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación práctica del Análisis de Componentes Principales con Kernel (Kernel PCA)

Este script demuestra la aplicación del Kernel PCA en conjuntos de datos con estructuras
no lineales, comparando diferentes kernels y visualizando los resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles, make_moons, make_swiss_roll, load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances
import seaborn as sns
import time
from mpl_toolkits.mplot3d import Axes3D

# Configuración para visualizaciones más estéticas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def generar_datos_no_lineales():
    """
    Genera varios conjuntos de datos con estructuras no lineales para demostrar Kernel PCA.
    
    Returns:
        dict: Diccionario con los conjuntos de datos generados.
    """
    print("1. Generando conjuntos de datos con estructuras no lineales...")
    
    # Generar datos circulares concéntricos
    X_circles, y_circles = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=42)
    
    # Generar datos en forma de media luna
    X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=42)
    
    # Generar datos en forma de espiral suiza (3D)
    X_swiss_roll, y_swiss_roll = make_swiss_roll(n_samples=500, noise=0.05, random_state=42)
    X_swiss_roll = X_swiss_roll[:, [0, 2]]  # Usar solo 2 dimensiones para simplificar
    
    # Crear un conjunto de datos sintético con estructura no lineal más compleja
    np.random.seed(42)
    
    # Generar datos en forma de S
    t = np.random.uniform(0, 2*np.pi, 500)
    X_s_curve = np.zeros((500, 2))
    X_s_curve[:, 0] = np.sin(t)
    X_s_curve[:, 1] = np.sign(np.sin(t)) * (np.cos(t) - 1)
    X_s_curve += np.random.normal(0, 0.05, X_s_curve.shape)
    y_s_curve = np.zeros(500)
    
    # Mostrar información sobre los conjuntos de datos
    datasets = {
        'Círculos': (X_circles, y_circles),
        'Medias Lunas': (X_moons, y_moons),
        'Rollo Suizo 2D': (X_swiss_roll, y_swiss_roll),
        'Curva S': (X_s_curve, y_s_curve)
    }
    
    for name, (X, y) in datasets.items():
        print(f"   - {name}: {X.shape[0]} muestras, {X.shape[1]} dimensiones, {len(np.unique(y))} clases")
    
    return datasets

def visualizar_datos_originales(datasets):
    """
    Visualiza los conjuntos de datos originales.
    
    Args:
        datasets (dict): Diccionario con los conjuntos de datos.
    """
    print("\n2. Visualizando conjuntos de datos originales...")
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, (X, y)) in enumerate(datasets.items()):
        plt.subplot(2, 2, i+1)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                             edgecolor='k', s=40, alpha=0.8)
        
        if len(np.unique(y)) > 1:
            plt.colorbar(scatter, label='Clase')
        
        plt.title(f'Conjunto de Datos: {name}')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kernel_pca_datos_originales.png', dpi=300, bbox_inches='tight')
    plt.show()

def aplicar_pca_y_kernel_pca(X, kernels=['linear', 'poly', 'rbf', 'sigmoid'], n_components=2):
    """
    Aplica PCA tradicional y Kernel PCA con diferentes kernels a los datos.
    
    Args:
        X (numpy.ndarray): Datos de entrada.
        kernels (list): Lista de kernels a utilizar.
        n_components (int): Número de componentes a retener.
    
    Returns:
        tuple: (pca, kpcas, X_pca, X_kpcas) donde pca y kpcas son los modelos ajustados,
               y X_pca y X_kpcas son los datos transformados.
    """
    # Estandarizar los datos
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Aplicar PCA tradicional
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    
    # Aplicar Kernel PCA con diferentes kernels
    kpcas = {}
    X_kpcas = {}
    
    for kernel in kernels:
        # Configurar parámetros específicos para cada kernel
        if kernel == 'poly':
            kpca = KernelPCA(n_components=n_components, kernel=kernel, degree=3, gamma=10)
        elif kernel == 'rbf':
            kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=10)
        elif kernel == 'sigmoid':
            kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=1, coef0=1)
        else:  # linear
            kpca = KernelPCA(n_components=n_components, kernel=kernel)
        
        # Ajustar y transformar
        X_kpca = kpca.fit_transform(X_std)
        
        kpcas[kernel] = kpca
        X_kpcas[kernel] = X_kpca
    
    return pca, kpcas, X_pca, X_kpcas

def visualizar_transformaciones(datasets, kernels=['linear', 'poly', 'rbf', 'sigmoid']):
    """
    Visualiza las transformaciones de los datos usando PCA tradicional y Kernel PCA.
    
    Args:
        datasets (dict): Diccionario con los conjuntos de datos.
        kernels (list): Lista de kernels a utilizar.
    """
    print("\n3. Visualizando transformaciones con diferentes kernels...")
    
    for name, (X, y) in datasets.items():
        print(f"\n   - Procesando conjunto de datos: {name}")
        
        # Aplicar PCA y Kernel PCA
        pca, kpcas, X_pca, X_kpcas = aplicar_pca_y_kernel_pca(X, kernels)
        
        # Visualizar resultados
        plt.figure(figsize=(15, 10))
        
        # PCA tradicional
        plt.subplot(2, 3, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                             edgecolor='k', s=40, alpha=0.8)
        
        if len(np.unique(y)) > 1:
            plt.colorbar(scatter, label='Clase')
        
        plt.title(f'PCA Tradicional')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        
        # Kernel PCA para cada kernel
        for i, kernel in enumerate(kernels):
            plt.subplot(2, 3, i+2)
            scatter = plt.scatter(X_kpcas[kernel][:, 0], X_kpcas[kernel][:, 1], c=y, cmap='viridis', 
                                 edgecolor='k', s=40, alpha=0.8)
            
            if len(np.unique(y)) > 1:
                plt.colorbar(scatter, label='Clase')
            
            plt.title(f'Kernel PCA ({kernel})')
            plt.xlabel('KPC1')
            plt.ylabel('KPC2')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Transformaciones para {name}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'kernel_pca_{name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()

def analizar_parametros_kernel(X, y, kernel='rbf'):
    """
    Analiza el efecto de los parámetros del kernel en la transformación.
    
    Args:
        X (numpy.ndarray): Datos de entrada.
        y (numpy.ndarray): Etiquetas de clase.
        kernel (str): Kernel a analizar.
    """
    print(f"\n4. Analizando parámetros del kernel {kernel}...")
    
    # Estandarizar los datos
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Configurar parámetros a analizar
    if kernel == 'rbf':
        param_name = 'gamma'
        param_values = [0.01, 0.1, 1, 10, 100]
        param_label = 'Gamma'
    elif kernel == 'poly':
        param_name = 'degree'
        param_values = [2, 3, 4, 5, 6]
        param_label = 'Grado'
    elif kernel == 'sigmoid':
        param_name = 'gamma'
        param_values = [0.01, 0.1, 1, 10, 100]
        param_label = 'Gamma'
    else:
        print(f"   - El kernel {kernel} no tiene parámetros relevantes para analizar.")
        return
    
    # Aplicar Kernel PCA con diferentes valores del parámetro
    plt.figure(figsize=(15, 10))
    
    for i, param_value in enumerate(param_values):
        # Configurar Kernel PCA
        if kernel == 'rbf':
            kpca = KernelPCA(n_components=2, kernel=kernel, gamma=param_value)
        elif kernel == 'poly':
            kpca = KernelPCA(n_components=2, kernel=kernel, degree=param_value, gamma=1)
        elif kernel == 'sigmoid':
            kpca = KernelPCA(n_components=2, kernel=kernel, gamma=param_value, coef0=1)
        
        # Ajustar y transformar
        X_kpca = kpca.fit_transform(X_std)
        
        # Visualizar
        plt.subplot(2, 3, i+1)
        scatter = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', 
                             edgecolor='k', s=40, alpha=0.8)
        
        if len(np.unique(y)) > 1:
            plt.colorbar(scatter, label='Clase')
        
        plt.title(f'{param_label} = {param_value}')
        plt.xlabel('KPC1')
        plt.ylabel('KPC2')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Efecto del Parámetro {param_label} en Kernel PCA ({kernel})', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'kernel_pca_parametro_{kernel}_{param_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluar_preservacion_distancias(X, kernels=['linear', 'poly', 'rbf', 'sigmoid']):
    """
    Evalúa cómo cada método preserva las distancias entre puntos.
    
    Args:
        X (numpy.ndarray): Datos de entrada.
        kernels (list): Lista de kernels a utilizar.
    """
    print("\n5. Evaluando preservación de distancias...")
    
    # Estandarizar los datos
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Calcular matriz de distancias en el espacio original
    D_original = pairwise_distances(X_std)
    
    # Aplicar PCA y Kernel PCA
    pca, kpcas, X_pca, X_kpcas = aplicar_pca_y_kernel_pca(X_std, kernels)
    
    # Calcular matriz de distancias en los espacios transformados
    D_pca = pairwise_distances(X_pca)
    
    D_kpcas = {}
    for kernel in kernels:
        D_kpcas[kernel] = pairwise_distances(X_kpcas[kernel])
    
    # Calcular correlación de Spearman entre matrices de distancia
    from scipy.stats import spearmanr
    
    corr_pca = spearmanr(D_original.flatten(), D_pca.flatten())[0]
    
    corr_kpcas = {}
    for kernel in kernels:
        corr_kpcas[kernel] = spearmanr(D_original.flatten(), D_kpcas[kernel].flatten())[0]
    
    # Visualizar resultados
    methods = ['PCA'] + [f'Kernel PCA ({k})' for k in kernels]
    correlations = [corr_pca] + [corr_kpcas[k] for k in kernels]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, correlations, color='skyblue')
    plt.title('Preservación de Distancias (Correlación de Spearman)')
    plt.ylabel('Correlación con Distancias Originales')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kernel_pca_preservacion_distancias.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Imprimir resultados
    print("\n   - Correlación de distancias con el espacio original:")
    print(f"     * PCA tradicional: {corr_pca:.4f}")
    for kernel in kernels:
        print(f"     * Kernel PCA ({kernel}): {corr_kpcas[kernel]:.4f}")

def aplicar_a_digitos():
    """
    Aplica Kernel PCA al conjunto de datos de dígitos manuscritos.
    """
    print("\n6. Aplicando Kernel PCA al conjunto de datos de dígitos manuscritos...")
    
    # Cargar conjunto de datos de dígitos
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    print(f"   - Dimensiones del conjunto de datos: {X.shape}")
    print(f"   - Número de clases: {len(np.unique(y))}")
    
    # Visualizar algunos ejemplos
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(digits.images[i], cmap='gray')
        plt.title(f'Dígito: {y[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('kernel_pca_digitos_ejemplos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Aplicar PCA y Kernel PCA
    kernels = ['linear', 'poly', 'rbf']
    pca, kpcas, X_pca, X_kpcas = aplicar_pca_y_kernel_pca(X, kernels)
    
    # Visualizar resultados
    plt.figure(figsize=(15, 10))
    
    # PCA tradicional
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', 
                         edgecolor='k', s=40, alpha=0.8)
    plt.colorbar(scatter, label='Dígito')
    plt.title('PCA Tradicional')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    # Kernel PCA para cada kernel
    for i, kernel in enumerate(kernels):
        plt.subplot(2, 2, i+2)
        scatter = plt.scatter(X_kpcas[kernel][:, 0], X_kpcas[kernel][:, 1], c=y, cmap='tab10', 
                             edgecolor='k', s=40, alpha=0.8)
        plt.colorbar(scatter, label='Dígito')
        plt.title(f'Kernel PCA ({kernel})')
        plt.xlabel('KPC1')
        plt.ylabel('KPC2')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Proyección de Dígitos Manuscritos', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('kernel_pca_digitos_proyeccion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analizar separabilidad de clases
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    print("\n   - Evaluando separabilidad de clases con clasificador KNN:")
    
    # Función para evaluar
    def evaluate_projection(X_proj, name):
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X_proj, y, cv=5)
        print(f"     * {name}: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores.mean()
    
    # Evaluar cada proyección
    accuracies = [evaluate_projection(X_pca, "PCA tradicional")]
    methods = ['PCA']
    
    for kernel in kernels:
        acc = evaluate_projection(X_kpcas[kernel], f"Kernel PCA ({kernel})")
        accuracies.append(acc)
        methods.append(f'Kernel PCA ({kernel})')
    
    # Visualizar resultados
    plt.figure(figsize=(10, 6))
    plt.bar(methods, accuracies, color='lightgreen')
    plt.title('Precisión de Clasificación KNN en Proyecciones')
    plt.ylabel('Precisión (Validación Cruzada)')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kernel_pca_digitos_clasificacion.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Función principal que ejecuta el flujo completo del análisis Kernel PCA."""
    print("=== Análisis de Componentes Principales con Kernel (Kernel PCA) ===\n")
    
    # Generar conjuntos de datos no lineales
    datasets = generar_datos_no_lineales()
    
    # Visualizar datos originales
    visualizar_datos_originales(datasets)
    
    # Visualizar transformaciones con diferentes kernels
    visualizar_transformaciones(datasets)
    
    # Analizar parámetros del kernel RBF
    X_circles, y_circles = datasets['Círculos']
    analizar_parametros_kernel(X_circles, y_circles, kernel='rbf')
    
    # Analizar parámetros del kernel polinomial
    X_moons, y_moons = datasets['Medias Lunas']
    analizar_parametros_kernel(X_moons, y_moons, kernel='poly')
    
    # Evaluar preservación de distancias
    X_s_curve, y_s_curve = datasets['Curva S']
    evaluar_preservacion_distancias(X_s_curve)
    
    # Aplicar a conjunto de datos de dígitos manuscritos
    aplicar_a_digitos()
    
    print("\n=== Análisis Kernel PCA Completado ===")

if __name__ == "__main__":
    main()
