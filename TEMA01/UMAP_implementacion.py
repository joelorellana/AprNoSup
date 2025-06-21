#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación práctica de UMAP (Uniform Manifold Approximation and Projection)

Este script demuestra la aplicación de UMAP para visualización y reducción de dimensionalidad,
explorando el efecto de diferentes hiperparámetros y comparando con otras técnicas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, fetch_openml, make_blobs, make_moons, make_swiss_roll
from sklearn.model_selection import train_test_split
import warnings
from mpl_toolkits.mplot3d import Axes3D

# Importar UMAP (requiere instalación: pip install umap-learn)
try:
    import umap
except ImportError:
    print("Error: El paquete 'umap-learn' no está instalado.")
    print("Instálelo con: pip install umap-learn")
    exit(1)

# Suprimir advertencias
warnings.filterwarnings("ignore")

# Configuración para visualizaciones más estéticas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def cargar_datos(dataset_name='digits', n_samples=None):
    """
    Carga un conjunto de datos para demostrar UMAP.
    
    Args:
        dataset_name (str): Nombre del conjunto de datos ('digits', 'mnist', 'fashion_mnist', 'synthetic').
        n_samples (int, optional): Número de muestras a seleccionar. Si es None, se usan todas.
    
    Returns:
        tuple: (X, y, feature_names, class_names) donde X son los datos, y son las etiquetas,
               feature_names son los nombres de las características, y class_names son los nombres de las clases.
    """
    print(f"1. Cargando conjunto de datos: {dataset_name}...")
    
    if dataset_name == 'digits':
        # Conjunto de datos de dígitos manuscritos (8x8)
        digits = load_digits()
        X = digits.data
        y = digits.target
        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        class_names = [str(i) for i in range(10)]
        
    elif dataset_name == 'mnist':
        # Conjunto de datos MNIST (28x28)
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.astype('float32')
        y = mnist.target.astype('int')
        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        class_names = [str(i) for i in range(10)]
        
    elif dataset_name == 'synthetic':
        # Datos sintéticos con estructura no lineal
        n_samples_total = 1000 if n_samples is None else n_samples
        
        # Generar datos en forma de media luna
        X_moons, y_moons = make_moons(n_samples=n_samples_total//2, noise=0.05, random_state=42)
        
        # Generar datos en forma de círculos concéntricos
        X_circles, y_circles = make_circles(n_samples=n_samples_total//2, noise=0.05, factor=0.5, random_state=42)
        
        # Combinar los conjuntos de datos
        X = np.vstack([X_moons, X_circles])
        y = np.hstack([y_moons, y_circles + 2])  # Asignar clases 0,1 a lunas y 2,3 a círculos
        
        feature_names = ['feature_1', 'feature_2']
        class_names = ['Luna 1', 'Luna 2', 'Círculo Interior', 'Círculo Exterior']
        
    else:
        raise ValueError(f"Conjunto de datos '{dataset_name}' no reconocido.")
    
    # Submuestrear si es necesario
    if n_samples is not None and n_samples < X.shape[0]:
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    print(f"   - Dimensiones del conjunto de datos: {X.shape}")
    print(f"   - Número de clases: {len(np.unique(y))}")
    
    return X, y, feature_names, class_names

def visualizar_ejemplos(X, y, dataset_name, class_names):
    """
    Visualiza algunos ejemplos del conjunto de datos.
    
    Args:
        X (numpy.ndarray): Datos.
        y (numpy.ndarray): Etiquetas.
        dataset_name (str): Nombre del conjunto de datos.
        class_names (list): Nombres de las clases.
    """
    print("\n2. Visualizando ejemplos del conjunto de datos...")
    
    if dataset_name in ['digits', 'mnist']:
        # Determinar dimensiones de la imagen
        if dataset_name == 'digits':
            img_dim = 8
        else:  # mnist
            img_dim = 28
        
        # Seleccionar un ejemplo de cada clase
        plt.figure(figsize=(15, 8))
        for i, class_idx in enumerate(np.unique(y)):
            # Encontrar un ejemplo de esta clase
            idx = np.where(y == class_idx)[0][0]
            
            plt.subplot(2, 5, i+1)
            plt.imshow(X[idx].reshape(img_dim, img_dim), cmap='gray')
            plt.title(f'Clase: {class_names[i]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'umap_{dataset_name}_ejemplos.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    elif dataset_name == 'synthetic':
        # Para datos sintéticos, visualizar las 2 dimensiones
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
        plt.colorbar(scatter, label='Clase')
        plt.title('Visualización de Datos Sintéticos')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('umap_synthetic_ejemplos.png', dpi=300, bbox_inches='tight')
        plt.show()

def aplicar_umap(X, n_neighbors_list=[5, 15, 30, 50], min_dist_list=[0.0, 0.1, 0.5, 0.99], n_components=2, random_state=42):
    """
    Aplica UMAP con diferentes hiperparámetros.
    
    Args:
        X (numpy.ndarray): Datos.
        n_neighbors_list (list): Lista de valores de n_neighbors a probar.
        min_dist_list (list): Lista de valores de min_dist a probar.
        n_components (int): Número de componentes a retener.
        random_state (int): Semilla para reproducibilidad.
    
    Returns:
        tuple: (umap_results, X_std) donde umap_results es un diccionario con los resultados de UMAP
               para diferentes hiperparámetros, y X_std son los datos estandarizados.
    """
    print("\n3. Aplicando UMAP con diferentes hiperparámetros...")
    
    # Estandarizar los datos
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Aplicar UMAP con diferentes valores de n_neighbors
    umap_results = {}
    
    # Variar n_neighbors con min_dist fijo
    min_dist_fixed = 0.1
    for n_neighbors in n_neighbors_list:
        print(f"   - Ejecutando UMAP con n_neighbors={n_neighbors}, min_dist={min_dist_fixed}...")
        
        start_time = time.time()
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist_fixed, 
                           n_components=n_components, random_state=random_state)
        X_umap = reducer.fit_transform(X_std)
        elapsed_time = time.time() - start_time
        
        umap_results[f'n_neighbors_{n_neighbors}'] = {
            'embedding': X_umap,
            'time': elapsed_time,
            'reducer': reducer
        }
        
        print(f"     * Tiempo: {elapsed_time:.2f} segundos")
    
    # Variar min_dist con n_neighbors fijo
    n_neighbors_fixed = 15
    for min_dist in min_dist_list:
        print(f"   - Ejecutando UMAP con n_neighbors={n_neighbors_fixed}, min_dist={min_dist}...")
        
        start_time = time.time()
        reducer = umap.UMAP(n_neighbors=n_neighbors_fixed, min_dist=min_dist, 
                           n_components=n_components, random_state=random_state)
        X_umap = reducer.fit_transform(X_std)
        elapsed_time = time.time() - start_time
        
        umap_results[f'min_dist_{min_dist}'] = {
            'embedding': X_umap,
            'time': elapsed_time,
            'reducer': reducer
        }
        
        print(f"     * Tiempo: {elapsed_time:.2f} segundos")
    
    return umap_results, X_std

def visualizar_resultados_umap(umap_results, y, class_names, title_prefix=''):
    """
    Visualiza los resultados de UMAP.
    
    Args:
        umap_results (dict): Resultados de UMAP para diferentes hiperparámetros.
        y (numpy.ndarray): Etiquetas.
        class_names (list): Nombres de las clases.
        title_prefix (str): Prefijo para los títulos de los gráficos.
    """
    print("\n4. Visualizando resultados de UMAP...")
    
    # Visualizar resultados para diferentes valores de n_neighbors
    n_neighbors_keys = [k for k in umap_results.keys() if k.startswith('n_neighbors')]
    
    if n_neighbors_keys:
        plt.figure(figsize=(15, 10))
        
        for i, key in enumerate(n_neighbors_keys):
            n_neighbors = key.split('_')[1]
            X_umap = umap_results[key]['embedding']
            
            plt.subplot(2, 2, i+1)
            scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
            
            # Añadir leyenda si no hay demasiadas clases
            if len(np.unique(y)) <= 10:
                handles, labels = scatter.legend_elements()
                legend = plt.legend(handles, class_names, title="Clases", loc="best")
            
            plt.title(f'UMAP (n_neighbors={n_neighbors}, min_dist=0.1)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{title_prefix}umap_n_neighbors.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Visualizar resultados para diferentes valores de min_dist
    min_dist_keys = [k for k in umap_results.keys() if k.startswith('min_dist')]
    
    if min_dist_keys:
        plt.figure(figsize=(15, 10))
        
        for i, key in enumerate(min_dist_keys):
            min_dist = key.split('_')[1]
            X_umap = umap_results[key]['embedding']
            
            plt.subplot(2, 2, i+1)
            scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
            
            # Añadir leyenda si no hay demasiadas clases
            if len(np.unique(y)) <= 10:
                handles, labels = scatter.legend_elements()
                legend = plt.legend(handles, class_names, title="Clases", loc="best")
            
            plt.title(f'UMAP (n_neighbors=15, min_dist={min_dist})')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{title_prefix}umap_min_dist.png', dpi=300, bbox_inches='tight')
        plt.show()

def comparar_tecnicas(X_std, y, class_names, title_prefix=''):
    """
    Compara UMAP con PCA y t-SNE.
    
    Args:
        X_std (numpy.ndarray): Datos estandarizados.
        y (numpy.ndarray): Etiquetas.
        class_names (list): Nombres de las clases.
        title_prefix (str): Prefijo para los títulos de los gráficos.
    """
    print("\n5. Comparando UMAP con PCA y t-SNE...")
    
    # Aplicar PCA
    start_time = time.time()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    pca_time = time.time() - start_time
    print(f"   - PCA completado en {pca_time:.2f} segundos")
    
    # Aplicar t-SNE
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate='auto', random_state=42)
    X_tsne = tsne.fit_transform(X_std)
    tsne_time = time.time() - start_time
    print(f"   - t-SNE completado en {tsne_time:.2f} segundos")
    
    # Aplicar UMAP
    start_time = time.time()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_std)
    umap_time = time.time() - start_time
    print(f"   - UMAP completado en {umap_time:.2f} segundos")
    
    # Visualizar resultados
    plt.figure(figsize=(18, 6))
    
    # PCA
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
    
    # Añadir leyenda si no hay demasiadas clases
    if len(np.unique(y)) <= 10:
        handles, labels = scatter.legend_elements()
        legend = plt.legend(handles, class_names, title="Clases", loc="best")
    
    plt.title(f'PCA ({pca_time:.2f}s)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    # t-SNE
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
    
    # Añadir leyenda si no hay demasiadas clases
    if len(np.unique(y)) <= 10:
        handles, labels = scatter.legend_elements()
        legend = plt.legend(handles, class_names, title="Clases", loc="best")
    
    plt.title(f't-SNE ({tsne_time:.2f}s)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, alpha=0.3)
    
    # UMAP
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
    
    # Añadir leyenda si no hay demasiadas clases
    if len(np.unique(y)) <= 10:
        handles, labels = scatter.legend_elements()
        legend = plt.legend(handles, class_names, title="Clases", loc="best")
    
    plt.title(f'UMAP ({umap_time:.2f}s)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title_prefix}comparacion_tecnicas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Comparar tiempos de ejecución
    plt.figure(figsize=(10, 6))
    plt.bar(['PCA', 't-SNE', 'UMAP'], [pca_time, tsne_time, umap_time], color=['blue', 'orange', 'green'])
    plt.title('Comparación de Tiempos de Ejecución')
    plt.ylabel('Tiempo (segundos)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title_prefix}comparacion_tiempos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluar separación de clases
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    print("\n   - Evaluando separación de clases con clasificador KNN:")
    
    # Dividir datos en entrenamiento y prueba
    X_pca_train, X_pca_test, X_tsne_train, X_tsne_test, X_umap_train, X_umap_test, y_train, y_test = train_test_split(
        X_pca, X_tsne, X_umap, y, test_size=0.3, random_state=42, stratify=y)
    
    # Clasificador KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Evaluar en proyección PCA
    knn.fit(X_pca_train, y_train)
    pca_score = knn.score(X_pca_test, y_test)
    print(f"     * Precisión con PCA: {pca_score:.4f}")
    
    # Evaluar en proyección t-SNE
    knn.fit(X_tsne_train, y_train)
    tsne_score = knn.score(X_tsne_test, y_test)
    print(f"     * Precisión con t-SNE: {tsne_score:.4f}")
    
    # Evaluar en proyección UMAP
    knn.fit(X_umap_train, y_train)
    umap_score = knn.score(X_umap_test, y_test)
    print(f"     * Precisión con UMAP: {umap_score:.4f}")
    
    # Visualizar comparación de precisión
    plt.figure(figsize=(10, 6))
    plt.bar(['PCA', 't-SNE', 'UMAP'], [pca_score, tsne_score, umap_score], color=['blue', 'orange', 'green'])
    plt.title('Comparación de Precisión de Clasificación')
    plt.ylabel('Precisión')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title_prefix}comparacion_precision.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Función principal que ejecuta el flujo completo del análisis UMAP."""
    print("=== Análisis de Uniform Manifold Approximation and Projection (UMAP) ===\n")
    
    # Cargar y visualizar datos
    X, y, feature_names, class_names = cargar_datos('digits')
    visualizar_ejemplos(X, y, 'digits', class_names)
    
    # Aplicar UMAP con diferentes parámetros
    umap_results, X_std = aplicar_umap(X)
    
    # Visualizar resultados
    visualizar_resultados_umap(umap_results, y, class_names)
    
    # Comparar con otras técnicas
    comparar_tecnicas(X_std, y, class_names)
    
    print("\n=== Análisis UMAP Completado ===")

if __name__ == "__main__":
    main()
