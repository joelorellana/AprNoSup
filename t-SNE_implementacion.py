#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación práctica de t-SNE (t-Distributed Stochastic Neighbor Embedding)

Este script demuestra la aplicación de t-SNE para visualización de datos de alta dimensionalidad,
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

# Suprimir advertencias
warnings.filterwarnings("ignore")

# Configuración para visualizaciones más estéticas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def cargar_datos(dataset_name='digits', n_samples=None):
    """
    Carga un conjunto de datos para demostrar t-SNE.
    
    Args:
        dataset_name (str): Nombre del conjunto de datos ('digits', 'mnist', 'fashion_mnist', 'synthetic').
        n_samples (int, optional): Número de muestras a seleccionar. Si es None, se usan todas.
    
    Returns:
        tuple: (X, y, feature_names) donde X son los datos, y son las etiquetas,
               y feature_names son los nombres de las características (si están disponibles).
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
        
    elif dataset_name == 'fashion_mnist':
        # Conjunto de datos Fashion MNIST
        fashion_mnist = fetch_openml('Fashion-MNIST', version=1, parser='auto')
        X = fashion_mnist.data.astype('float32')
        y = fashion_mnist.target.astype('int')
        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 
                      'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']
        
    elif dataset_name == 'synthetic':
        # Datos sintéticos con estructura de cluster
        n_samples_total = 1000 if n_samples is None else n_samples
        n_features = 50
        n_clusters = 5
        
        X, y = make_blobs(n_samples=n_samples_total, n_features=n_features, 
                         centers=n_clusters, cluster_std=2.0, random_state=42)
        
        # Añadir no linealidad
        X = np.hstack([X, 0.1 * np.sin(X[:, 0:1] * 5) + 0.1 * np.cos(X[:, 1:2] * 5)])
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        class_names = [f'Cluster {i}' for i in range(n_clusters)]
        
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
    
    if dataset_name in ['digits', 'mnist', 'fashion_mnist']:
        # Determinar dimensiones de la imagen
        if dataset_name == 'digits':
            img_dim = 8
        else:  # mnist o fashion_mnist
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
        plt.savefig(f't-sne_{dataset_name}_ejemplos.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    elif dataset_name == 'synthetic':
        # Para datos sintéticos, visualizar las primeras 3 dimensiones
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='tab10', s=30, alpha=0.8)
        plt.colorbar(scatter, label='Clase')
        
        ax.set_title('Visualización 3D de Datos Sintéticos')
        ax.set_xlabel('Dimensión 1')
        ax.set_ylabel('Dimensión 2')
        ax.set_zlabel('Dimensión 3')
        
        plt.tight_layout()
        plt.savefig('t-sne_synthetic_ejemplos.png', dpi=300, bbox_inches='tight')
        plt.show()

def aplicar_tsne(X, perplexities=[5, 30, 50], n_iter=1000, learning_rates=[200, 1000], random_state=42):
    """
    Aplica t-SNE con diferentes hiperparámetros.
    
    Args:
        X (numpy.ndarray): Datos.
        perplexities (list): Lista de valores de perplejidad a probar.
        n_iter (int): Número de iteraciones.
        learning_rates (list): Lista de tasas de aprendizaje a probar.
        random_state (int): Semilla para reproducibilidad.
    
    Returns:
        dict: Diccionario con los resultados de t-SNE para diferentes hiperparámetros.
    """
    print("\n3. Aplicando t-SNE con diferentes hiperparámetros...")
    
    # Estandarizar los datos
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Aplicar t-SNE con diferentes perplejidades
    tsne_results = {}
    
    # Primero, variar la perplejidad con tasa de aprendizaje fija
    for perplexity in perplexities:
        print(f"   - Ejecutando t-SNE con perplejidad={perplexity}, learning_rate=auto...")
        
        start_time = time.time()
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                   learning_rate='auto', random_state=random_state)
        X_tsne = tsne.fit_transform(X_std)
        elapsed_time = time.time() - start_time
        
        tsne_results[f'perplexity_{perplexity}'] = {
            'embedding': X_tsne,
            'time': elapsed_time,
            'kl_divergence': tsne.kl_divergence_
        }
        
        print(f"     * Tiempo: {elapsed_time:.2f} segundos")
        print(f"     * Divergencia KL: {tsne.kl_divergence_:.4f}")
    
    # Luego, variar la tasa de aprendizaje con perplejidad fija
    best_perplexity = perplexities[1]  # Usar la perplejidad intermedia
    
    for lr in learning_rates:
        print(f"   - Ejecutando t-SNE con perplejidad={best_perplexity}, learning_rate={lr}...")
        
        start_time = time.time()
        tsne = TSNE(n_components=2, perplexity=best_perplexity, n_iter=n_iter, 
                   learning_rate=lr, random_state=random_state)
        X_tsne = tsne.fit_transform(X_std)
        elapsed_time = time.time() - start_time
        
        tsne_results[f'lr_{lr}'] = {
            'embedding': X_tsne,
            'time': elapsed_time,
            'kl_divergence': tsne.kl_divergence_
        }
        
        print(f"     * Tiempo: {elapsed_time:.2f} segundos")
        print(f"     * Divergencia KL: {tsne.kl_divergence_:.4f}")
    
    return tsne_results, X_std

def visualizar_resultados_tsne(tsne_results, y, class_names, title_prefix=''):
    """
    Visualiza los resultados de t-SNE.
    
    Args:
        tsne_results (dict): Resultados de t-SNE para diferentes hiperparámetros.
        y (numpy.ndarray): Etiquetas.
        class_names (list): Nombres de las clases.
        title_prefix (str): Prefijo para los títulos de los gráficos.
    """
    print("\n4. Visualizando resultados de t-SNE...")
    
    # Visualizar resultados para diferentes perplejidades
    perplexity_keys = [k for k in tsne_results.keys() if k.startswith('perplexity')]
    
    if perplexity_keys:
        plt.figure(figsize=(15, 10))
        
        for i, key in enumerate(perplexity_keys):
            perplexity = key.split('_')[1]
            X_tsne = tsne_results[key]['embedding']
            kl_divergence = tsne_results[key]['kl_divergence']
            
            plt.subplot(2, 2, i+1)
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
            
            # Añadir leyenda si no hay demasiadas clases
            if len(np.unique(y)) <= 10:
                handles, labels = scatter.legend_elements()
                legend = plt.legend(handles, class_names, title="Clases", loc="best")
            
            plt.title(f't-SNE (Perplejidad={perplexity}, KL={kl_divergence:.2f})')
            plt.xlabel('Dimensión 1')
            plt.ylabel('Dimensión 2')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{title_prefix}t-sne_perplejidades.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Visualizar resultados para diferentes tasas de aprendizaje
    lr_keys = [k for k in tsne_results.keys() if k.startswith('lr')]
    
    if lr_keys:
        plt.figure(figsize=(15, 5))
        
        for i, key in enumerate(lr_keys):
            lr = key.split('_')[1]
            X_tsne = tsne_results[key]['embedding']
            kl_divergence = tsne_results[key]['kl_divergence']
            
            plt.subplot(1, 2, i+1)
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
            
            # Añadir leyenda si no hay demasiadas clases
            if len(np.unique(y)) <= 10:
                handles, labels = scatter.legend_elements()
                legend = plt.legend(handles, class_names, title="Clases", loc="best")
            
            plt.title(f't-SNE (Learning Rate={lr}, KL={kl_divergence:.2f})')
            plt.xlabel('Dimensión 1')
            plt.ylabel('Dimensión 2')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{title_prefix}t-sne_learning_rates.png', dpi=300, bbox_inches='tight')
        plt.show()

def comparar_con_pca(X_std, y, class_names, title_prefix=''):
    """
    Compara t-SNE con PCA.
    
    Args:
        X_std (numpy.ndarray): Datos estandarizados.
        y (numpy.ndarray): Etiquetas.
        class_names (list): Nombres de las clases.
        title_prefix (str): Prefijo para los títulos de los gráficos.
    """
    print("\n5. Comparando t-SNE con PCA...")
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # Aplicar t-SNE con hiperparámetros óptimos
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate='auto', random_state=42)
    X_tsne = tsne.fit_transform(X_std)
    
    # Visualizar resultados
    plt.figure(figsize=(15, 6))
    
    # PCA
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
    
    # Añadir leyenda si no hay demasiadas clases
    if len(np.unique(y)) <= 10:
        handles, labels = scatter.legend_elements()
        legend = plt.legend(handles, class_names, title="Clases", loc="best")
    
    plt.title(f'PCA (Varianza Explicada: {pca.explained_variance_ratio_.sum():.2f})')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True, alpha=0.3)
    
    # t-SNE
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
    
    # Añadir leyenda si no hay demasiadas clases
    if len(np.unique(y)) <= 10:
        handles, labels = scatter.legend_elements()
        legend = plt.legend(handles, class_names, title="Clases", loc="best")
    
    plt.title(f't-SNE (KL Divergencia: {tsne.kl_divergence_:.2f})')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title_prefix}comparacion_pca_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluar separación de clases
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    print("\n   - Evaluando separación de clases con clasificador KNN:")
    
    # Dividir datos en entrenamiento y prueba
    X_pca_train, X_pca_test, X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(
        X_pca, X_tsne, y, test_size=0.3, random_state=42, stratify=y)
    
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
    
    # Visualizar comparación
    plt.figure(figsize=(8, 5))
    plt.bar(['PCA', 't-SNE'], [pca_score, tsne_score], color=['skyblue', 'salmon'])
    plt.title('Comparación de Precisión de Clasificación')
    plt.ylabel('Precisión')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title_prefix}comparacion_precision.png', dpi=300, bbox_inches='tight')
    plt.show()

def explorar_inicializacion_tsne(X_std, y, class_names, title_prefix=''):
    """
    Explora el efecto de diferentes inicializaciones en t-SNE.
    
    Args:
        X_std (numpy.ndarray): Datos estandarizados.
        y (numpy.ndarray): Etiquetas.
        class_names (list): Nombres de las clases.
        title_prefix (str): Prefijo para los títulos de los gráficos.
    """
    print("\n6. Explorando el efecto de la inicialización en t-SNE...")
    
    # Inicialización aleatoria
    tsne_random = TSNE(n_components=2, perplexity=30, n_iter=1000, 
                      init='random', random_state=42)
    X_tsne_random = tsne_random.fit_transform(X_std)
    
    # Inicialización con PCA
    tsne_pca = TSNE(n_components=2, perplexity=30, n_iter=1000, 
                   init='pca', random_state=42)
    X_tsne_pca = tsne_pca.fit_transform(X_std)
    
    # Visualizar resultados
    plt.figure(figsize=(15, 6))
    
    # Inicialización aleatoria
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_tsne_random[:, 0], X_tsne_random[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
    
    # Añadir leyenda si no hay demasiadas clases
    if len(np.unique(y)) <= 10:
        handles, labels = scatter.legend_elements()
        legend = plt.legend(handles, class_names, title="Clases", loc="best")
    
    plt.title(f't-SNE (Inicialización Aleatoria, KL={tsne_random.kl_divergence_:.2f})')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid(True, alpha=0.3)
    
    # Inicialización con PCA
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_tsne_pca[:, 0], X_tsne_pca[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
    
    # Añadir leyenda si no hay demasiadas clases
    if len(np.unique(y)) <= 10:
        handles, labels = scatter.legend_elements()
        legend = plt.legend(handles, class_names, title="Clases", loc="best")
    
    plt.title(f't-SNE (Inicialización PCA, KL={tsne_pca.kl_divergence_:.2f})')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title_prefix}tsne_inicializacion.png', dpi=300, bbox_inches='tight')

def visualizar_evolucion_tsne(X_std, y, class_names, title_prefix=''):
    """
    Visualiza la evolución de t-SNE a lo largo de las iteraciones.
    
    Args:
        X_std (numpy.ndarray): Datos estandarizados.
        y (numpy.ndarray): Etiquetas.
        class_names (list): Nombres de las clases.
        title_prefix (str): Prefijo para los títulos de los gráficos.
    """
    print("\n7. Visualizando la evolución de t-SNE a lo largo de las iteraciones...")
    
    # Lista de números de iteraciones a probar
    n_iters = [250, 500, 1000]
    
    # Aplicar t-SNE con diferentes números de iteraciones
    tsne_evolutions = {}
    
    for n_iter in n_iters:
        print(f"   - Ejecutando t-SNE con {n_iter} iteraciones...")
        
        tsne = TSNE(n_components=2, perplexity=30, n_iter=n_iter, 
                   learning_rate='auto', random_state=42)
        X_tsne = tsne.fit_transform(X_std)
        
        tsne_evolutions[n_iter] = {
            'embedding': X_tsne,
            'kl_divergence': tsne.kl_divergence_
        }
        
        print(f"     * KL Divergencia: {tsne.kl_divergence_:.4f}")
    
    # Visualizar evolución
    plt.figure(figsize=(15, 10))
    
    for i, n_iter in enumerate(n_iters):
        X_tsne = tsne_evolutions[n_iter]['embedding']
        kl_divergence = tsne_evolutions[n_iter]['kl_divergence']
        
        plt.subplot(2, 2, i+1)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.8)
        
        # Añadir leyenda si no hay demasiadas clases
        if len(np.unique(y)) <= 10:
            handles, labels = scatter.legend_elements()
            legend = plt.legend(handles, class_names, title="Clases", loc="best")
        
        plt.title(f't-SNE ({n_iter} iteraciones, KL={kl_divergence:.2f})')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title_prefix}tsne_evolucion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualizar convergencia
    plt.figure(figsize=(10, 6))
    plt.plot(n_iters, [tsne_evolutions[n_iter]['kl_divergence'] for n_iter in n_iters], 
            'o-', linewidth=2, markersize=10)
    plt.title('Convergencia de t-SNE')
    plt.xlabel('Número de Iteraciones')
    plt.ylabel('KL Divergencia')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title_prefix}tsne_convergencia.png', dpi=300, bbox_inches='tight')
    plt.show()

def caso_practico_mnist(n_samples=2000):
    """
    Caso práctico: Aplicación de t-SNE al conjunto de datos MNIST.
    
    Args:
        n_samples (int): Número de muestras a utilizar.
    """
    print("\n=== Caso Práctico: Visualización de Dígitos MNIST con t-SNE ===")
    
    # Cargar datos MNIST
    X, y, feature_names, class_names = cargar_datos('mnist', n_samples=n_samples)
    
    # Visualizar ejemplos
    visualizar_ejemplos(X, y, 'mnist', class_names)
    
    # Aplicar t-SNE con diferentes perplejidades
    tsne_results, X_std = aplicar_tsne(X, perplexities=[5, 30, 50])
    
    # Visualizar resultados
    visualizar_resultados_tsne(tsne_results, y, class_names, 'mnist_')
    
    # Comparar con PCA
    comparar_con_pca(X_std, y, class_names, 'mnist_')
    
    # Explorar inicialización
    explorar_inicializacion_tsne(X_std, y, class_names, 'mnist_')
    
    # Visualizar evolución
    visualizar_evolucion_tsne(X_std, y, class_names, 'mnist_')
    
    print("\n=== Caso Práctico MNIST Completado ===")

def caso_practico_fashion_mnist(n_samples=2000):
    """
    Caso práctico: Aplicación de t-SNE al conjunto de datos Fashion MNIST.
    
    Args:
        n_samples (int): Número de muestras a utilizar.
    """
    print("\n=== Caso Práctico: Visualización de Fashion MNIST con t-SNE ===")
    
    # Cargar datos Fashion MNIST
    X, y, feature_names, class_names = cargar_datos('fashion_mnist', n_samples=n_samples)
    
    # Visualizar ejemplos
    visualizar_ejemplos(X, y, 'fashion_mnist', class_names)
    
    # Aplicar t-SNE con diferentes perplejidades
    tsne_results, X_std = aplicar_tsne(X, perplexities=[5, 30, 50])
    
    # Visualizar resultados
    visualizar_resultados_tsne(tsne_results, y, class_names, 'fashion_mnist_')
    
    # Comparar con PCA
    comparar_con_pca(X_std, y, class_names, 'fashion_mnist_')
    
    print("\n=== Caso Práctico Fashion MNIST Completado ===")

def main():
    """Función principal que ejecuta el flujo completo del análisis t-SNE."""
    print("=== Análisis de t-Distributed Stochastic Neighbor Embedding (t-SNE) ===\n")
    
    # Caso práctico con dígitos
    X, y, feature_names, class_names = cargar_datos('digits')
    visualizar_ejemplos(X, y, 'digits', class_names)
    tsne_results, X_std = aplicar_tsne(X)
    visualizar_resultados_tsne(tsne_results, y, class_names)
    comparar_con_pca(X_std, y, class_names)
    explorar_inicializacion_tsne(X_std, y, class_names)
    visualizar_evolucion_tsne(X_std, y, class_names)
    
    # Caso práctico con MNIST (submuestreado)
    # Descomentar para ejecutar (puede ser computacionalmente intensivo)
    # caso_practico_mnist(n_samples=2000)
    
    # Caso práctico con Fashion MNIST (submuestreado)
    # Descomentar para ejecutar (puede ser computacionalmente intensivo)
    # caso_practico_fashion_mnist(n_samples=2000)
    
    print("\n=== Análisis t-SNE Completado ===")

if __name__ == "__main__":
    main()
