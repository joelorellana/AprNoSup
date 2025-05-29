# Pipeline Avanzado de Clustering: Preprocesamiento, Paralelización y Visualización
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025
# Nivel: Maestría en Ciencia de Datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from joblib import Parallel, delayed, Memory
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Configuración para paralelización y caché
n_jobs = -1  # Usar todos los núcleos disponibles
cachedir = './cache'
if not os.path.exists(cachedir):
    os.makedirs(cachedir)
memory = Memory(cachedir, verbose=0)

# 1. Carga y exploración de datos de alta dimensionalidad
# ------------------------------------------------------------------------------
print("1. Carga y exploración de datos de alta dimensionalidad")

# Cargamos el dataset MNIST para demostrar el pipeline en datos de alta dimensionalidad
print("Cargando dataset MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Tomamos una muestra para acelerar los cálculos (en un entorno real podríamos usar todo el dataset)
sample_size = 10000
random_state = 42
np.random.seed(random_state)
indices = np.random.choice(X.shape[0], sample_size, replace=False)
X_sample = X.iloc[indices].values if hasattr(X, 'iloc') else X[indices]
y_sample = y.iloc[indices].values if hasattr(y, 'iloc') else y[indices]

print(f"Dimensiones originales: {X_sample.shape}")
print(f"Número de clases: {len(np.unique(y_sample))}")

# Visualizamos algunas imágenes
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_sample[i].reshape(28, 28), cmap='gray')
    plt.title(f"Etiqueta: {y_sample[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 2. Definición del pipeline avanzado de clustering
# ------------------------------------------------------------------------------
print("\n2. Definición del pipeline avanzado de clustering")

class ClusteringPipeline:
    """
    Pipeline avanzado para clustering que incluye:
    - Preprocesamiento (escalado, manejo de valores atípicos)
    - Selección de características (varianza, filtros)
    - Reducción de dimensionalidad (PCA, SVD, t-SNE, UMAP)
    - Clustering (K-means, jerárquico, DBSCAN)
    - Evaluación (silueta, Calinski-Harabasz, Davies-Bouldin)
    - Paralelización para grandes volúmenes de datos
    """
    
    def __init__(self, random_state=42, n_jobs=-1, memory=None):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.memory = memory
        self.results_ = {}
        
    def preprocess(self, X, scaler_type='standard', remove_outliers=False, outlier_threshold=3):
        """Preprocesamiento de datos"""
        print("Preprocesando datos...")
        
        # Manejo de valores faltantes
        if hasattr(X, 'isna'):
            X_clean = X.copy()
            # Imputamos con la media
            for col in X_clean.columns:
                if X_clean[col].isna().sum() > 0:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].mean())
        else:
            X_clean = X.copy()
            # Si es un array de numpy, reemplazamos NaN con 0
            if np.isnan(X_clean).any():
                X_clean = np.nan_to_num(X_clean)
        
        # Escalado de datos
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Tipo de escalador no reconocido: {scaler_type}")
        
        X_scaled = scaler.fit_transform(X_clean)
        
        # Eliminación de valores atípicos (opcional)
        if remove_outliers:
            if scaler_type == 'standard':
                # Para datos estandarizados, los outliers están a más de threshold desviaciones estándar
                mask = np.all(np.abs(X_scaled) < outlier_threshold, axis=1)
                X_scaled = X_scaled[mask]
                print(f"Se eliminaron {len(X_clean) - len(X_scaled)} valores atípicos")
        
        self.X_preprocessed_ = X_scaled
        return X_scaled
    
    def select_features(self, X, method='variance', n_features=100, **kwargs):
        """Selección de características"""
        print(f"Seleccionando {n_features} características usando método '{method}'...")
        
        if method == 'variance':
            # Eliminamos características con varianza cercana a cero
            selector = VarianceThreshold(threshold=kwargs.get('threshold', 0.01))
            X_selected = selector.fit_transform(X)
            
            # Si aún hay demasiadas características, seleccionamos las top n_features con mayor varianza
            if X_selected.shape[1] > n_features:
                variances = selector.variances_
                top_indices = np.argsort(variances)[-n_features:]
                X_selected = X[:, top_indices]
                
        elif method == 'kbest':
            # Seleccionamos las mejores características según una métrica (requiere etiquetas)
            if 'y' not in kwargs:
                raise ValueError("El método 'kbest' requiere etiquetas (y)")
            
            selector = SelectKBest(f_classif, k=n_features)
            X_selected = selector.fit_transform(X, kwargs['y'])
            
        elif method == 'none':
            # No aplicamos selección de características
            X_selected = X
            
        else:
            raise ValueError(f"Método de selección no reconocido: {method}")
        
        print(f"Dimensiones después de selección: {X_selected.shape}")
        self.X_selected_ = X_selected
        return X_selected
    
    def reduce_dimensionality(self, X, method='pca', n_components=50, **kwargs):
        """Reducción de dimensionalidad"""
        print(f"Reduciendo dimensionalidad a {n_components} componentes usando '{method}'...")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            X_reduced = reducer.fit_transform(X)
            self.explained_variance_ = reducer.explained_variance_ratio_
            print(f"Varianza explicada: {sum(self.explained_variance_):.4f}")
            
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            X_reduced = reducer.fit_transform(X)
            self.explained_variance_ = reducer.explained_variance_ratio_
            print(f"Varianza explicada: {sum(self.explained_variance_):.4f}")
            
        elif method == 'tsne':
            # t-SNE es computacionalmente costoso
            reducer = TSNE(n_components=n_components, 
                          random_state=self.random_state,
                          **kwargs)
            X_reduced = reducer.fit_transform(X)
            
        elif method == 'umap':
            # UMAP es más rápido que t-SNE y preserva mejor la estructura global
            reducer = umap.UMAP(n_components=n_components,
                               random_state=self.random_state,
                               **kwargs)
            X_reduced = reducer.fit_transform(X)
            
        else:
            raise ValueError(f"Método de reducción no reconocido: {method}")
        
        print(f"Dimensiones después de reducción: {X_reduced.shape}")
        self.X_reduced_ = X_reduced
        self.reducer_ = reducer
        return X_reduced
    
    def cluster(self, X, method='kmeans', n_clusters=10, **kwargs):
        """Aplicación de algoritmos de clustering"""
        print(f"Aplicando clustering '{method}' con {n_clusters} clusters...")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, 
                              random_state=self.random_state,
                              n_init=10,
                              **kwargs)
            labels = clusterer.fit_predict(X)
            self.centroids_ = clusterer.cluster_centers_
            
        elif method == 'minibatch_kmeans':
            # Versión escalable de K-means para grandes volúmenes de datos
            clusterer = MiniBatchKMeans(n_clusters=n_clusters,
                                       random_state=self.random_state,
                                       batch_size=kwargs.get('batch_size', 1000),
                                       **kwargs)
            labels = clusterer.fit_predict(X)
            self.centroids_ = clusterer.cluster_centers_
            
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters,
                                              linkage=kwargs.get('linkage', 'ward'),
                                              **kwargs)
            labels = clusterer.fit_predict(X)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=kwargs.get('eps', 0.5),
                              min_samples=kwargs.get('min_samples', 5),
                              n_jobs=self.n_jobs,
                              **kwargs)
            labels = clusterer.fit_predict(X)
            
        else:
            raise ValueError(f"Método de clustering no reconocido: {method}")
        
        self.labels_ = labels
        self.clusterer_ = clusterer
        
        # Contamos elementos por cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Distribución de clusters: {dict(zip(unique_labels, counts))}")
        
        return labels
    
    def evaluate(self, X, labels):
        """Evaluación de la calidad del clustering"""
        print("Evaluando calidad del clustering...")
        
        # Verificamos si hay suficientes clusters para calcular métricas
        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            print("Se necesitan al menos 2 clusters para calcular métricas")
            return {}
        
        # Verificamos si hay etiquetas de ruido (-1) en DBSCAN
        if -1 in np.unique(labels):
            # Filtramos puntos de ruido para calcular métricas
            mask = labels != -1
            X_filtered = X[mask]
            labels_filtered = labels[mask]
            
            # Verificamos si hay suficientes clusters después de filtrar
            if len(np.unique(labels_filtered)) < 2:
                print("Insuficientes clusters después de filtrar ruido")
                return {}
                
            metrics = {
                'silhouette': silhouette_score(X_filtered, labels_filtered),
                'calinski_harabasz': calinski_harabasz_score(X_filtered, labels_filtered),
                'davies_bouldin': davies_bouldin_score(X_filtered, labels_filtered)
            }
        else:
            metrics = {
                'silhouette': silhouette_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels)
            }
        
        print(f"Métricas de evaluación:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
            
        self.metrics_ = metrics
        return metrics
    
    def optimize_hyperparameters(self, X, param_grid, n_jobs=1):
        """Búsqueda de hiperparámetros óptimos para clustering"""
        print("Optimizando hiperparámetros...")
        
        # Generamos todas las combinaciones de parámetros
        param_combinations = list(ParameterGrid(param_grid))
        print(f"Evaluando {len(param_combinations)} combinaciones de parámetros...")
        
        # Evaluamos secuencialmente para evitar problemas de serialización
        results = []
        for params in param_combinations:
            # Hacemos una copia de los parámetros
            params_copy = params.copy()
            method = params_copy.pop('method', 'kmeans')
            n_clusters = params_copy.pop('n_clusters', 8)
            
            try:
                # Aplicamos clustering con los parámetros específicos
                labels = self.cluster(X, method=method, n_clusters=n_clusters, **params_copy)
                metrics = self.evaluate(X, labels)
                
                results.append({
                    'params': params,
                    'silhouette': metrics.get('silhouette', -1),
                    'calinski_harabasz': metrics.get('calinski_harabasz', -1),
                    'davies_bouldin': metrics.get('davies_bouldin', -1),
                    'labels': labels
                })
                print(f"Parámetros: {params}, Silueta: {metrics.get('silhouette', -1):.4f}")
            except Exception as e:
                print(f"Error con parámetros {params}: {str(e)}")
                results.append({
                    'params': params,
                    'silhouette': -1,
                    'calinski_harabasz': -1,
                    'davies_bouldin': float('inf'),
                    'error': str(e)
                })
        
        # Ordenamos por silhouette score (mayor es mejor)
        results = sorted(results, key=lambda x: x['silhouette'], reverse=True)
        
        self.hyperparameter_results_ = results
        best_result = results[0]
        
        print(f"\nMejores parámetros encontrados:")
        for key, value in best_result['params'].items():
            print(f"- {key}: {value}")
        print(f"Silhouette score: {best_result['silhouette']:.4f}")
        
        # Aplicamos los mejores parámetros
        best_params = best_result['params'].copy()
        method = best_params.pop('method')
        n_clusters = best_params.pop('n_clusters')
        
        # Filtramos los parámetros específicos para cada algoritmo
        if method == 'kmeans' or method == 'minibatch_kmeans':
            # Parámetros válidos para K-means
            valid_params = {k: v for k, v in best_params.items() 
                          if k in ['init', 'max_iter', 'tol', 'n_init']}
        elif method == 'hierarchical':
            # Parámetros válidos para clustering jerárquico
            valid_params = {k: v for k, v in best_params.items() 
                          if k in ['linkage', 'affinity', 'compute_full_tree']}
        elif method == 'dbscan':
            # Parámetros válidos para DBSCAN
            valid_params = {k: v for k, v in best_params.items() 
                          if k in ['eps', 'min_samples', 'metric', 'leaf_size']}
        else:
            valid_params = {}
            
        self.cluster(X, method=method, n_clusters=n_clusters, **valid_params)
        
        return best_result
    
    def run_pipeline(self, X, y=None, preprocess_params=None, feature_params=None, 
                    reduction_params=None, cluster_params=None, optimize=False):
        """Ejecuta el pipeline completo"""
        start_time = time.time()
        print("Iniciando pipeline de clustering...")
        
        # Parámetros por defecto
        if preprocess_params is None:
            preprocess_params = {'scaler_type': 'standard', 'remove_outliers': False}
            
        if feature_params is None:
            feature_params = {'method': 'variance', 'n_features': min(100, X.shape[1])}
            
        if reduction_params is None:
            reduction_params = {'method': 'pca', 'n_components': min(50, X.shape[1])}
            
        if cluster_params is None:
            cluster_params = {'method': 'kmeans', 'n_clusters': 10}
        
        # Paso 1: Preprocesamiento
        X_preprocessed = self.preprocess(X, **preprocess_params)
        
        # Paso 2: Selección de características
        if y is not None and feature_params.get('method') == 'kbest':
            feature_params['y'] = y
        X_selected = self.select_features(X_preprocessed, **feature_params)
        
        # Paso 3: Reducción de dimensionalidad
        X_reduced = self.reduce_dimensionality(X_selected, **reduction_params)
        
        # Paso 4: Clustering
        if optimize:
            # Si optimize es True, cluster_params debe ser un grid de parámetros
            best_result = self.optimize_hyperparameters(X_reduced, cluster_params)
            labels = best_result['labels']
        else:
            labels = self.cluster(X_reduced, **cluster_params)
        
        # Paso 5: Evaluación
        metrics = self.evaluate(X_reduced, labels)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Pipeline completado en {execution_time:.2f} segundos")
        
        # Guardamos resultados
        self.results_ = {
            'X_preprocessed': X_preprocessed,
            'X_selected': X_selected,
            'X_reduced': X_reduced,
            'labels': labels,
            'metrics': metrics,
            'execution_time': execution_time
        }
        
        return self.results_

# 3. Aplicación del pipeline a los datos MNIST
# ------------------------------------------------------------------------------
print("\n3. Aplicación del pipeline a los datos MNIST")

# Creamos una instancia del pipeline
pipeline = ClusteringPipeline(random_state=42, n_jobs=n_jobs, memory=memory)

# Definimos los parámetros del pipeline
preprocess_params = {
    'scaler_type': 'standard',
    'remove_outliers': False
}

feature_params = {
    'method': 'variance',
    'n_features': 200  # Seleccionamos las 200 características con mayor varianza
}

reduction_params = {
    'method': 'pca',
    'n_components': 50  # Reducimos a 50 dimensiones con PCA
}

# Ejecutamos el pipeline sin optimización
results = pipeline.run_pipeline(
    X_sample,
    preprocess_params=preprocess_params,
    feature_params=feature_params,
    reduction_params=reduction_params,
    cluster_params={'method': 'kmeans', 'n_clusters': 10}
)

# 4. Visualización de resultados con PCA
# ------------------------------------------------------------------------------
print("\n4. Visualización de resultados con PCA")

# Obtenemos los datos reducidos y las etiquetas
X_reduced = results['X_reduced']
labels = results['labels']

# Visualizamos los primeros dos componentes principales
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('Clustering de dígitos MNIST usando PCA (2 componentes)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.grid(True)
plt.show()

# Visualizamos los clusters vs las etiquetas reales
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_sample, cmap='tab10', s=50, alpha=0.7)
plt.colorbar(scatter, label='Dígito real')
plt.title('Dígitos reales en espacio PCA (2 componentes)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.grid(True)
plt.show()

# 5. Comparación de técnicas de reducción de dimensionalidad
# ------------------------------------------------------------------------------
print("\n5. Comparación de técnicas de reducción de dimensionalidad")

# Comparamos PCA, t-SNE y UMAP
reduction_methods = ['pca', 'tsne', 'umap']
reduced_data = {}

for method in reduction_methods:
    print(f"\nAplicando {method.upper()}...")
    
    # Configuramos parámetros específicos para cada método
    if method == 'pca':
        params = {'n_components': 2}
    elif method == 'tsne':
        params = {'n_components': 2, 'perplexity': 30, 'n_iter': 1000}
    elif method == 'umap':
        params = {'n_components': 2, 'n_neighbors': 15, 'min_dist': 0.1}
    
    # Aplicamos reducción de dimensionalidad
    start_time = time.time()
    X_method = pipeline.reduce_dimensionality(pipeline.X_selected_, method=method, **params)
    end_time = time.time()
    
    reduced_data[method] = {
        'data': X_method,
        'time': end_time - start_time
    }
    
    print(f"Tiempo de ejecución: {reduced_data[method]['time']:.2f} segundos")

# Visualizamos los resultados
plt.figure(figsize=(18, 6))

for i, method in enumerate(reduction_methods):
    plt.subplot(1, 3, i+1)
    scatter = plt.scatter(
        reduced_data[method]['data'][:, 0],
        reduced_data[method]['data'][:, 1],
        c=y_sample,
        cmap='tab10',
        s=30,
        alpha=0.7
    )
    plt.title(f'{method.upper()}\nTiempo: {reduced_data[method]["time"]:.2f}s')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid(True)
    
    # Añadimos leyenda solo en el último gráfico
    if i == len(reduction_methods) - 1:
        plt.colorbar(scatter, label='Dígito')

plt.tight_layout()
plt.show()

# 6. Optimización de hiperparámetros para clustering
# ------------------------------------------------------------------------------
print("\n6. Optimización de hiperparámetros para clustering")

# Definimos un grid de parámetros para optimización
param_grid = {
    'method': ['kmeans', 'minibatch_kmeans', 'hierarchical'],
    'n_clusters': [8, 10, 12],
    # Parámetros específicos para cada método
    'init': ['k-means++'],  # Para K-means
    'linkage': ['ward', 'average']  # Para jerárquico
}

# Optimizamos hiperparámetros (esto puede tomar tiempo)
print("Optimizando hiperparámetros (esto puede tomar tiempo)...")
best_result = pipeline.optimize_hyperparameters(X_reduced, param_grid, n_jobs=n_jobs)

# Visualizamos los resultados con los mejores parámetros
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=best_result['labels'], cmap='tab10', s=50, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Clustering óptimo (Silhouette: {best_result["silhouette"]:.4f})')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.grid(True)
plt.show()

# 7. Análisis de clusters y evaluación de resultados
# ------------------------------------------------------------------------------
print("\n7. Análisis de clusters y evaluación de resultados")

# Creamos una matriz de confusión entre clusters y dígitos reales
confusion_matrix = pd.crosstab(
    pipeline.labels_,
    y_sample,
    rownames=['Cluster'],
    colnames=['Dígito']
)

print("Matriz de confusión (clusters vs dígitos reales):")
print(confusion_matrix)

# Visualizamos la matriz de confusión
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Matriz de confusión: Clusters vs Dígitos reales')
plt.show()

# Calculamos pureza de clusters
def calculate_purity(confusion_matrix):
    return np.sum(np.max(confusion_matrix.values, axis=1)) / np.sum(confusion_matrix.values)

purity = calculate_purity(confusion_matrix)
print(f"Pureza de clusters: {purity:.4f}")

# Visualizamos ejemplos representativos de cada cluster
def plot_cluster_representatives(X, labels, n_clusters, n_samples=5):
    plt.figure(figsize=(15, 2 * n_clusters))
    
    for cluster in range(n_clusters):
        # Obtenemos índices de este cluster
        indices = np.where(labels == cluster)[0]
        
        if len(indices) == 0:
            continue
            
        # Seleccionamos n_samples aleatorios
        sample_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(n_clusters, n_samples, cluster * n_samples + i + 1)
            plt.imshow(X[idx].reshape(28, 28), cmap='gray')
            plt.title(f'C{cluster}, D{y_sample[idx]}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualizamos representantes de cada cluster
n_clusters = len(np.unique(pipeline.labels_))
plot_cluster_representatives(X_sample, pipeline.labels_, n_clusters)

# 8. Conclusiones y ejercicios propuestos
# ------------------------------------------------------------------------------
print("\n8. Conclusiones y ejercicios propuestos")

print("Conclusiones:")
print("1. Hemos implementado un pipeline completo de clustering que incluye:")
print("   - Preprocesamiento y selección de características")
print("   - Reducción de dimensionalidad con PCA, t-SNE y UMAP")
print("   - Clustering con diferentes algoritmos")
print("   - Optimización de hiperparámetros")
print("   - Evaluación de resultados")
print("2. Las técnicas de visualización avanzadas (t-SNE, UMAP) proporcionan mejores separaciones visuales que PCA")
print("3. La paralelización permite procesar eficientemente grandes volúmenes de datos")
print("4. La optimización de hiperparámetros mejora significativamente la calidad del clustering")

print("\nEjercicios propuestos:")
print("1. Implementar técnicas adicionales de selección de características (PCA supervisado, autoencoders)")
print("2. Comparar el rendimiento de diferentes algoritmos de clustering en conjuntos de datos más complejos")
print("3. Desarrollar un método para determinar automáticamente el número óptimo de clusters")
print("4. Implementar técnicas de ensemble clustering para mejorar la robustez")
print("5. Aplicar el pipeline a un problema real de su dominio profesional")
