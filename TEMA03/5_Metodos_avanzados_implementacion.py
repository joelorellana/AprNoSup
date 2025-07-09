# Métodos avanzados de clustering: DBSCAN y Clustering Espectral
# Implementación y aplicación a datos complejos
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Métodos Avanzados de Clustering: DBSCAN y Clustering Espectral")
print("=============================================================")

# 1. Generación de conjuntos de datos no lineales
# ------------------------------------------------------------------------------
print("\n1. Generación de conjuntos de datos no lineales")

# Semilla para reproducibilidad
np.random.seed(42)

# Generamos varios conjuntos de datos con diferentes características
datasets = {
    'Anillos concéntricos': make_circles(n_samples=1000, factor=0.5, noise=0.05),
    'Lunas crecientes': make_moons(n_samples=1000, noise=0.05),
    'Clusters gaussianos': make_blobs(n_samples=1000, centers=4, cluster_std=0.5, random_state=42),
    'Clusters de densidad variable': None
}

# Creamos un dataset con clusters de densidad variable manualmente
X1, _ = make_blobs(n_samples=500, centers=[[0, 0]], cluster_std=0.5, random_state=42)
X2, _ = make_blobs(n_samples=100, centers=[[5, 5]], cluster_std=1.0, random_state=42)
X3, _ = make_blobs(n_samples=300, centers=[[2, 8]], cluster_std=0.65, random_state=42)
X_density = np.vstack([X1, X2, X3])
datasets['Clusters de densidad variable'] = (X_density, np.hstack([np.zeros(500), np.ones(100), 2*np.ones(300)]))

# Visualizamos los datasets
plt.figure(figsize=(15, 12))
for i, (name, (X, y)) in enumerate(datasets.items()):
    plt.subplot(2, 2, i+1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
    plt.title(name)
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Aplicación de K-means como referencia
# ------------------------------------------------------------------------------
print("\n2. Aplicación de K-means como referencia")

# Aplicamos K-means a cada dataset
kmeans_results = {}
for name, (X, y) in datasets.items():
    # Determinamos el número real de clusters
    n_clusters = len(np.unique(y))
    
    # Aplicamos K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans.fit_predict(X)
    
    # Calculamos métricas
    silhouette = silhouette_score(X, labels_kmeans) if n_clusters > 1 else 0
    davies_bouldin = davies_bouldin_score(X, labels_kmeans) if n_clusters > 1 else 0
    
    # Guardamos resultados
    kmeans_results[name] = {
        'labels': labels_kmeans,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin
    }
    
    print(f"K-means en {name}:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")

# 3. Determinación de parámetros para DBSCAN
# ------------------------------------------------------------------------------
print("\n3. Determinación de parámetros para DBSCAN")

# Función para encontrar el valor óptimo de epsilon usando el método del codo
def find_optimal_eps(X, min_samples=5, k=min_samples):
    # Calculamos las distancias al k-ésimo vecino más cercano
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    
    # Ordenamos las distancias al k-ésimo vecino
    k_dist = np.sort(distances[:, k-1])
    
    # Visualizamos el gráfico del codo
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_dist)), k_dist)
    plt.axhline(y=np.mean(k_dist), color='r', linestyle='--', 
                label=f'Media: {np.mean(k_dist):.4f}')
    plt.axhline(y=np.median(k_dist), color='g', linestyle='--', 
                label=f'Mediana: {np.median(k_dist):.4f}')
    plt.title(f'Distancias al {k}-ésimo vecino más cercano (ordenadas)')
    plt.xlabel('Puntos (ordenados por distancia)')
    plt.ylabel(f'Distancia al {k}-ésimo vecino')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Devolvemos la mediana como una estimación razonable de epsilon
    return np.median(k_dist)

# Analizamos el dataset de lunas crecientes como ejemplo
X_moons = datasets['Lunas crecientes'][0]
optimal_eps = find_optimal_eps(X_moons, min_samples=5)
print(f"Valor óptimo estimado de epsilon para 'Lunas crecientes': {optimal_eps:.4f}")

# 4. Aplicación de DBSCAN
# ------------------------------------------------------------------------------
print("\n4. Aplicación de DBSCAN")

# Aplicamos DBSCAN a cada dataset con parámetros adaptados
dbscan_results = {}
for name, (X, y) in datasets.items():
    # Estimamos epsilon para este dataset
    eps = find_optimal_eps(X, min_samples=5)
    
    # Aplicamos DBSCAN
    start_time = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X)
    execution_time = time.time() - start_time
    
    # Contamos el número de clusters y puntos de ruido
    n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = list(labels_dbscan).count(-1)
    
    # Calculamos métricas si hay más de un cluster y no todos son ruido
    if n_clusters > 1 and len(labels_dbscan) > n_noise:
        # Filtramos los puntos de ruido para calcular las métricas
        mask = labels_dbscan != -1
        if sum(mask) > n_clusters:  # Aseguramos que hay suficientes puntos
            silhouette = silhouette_score(X[mask], labels_dbscan[mask])
            davies_bouldin = davies_bouldin_score(X[mask], labels_dbscan[mask])
        else:
            silhouette = np.nan
            davies_bouldin = np.nan
    else:
        silhouette = np.nan
        davies_bouldin = np.nan
    
    # Guardamos resultados
    dbscan_results[name] = {
        'labels': labels_dbscan,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'eps': eps,
        'execution_time': execution_time
    }
    
    print(f"DBSCAN en {name} (eps={eps:.4f}, min_samples=5):")
    print(f"  Número de clusters encontrados: {n_clusters}")
    print(f"  Número de puntos de ruido: {n_noise} ({100*n_noise/len(labels_dbscan):.2f}%)")
    if not np.isnan(silhouette):
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"  Tiempo de ejecución: {execution_time:.4f} segundos")

# 5. Aplicación de Clustering Espectral
# ------------------------------------------------------------------------------
print("\n5. Aplicación de Clustering Espectral")

# Aplicamos Clustering Espectral a cada dataset
spectral_results = {}
for name, (X, y) in datasets.items():
    # Determinamos el número real de clusters
    n_clusters = len(np.unique(y))
    
    # Aplicamos Clustering Espectral
    start_time = time.time()
    spectral = SpectralClustering(n_clusters=n_clusters, 
                                  affinity='nearest_neighbors',
                                  assign_labels='kmeans',
                                  random_state=42)
    labels_spectral = spectral.fit_predict(X)
    execution_time = time.time() - start_time
    
    # Calculamos métricas
    silhouette = silhouette_score(X, labels_spectral) if n_clusters > 1 else 0
    davies_bouldin = davies_bouldin_score(X, labels_spectral) if n_clusters > 1 else 0
    
    # Guardamos resultados
    spectral_results[name] = {
        'labels': labels_spectral,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'execution_time': execution_time
    }
    
    print(f"Clustering Espectral en {name} (n_clusters={n_clusters}):")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"  Tiempo de ejecución: {execution_time:.4f} segundos")

# 6. Visualización comparativa de resultados
# ------------------------------------------------------------------------------
print("\n6. Visualización comparativa de resultados")

# Función para visualizar los resultados de clustering
def plot_clustering_comparison(datasets, kmeans_results, dbscan_results, spectral_results):
    for name, (X, y) in datasets.items():
        plt.figure(figsize=(20, 5))
        
        # Datos originales
        plt.subplot(1, 4, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
        plt.title(f'Datos originales\n{name}')
        plt.grid(True)
        
        # K-means
        plt.subplot(1, 4, 2)
        plt.scatter(X[:, 0], X[:, 1], c=kmeans_results[name]['labels'], cmap='viridis', s=30, alpha=0.7)
        plt.title(f'K-means\nSilhouette: {kmeans_results[name]["silhouette"]:.4f}')
        plt.grid(True)
        
        # DBSCAN
        plt.subplot(1, 4, 3)
        labels_dbscan = dbscan_results[name]['labels']
        # Usamos un colormap que reserva el color negro para el ruido (-1)
        colors = np.array(['black'] + list(plt.cm.viridis(np.linspace(0, 1, 20))))
        colors_mapped = colors[labels_dbscan + 1]
        plt.scatter(X[:, 0], X[:, 1], c=colors_mapped, s=30, alpha=0.7)
        silhouette = dbscan_results[name]['silhouette']
        silhouette_text = f"Silhouette: {silhouette:.4f}" if not np.isnan(silhouette) else "Silhouette: N/A"
        plt.title(f'DBSCAN\n{silhouette_text}\nRuido: {dbscan_results[name]["n_noise"]}')
        plt.grid(True)
        
        # Clustering Espectral
        plt.subplot(1, 4, 4)
        plt.scatter(X[:, 0], X[:, 1], c=spectral_results[name]['labels'], cmap='viridis', s=30, alpha=0.7)
        plt.title(f'Clustering Espectral\nSilhouette: {spectral_results[name]["silhouette"]:.4f}')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Visualizamos los resultados
plot_clustering_comparison(datasets, kmeans_results, dbscan_results, spectral_results)

# 7. Análisis de rendimiento y métricas
# ------------------------------------------------------------------------------
print("\n7. Análisis de rendimiento y métricas")

# Creamos un DataFrame para comparar métricas
metrics_df = pd.DataFrame(columns=['Dataset', 'Algoritmo', 'Silhouette', 'Davies-Bouldin', 'Tiempo (s)'])

# Añadimos datos para cada dataset y algoritmo
row_idx = 0
for name in datasets.keys():
    # K-means
    metrics_df.loc[row_idx] = [
        name, 'K-means', 
        kmeans_results[name]['silhouette'], 
        kmeans_results[name]['davies_bouldin'],
        0  # No medimos tiempo para K-means
    ]
    row_idx += 1
    
    # DBSCAN
    metrics_df.loc[row_idx] = [
        name, 'DBSCAN', 
        dbscan_results[name]['silhouette'], 
        dbscan_results[name]['davies_bouldin'],
        dbscan_results[name]['execution_time']
    ]
    row_idx += 1
    
    # Clustering Espectral
    metrics_df.loc[row_idx] = [
        name, 'Espectral', 
        spectral_results[name]['silhouette'], 
        spectral_results[name]['davies_bouldin'],
        spectral_results[name]['execution_time']
    ]
    row_idx += 1

# Mostramos la tabla de métricas
print("Comparación de métricas por algoritmo y dataset:")
print(metrics_df.to_string(index=False))

# Visualizamos las métricas
plt.figure(figsize=(15, 10))

# Silhouette Score
plt.subplot(2, 1, 1)
metrics_pivot = metrics_df.pivot(index='Dataset', columns='Algoritmo', values='Silhouette')
metrics_pivot.plot(kind='bar', ax=plt.gca())
plt.title('Comparación de Silhouette Score por dataset y algoritmo')
plt.ylabel('Silhouette Score (mayor es mejor)')
plt.grid(True)
plt.legend(title='Algoritmo')

# Davies-Bouldin Index
plt.subplot(2, 1, 2)
metrics_pivot = metrics_df.pivot(index='Dataset', columns='Algoritmo', values='Davies-Bouldin')
metrics_pivot.plot(kind='bar', ax=plt.gca())
plt.title('Comparación de Davies-Bouldin Index por dataset y algoritmo')
plt.ylabel('Davies-Bouldin Index (menor es mejor)')
plt.grid(True)
plt.legend(title='Algoritmo')

plt.tight_layout()
plt.show()

# 8. Caso práctico: Segmentación de imágenes simplificada
# ------------------------------------------------------------------------------
print("\n8. Caso práctico: Segmentación de imágenes simplificada")

# Creamos una imagen sintética simple
def create_synthetic_image(size=100):
    # Creamos una imagen con diferentes regiones
    image = np.zeros((size, size, 3))
    
    # Fondo
    image[:, :] = [0.8, 0.8, 0.8]  # Gris claro
    
    # Círculo central
    center = size // 2
    radius = size // 4
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    circle_mask = dist_from_center <= radius
    image[circle_mask] = [0.2, 0.4, 0.8]  # Azul
    
    # Rectángulo
    rect_start = size // 6
    rect_end = size // 3
    image[rect_start:rect_end, rect_start:rect_end] = [0.8, 0.2, 0.2]  # Rojo
    
    # Triángulo
    for i in range(size // 3):
        width = i // 2
        pos = size - size // 3 + i
        if pos < size and pos - width >= 0 and pos + width < size:
            image[pos, pos-width:pos+width] = [0.2, 0.8, 0.2]  # Verde
    
    # Añadimos un poco de ruido
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    return image

# Creamos y visualizamos la imagen
synthetic_image = create_synthetic_image(size=100)
plt.figure(figsize=(6, 6))
plt.imshow(synthetic_image)
plt.title('Imagen sintética')
plt.axis('off')
plt.show()

# Preparamos los datos para clustering
# Convertimos la imagen a una matriz de píxeles
pixels = synthetic_image.reshape(-1, 3)

# Añadimos información espacial (coordenadas x, y)
h, w = synthetic_image.shape[:2]
x, y = np.meshgrid(np.arange(w), np.arange(h))
spatial_features = np.column_stack([y.flatten(), x.flatten()])

# Normalizamos características espaciales y de color
color_features = pixels
spatial_features_scaled = spatial_features / np.max(spatial_features)
color_features_scaled = color_features

# Combinamos características con diferentes pesos
# Damos más peso a las características de color
features = np.column_stack([color_features_scaled * 0.8, spatial_features_scaled * 0.2])

# Aplicamos los algoritmos de clustering
n_segments = 4  # Número esperado de segmentos

# K-means
kmeans = KMeans(n_clusters=n_segments, random_state=42)
kmeans_labels = kmeans.fit_predict(features)
kmeans_image = kmeans_labels.reshape(h, w)

# DBSCAN
# Estimamos epsilon
eps = find_optimal_eps(features, min_samples=10, k=10)
dbscan = DBSCAN(eps=eps, min_samples=10)
dbscan_labels = dbscan.fit_predict(features)
# Mapeamos el ruido (-1) a un nuevo segmento
dbscan_labels = dbscan_labels + 1  # Desplazamos todos los valores
dbscan_labels[dbscan_labels == 0] = n_segments + 1  # El ruido es ahora un segmento adicional
dbscan_image = dbscan_labels.reshape(h, w)

# Clustering Espectral
spectral = SpectralClustering(n_clusters=n_segments, affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral.fit_predict(features)
spectral_image = spectral_labels.reshape(h, w)

# Visualizamos los resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(synthetic_image)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(kmeans_image, cmap='viridis')
plt.title('Segmentación K-means')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(dbscan_image, cmap='viridis')
plt.title(f'Segmentación DBSCAN\n{len(np.unique(dbscan_labels))} segmentos')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(spectral_image, cmap='viridis')
plt.title('Segmentación Espectral')
plt.axis('off')

plt.tight_layout()
plt.show()

# 9. Conclusiones
# ------------------------------------------------------------------------------
print("\n9. Conclusiones")

print("""
Conclusiones del análisis de métodos avanzados de clustering:

1. Rendimiento en diferentes formas de clusters:
   - K-means funciona bien con clusters convexos y de tamaño similar.
   - DBSCAN destaca en la identificación de clusters de formas arbitrarias y manejo de ruido.
   - Clustering Espectral es efectivo para formas complejas y no lineales.

2. Selección de parámetros:
   - La selección de epsilon en DBSCAN es crítica y puede automatizarse con el método del codo.
   - El Clustering Espectral requiere conocer el número de clusters a priori.

3. Eficiencia computacional:
   - K-means es generalmente el más rápido.
   - DBSCAN tiene complejidad variable dependiendo de la estructura de datos.
   - Clustering Espectral es computacionalmente más intensivo debido a la descomposición espectral.

4. Aplicaciones específicas:
   - Para segmentación de imágenes, tanto DBSCAN como Clustering Espectral ofrecen ventajas
     sobre K-means al capturar mejor las regiones con formas irregulares.
   - Para detección de anomalías, DBSCAN proporciona una identificación natural de puntos de ruido.

5. Recomendaciones prácticas:
   - Utilizar múltiples algoritmos y comparar resultados.
   - Evaluar con métricas internas como Silhouette y Davies-Bouldin.
   - Considerar la interpretabilidad de los clusters para aplicaciones específicas.
""")

# Ejercicios propuestos
print("\nEjercicios propuestos:")
print("""
1. Implementar HDBSCAN, una extensión de DBSCAN que maneja mejor clusters de densidad variable.

2. Explorar diferentes funciones de afinidad para Clustering Espectral (RBF, vecinos más cercanos, etc.)
   y analizar su impacto en los resultados.

3. Aplicar estos algoritmos a un conjunto de datos real de alta dimensionalidad, como datos genómicos
   o de procesamiento de lenguaje natural, evaluando su rendimiento.

4. Desarrollar un enfoque de ensemble que combine los resultados de múltiples algoritmos de clustering
   para obtener una solución más robusta.

5. Implementar una versión incremental de DBSCAN para manejar conjuntos de datos que no caben en memoria.
""")
