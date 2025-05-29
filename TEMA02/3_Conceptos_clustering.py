# Conceptos básicos y aplicaciones del clustering
# Ejemplo práctico: Visualización y análisis exploratorio de datos para clustering
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Generación de conjuntos de datos sintéticos para ilustrar conceptos de clustering
# ------------------------------------------------------------------------------

# Generamos tres tipos de datasets para mostrar diferentes desafíos en clustering
print("Generando conjuntos de datos sintéticos...")

# Dataset 1: Clusters bien definidos (ideal para K-means)
n_samples = 1000
X_blobs, y_blobs = make_blobs(n_samples=n_samples, centers=4, 
                             cluster_std=0.6, random_state=42)

# Dataset 2: Clusters no lineales en forma de media luna (desafío para K-means)
X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.08, random_state=42)

# Dataset 3: Clusters concéntricos (imposible para K-means)
X_circles, y_circles = make_circles(n_samples=n_samples, noise=0.08, factor=0.5, random_state=42)

# Visualización de los tres datasets
plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='viridis', alpha=0.7, s=30)
plt.title('Clusters bien definidos')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

plt.subplot(132)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.7, s=30)
plt.title('Clusters no lineales')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

plt.subplot(133)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', alpha=0.7, s=30)
plt.title('Clusters concéntricos')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

plt.tight_layout()
plt.show()

# 2. Análisis de métricas de distancia
# ------------------------------------------------------------------------------
print("\nAnálisis de métricas de distancia...")

# Creamos un pequeño conjunto de puntos para ilustrar diferentes métricas
points = np.array([
    [0, 0],  # Punto A
    [0, 4],  # Punto B
    [3, 0],  # Punto C
    [3, 4]   # Punto D
])

# Etiquetas para los puntos
point_labels = ['A', 'B', 'C', 'D']

# Visualización de los puntos
plt.figure(figsize=(10, 8))
plt.scatter(points[:, 0], points[:, 1], c='blue', s=100)

# Añadimos etiquetas a los puntos
for i, label in enumerate(point_labels):
    plt.annotate(label, (points[i, 0] + 0.1, points[i, 1] + 0.1), fontsize=15)

# Dibujamos líneas para ilustrar distancias
# Distancia euclidiana entre A y D
plt.plot([points[0, 0], points[3, 0]], [points[0, 1], points[3, 1]], 'r--', label='Euclidiana A-D: 5.0')

# Distancia de Manhattan entre A y D
plt.plot([points[0, 0], points[0, 0]], [points[0, 1], points[3, 1]], 'g:', linewidth=2)
plt.plot([points[0, 0], points[3, 0]], [points[3, 1], points[3, 1]], 'g:', linewidth=2)
plt.annotate('Manhattan A-D: 7.0', xy=(1.5, 2), fontsize=12, color='green')

# Añadimos una cuadrícula
plt.grid(True)
plt.xlim(-1, 4)
plt.ylim(-1, 5)
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Ilustración de métricas de distancia')
plt.legend()
plt.show()

# Calculamos y mostramos la matriz de distancias euclidianas
dist_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        dist_matrix[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

print("Matriz de distancias euclidianas:")
df_dist = pd.DataFrame(dist_matrix, index=point_labels, columns=point_labels)
print(df_dist)

# 3. Evaluación de la calidad del clustering: Coeficiente de silueta
# ------------------------------------------------------------------------------
print("\nEvaluación de clustering mediante coeficiente de silueta...")

# Usamos el dataset de blobs para ilustrar el coeficiente de silueta
from sklearn.cluster import KMeans

# Aplicamos K-means con diferentes números de clusters
range_n_clusters = [2, 3, 4, 5, 6]
silhouette_avg_scores = []

for n_clusters in range_n_clusters:
    # Aplicamos K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_blobs)
    
    # Calculamos el coeficiente de silueta promedio
    silhouette_avg = silhouette_score(X_blobs, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)
    
    print(f"Para n_clusters = {n_clusters}, el coeficiente de silueta promedio es: {silhouette_avg:.3f}")
    
    # Calculamos el coeficiente de silueta para cada muestra
    sample_silhouette_values = silhouette_samples(X_blobs, cluster_labels)
    
    # Visualización del coeficiente de silueta
    plt.figure(figsize=(8, 6))
    y_lower = 10
    
    for i in range(n_clusters):
        # Obtenemos las muestras del cluster i y las ordenamos
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # Etiquetamos los clusters
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
        
        # Calculamos el nuevo y_lower para el próximo cluster
        y_lower = y_upper + 10
    
    plt.title(f'Visualización del coeficiente de silueta para n_clusters = {n_clusters}')
    plt.xlabel('Valores del coeficiente de silueta')
    plt.ylabel('Etiqueta del cluster')
    
    # Línea vertical para el valor promedio del coeficiente de silueta
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])  # Eliminamos las etiquetas del eje y
    plt.show()

# Graficamos el coeficiente de silueta promedio para diferentes números de clusters
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_avg_scores, 'o-', linewidth=2)
plt.xlabel('Número de clusters')
plt.ylabel('Coeficiente de silueta promedio')
plt.title('Evolución del coeficiente de silueta según el número de clusters')
plt.grid(True)
plt.show()

# 4. Visualización de alta dimensionalidad: Reducción dimensional con PCA
# ------------------------------------------------------------------------------
print("\nVisualización de datos de alta dimensionalidad mediante PCA...")

# Generamos datos en 5 dimensiones
n_samples = 500
n_features = 5
n_clusters = 3

X_high_dim, y_high_dim = make_blobs(n_samples=n_samples, 
                                    n_features=n_features, 
                                    centers=n_clusters,
                                    random_state=42)

# Estandarizamos los datos
scaler = StandardScaler()
X_high_dim_scaled = scaler.fit_transform(X_high_dim)

# Aplicamos PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_high_dim_scaled)

# Visualizamos los resultados
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X_high_dim_scaled[:, 0], X_high_dim_scaled[:, 1], c=y_high_dim, cmap='viridis', alpha=0.7)
plt.title('Datos originales (primeras 2 dimensiones)')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')

plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_high_dim, cmap='viridis', alpha=0.7)
plt.title('Datos reducidos con PCA')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')

plt.tight_layout()
plt.show()

# Mostramos la varianza explicada por cada componente
explained_variance = pca.explained_variance_ratio_
print(f"Varianza explicada por cada componente principal: {explained_variance}")
print(f"Varianza acumulada: {np.sum(explained_variance):.2f}")

# 5. Desafío de la maldición de la dimensionalidad
# ------------------------------------------------------------------------------
print("\nIlustración de la maldición de la dimensionalidad...")

# Generamos puntos aleatorios en espacios de diferentes dimensiones
dimensions = [2, 10, 50, 100, 500]
n_points = 1000
distances = []

for dim in dimensions:
    # Generamos puntos aleatorios en un hipercubo unitario
    points = np.random.random((n_points, dim))
    
    # Calculamos las distancias entre el primer punto y todos los demás
    dist = np.sqrt(np.sum((points[0] - points[1:])**2, axis=1))
    
    # Guardamos las distancias
    distances.append(dist)

# Visualizamos la distribución de distancias para cada dimensión
plt.figure(figsize=(12, 6))
for i, dim in enumerate(dimensions):
    sns.kdeplot(distances[i], label=f'Dimensión {dim}')

plt.title('Distribución de distancias en espacios de diferentes dimensiones')
plt.xlabel('Distancia euclidiana')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)
plt.show()

print("Estadísticas de distancias por dimensión:")
for i, dim in enumerate(dimensions):
    print(f"Dimensión {dim}: Media = {np.mean(distances[i]):.3f}, Desviación estándar = {np.std(distances[i]):.3f}")

print("\nObservación: A medida que aumenta la dimensionalidad, las distancias tienden a homogeneizarse,")
print("dificultando la identificación de vecinos cercanos y lejanos, lo que afecta al clustering.")

# 6. Conclusiones y ejercicios propuestos
# ------------------------------------------------------------------------------
print("\nConclusiones y ejercicios propuestos:")
print("1. Hemos explorado diferentes tipos de datos y los desafíos que presentan para el clustering.")
print("2. Analizamos métricas de distancia y su impacto en la formación de clusters.")
print("3. Evaluamos la calidad del clustering mediante el coeficiente de silueta.")
print("4. Visualizamos datos de alta dimensionalidad mediante técnicas de reducción dimensional.")
print("5. Ilustramos el problema de la maldición de la dimensionalidad.")

print("\nEjercicios propuestos:")
print("1. Experimente con otros conjuntos de datos y analice qué algoritmos de clustering son más adecuados.")
print("2. Implemente diferentes métricas de distancia y compare su impacto en los resultados del clustering.")
print("3. Explore otras técnicas de reducción dimensional como t-SNE o UMAP y compare con PCA.")
print("4. Desarrolle un caso práctico de segmentación de clientes utilizando datos reales de transacciones.")
