# DBSCAN: Clustering para Formas Complejas
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("# DBSCAN: Clustering para Formas Complejas")
print("=" * 80)

# 1. Limitaciones de K-means y clustering jerárquico
# ------------------------------------------------------------------------------
print("\n1. Limitaciones de K-means y clustering jerárquico")

# Generamos datasets con formas no convexas
n_samples = 500
np.random.seed(42)

# Dataset 1: Dos lunas
X_moons, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

# Dataset 2: Círculos concéntricos
X_circles, _ = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)

# Aplicamos K-means y comparamos con DBSCAN
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Lunas con K-means
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_moons)
axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_kmeans, cmap='viridis', s=30)
axes[0, 0].set_title('K-means en datos de lunas')

# Lunas con DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_moons)
axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_dbscan, cmap='viridis', s=30)
axes[0, 1].set_title('DBSCAN en datos de lunas')

# Círculos con K-means
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_circles)
axes[1, 0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_kmeans, cmap='viridis', s=30)
axes[1, 0].set_title('K-means en datos de círculos')

# Círculos con DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X_circles)
axes[1, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=y_dbscan, cmap='viridis', s=30)
axes[1, 1].set_title('DBSCAN en datos de círculos')

plt.tight_layout()
plt.show()

print("Observaciones:")
print("- K-means falla al detectar clusters de formas no convexas")
print("- DBSCAN puede identificar clusters de formas arbitrarias")

# 2. Fundamentos de DBSCAN
# ------------------------------------------------------------------------------
print("\n2. Fundamentos de DBSCAN")

print("DBSCAN (Density-Based Spatial Clustering of Applications with Noise):")
print("- Basado en la densidad de puntos en el espacio")
print("- Parámetros clave: eps (radio de vecindad) y min_samples (puntos mínimos)")
print("- Tipos de puntos: core, border y noise")

# Visualizamos el concepto de eps y min_samples
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
point_idx = 150  # Elegimos un punto para ilustrar
point = X[point_idx]
distances = np.sqrt(np.sum((X - point)**2, axis=1))
neighbors_idx = np.where(distances <= 0.5)[0]  # eps = 0.5

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='lightgray', s=40)
plt.scatter(X[neighbors_idx, 0], X[neighbors_idx, 1], c='green', s=40)
plt.scatter(point[0], point[1], c='red', s=100, marker='*')
circle = plt.Circle((point[0], point[1]), 0.5, fill=False, color='red')
plt.gca().add_patch(circle)
plt.title(f'Concepto de eps: Radio = 0.5, Vecinos = {len(neighbors_idx)}')
plt.show()

# 3. Selección de parámetros para DBSCAN
# ------------------------------------------------------------------------------
print("\n3. Selección de parámetros para DBSCAN")

# Método del vecino más cercano para estimar eps
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_moons)
distances, _ = neigh.kneighbors(X_moons)
distances = np.sort(distances[:, 4])  # Distancia al 5º vecino

plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.axhline(y=0.15, color='r', linestyle='--')
plt.title('Distancias al 5º vecino más cercano')
plt.xlabel('Puntos ordenados por distancia')
plt.ylabel('Distancia')
plt.show()

print("Método para seleccionar eps:")
print("1. Elegir min_samples (típicamente entre 3 y 5 para 2D)")
print("2. Calcular la distancia al k-ésimo vecino más cercano")
print("3. Buscar el 'codo' en la gráfica para determinar eps")

# 4. Comparación de algoritmos en datos con ruido
# ------------------------------------------------------------------------------
print("\n4. Comparación de algoritmos en datos con ruido")

# Generamos datos con outliers
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
outliers = np.random.uniform(-10, 10, (15, 2))  # Añadimos outliers
X_with_outliers = np.vstack([X_blobs, outliers])

# Comparamos algoritmos
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# K-means
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_with_outliers)
axes[0].scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=y_kmeans, cmap='viridis', s=30)
axes[0].set_title('K-means con outliers')

# Jerárquico
hierarchical = AgglomerativeClustering(n_clusters=3)
y_hierarchical = hierarchical.fit_predict(X_with_outliers)
axes[1].scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=y_hierarchical, cmap='viridis', s=30)
axes[1].set_title('Jerárquico con outliers')

# DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=5)
y_dbscan = dbscan.fit_predict(X_with_outliers)
axes[2].scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=y_dbscan, cmap='viridis', s=30)
axes[2].set_title('DBSCAN con outliers')

plt.tight_layout()
plt.show()

print("Ventaja de DBSCAN: Identifica outliers como ruido (etiqueta -1)")

# 5. Aplicación: Clustering geoespacial
# ------------------------------------------------------------------------------
print("\n5. Aplicación: Clustering geoespacial")

# Creamos datos simulados de ubicaciones
np.random.seed(42)
n_points = 200

# Simulamos coordenadas de tres ciudades
city1 = np.random.normal(loc=[0, 0], scale=0.1, size=(n_points, 2))
city2 = np.random.normal(loc=[1, 1], scale=0.1, size=(n_points, 2))
city3 = np.random.normal(loc=[0, 1], scale=0.1, size=(n_points, 2))

# Añadimos algunos puntos aleatorios (rural)
rural = np.random.uniform(low=-0.5, high=1.5, size=(50, 2))

# Combinamos todos los puntos
locations = np.vstack([city1, city2, city3, rural])

# Aplicamos DBSCAN para identificar ciudades
dbscan = DBSCAN(eps=0.15, min_samples=10)
clusters = dbscan.fit_predict(locations)

# Visualizamos los resultados
plt.figure(figsize=(10, 8))
plt.scatter(locations[:, 0], locations[:, 1], c=clusters, cmap='viridis', s=30)
plt.title('Clustering geoespacial con DBSCAN')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

print("Aplicaciones de DBSCAN en datos geoespaciales:")
print("- Identificación de áreas urbanas vs. rurales")
print("- Detección de puntos de interés")
print("- Análisis de patrones de tráfico")

# 6. Conclusiones y ejercicios propuestos
# ------------------------------------------------------------------------------
print("\n6. Conclusiones y ejercicios propuestos")

print("Ventajas de DBSCAN:")
print("- No requiere especificar el número de clusters a priori")
print("- Puede encontrar clusters de formas arbitrarias")
print("- Robusto frente a outliers")
print("- Funciona bien con clusters de densidad similar")

print("\nDesventajas de DBSCAN:")
print("- Sensible a los parámetros eps y min_samples")
print("- Dificultad con clusters de densidades muy diferentes")
print("- Problemas con datos de alta dimensionalidad")

print("\nEjercicios propuestos:")
print("1. Experimente con diferentes valores de eps y min_samples")
print("2. Aplique DBSCAN a un conjunto de datos geoespaciales real")
print("3. Compare DBSCAN con OPTICS y HDBSCAN")
print("4. Implemente una versión de DBSCAN que maneje clusters de diferentes densidades")
