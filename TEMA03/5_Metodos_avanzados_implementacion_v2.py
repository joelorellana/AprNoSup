# Métodos avanzados de clustering: DBSCAN y Clustering Espectral
# Implementación con datasets reales
# Autor: UDB - Aprendizaje No Supervisado

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("Métodos Avanzados de Clustering: DBSCAN y Clustering Espectral")
print("=============================================================")

# 1. Carga y preparación de datos reales
# ------------------------------------------------------------------------------
print("\n1. Carga y preparación de datos reales")

# Cargamos el dataset de vinos
wine = load_wine()
X_wine = wine.data
y_wine = wine.target

print(f"Dataset de vinos: {X_wine.shape[0]} muestras, {X_wine.shape[1]} características")
print(f"Clases: {np.unique(y_wine)}")
print(f"Distribución de clases: {np.bincount(y_wine)}")

# Escalamos los datos
scaler = StandardScaler()
X_wine_scaled = scaler.fit_transform(X_wine)

# Reducimos dimensionalidad para visualización
pca = PCA(n_components=2)
X_wine_pca = pca.fit_transform(X_wine_scaled)

# Visualizamos el dataset
plt.figure(figsize=(10, 8))
plt.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=y_wine, cmap='viridis', s=50, alpha=0.8)
plt.title('Dataset de vinos (PCA)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
plt.colorbar(label='Clase de vino')
plt.grid(True)
plt.show()

# 2. Determinación de parámetros para DBSCAN
# ------------------------------------------------------------------------------
print("\n2. Determinación de parámetros para DBSCAN")

# Calculamos distancias al k-ésimo vecino más cercano para estimar epsilon
k = 5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(X_wine_scaled)
distances, _ = neigh.kneighbors(X_wine_scaled)
distances = np.sort(distances[:, k-1])

# Visualizamos el gráfico del codo para epsilon
plt.figure(figsize=(10, 6))
plt.plot(range(len(distances)), distances)
plt.axhline(y=0.5, color='r', linestyle='--', label='Epsilon seleccionado = 0.5')
plt.title('Distancias al k-ésimo vecino más cercano (k=5)')
plt.xlabel('Muestras (ordenadas por distancia)')
plt.ylabel('Distancia')
plt.grid(True)
plt.legend()
plt.show()

# 3. Aplicación de DBSCAN al dataset de vinos
# ------------------------------------------------------------------------------
print("\n3. Aplicación de DBSCAN al dataset de vinos")

# Aplicamos DBSCAN con los parámetros determinados
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_wine_scaled)

# Contamos clusters y ruido
n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

print(f"Número de clusters encontrados por DBSCAN: {n_clusters}")
print(f"Número de puntos de ruido: {n_noise} ({100*n_noise/len(labels_dbscan):.2f}%)")

# Visualizamos los resultados de DBSCAN
plt.figure(figsize=(10, 8))
colors = np.array(['black'] + list(plt.cm.viridis(np.linspace(0, 1, 10))))
colors_mapped = colors[labels_dbscan + 1]
plt.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=colors_mapped, s=50, alpha=0.8)
plt.title(f'Clustering con DBSCAN: {n_clusters} clusters, {n_noise} puntos de ruido')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
plt.grid(True)
plt.show()

# 4. Aplicación de Clustering Espectral al dataset de vinos
# ------------------------------------------------------------------------------
print("\n4. Aplicación de Clustering Espectral al dataset de vinos")

# Aplicamos Clustering Espectral
n_clusters_spectral = 3  # Sabemos que hay 3 clases de vino
spectral = SpectralClustering(n_clusters=n_clusters_spectral, 
                             affinity='nearest_neighbors',
                             assign_labels='kmeans',
                             random_state=42)
labels_spectral = spectral.fit_predict(X_wine_scaled)

# Calculamos silhouette score
silhouette = silhouette_score(X_wine_scaled, labels_spectral)
print(f"Silhouette Score para Clustering Espectral: {silhouette:.4f}")

# Visualizamos los resultados de Clustering Espectral
plt.figure(figsize=(10, 8))
plt.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=labels_spectral, cmap='viridis', s=50, alpha=0.8)
plt.title(f'Clustering Espectral: {n_clusters_spectral} clusters\nSilhouette Score: {silhouette:.4f}')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# 5. Aplicación a un segundo dataset: Cáncer de mama
# ------------------------------------------------------------------------------
print("\n5. Aplicación a un segundo dataset: Cáncer de mama")

# Cargamos el dataset de cáncer de mama
breast_cancer = load_breast_cancer()
X_cancer = breast_cancer.data
y_cancer = breast_cancer.target

print(f"Dataset de cáncer de mama: {X_cancer.shape[0]} muestras, {X_cancer.shape[1]} características")
print(f"Clases: {np.unique(y_cancer)}")
print(f"Distribución de clases: {np.bincount(y_cancer)}")

# Escalamos los datos
X_cancer_scaled = StandardScaler().fit_transform(X_cancer)

# Reducimos dimensionalidad para visualización
pca_cancer = PCA(n_components=2)
X_cancer_pca = pca_cancer.fit_transform(X_cancer_scaled)

# Aplicamos DBSCAN
dbscan_cancer = DBSCAN(eps=0.6, min_samples=10)
labels_dbscan_cancer = dbscan_cancer.fit_predict(X_cancer_scaled)

# Contamos clusters y ruido
n_clusters_cancer = len(set(labels_dbscan_cancer)) - (1 if -1 in labels_dbscan_cancer else 0)
n_noise_cancer = list(labels_dbscan_cancer).count(-1)

# Aplicamos Clustering Espectral
spectral_cancer = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
labels_spectral_cancer = spectral_cancer.fit_predict(X_cancer_scaled)

# Visualizamos los resultados comparativos
plt.figure(figsize=(15, 5))

# Datos originales
plt.subplot(1, 3, 1)
plt.scatter(X_cancer_pca[:, 0], X_cancer_pca[:, 1], c=y_cancer, cmap='viridis', s=30, alpha=0.7)
plt.title('Datos originales\nCáncer de mama')
plt.grid(True)

# DBSCAN
plt.subplot(1, 3, 2)
colors_cancer = np.array(['black'] + list(plt.cm.viridis(np.linspace(0, 1, 10))))
colors_mapped_cancer = colors_cancer[labels_dbscan_cancer + 1]
plt.scatter(X_cancer_pca[:, 0], X_cancer_pca[:, 1], c=colors_mapped_cancer, s=30, alpha=0.7)
plt.title(f'DBSCAN\n{n_clusters_cancer} clusters, {n_noise_cancer} ruido')
plt.grid(True)

# Clustering Espectral
plt.subplot(1, 3, 3)
plt.scatter(X_cancer_pca[:, 0], X_cancer_pca[:, 1], c=labels_spectral_cancer, cmap='viridis', s=30, alpha=0.7)
plt.title('Clustering Espectral\n2 clusters')
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Conclusiones
# ------------------------------------------------------------------------------
print("\n6. Conclusiones")

print("""
Conclusiones del análisis con datasets reales:

1. DBSCAN:
   - Identifica automáticamente el número de clusters basado en la densidad de los datos.
   - Detecta puntos de ruido que podrían ser anomalías o casos atípicos.
   - La selección de parámetros (eps, min_samples) es crítica y debe adaptarse a cada dataset.

2. Clustering Espectral:
   - Captura estructuras no lineales en los datos que K-means podría no detectar.
   - Requiere especificar el número de clusters a priori.
   - Puede ser computacionalmente intensivo para datasets grandes.

3. Comparación en datasets reales:
   - En el dataset de vinos, ambos métodos identifican estructuras que se alinean parcialmente
     con las clases reales.
   - En el dataset de cáncer de mama, el Clustering Espectral parece capturar mejor la estructura
     binaria natural de los datos.
""")
