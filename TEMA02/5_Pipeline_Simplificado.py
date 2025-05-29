# Pipeline Simplificado de Clustering: Versión optimizada para ejecución rápida
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025
# Nivel: Maestría en Ciencia de Datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Pipeline Simplificado de Clustering: Preprocesamiento, Reducción de Dimensionalidad y Visualización")

# 1. Carga de datos (usamos digits en lugar de MNIST para mayor rapidez)
print("\n1. Carga de datos")
digits = load_digits()
X = digits.data
y = digits.target

print(f"Dimensiones originales: {X.shape}")
print(f"Número de clases: {len(np.unique(y))}")

# Visualizamos algunas imágenes
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Dígito: {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 2. Preprocesamiento
print("\n2. Preprocesamiento")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Datos escalados correctamente")

# 3. Reducción de dimensionalidad
print("\n3. Reducción de dimensionalidad")

# PCA
print("Aplicando PCA...")
start_time = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_time = time.time() - start_time
print(f"PCA completado en {pca_time:.2f} segundos")
print(f"Varianza explicada: {sum(pca.explained_variance_ratio_):.4f}")

# t-SNE (con menos iteraciones para mayor rapidez)
print("Aplicando t-SNE...")
start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
tsne_time = time.time() - start_time
print(f"t-SNE completado en {tsne_time:.2f} segundos")

# UMAP
print("Aplicando UMAP...")
start_time = time.time()
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
umap_time = time.time() - start_time
print(f"UMAP completado en {umap_time:.2f} segundos")

# 4. Clustering con K-means
print("\n4. Clustering con K-means")
n_clusters = 10  # Sabemos que hay 10 dígitos

# Aplicamos K-means a los datos reducidos por PCA
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
labels_pca = kmeans_pca.fit_predict(X_pca)
silhouette_pca = silhouette_score(X_pca, labels_pca)
print(f"Silhouette score con PCA: {silhouette_pca:.4f}")

# Aplicamos K-means a los datos reducidos por t-SNE
kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=42)
labels_tsne = kmeans_tsne.fit_predict(X_tsne)
silhouette_tsne = silhouette_score(X_tsne, labels_tsne)
print(f"Silhouette score con t-SNE: {silhouette_tsne:.4f}")

# Aplicamos K-means a los datos reducidos por UMAP
kmeans_umap = KMeans(n_clusters=n_clusters, random_state=42)
labels_umap = kmeans_umap.fit_predict(X_umap)
silhouette_umap = silhouette_score(X_umap, labels_umap)
print(f"Silhouette score con UMAP: {silhouette_umap:.4f}")

# 5. Visualización comparativa
print("\n5. Visualización comparativa")

# Función para visualizar clusters y etiquetas reales
def plot_clusters_and_labels(X_reduced, labels, y, title, silhouette):
    plt.figure(figsize=(15, 6))
    
    # Visualizamos clusters
    plt.subplot(121)
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab10', s=30, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Clusters K-means\nSilhouette: {silhouette:.4f}')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid(True)
    
    # Visualizamos etiquetas reales
    plt.subplot(122)
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
    plt.colorbar(scatter, label='Dígito real')
    plt.title(f'Etiquetas reales\n{title}')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Visualizamos resultados para cada método de reducción
plot_clusters_and_labels(X_pca, labels_pca, y, f'PCA (tiempo: {pca_time:.2f}s)', silhouette_pca)
plot_clusters_and_labels(X_tsne, labels_tsne, y, f't-SNE (tiempo: {tsne_time:.2f}s)', silhouette_tsne)
plot_clusters_and_labels(X_umap, labels_umap, y, f'UMAP (tiempo: {umap_time:.2f}s)', silhouette_umap)

# 6. Análisis de pureza de clusters
print("\n6. Análisis de pureza de clusters")

def calculate_cluster_purity(labels, y):
    # Creamos matriz de confusión
    confusion = pd.crosstab(labels, y)
    
    # Calculamos pureza
    purity = np.sum(np.max(confusion, axis=1)) / len(labels)
    
    return purity, confusion

# Calculamos pureza para cada método
purity_pca, confusion_pca = calculate_cluster_purity(labels_pca, y)
purity_tsne, confusion_tsne = calculate_cluster_purity(labels_tsne, y)
purity_umap, confusion_umap = calculate_cluster_purity(labels_umap, y)

print(f"Pureza de clusters con PCA: {purity_pca:.4f}")
print(f"Pureza de clusters con t-SNE: {purity_tsne:.4f}")
print(f"Pureza de clusters con UMAP: {purity_umap:.4f}")

# Visualizamos matriz de confusión del mejor método
best_method = max([("PCA", purity_pca), ("t-SNE", purity_tsne), ("UMAP", purity_umap)], key=lambda x: x[1])
print(f"\nMejor método: {best_method[0]} con pureza {best_method[1]:.4f}")

if best_method[0] == "PCA":
    confusion = confusion_pca
elif best_method[0] == "t-SNE":
    confusion = confusion_tsne
else:
    confusion = confusion_umap

plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d')
plt.title(f'Matriz de confusión: Clusters vs Dígitos reales ({best_method[0]})')
plt.xlabel('Dígito real')
plt.ylabel('Cluster')
plt.show()

# 7. Conclusiones
print("\n7. Conclusiones")
print("1. Comparación de métodos de reducción de dimensionalidad:")
print(f"   - PCA: Rápido ({pca_time:.2f}s), pero menor separación de clusters")
print(f"   - t-SNE: Más lento ({tsne_time:.2f}s), mejor separación visual")
print(f"   - UMAP: Balance entre velocidad ({umap_time:.2f}s) y calidad de separación")

print("\n2. Evaluación de clustering:")
print(f"   - Silhouette score: PCA={silhouette_pca:.4f}, t-SNE={silhouette_tsne:.4f}, UMAP={silhouette_umap:.4f}")
print(f"   - Pureza: PCA={purity_pca:.4f}, t-SNE={purity_tsne:.4f}, UMAP={purity_umap:.4f}")

print("\n3. Recomendaciones para clustering de alta dimensionalidad:")
print("   - Para exploración rápida: PCA")
print("   - Para visualización de alta calidad: t-SNE o UMAP")
print("   - Para mejor balance entre velocidad y calidad: UMAP")

print("\nEste pipeline simplificado demuestra los conceptos clave del pipeline avanzado")
print("en un formato más rápido y accesible para experimentación.")
