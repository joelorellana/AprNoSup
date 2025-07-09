# Estrategias para la evaluación y optimización de modelos de clustering
# Implementación con datasets reales
# Autor: UDB - Aprendizaje No Supervisado

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("Estrategias para la Evaluación y Optimización de Modelos de Clustering")
print("=====================================================================")

# 1. Carga y preparación de datos
# ------------------------------------------------------------------------------
print("\n1. Carga y preparación de datos")

# Cargamos el dataset de iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset Iris: {X.shape[0]} muestras, {X.shape[1]} características")
print(f"Características: {feature_names}")
print(f"Clases: {target_names}")
print(f"Distribución de clases: {np.bincount(y)}")

# Escalamos los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualizamos el dataset
plt.figure(figsize=(10, 8))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.title('Dataset Iris (PCA)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
plt.colorbar(label='Especie')
plt.grid(True)
plt.show()

# 2. Determinación del número óptimo de clusters
# ------------------------------------------------------------------------------
print("\n2. Determinación del número óptimo de clusters")

# Método del codo (Elbow Method)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []
range_n_clusters = range(2, 11)

for n_clusters in range_n_clusters:
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    
    # Calculamos métricas
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels))

# Visualizamos los resultados
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Método del codo
axes[0, 0].plot(range_n_clusters, inertias, 'o-', linewidth=2)
axes[0, 0].set_title('Método del Codo')
axes[0, 0].set_xlabel('Número de clusters')
axes[0, 0].set_ylabel('Inercia')
axes[0, 0].grid(True)

# Silhouette Score
axes[0, 1].plot(range_n_clusters, silhouette_scores, 'o-', linewidth=2)
axes[0, 1].set_title('Coeficiente de Silueta')
axes[0, 1].set_xlabel('Número de clusters')
axes[0, 1].set_ylabel('Silhouette Score (mayor es mejor)')
axes[0, 1].grid(True)

# Davies-Bouldin Index
axes[1, 0].plot(range_n_clusters, davies_bouldin_scores, 'o-', linewidth=2)
axes[1, 0].set_title('Índice Davies-Bouldin')
axes[1, 0].set_xlabel('Número de clusters')
axes[1, 0].set_ylabel('Davies-Bouldin (menor es mejor)')
axes[1, 0].grid(True)

# Calinski-Harabasz Index
axes[1, 1].plot(range_n_clusters, calinski_harabasz_scores, 'o-', linewidth=2)
axes[1, 1].set_title('Índice Calinski-Harabasz')
axes[1, 1].set_xlabel('Número de clusters')
axes[1, 1].set_ylabel('Calinski-Harabasz (mayor es mejor)')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Identificamos el número óptimo de clusters según cada métrica
optimal_n_inertia = range_n_clusters[np.argmax(np.diff(np.diff(inertias)))+1]
optimal_n_silhouette = range_n_clusters[np.argmax(silhouette_scores)]
optimal_n_davies = range_n_clusters[np.argmin(davies_bouldin_scores)]
optimal_n_calinski = range_n_clusters[np.argmax(calinski_harabasz_scores)]

print(f"Número óptimo de clusters según:")
print(f"- Método del codo: {optimal_n_inertia}")
print(f"- Coeficiente de Silueta: {optimal_n_silhouette}")
print(f"- Índice Davies-Bouldin: {optimal_n_davies}")
print(f"- Índice Calinski-Harabasz: {optimal_n_calinski}")

# 3. Evaluación de clustering con métricas externas
# ------------------------------------------------------------------------------
print("\n3. Evaluación de clustering con métricas externas")

# Aplicamos K-means con el número óptimo de clusters
n_clusters = 3  # Sabemos que Iris tiene 3 clases
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Calculamos métricas externas
ari = adjusted_rand_score(y, labels_kmeans)
nmi = normalized_mutual_info_score(y, labels_kmeans)

print(f"Métricas externas para K-means con {n_clusters} clusters:")
print(f"- Adjusted Rand Index: {ari:.4f} (1.0 es coincidencia perfecta)")
print(f"- Normalized Mutual Information: {nmi:.4f} (1.0 es coincidencia perfecta)")

# Visualizamos la comparación entre clusters y etiquetas reales
plt.figure(figsize=(15, 6))

# Clusters predichos
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='viridis', s=50, alpha=0.8)
plt.title(f'Clusters K-means (k={n_clusters})')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.grid(True)

# Etiquetas reales
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.title('Clases reales')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Especie')
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. Validación de estabilidad del clustering
# ------------------------------------------------------------------------------
print("\n4. Validación de estabilidad del clustering")

# Función para evaluar la estabilidad mediante submuestreo
def evaluate_stability(X, n_clusters, n_samples=10, sample_size=0.8, random_state=42):
    np.random.seed(random_state)
    ari_scores = []
    
    # Generamos un modelo de referencia
    kmeans_ref = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_ref = kmeans_ref.fit_predict(X)
    
    # Realizamos submuestreo múltiples veces
    for i in range(n_samples):
        # Seleccionamos un subconjunto aleatorio
        indices = np.random.choice(len(X), size=int(sample_size*len(X)), replace=False)
        X_sample = X[indices]
        
        # Aplicamos clustering al subconjunto
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state+i, n_init=10)
        kmeans.fit(X_sample)
        
        # Predecimos las etiquetas para todos los datos
        labels_full = kmeans.predict(X)
        
        # Calculamos ARI entre las etiquetas de referencia y las nuevas
        ari = adjusted_rand_score(labels_ref, labels_full)
        ari_scores.append(ari)
    
    return np.mean(ari_scores), np.std(ari_scores)

# Evaluamos la estabilidad para diferentes números de clusters
stability_means = []
stability_stds = []

for n_clusters in range(2, 6):
    mean_ari, std_ari = evaluate_stability(X_scaled, n_clusters)
    stability_means.append(mean_ari)
    stability_stds.append(std_ari)
    print(f"Estabilidad para k={n_clusters}: ARI medio = {mean_ari:.4f} ± {std_ari:.4f}")

# Visualizamos la estabilidad
plt.figure(figsize=(10, 6))
plt.errorbar(range(2, 6), stability_means, yerr=stability_stds, fmt='o-', capsize=5)
plt.title('Estabilidad del clustering para diferentes números de clusters')
plt.xlabel('Número de clusters')
plt.ylabel('ARI medio (mayor es más estable)')
plt.grid(True)
plt.show()

# 5. Optimización de parámetros para clustering jerárquico
# ------------------------------------------------------------------------------
print("\n5. Optimización de parámetros para clustering jerárquico")

# Evaluamos diferentes métodos de linkage
linkage_methods = ['ward', 'complete', 'average', 'single']
linkage_scores = []

for method in linkage_methods:
    # Aplicamos clustering jerárquico
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels_hierarchical = hierarchical.fit_predict(X_scaled)
    
    # Calculamos métricas
    silhouette = silhouette_score(X_scaled, labels_hierarchical)
    ari = adjusted_rand_score(y, labels_hierarchical)
    
    linkage_scores.append({
        'method': method,
        'silhouette': silhouette,
        'ari': ari
    })
    
    print(f"Método de linkage '{method}':")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  ARI: {ari:.4f}")

# Visualizamos dendrogramas para diferentes métodos
plt.figure(figsize=(15, 10))
for i, method in enumerate(linkage_methods):
    plt.subplot(2, 2, i+1)
    Z = linkage(X_scaled, method=method)
    dendrogram(Z)
    plt.title(f'Dendrograma ({method})')
    plt.xlabel('Muestras')
    plt.ylabel('Distancia')
    plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Conclusiones
# ------------------------------------------------------------------------------
print("\n6. Conclusiones")

print("""
Conclusiones sobre evaluación y optimización de clustering:

1. Determinación del número óptimo de clusters:
   - Diferentes métricas pueden sugerir diferentes números óptimos de clusters.
   - Es importante considerar múltiples enfoques y el conocimiento del dominio.
   - Para el dataset Iris, la mayoría de las métricas sugieren 3 clusters, lo que
     coincide con el número real de especies.

2. Métricas de evaluación:
   - Las métricas internas como silhouette y Davies-Bouldin evalúan la calidad del
     clustering sin referencia a etiquetas externas.
   - Las métricas externas como ARI y NMI permiten comparar con etiquetas conocidas,
     útil para validación en datasets de referencia.

3. Estabilidad del clustering:
   - La estabilidad es crucial para confiar en los resultados del clustering.
   - El submuestreo aleatorio permite evaluar cuán consistentes son los clusters.
   - Mayor estabilidad generalmente indica un número de clusters más robusto.

4. Optimización de parámetros:
   - Diferentes métodos de linkage en clustering jerárquico producen resultados distintos.
   - La selección de parámetros debe basarse tanto en métricas cuantitativas como en
     la interpretabilidad de los resultados.

5. Recomendaciones prácticas:
   - Utilizar múltiples algoritmos y comparar sus resultados.
   - Combinar métricas internas y externas cuando sea posible.
   - Validar la estabilidad de los clusters mediante técnicas de remuestreo.
   - Priorizar la interpretabilidad de los resultados en el contexto del problema.
""")
