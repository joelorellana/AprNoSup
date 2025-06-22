# K-means: Fundamentos e Implementación
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.datasets import load_wine, make_blobs, make_moons
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("# K-means: Fundamentos e Implementación")
print("=" * 80)

# 1. Introducción al clustering y K-means
# ------------------------------------------------------------------------------
print("\n1. Introducción al clustering y K-means")

print("El clustering es una técnica de aprendizaje no supervisado que agrupa datos similares.")
print("K-means es uno de los algoritmos de clustering más populares debido a su simplicidad y eficiencia.")
print("Funciona dividiendo los datos en K grupos, donde cada grupo está representado por su centroide.")
print("El algoritmo busca minimizar la suma de distancias cuadráticas entre los puntos y sus centroides.")

# 2. Implementación paso a paso de K-means
# ------------------------------------------------------------------------------
print("\n2. Implementación paso a paso de K-means")

# Generamos datos sintéticos para ilustrar el algoritmo
np.random.seed(42)
n_samples = 300
n_clusters = 3

X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, 
                      cluster_std=[1.0, 1.2, 0.8], random_state=42)

# Visualizamos los datos originales
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], s=50, c='gray', alpha=0.5)
plt.title('Datos originales')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.grid(True)
plt.show()

# Implementación paso a paso de K-means
print("\nImplementación paso a paso de K-means:")

# Paso 1: Inicialización de centroides (mostramos k-means++)
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=42)
kmeans.fit(X)

# Visualizamos la inicialización
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], s=50, c='gray', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='X', s=200, label='Centroides iniciales')
plt.title('Inicialización de centroides (k-means++)')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)
plt.show()

# Paso 2: Asignación de puntos a centroides
labels = kmeans.labels_

# Visualizamos la asignación
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='X', s=200, label='Centroides')
plt.title('Asignación de puntos a centroides')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)
plt.show()

# Evaluamos la calidad del clustering
inertia = kmeans.inertia_
silhouette = silhouette_score(X, labels)
calinski = calinski_harabasz_score(X, labels)

print(f"Inercia (suma de distancias cuadráticas): {inertia:.2f}")
print(f"Coeficiente de silueta: {silhouette:.3f}")
print(f"Índice Calinski-Harabasz: {calinski:.2f}")

# 3. Determinación del número óptimo de clusters
# ------------------------------------------------------------------------------
print("\n3. Determinación del número óptimo de clusters")

# Cargamos un dataset real: Wine
wine = load_wine()
X_wine = wine.data
y_wine = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print(f"Dataset Wine: {X_wine.shape[0]} muestras, {X_wine.shape[1]} características")
print(f"Clases: {target_names}")

# Escalamos los datos
scaler = StandardScaler()
X_wine_scaled = scaler.fit_transform(X_wine)

# Método del codo
inertias = []
silhouette_scores = []
calinski_scores = []
range_n_clusters = range(2, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_wine_scaled)
    inertias.append(kmeans.inertia_)
    
    # Calculamos métricas de calidad
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_wine_scaled, labels))
    calinski_scores.append(calinski_harabasz_score(X_wine_scaled, labels))

# Visualizamos el método del codo
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range_n_clusters, inertias, 'o-', linewidth=2)
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.grid(True)

plt.subplot(132)
plt.plot(range_n_clusters, silhouette_scores, 'o-', linewidth=2)
plt.xlabel('Número de clusters')
plt.ylabel('Coeficiente de silueta')
plt.title('Análisis de silueta')
plt.grid(True)

plt.subplot(133)
plt.plot(range_n_clusters, calinski_scores, 'o-', linewidth=2)
plt.xlabel('Número de clusters')
plt.ylabel('Índice Calinski-Harabasz')
plt.title('Índice Calinski-Harabasz')
plt.grid(True)

plt.tight_layout()
plt.show()

# Identificamos el número óptimo de clusters
optimal_k = 3  # Basado en el conocimiento previo y las métricas

print(f"Número óptimo de clusters según las métricas: {optimal_k}")
print(f"Coincide con el número real de clases en el dataset: {len(np.unique(y_wine))}")

# 4. Impacto del escalado de datos en K-means
# ------------------------------------------------------------------------------
print("\n4. Impacto del escalado de datos en K-means")

# Generamos datos con diferentes escalas
np.random.seed(42)
n_samples = 300
X_uneven = np.random.randn(n_samples, 2)
X_uneven[:, 0] = X_uneven[:, 0] * 10  # Amplificamos la primera dimensión

# Aplicamos diferentes técnicas de escalado
scalers = {
    'Sin escalar': X_uneven,
    'StandardScaler': StandardScaler().fit_transform(X_uneven),
    'MinMaxScaler': MinMaxScaler().fit_transform(X_uneven)
}

# Aplicamos K-means a cada versión de los datos
plt.figure(figsize=(15, 5))
i = 1

for name, X_scaled in scalers.items():
    # Aplicamos K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculamos métricas
    silhouette_avg = silhouette_score(X_scaled, labels)
    
    # Visualizamos
    plt.subplot(1, 3, i)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=40, alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               c='red', marker='X', s=100)
    plt.title(f'{name}\nSilueta: {silhouette_avg:.3f}')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.grid(True)
    i += 1

plt.tight_layout()
plt.show()

print("Observaciones sobre el escalado:")
print("- Sin escalar: La primera característica domina debido a su mayor escala")
print("- StandardScaler: Normaliza a media 0 y desviación estándar 1")
print("- MinMaxScaler: Escala los datos al rango [0,1]")
print("El escalado adecuado es crucial para K-means ya que utiliza distancias euclidianas")

# 5. Visualización e interpretación de clusters
# ------------------------------------------------------------------------------
print("\n5. Visualización e interpretación de clusters")

# Aplicamos K-means al dataset Wine con el número óptimo de clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
labels = kmeans.fit_predict(X_wine_scaled)

# Comparamos con las etiquetas reales
contingency = pd.crosstab(labels, y_wine, rownames=['Cluster'], colnames=['Clase'])
print("Tabla de contingencia (clusters vs clases reales):")
print(contingency)

# Analizamos los centroides
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=feature_names)
centroids_df.index = [f'Cluster {i}' for i in range(optimal_k)]

# Visualizamos los centroides como un mapa de calor
plt.figure(figsize=(14, 6))
sns.heatmap(centroids_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Características de los centroides')
plt.tight_layout()
plt.show()

# Visualizamos los clusters en un espacio reducido (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_wine_pca = pca.fit_transform(X_wine_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
centroids_pca = pca.transform(centroids)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroides')
plt.title('Clustering K-means del dataset Wine (PCA)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()
plt.grid(True)
plt.colorbar(scatter, label='Cluster')
plt.show()

print("Interpretación de los clusters:")
for i in range(optimal_k):
    dominant_class = contingency.loc[i].idxmax()
    dominant_class_name = target_names[dominant_class]
    dominant_features = centroids_df.loc[f'Cluster {i}'].nlargest(3).index.tolist()
    print(f"Cluster {i}: Principalmente {dominant_class_name}")
    print(f"  Características dominantes: {', '.join(dominant_features)}")

# 6. Limitaciones de K-means con formas no convexas
# ------------------------------------------------------------------------------
print("\n6. Limitaciones de K-means con formas no convexas")

# Generamos datos con formas no convexas
X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Aplicamos K-means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
moon_labels = kmeans.fit_predict(X_moons)

# Visualizamos los resultados
plt.figure(figsize=(10, 8))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=moon_labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='X', s=200, label='Centroides')
plt.title('K-means en datos con forma de lunas')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)
plt.show()

print("Limitaciones de K-means:")
print("1. Asume clusters de forma convexa y tamaño similar")
print("2. Sensible a valores atípicos (outliers)")
print("3. Requiere especificar el número de clusters a priori")
print("4. Puede converger a mínimos locales, dependiendo de la inicialización")
print("5. No adecuado para clusters de formas complejas o densidades variables")

# 7. Conclusiones y ejercicios propuestos
# ------------------------------------------------------------------------------
print("\n7. Conclusiones y ejercicios propuestos")

print("Conclusiones:")
print("- K-means es un algoritmo eficiente y simple para clustering")
print("- El escalado de datos es crucial para obtener buenos resultados")
print("- La selección del número óptimo de clusters puede guiarse por métricas como silueta")
print("- K-means tiene limitaciones con formas no convexas y outliers")

print("\nEjercicios propuestos:")
print("1. Implemente K-means desde cero en Python y compare con la implementación de scikit-learn")
print("2. Experimente con diferentes inicializaciones (random vs k-means++) y compare resultados")
print("3. Aplique K-means a un dataset real de su interés y analice los resultados")
print("4. Compare el rendimiento de K-means y Mini-batch K-means para grandes conjuntos de datos")
print("5. Investigue técnicas para manejar outliers en K-means")
