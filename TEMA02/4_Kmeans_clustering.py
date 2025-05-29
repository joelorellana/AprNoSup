# K-means: Implementación avanzada y aplicaciones industriales
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Implementación detallada de K-means
# ------------------------------------------------------------------------------
print("1. Implementación detallada de K-means")

# Generamos datos sintéticos para el ejemplo
n_samples = 1500
n_features = 2
n_clusters = 4
random_state = 42

# Creamos clusters con diferentes varianzas para mostrar limitaciones de K-means
X, y_true = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=n_clusters,
                      cluster_std=[1.0, 2.5, 0.5, 3.0],
                      random_state=random_state)

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

# Inicialización: seleccionamos K centroides iniciales (método K-means++)
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualizamos los resultados finales
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroides')
plt.title('Resultado de K-means')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)
plt.show()

# Evaluamos la calidad del clustering
inertia = kmeans.inertia_
silhouette_avg = silhouette_score(X, labels)
print(f"Inercia (suma de distancias cuadráticas): {inertia:.2f}")
print(f"Coeficiente de silueta: {silhouette_avg:.3f}")

# 2. Determinación del número óptimo de clusters
# ------------------------------------------------------------------------------
print("\n2. Determinación del número óptimo de clusters")

# Método del codo
inertias = []
silhouette_scores = []
range_n_clusters = range(2, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    
    # Solo calculamos silueta para n_clusters >= 2
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Visualizamos el método del codo
plt.figure(figsize=(14, 6))

plt.subplot(121)
plt.plot(range_n_clusters, inertias, 'o-', linewidth=2)
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.grid(True)

plt.subplot(122)
plt.plot(range_n_clusters, silhouette_scores, 'o-', linewidth=2)
plt.xlabel('Número de clusters')
plt.ylabel('Coeficiente de silueta')
plt.title('Análisis de silueta')
plt.grid(True)

plt.tight_layout()
plt.show()

# Identificamos el número óptimo de clusters
optimal_k_elbow = 4  # Valor aproximado donde se observa el "codo"
optimal_k_silhouette = np.argmax(silhouette_scores) + 2  # +2 porque empezamos desde 2

print(f"Número óptimo de clusters según método del codo: {optimal_k_elbow}")
print(f"Número óptimo de clusters según coeficiente de silueta: {optimal_k_silhouette}")

# 3. Impacto del preprocesamiento en K-means
# ------------------------------------------------------------------------------
print("\n3. Impacto del preprocesamiento en K-means")

# Generamos datos con diferentes escalas
n_samples = 1000
n_features = 2
n_clusters = 3

# Creamos datos con diferentes escalas
X_uneven = np.random.randn(n_samples, n_features)
X_uneven[:, 0] = X_uneven[:, 0] * 10  # Amplificamos la primera dimensión

# Aplicamos diferentes técnicas de escalado
scalers = {
    'Sin escalar': X_uneven,
    'StandardScaler': StandardScaler().fit_transform(X_uneven),
    'MinMaxScaler': MinMaxScaler().fit_transform(X_uneven),
    'RobustScaler': RobustScaler().fit_transform(X_uneven)
}

# Visualizamos los datos con diferentes escalados
plt.figure(figsize=(16, 12))
i = 1

for name, X_scaled in scalers.items():
    # Aplicamos K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    # Calculamos métricas
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(X_scaled, labels)
    
    # Visualizamos
    plt.subplot(2, 2, i)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=40, alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X')
    plt.title(f'{name}\nSilueta: {silhouette_avg:.3f}, Inercia: {inertia:.2f}')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.grid(True)
    i += 1

plt.tight_layout()
plt.show()

print("Observación: El escalado de datos tiene un impacto significativo en los resultados de K-means.")
print("StandardScaler suele ser la opción más recomendada para K-means, especialmente con variables de diferentes escalas.")

# 4. Comparación de variantes de K-means: K-means vs Mini-batch K-means
# ------------------------------------------------------------------------------
print("\n4. Comparación de variantes de K-means: K-means vs Mini-batch K-means")

# Generamos un conjunto de datos más grande para evaluar rendimiento
n_samples_large = 100000
n_features = 2
n_clusters = 5

X_large, _ = make_blobs(n_samples=n_samples_large, 
                       n_features=n_features,
                       centers=n_clusters,
                       random_state=random_state)

# Comparamos tiempo de ejecución
algorithms = {
    'K-means': KMeans(n_clusters=n_clusters, random_state=random_state),
    'Mini-batch K-means (batch_size=100)': MiniBatchKMeans(n_clusters=n_clusters, 
                                                          batch_size=100,
                                                          random_state=random_state),
    'Mini-batch K-means (batch_size=1000)': MiniBatchKMeans(n_clusters=n_clusters, 
                                                           batch_size=1000,
                                                           random_state=random_state)
}

results = {}

for name, algorithm in algorithms.items():
    # Medimos tiempo de ejecución
    start_time = time.time()
    algorithm.fit(X_large)
    end_time = time.time()
    
    # Guardamos resultados
    results[name] = {
        'time': end_time - start_time,
        'inertia': algorithm.inertia_,
        'labels': algorithm.labels_,
        'centroids': algorithm.cluster_centers_
    }
    
    print(f"{name}: Tiempo de ejecución = {results[name]['time']:.2f} segundos, Inercia = {results[name]['inertia']:.2f}")

# Visualizamos comparación de rendimiento
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [results[name]['time'] for name in results.keys()], color='skyblue')
plt.title('Comparación de tiempo de ejecución')
plt.ylabel('Tiempo (segundos)')
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.show()

# Visualizamos comparación de calidad (inercia)
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [results[name]['inertia'] for name in results.keys()], color='lightgreen')
plt.title('Comparación de inercia')
plt.ylabel('Inercia')
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.show()

print("Observación: Mini-batch K-means ofrece un equilibrio entre eficiencia computacional y calidad de los resultados.")
print("Es especialmente útil para conjuntos de datos muy grandes donde K-means estándar sería prohibitivo.")

# 5. Caso de estudio: Segmentación de clientes (RFM Analysis)
# ------------------------------------------------------------------------------
print("\n5. Caso de estudio: Segmentación de clientes (RFM Analysis)")

# Generamos datos sintéticos que simulan un análisis RFM (Recency, Frequency, Monetary)
n_customers = 1000

# Simulamos datos RFM con distribuciones realistas
np.random.seed(random_state)

# Recency: días desde la última compra (valores menores son mejores)
recency = np.random.exponential(scale=30, size=n_customers).astype(int) + 1

# Frequency: número de compras en el período
frequency = np.random.lognormal(mean=1.1, sigma=1, size=n_customers).astype(int) + 1

# Monetary: valor promedio de compra
monetary = np.random.lognormal(mean=4.5, sigma=0.7, size=n_customers)

# Creamos el DataFrame
df_rfm = pd.DataFrame({
    'CustomerID': range(1, n_customers + 1),
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary
})

# Mostramos los primeros registros
print("Muestra de datos RFM:")
print(df_rfm.head())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df_rfm.describe())

# Preprocesamos los datos
X_rfm = df_rfm[['Recency', 'Frequency', 'Monetary']].copy()

# Invertimos Recency (valores menores son mejores, pero para clustering queremos que valores mayores sean mejores)
X_rfm['Recency'] = X_rfm['Recency'].max() - X_rfm['Recency']

# Escalamos los datos
scaler = StandardScaler()
X_rfm_scaled = scaler.fit_transform(X_rfm)

# Aplicamos K-means
n_clusters = 4  # Típicamente se usan 4-5 segmentos en RFM
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
df_rfm['Cluster'] = kmeans.fit_predict(X_rfm_scaled)

# Analizamos los centroides
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=['Recency', 'Frequency', 'Monetary'])
centroids_df['Recency'] = X_rfm['Recency'].max() - centroids_df['Recency']  # Revertimos la transformación
centroids_df.index.name = 'Cluster'
centroids_df.index = centroids_df.index.map(str)

print("\nCentroides de los clusters (valores originales):")
print(centroids_df)

# Visualizamos los clusters
plt.figure(figsize=(16, 12))

# Recency vs Frequency
plt.subplot(221)
sns.scatterplot(x='Recency', y='Frequency', hue='Cluster', data=df_rfm, palette='viridis', alpha=0.7)
plt.title('Recency vs Frequency')
plt.xlabel('Recency (días desde última compra)')
plt.ylabel('Frequency (número de compras)')

# Recency vs Monetary
plt.subplot(222)
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=df_rfm, palette='viridis', alpha=0.7)
plt.title('Recency vs Monetary')
plt.xlabel('Recency (días desde última compra)')
plt.ylabel('Monetary (valor promedio)')

# Frequency vs Monetary
plt.subplot(223)
sns.scatterplot(x='Frequency', y='Monetary', hue='Cluster', data=df_rfm, palette='viridis', alpha=0.7)
plt.title('Frequency vs Monetary')
plt.xlabel('Frequency (número de compras)')
plt.ylabel('Monetary (valor promedio)')

# Distribución de clusters
plt.subplot(224)
cluster_counts = df_rfm['Cluster'].value_counts().sort_index()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.title('Distribución de clientes por cluster')
plt.xlabel('Cluster')
plt.ylabel('Número de clientes')

plt.tight_layout()
plt.show()

# Interpretación de los clusters
print("\nInterpretación de los clusters:")
cluster_interpretation = {
    '0': 'Clientes ocasionales de bajo valor',
    '1': 'Clientes frecuentes de valor medio',
    '2': 'Clientes VIP (alta frecuencia y alto valor)',
    '3': 'Clientes inactivos (baja recencia)'
}

# Calculamos métricas promedio por cluster
cluster_summary = df_rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'})

cluster_summary['Interpretation'] = [cluster_interpretation[str(i)] for i in range(n_clusters)]
print(cluster_summary)

# 6. Limitaciones de K-means y consideraciones prácticas
# ------------------------------------------------------------------------------
print("\n6. Limitaciones de K-means y consideraciones prácticas")

# Generamos datos que ilustran limitaciones de K-means
from sklearn.datasets import make_moons, make_circles

# Generamos tres conjuntos de datos desafiantes
n_samples = 1000

# Dataset 1: Clusters no convexos (lunas)
X_moons, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)

# Dataset 2: Clusters concéntricos
X_circles, _ = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=random_state)

# Dataset 3: Clusters de diferentes densidades y tamaños
X_blobs, _ = make_blobs(n_samples=[700, 300], 
                       centers=[[0, 0], [3, 3]], 
                       cluster_std=[1.0, 0.5],
                       random_state=random_state)

datasets = {
    'Clusters no convexos (lunas)': X_moons,
    'Clusters concéntricos': X_circles,
    'Clusters de diferentes densidades': X_blobs
}

# Aplicamos K-means a cada conjunto de datos
plt.figure(figsize=(18, 6))
i = 1

for name, X in datasets.items():
    # Aplicamos K-means con k=2
    kmeans = KMeans(n_clusters=2, random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    # Visualizamos
    plt.subplot(1, 3, i)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               c='red', s=200, marker='X')
    plt.title(name)
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.grid(True)
    i += 1

plt.tight_layout()
plt.show()

print("Limitaciones de K-means:")
print("1. Asume clusters de forma convexa y tamaño similar")
print("2. Sensible a valores atípicos (outliers)")
print("3. Requiere especificar el número de clusters a priori")
print("4. Puede converger a mínimos locales, dependiendo de la inicialización")
print("5. No adecuado para clusters de formas complejas o densidades variables")

# 7. Ejercicios propuestos
# ------------------------------------------------------------------------------
print("\n7. Ejercicios propuestos")
print("1. Experimente con diferentes inicializaciones de K-means (random, k-means++) y compare resultados.")
print("2. Implemente y compare K-means con K-medoids en presencia de outliers.")
print("3. Desarrolle un caso de segmentación de clientes utilizando datos reales y proponga estrategias de marketing para cada segmento.")
print("4. Compare el rendimiento de K-means y Mini-batch K-means para diferentes tamaños de conjuntos de datos.")
print("5. Investigue y aplique métodos automáticos para determinar el número óptimo de clusters (Gap statistic, Silhouette, etc.).")
