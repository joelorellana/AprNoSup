# Clustering Jerárquico: Técnicas Avanzadas
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("# Clustering Jerárquico: Técnicas Avanzadas")
print("=" * 80)

# 1. Introducción al clustering jerárquico
# ------------------------------------------------------------------------------
print("\n1. Introducción al clustering jerárquico")

print("El clustering jerárquico agrupa datos en una estructura de árbol o jerarquía.")
print("Existen dos enfoques principales:")
print("- Aglomerativo (bottom-up): Comienza con cada punto como un cluster y los fusiona")
print("- Divisivo (top-down): Comienza con un cluster y lo divide recursivamente")
print("\nEn este ejemplo nos centraremos en el enfoque aglomerativo, que es el más común.")

# 2. Implementación con diferentes métodos de linkage
# ------------------------------------------------------------------------------
print("\n2. Implementación con diferentes métodos de linkage")

# Generamos datos sintéticos para ilustrar
np.random.seed(42)
n_samples = 150
n_clusters = 3

# Creamos clusters con diferentes varianzas
X = np.vstack([
    np.random.randn(n_samples, 2) * 0.5 + np.array([0, 0]),
    np.random.randn(n_samples, 2) * 1.0 + np.array([5, 5]),
    np.random.randn(n_samples, 2) * 0.3 + np.array([5, 0])
])

# Visualizamos los datos
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.7)
plt.title('Datos para clustering jerárquico')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.grid(True)
plt.show()

# Diferentes métodos de linkage
linkage_methods = ['single', 'complete', 'average', 'ward']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

plt.figure(figsize=(15, 10))

for i, method in enumerate(linkage_methods):
    # Calculamos la matriz de linkage
    Z = linkage(X, method=method)
    
    # Visualizamos el dendrograma
    plt.subplot(2, 2, i+1)
    dendrogram(Z, truncate_mode='level', p=3)
    plt.title(f'Dendrograma ({method})')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Distancia')
    plt.grid(True)

plt.tight_layout()
plt.show()

print("Métodos de linkage:")
print("- Single: Distancia mínima entre puntos de dos clusters")
print("- Complete: Distancia máxima entre puntos de dos clusters")
print("- Average: Promedio de distancias entre todos los pares de puntos")
print("- Ward: Minimiza la varianza dentro de los clusters")

# Comparamos los resultados de clustering con diferentes métodos
plt.figure(figsize=(15, 10))

for i, method in enumerate(linkage_methods):
    # Aplicamos clustering jerárquico
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = clustering.fit_predict(X)
    
    # Calculamos silueta
    silhouette = silhouette_score(X, labels)
    
    # Visualizamos los clusters
    plt.subplot(2, 2, i+1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(f'Clusters con {method} linkage\nSilueta: {silhouette:.3f}')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.grid(True)

plt.tight_layout()
plt.show()

# 3. Visualización e interpretación de dendrogramas
# ------------------------------------------------------------------------------
print("\n3. Visualización e interpretación de dendrogramas")

# Calculamos la matriz de linkage con el método ward
Z = linkage(X, method='ward')

# Visualizamos el dendrograma completo
plt.figure(figsize=(12, 8))
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.title('Dendrograma completo (método Ward)')
plt.xlabel('Índice de muestra')
plt.ylabel('Distancia')
plt.axhline(y=15, color='r', linestyle='--', label='Umbral de corte')
plt.legend()
plt.grid(True)
plt.show()

print("Interpretación del dendrograma:")
print("- El eje x muestra las muestras individuales o clusters")
print("- El eje y muestra la distancia a la que se fusionan los clusters")
print("- Las líneas verticales representan clusters que se unen")
print("- La altura de la línea horizontal indica la distancia entre clusters")
print("- Un corte horizontal del dendrograma determina el número de clusters")

# 4. Determinación del punto de corte óptimo
# ------------------------------------------------------------------------------
print("\n4. Determinación del punto de corte óptimo")

# Evaluamos diferentes números de clusters
max_clusters = 10
silhouette_scores = []
distances = []

for n_clusters in range(2, max_clusters + 1):
    # Cortamos el dendrograma para obtener n_clusters
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Calculamos la silueta
    silhouette = silhouette_score(X, labels)
    silhouette_scores.append(silhouette)
    
    # Guardamos la distancia de fusión
    distances.append(Z[-(n_clusters-1), 2])

# Visualizamos los resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-', linewidth=2)
plt.xlabel('Número de clusters')
plt.ylabel('Coeficiente de silueta')
plt.title('Análisis de silueta')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), distances, 'o-', linewidth=2)
plt.xlabel('Número de clusters')
plt.ylabel('Distancia de fusión')
plt.title('Método del codo para dendrogramas')
plt.grid(True)

plt.tight_layout()
plt.show()

# Determinamos el número óptimo de clusters
optimal_n_clusters = 3  # Basado en el análisis anterior
print(f"Número óptimo de clusters según el análisis: {optimal_n_clusters}")

# Aplicamos clustering con el número óptimo
labels = fcluster(Z, optimal_n_clusters, criterion='maxclust')

# Visualizamos los clusters finales
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.title(f'Clustering jerárquico con {optimal_n_clusters} clusters')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.grid(True)
plt.colorbar(label='Cluster')
plt.show()

# 5. Matrices de distancia personalizadas
# ------------------------------------------------------------------------------
print("\n5. Matrices de distancia personalizadas")

# Descargamos datos financieros
print("Descargando datos financieros...")
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Datos de un año

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculamos los rendimientos diarios
returns = data.pct_change().dropna()

print(f"Datos financieros: {returns.shape[0]} días, {returns.shape[1]} acciones")
print(f"Acciones: {', '.join(tickers)}")

# Calculamos la matriz de correlación
corr_matrix = returns.corr()

# Visualizamos la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Matriz de correlación entre acciones')
plt.tight_layout()
plt.show()

# Convertimos la correlación en una medida de distancia
# Distancia = 1 - correlación (valores más altos indican menor correlación)
distance_matrix = 1 - corr_matrix

# Aplicamos clustering jerárquico con la matriz de distancia personalizada
Z_finance = linkage(squareform(distance_matrix), method='ward')

# Visualizamos el dendrograma
plt.figure(figsize=(12, 8))
dendrogram(Z_finance, labels=tickers, leaf_rotation=90, leaf_font_size=12)
plt.title('Dendrograma de acciones basado en correlaciones')
plt.xlabel('Acciones')
plt.ylabel('Distancia (1 - correlación)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Determinamos clusters de acciones
n_clusters_finance = 4  # Basado en el dendrograma
labels_finance = fcluster(Z_finance, n_clusters_finance, criterion='maxclust')

# Creamos un DataFrame con los resultados
stocks_clusters = pd.DataFrame({'Acción': tickers, 'Cluster': labels_finance})
print("Clusters de acciones:")
for cluster in range(1, n_clusters_finance + 1):
    stocks = stocks_clusters[stocks_clusters['Cluster'] == cluster]['Acción'].tolist()
    print(f"Cluster {cluster}: {', '.join(stocks)}")

# 6. Comparación con K-means
# ------------------------------------------------------------------------------
print("\n6. Comparación con K-means")

# Aplicamos K-means a los datos financieros
kmeans = KMeans(n_clusters=n_clusters_finance, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(distance_matrix) + 1  # +1 para que empiece en 1 como fcluster

# Comparamos los resultados
comparison = pd.DataFrame({
    'Acción': tickers,
    'Cluster Jerárquico': labels_finance,
    'Cluster K-means': kmeans_labels
})

print("Comparación de clusters:")
print(comparison)

# Calculamos la concordancia entre ambos métodos
concordance = sum(comparison['Cluster Jerárquico'] == comparison['Cluster K-means']) / len(tickers)
print(f"Concordancia entre métodos: {concordance:.2%}")

# Visualizamos los clusters en un espacio bidimensional usando MDS
from sklearn.manifold import MDS

# Aplicamos MDS para visualizar las distancias en 2D
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
positions = mds.fit_transform(distance_matrix)

# Visualizamos los clusters de ambos métodos
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
for i, ticker in enumerate(tickers):
    plt.scatter(positions[i, 0], positions[i, 1], s=100, 
               color=plt.cm.viridis(labels_finance[i] / (n_clusters_finance + 1)))
    plt.text(positions[i, 0], positions[i, 1], ticker, fontsize=12)
plt.title('Clustering Jerárquico')
plt.grid(True)

plt.subplot(1, 2, 2)
for i, ticker in enumerate(tickers):
    plt.scatter(positions[i, 0], positions[i, 1], s=100, 
               color=plt.cm.viridis(kmeans_labels[i] / (n_clusters_finance + 1)))
    plt.text(positions[i, 0], positions[i, 1], ticker, fontsize=12)
plt.title('Clustering K-means')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Ventajas del clustering jerárquico vs K-means:")
print("1. No requiere especificar el número de clusters a priori")
print("2. Proporciona una jerarquía completa de los datos")
print("3. Puede usar matrices de distancia personalizadas")
print("4. No asume formas específicas de clusters")
print("5. Es determinístico (siempre produce el mismo resultado)")

print("\nDesventajas del clustering jerárquico vs K-means:")
print("1. Mayor complejidad computacional (O(n²) vs O(n))")
print("2. Mayor uso de memoria")
print("3. Puede ser difícil de interpretar para grandes conjuntos de datos")
print("4. No tiene un mecanismo para reasignar puntos una vez asignados")

# 7. Conclusiones y ejercicios propuestos
# ------------------------------------------------------------------------------
print("\n7. Conclusiones y ejercicios propuestos")

print("Conclusiones:")
print("- El clustering jerárquico es una técnica poderosa para descubrir estructuras anidadas")
print("- La elección del método de linkage afecta significativamente los resultados")
print("- Los dendrogramas proporcionan una visualización intuitiva de la estructura jerárquica")
print("- Las matrices de distancia personalizadas permiten aplicar clustering a diversos tipos de datos")
print("- El clustering jerárquico y K-means pueden complementarse para análisis más robustos")

print("\nEjercicios propuestos:")
print("1. Aplique clustering jerárquico a un conjunto de datos de su interés")
print("2. Compare los resultados de los diferentes métodos de linkage")
print("3. Implemente una matriz de distancia personalizada para un problema específico")
print("4. Combine clustering jerárquico con técnicas de reducción de dimensionalidad")
print("5. Desarrolle un método para evaluar la estabilidad de los clusters jerárquicos")
