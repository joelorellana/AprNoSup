# Clustering Jerárquico: Implementación avanzada y aplicaciones industriales
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs, make_moons, fetch_olivetti_faces
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Fundamentos del clustering jerárquico
# ------------------------------------------------------------------------------
print("1. Fundamentos del clustering jerárquico")

# Generamos datos sintéticos para el ejemplo
n_samples = 150
n_features = 2
n_clusters = 4
random_state = 42

# Creamos clusters con diferentes varianzas
X, y_true = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=n_clusters,
                      cluster_std=[1.0, 2.0, 0.5, 1.5],
                      random_state=random_state)

# Visualizamos los datos originales
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], s=50, c='gray', alpha=0.5)
plt.title('Datos originales')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.grid(True)
plt.show()

# Calculamos la matriz de linkage (enfoque aglomerativo)
print("\nCalculando matriz de linkage con diferentes métodos...")

# Función para visualizar dendrogramas con diferentes métodos de linkage
def plot_dendrogram(methods):
    plt.figure(figsize=(20, 10))
    for i, method in enumerate(methods):
        plt.subplot(1, len(methods), i+1)
        Z = linkage(X, method=method)
        dendrogram(Z, leaf_rotation=90)
        plt.title(f'Dendrograma ({method})')
        plt.xlabel('Índice de muestra')
        plt.ylabel('Distancia')
    plt.tight_layout()
    plt.show()

# Visualizamos dendrogramas con diferentes métodos de linkage
methods = ['single', 'complete', 'average', 'ward']
plot_dendrogram(methods)

# Aplicamos clustering jerárquico con diferentes métodos de linkage
plt.figure(figsize=(20, 5))
for i, method in enumerate(methods):
    # Aplicamos clustering jerárquico
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = model.fit_predict(X)
    
    # Calculamos coeficiente de silueta
    silhouette_avg = silhouette_score(X, labels)
    
    # Visualizamos
    plt.subplot(1, 4, i+1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(f'Método: {method}\nSilueta: {silhouette_avg:.3f}')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.grid(True)

plt.tight_layout()
plt.show()

print("Observaciones sobre los métodos de linkage:")
print("- Single linkage: Tiende a formar clusters alargados (encadenamiento)")
print("- Complete linkage: Tiende a formar clusters compactos de tamaño similar")
print("- Average linkage: Ofrece un compromiso entre single y complete")
print("- Ward: Minimiza la varianza dentro de los clusters, similar a K-means")

# 2. Determinación del número óptimo de clusters
# ------------------------------------------------------------------------------
print("\n2. Determinación del número óptimo de clusters")

# Calculamos la matriz de linkage usando el método Ward
Z = linkage(X, method='ward')

# Visualizamos el dendrograma con una línea horizontal para corte
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Dendrograma para determinar número óptimo de clusters')
plt.xlabel('Índice de muestra')
plt.ylabel('Distancia')
plt.axhline(y=6, color='r', linestyle='--', label='Umbral de corte')
plt.legend()
plt.show()

# Analizamos las distancias de fusión
last_fusion_distances = Z[:, 2]
fusion_accelerations = np.diff(last_fusion_distances, 2)  # Segunda derivada

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(range(len(last_fusion_distances)), last_fusion_distances[::-1], 'o-')
plt.title('Distancias de fusión')
plt.xlabel('Número de clusters')
plt.ylabel('Distancia')
plt.grid(True)
plt.xticks(range(len(last_fusion_distances)), range(len(last_fusion_distances), 0, -1))

plt.subplot(122)
plt.plot(range(len(fusion_accelerations)), fusion_accelerations[::-1], 'o-')
plt.title('Aceleración de distancias de fusión')
plt.xlabel('Número de clusters')
plt.ylabel('Aceleración')
plt.grid(True)
plt.xticks(range(len(fusion_accelerations)), range(len(fusion_accelerations), 0, -1))

plt.tight_layout()
plt.show()

# Evaluamos diferentes números de clusters usando silueta
silhouette_scores = []
range_n_clusters = range(2, 11)

for n_clusters in range_n_clusters:
    # Obtenemos las etiquetas cortando el dendrograma
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    # Calculamos el coeficiente de silueta
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)
    
    print(f"Para n_clusters = {n_clusters}, el coeficiente de silueta es: {silhouette_avg:.3f}")

# Visualizamos los resultados
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, 'o-')
plt.title('Análisis de silueta para diferentes números de clusters')
plt.xlabel('Número de clusters')
plt.ylabel('Coeficiente de silueta')
plt.grid(True)
plt.show()

# Identificamos el número óptimo de clusters
optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f"Número óptimo de clusters según coeficiente de silueta: {optimal_n_clusters}")

# 3. Comparación con K-means y DBSCAN
# ------------------------------------------------------------------------------
print("\n3. Comparación con K-means y DBSCAN")

# Generamos datos que ilustran las fortalezas de diferentes algoritmos
datasets = {
    'Clusters convexos': make_blobs(n_samples=300, centers=3, random_state=random_state)[0],
    'Clusters no convexos': make_moons(n_samples=300, noise=0.1, random_state=random_state)[0]
}

# Aplicamos diferentes algoritmos a cada conjunto de datos
algorithms = {
    'K-means': lambda X, n_clusters: AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X),
    'Jerárquico (Ward)': lambda X, n_clusters: AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X),
    'Jerárquico (Single)': lambda X, n_clusters: AgglomerativeClustering(n_clusters=n_clusters, linkage='single').fit_predict(X),
    'DBSCAN': lambda X, n_clusters: DBSCAN(eps=0.3, min_samples=5).fit_predict(X)
}

# Visualizamos los resultados
plt.figure(figsize=(20, 10))
plot_idx = 1

for dataset_name, X in datasets.items():
    for algo_name, algorithm in algorithms.items():
        # Aplicamos el algoritmo
        if algo_name == 'DBSCAN':
            labels = algorithm(X, None)
        else:
            labels = algorithm(X, 2)  # 2 clusters para moon, 3 para blobs
        
        # Calculamos silueta si hay al menos 2 clusters y no hay ruido (-1)
        if len(np.unique(labels)) >= 2 and -1 not in labels:
            silhouette_avg = silhouette_score(X, labels)
            silhouette_text = f"Silueta: {silhouette_avg:.3f}"
        else:
            silhouette_text = "Silueta: N/A"
        
        # Visualizamos
        plt.subplot(2, 4, plot_idx)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.title(f'{dataset_name}\n{algo_name}\n{silhouette_text}')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.grid(True)
        plot_idx += 1

plt.tight_layout()
plt.show()

print("Observaciones:")
print("- K-means funciona bien con clusters convexos pero falla con formas complejas")
print("- Clustering jerárquico con Ward es similar a K-means")
print("- Single linkage puede capturar formas no convexas pero es sensible a ruido")
print("- DBSCAN es excelente para formas arbitrarias y detecta ruido automáticamente")

# 4. Caso de estudio: Análisis de expresiones faciales
# ------------------------------------------------------------------------------
print("\n4. Caso de estudio: Análisis de expresiones faciales")

# Cargamos el dataset Olivetti faces
faces = fetch_olivetti_faces(shuffle=True, random_state=random_state)
X_faces = faces.data
n_samples, n_features = X_faces.shape
n_faces = 20  # Número de caras a mostrar

# Visualizamos algunas imágenes
def plot_gallery(images, titles, h, w, n_row=2, n_col=10):
    plt.figure(figsize=(2. * n_col, 2. * n_row))
    for i in range(n_row * n_col):
        if i < len(images):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())
    plt.tight_layout()
    plt.show()

# Mostramos algunas caras
plot_gallery(X_faces[:n_faces], [f"Cara #{i}" for i in range(n_faces)], h=64, w=64)

# Reducimos dimensionalidad con PCA para visualización
pca = PCA(n_components=50, whiten=True, random_state=random_state)
X_pca = pca.fit_transform(X_faces)

print(f"Dimensionalidad original: {X_faces.shape[1]}")
print(f"Dimensionalidad reducida: {X_pca.shape[1]}")
print(f"Varianza explicada: {sum(pca.explained_variance_ratio_):.2f}")

# Aplicamos clustering jerárquico
n_clusters = 10  # Asumimos 10 personas diferentes
model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = model.fit_predict(X_pca)

# Visualizamos los resultados en espacio PCA
plt.figure(figsize=(12, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=50, alpha=0.8)
plt.colorbar(label='Cluster')
plt.title('Clustering jerárquico de caras en espacio PCA')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.grid(True)
plt.show()

# Visualizamos caras representativas de cada cluster
def plot_faces_by_cluster(X, labels, h, w, n_clusters):
    plt.figure(figsize=(2 * n_clusters, 4))
    for i in range(n_clusters):
        # Encontramos las caras en el cluster i
        cluster_faces = X[labels == i]
        if len(cluster_faces) > 0:
            # Seleccionamos una cara representativa (la más cercana al centroide)
            centroid = np.mean(cluster_faces, axis=0)
            distances = np.linalg.norm(cluster_faces - centroid, axis=1)
            representative_idx = np.argmin(distances)
            representative_face = cluster_faces[representative_idx]
            
            # Visualizamos
            plt.subplot(2, n_clusters // 2, i + 1)
            plt.imshow(representative_face.reshape((h, w)), cmap=plt.cm.gray)
            plt.title(f'Cluster {i}\n{len(cluster_faces)} caras')
            plt.xticks(())
            plt.yticks(())
    plt.tight_layout()
    plt.show()

# Mostramos caras representativas de cada cluster
plot_faces_by_cluster(X_faces, labels, h=64, w=64, n_clusters=n_clusters)

print("Análisis de clusters de caras:")
print(f"Se identificaron {n_clusters} grupos de caras similares")
print("Cada cluster puede representar a una persona o expresiones faciales similares")
print("Este tipo de análisis es útil en sistemas de reconocimiento facial y análisis de emociones")

# 5. Clustering jerárquico para análisis de genes
# ------------------------------------------------------------------------------
print("\n5. Clustering jerárquico para análisis de genes")

# Simulamos datos de expresión génica
n_genes = 100
n_conditions = 20
n_clusters = 4

# Generamos datos con patrones de expresión similares dentro de cada cluster
np.random.seed(random_state)
gene_data = np.zeros((n_genes, n_conditions))

# Creamos patrones para cada cluster
patterns = {
    0: lambda x: np.sin(x) + np.random.normal(0, 0.2, len(x)),  # Patrón sinusoidal
    1: lambda x: np.cos(x) + np.random.normal(0, 0.2, len(x)),  # Patrón cosinusoidal
    2: lambda x: x + np.random.normal(0, 0.2, len(x)),          # Patrón lineal creciente
    3: lambda x: -x + np.random.normal(0, 0.2, len(x))          # Patrón lineal decreciente
}

# Asignamos genes a clusters
gene_clusters = np.random.randint(0, n_clusters, n_genes)

# Generamos datos de expresión
x = np.linspace(0, 2*np.pi, n_conditions)
for i in range(n_genes):
    cluster = gene_clusters[i]
    gene_data[i] = patterns[cluster](x)

# Creamos un DataFrame para mejor manipulación
gene_names = [f"Gen_{i}" for i in range(n_genes)]
condition_names = [f"Cond_{i}" for i in range(n_conditions)]
df_genes = pd.DataFrame(gene_data, index=gene_names, columns=condition_names)

# Visualizamos algunos genes de cada cluster
plt.figure(figsize=(15, 10))
for cluster in range(n_clusters):
    plt.subplot(2, 2, cluster + 1)
    cluster_genes = np.where(gene_clusters == cluster)[0][:5]  # Tomamos 5 genes de cada cluster
    for gene_idx in cluster_genes:
        plt.plot(x, gene_data[gene_idx], alpha=0.7, label=f"Gen_{gene_idx}")
    plt.title(f'Cluster {cluster}: Patrón de expresión')
    plt.xlabel('Condición experimental')
    plt.ylabel('Nivel de expresión')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

# Aplicamos clustering jerárquico
# Calculamos la matriz de distancias (correlación)
from scipy.spatial.distance import pdist, squareform

# Usamos correlación como medida de distancia (1 - correlación)
gene_dist = pdist(gene_data, metric='correlation')
gene_dist_matrix = squareform(gene_dist)

# Calculamos linkage
Z = linkage(gene_dist, method='average')

# Visualizamos el dendrograma
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=gene_names, leaf_rotation=90, leaf_font_size=8)
plt.title('Dendrograma de genes')
plt.xlabel('Genes')
plt.ylabel('Distancia (1 - correlación)')
plt.axhline(y=0.8, color='r', linestyle='--', label='Umbral de corte')
plt.legend()
plt.tight_layout()
plt.show()

# Cortamos el dendrograma para obtener clusters
labels = fcluster(Z, t=0.8, criterion='distance')

# Visualizamos el mapa de calor con los clusters
plt.figure(figsize=(12, 10))
# Reordenamos el DataFrame según los clusters
df_genes['Cluster'] = labels
df_sorted = df_genes.sort_values('Cluster')
df_sorted = df_sorted.drop('Cluster', axis=1)

# Creamos el mapa de calor
sns.clustermap(df_sorted, 
              method='average', 
              metric='correlation',
              cmap='viridis',
              figsize=(12, 10),
              row_cluster=False,  # Ya están ordenados
              col_cluster=True)   # Agrupamos condiciones
plt.show()

print("Aplicaciones del clustering jerárquico en bioinformática:")
print("- Identificación de genes co-expresados")
print("- Descubrimiento de patrones temporales de expresión")
print("- Clasificación de enfermedades basada en perfiles de expresión")
print("- Análisis de interacciones proteína-proteína")

# 6. Clustering jerárquico para análisis de mercado
# ------------------------------------------------------------------------------
print("\n6. Clustering jerárquico para análisis de mercado")

# Simulamos datos de rendimientos de activos financieros
n_assets = 30
n_days = 252  # Aproximadamente un año de trading

# Generamos rendimientos correlacionados por sector
np.random.seed(random_state)

# Definimos sectores
sectors = {
    'Tecnología': range(0, 8),
    'Finanzas': range(8, 15),
    'Salud': range(15, 22),
    'Consumo': range(22, 30)
}

# Generamos matriz de correlación por bloques
corr_matrix = np.eye(n_assets)
for sector, indices in sectors.items():
    for i in indices:
        for j in indices:
            if i != j:
                corr_matrix[i, j] = 0.7 + np.random.normal(0, 0.1)  # Alta correlación intra-sector

# Aseguramos que la matriz sea definida positiva
corr_matrix = np.clip(corr_matrix, -1, 1)

# Forzamos simetría
corr_matrix = (corr_matrix + corr_matrix.T) / 2

# Añadimos un pequeño valor a la diagonal para garantizar que sea definida positiva
corr_matrix = corr_matrix + 1e-8 * np.eye(n_assets)

# Verificamos que sea definida positiva
min_eig = np.min(np.linalg.eigvals(corr_matrix))
if min_eig < 0:
    # Si aún no es definida positiva, ajustamos más
    corr_matrix = corr_matrix - 2 * min_eig * np.eye(n_assets)

# Generamos rendimientos correlacionados
from scipy.linalg import cholesky
L = cholesky(corr_matrix, lower=True)
uncorrelated_returns = np.random.normal(0, 1, size=(n_days, n_assets))
returns = uncorrelated_returns @ L.T

# Añadimos tendencias específicas por sector
for sector, indices in sectors.items():
    if sector == 'Tecnología':
        returns[:, indices] += np.linspace(0, 0.5, n_days).reshape(-1, 1)  # Tendencia alcista
    elif sector == 'Finanzas':
        returns[:, indices] -= np.linspace(0, 0.3, n_days).reshape(-1, 1)  # Tendencia bajista
    elif sector == 'Salud':
        returns[:, indices] += np.sin(np.linspace(0, 4*np.pi, n_days)).reshape(-1, 1) * 0.3  # Cíclico

# Creamos DataFrame
asset_names = [f"{sector}_{i}" for sector, indices in sectors.items() for i in indices]
df_returns = pd.DataFrame(returns, columns=asset_names)

# Visualizamos rendimientos acumulados
plt.figure(figsize=(15, 8))
(1 + df_returns).cumprod().plot(figsize=(15, 8), alpha=0.7)
plt.title('Rendimientos acumulados por activo')
plt.xlabel('Día de trading')
plt.ylabel('Rendimiento acumulado')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# Aplicamos clustering jerárquico basado en correlaciones
# Calculamos matriz de correlación
corr = df_returns.corr()

# Convertimos correlaciones a distancias (1 - correlación)
dist = 1 - corr

# Calculamos linkage
Z = linkage(squareform(dist), method='ward')

# Visualizamos el dendrograma
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=asset_names, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrograma de activos financieros')
plt.xlabel('Activos')
plt.ylabel('Distancia (1 - correlación)')
plt.tight_layout()
plt.show()

# Visualizamos el mapa de calor de correlaciones con clustering
plt.figure(figsize=(12, 10))
sns.clustermap(corr, 
              method='ward', 
              cmap='coolwarm',
              figsize=(12, 10),
              annot=False)
plt.title('Mapa de calor de correlaciones con clustering jerárquico')
plt.show()

print("Aplicaciones del clustering jerárquico en finanzas:")
print("- Construcción de carteras diversificadas")
print("- Identificación de sectores y subsectores")
print("- Análisis de contagio financiero")
print("- Detección de anomalías en mercados")

# 7. Ejercicios propuestos
# ------------------------------------------------------------------------------
print("\n7. Ejercicios propuestos")
print("1. Compare diferentes criterios de linkage (single, complete, average, ward) en presencia de ruido y outliers.")
print("2. Implemente un método automático para determinar el número óptimo de clusters basado en inconsistencia.")
print("3. Desarrolle un caso de estudio de segmentación de clientes comparando K-means y clustering jerárquico.")
print("4. Aplique clustering jerárquico a datos de series temporales utilizando Dynamic Time Warping como medida de distancia.")
print("5. Investigue y aplique técnicas de visualización avanzada para dendrogramas con grandes conjuntos de datos.")
