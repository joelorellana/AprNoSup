# Clustering: Aplicaciones Prácticas
# Autor: UDB - Aprendizaje No Supervisado
# Fecha: Mayo 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("# Clustering: Aplicaciones Prácticas")
print("=" * 80)

# 1. Segmentación de clientes (RFM)
# ------------------------------------------------------------------------------
print("\n1. Segmentación de clientes (RFM)")

# Generamos datos sintéticos de clientes con RFM
np.random.seed(42)
n_customers = 500

# Recency (días desde última compra): valores menores son mejores
recency = np.concatenate([
    np.random.normal(10, 5, size=int(n_customers*0.3)),  # Clientes recientes
    np.random.normal(30, 10, size=int(n_customers*0.5)),  # Clientes medios
    np.random.normal(60, 15, size=int(n_customers*0.2))   # Clientes antiguos
])

# Frequency (número de compras): valores mayores son mejores
frequency = np.concatenate([
    np.random.normal(30, 10, size=int(n_customers*0.2)),  # Clientes frecuentes
    np.random.normal(15, 5, size=int(n_customers*0.5)),   # Clientes medios
    np.random.normal(5, 2, size=int(n_customers*0.3))     # Clientes poco frecuentes
])

# Monetary (valor total de compras): valores mayores son mejores
monetary = np.concatenate([
    np.random.normal(1000, 300, size=int(n_customers*0.2)),  # Alto valor
    np.random.normal(500, 100, size=int(n_customers*0.5)),   # Valor medio
    np.random.normal(100, 50, size=int(n_customers*0.3))     # Bajo valor
])

# Creamos el DataFrame
customer_data = pd.DataFrame({
    'CustomerID': range(1, n_customers+1),
    'Recency': np.clip(recency, 1, 100).astype(int),
    'Frequency': np.clip(frequency, 1, 50).astype(int),
    'Monetary': np.clip(monetary, 10, 2000).astype(int)
})

print(customer_data.head())
print("\nEstadísticas descriptivas:")
print(customer_data.describe())

# Escalamos los datos para el clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(customer_data[['Recency', 'Frequency', 'Monetary']])

# Invertimos Recency (valores menores son mejores)
rfm_scaled[:, 0] = -rfm_scaled[:, 0]

# Aplicamos K-means para segmentar clientes
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Segment'] = kmeans.fit_predict(rfm_scaled)

# Analizamos los centroides
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids[:, 0] = -centroids[:, 0]  # Revertimos la inversión de Recency

# Creamos un DataFrame con los centroides
centroids_df = pd.DataFrame(centroids, columns=['Recency', 'Frequency', 'Monetary'])
centroids_df.index = ['Segmento ' + str(i) for i in range(4)]

print("\nCentroides de los segmentos:")
print(centroids_df)

# Visualizamos los segmentos
fig = plt.figure(figsize=(12, 10))

# Gráfico 3D
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(rfm_scaled[:, 0], rfm_scaled[:, 1], rfm_scaled[:, 2], 
                    c=customer_data['Segment'], cmap='viridis', s=40, alpha=0.6)
ax.set_xlabel('Recency (invertido)')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('Segmentación de clientes RFM')
plt.colorbar(scatter, label='Segmento')
plt.show()

# Interpretamos los segmentos
segment_names = {
    0: "Clientes de alto valor",
    1: "Clientes leales",
    2: "Clientes potenciales",
    3: "Clientes en riesgo"
}

for segment, name in segment_names.items():
    segment_data = customer_data[customer_data['Segment'] == segment]
    print(f"\nSegmento {segment} - {name}:")
    print(f"  Número de clientes: {len(segment_data)}")
    print(f"  Recency promedio: {segment_data['Recency'].mean():.1f} días")
    print(f"  Frequency promedio: {segment_data['Frequency'].mean():.1f} compras")
    print(f"  Monetary promedio: ${segment_data['Monetary'].mean():.2f}")

# 2. Clustering de texto
# ------------------------------------------------------------------------------
print("\n2. Clustering de texto")

# Creamos un pequeño corpus de texto
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Neural networks are inspired by the human brain",
    "Python is a popular programming language for data science",
    "Data science combines statistics and computer science",
    "Statistics helps us understand patterns in data",
    "Natural language processing helps computers understand human language",
    "Computer vision allows machines to interpret visual information",
    "Reinforcement learning is learning through trial and error",
    "Supervised learning uses labeled data for training"
]

# Vectorizamos los textos usando TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(documents)

print(f"Dimensiones de la matriz TF-IDF: {X_tfidf.shape}")
print(f"Número de características (palabras): {len(vectorizer.get_feature_names_out())}")

# Aplicamos K-means para agrupar documentos similares
kmeans = KMeans(n_clusters=3, random_state=42)
document_clusters = kmeans.fit_predict(X_tfidf)

# Mostramos los clusters
for cluster in range(3):
    print(f"\nCluster {cluster}:")
    for i, doc in enumerate(documents):
        if document_clusters[i] == cluster:
            print(f"  - {doc}")

# Extraemos las palabras más importantes por cluster
feature_names = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("\nPalabras clave por cluster:")
for cluster in range(3):
    print(f"Cluster {cluster}:", end=" ")
    top_words = [feature_names[ind] for ind in order_centroids[cluster, :5]]
    print(", ".join(top_words))

# Visualizamos los documentos en un espacio 2D
# Primero reducimos la dimensionalidad con PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=document_clusters, cmap='viridis', s=100)
for i, doc in enumerate(documents):
    plt.annotate(f"Doc {i+1}", (X_pca[i, 0], X_pca[i, 1]), 
                xytext=(5, 2), textcoords='offset points')
plt.title('Clustering de documentos')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()

# 3. Reducción de dimensionalidad + clustering
# ------------------------------------------------------------------------------
print("\n3. Reducción de dimensionalidad + clustering")

# Cargamos un dataset de dígitos (versión reducida de MNIST)
from sklearn.datasets import load_digits
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"Dataset de dígitos: {X_digits.shape[0]} muestras, {X_digits.shape[1]} características")

# Visualizamos algunos dígitos
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary')
    ax.set_title(f"Dígito: {y_digits[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Aplicamos PCA para reducir dimensionalidad
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_digits)

# Aplicamos t-SNE para visualización no lineal
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_digits)

# Aplicamos K-means en el espacio original
kmeans_original = KMeans(n_clusters=10, random_state=42)
clusters_original = kmeans_original.fit_predict(X_digits)

# Aplicamos K-means en el espacio PCA
kmeans_pca = KMeans(n_clusters=10, random_state=42)
clusters_pca = kmeans_pca.fit_predict(X_pca)

# Visualizamos los resultados
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Visualización con PCA
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', s=30, alpha=0.7)
axes[0].set_title('PCA: Colores por dígito real')
axes[0].grid(True)

# Visualización con t-SNE
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10', s=30, alpha=0.7)
axes[1].set_title('t-SNE: Colores por dígito real')
axes[1].grid(True)

plt.colorbar(scatter1, ax=axes[0], label='Dígito')
plt.colorbar(scatter2, ax=axes[1], label='Dígito')
plt.tight_layout()
plt.show()

# Comparamos la calidad del clustering
silhouette_original = silhouette_score(X_digits, clusters_original)
silhouette_pca = silhouette_score(X_pca, clusters_pca)

print(f"Silueta en espacio original: {silhouette_original:.3f}")
print(f"Silueta en espacio PCA: {silhouette_pca:.3f}")

# 4. Evaluación avanzada de clusters
# ------------------------------------------------------------------------------
print("\n4. Evaluación avanzada de clusters")

# Generamos datos sintéticos
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)

# Aplicamos diferentes algoritmos
algorithms = {
    'K-means': KMeans(n_clusters=4, random_state=42),
    'Jerárquico': AgglomerativeClustering(n_clusters=4),
    'DBSCAN': DBSCAN(eps=0.6, min_samples=5)
}

# Evaluamos cada algoritmo
results = {}
for name, algorithm in algorithms.items():
    # Aplicamos el algoritmo
    labels = algorithm.fit_predict(X)
    
    # Para DBSCAN, puede haber puntos de ruido (-1)
    if name == 'DBSCAN':
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"{name}: {n_clusters} clusters, {n_noise} puntos de ruido")
    else:
        print(f"{name}: {len(set(labels))} clusters")
    
    # Calculamos métricas
    if len(set(labels)) > 1:  # Necesitamos al menos 2 clusters para calcular silueta
        silhouette = silhouette_score(X, labels)
        print(f"  Coeficiente de silueta: {silhouette:.3f}")
    
    # Guardamos los resultados
    results[name] = labels

# Visualizamos los resultados
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (name, labels) in enumerate(results.items()):
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
    axes[i].set_title(name)
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# 5. Interpretación para toma de decisiones
# ------------------------------------------------------------------------------
print("\n5. Interpretación para toma de decisiones")

# Volvemos a los datos de clientes RFM
print("\nEstrategias de marketing basadas en segmentos de clientes:")

strategies = {
    "Clientes de alto valor": [
        "Programa de fidelización premium",
        "Ofertas exclusivas y anticipadas",
        "Atención personalizada"
    ],
    "Clientes leales": [
        "Programas de recompensas",
        "Cross-selling de productos complementarios",
        "Comunicación regular"
    ],
    "Clientes potenciales": [
        "Incentivos para aumentar frecuencia",
        "Promociones especiales",
        "Recordatorios personalizados"
    ],
    "Clientes en riesgo": [
        "Campaña de reactivación",
        "Descuentos especiales por retorno",
        "Encuestas de satisfacción"
    ]
}

for segment, strategy in strategies.items():
    print(f"\n{segment}:")
    for action in strategy:
        print(f"  - {action}")

# 6. Pipeline completo de clustering
# ------------------------------------------------------------------------------
print("\n6. Pipeline completo de clustering")

print("Pasos para un pipeline completo de clustering:")
print("1. Preparación de datos")
print("   - Limpieza de datos")
print("   - Manejo de valores faltantes")
print("   - Codificación de variables categóricas")
print("   - Escalado de características")

print("\n2. Exploración y reducción de dimensionalidad")
print("   - Análisis de componentes principales (PCA)")
print("   - t-SNE o UMAP para visualización")
print("   - Selección de características relevantes")

print("\n3. Selección del algoritmo y parámetros")
print("   - K-means para clusters convexos y similares en tamaño")
print("   - Jerárquico para estructuras anidadas")
print("   - DBSCAN para formas arbitrarias y detección de ruido")

print("\n4. Evaluación e interpretación")
print("   - Métricas internas (silueta, inercia)")
print("   - Validación con conocimiento del dominio")
print("   - Visualización de resultados")

print("\n5. Aplicación de resultados")
print("   - Segmentación de clientes")
print("   - Detección de anomalías")
print("   - Compresión de datos")
print("   - Recomendaciones personalizadas")

# Conclusiones y ejercicios propuestos
# ------------------------------------------------------------------------------
print("\n7. Conclusiones y ejercicios propuestos")

print("Conclusiones:")
print("- El clustering es una herramienta versátil con múltiples aplicaciones prácticas")
print("- La elección del algoritmo depende de la naturaleza de los datos y el objetivo")
print("- La interpretación de los clusters es crucial para la toma de decisiones")
print("- La combinación con técnicas de reducción de dimensionalidad mejora los resultados")

print("\nEjercicios propuestos:")
print("1. Aplique segmentación RFM a un dataset real de transacciones")
print("2. Implemente clustering de texto para agrupar noticias o tweets")
print("3. Compare diferentes técnicas de reducción de dimensionalidad antes del clustering")
print("4. Desarrolle un sistema de recomendación basado en clusters")
print("5. Implemente un pipeline completo para un caso de uso específico")
