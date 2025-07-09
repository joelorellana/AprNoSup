# 5. Métodos avanzados de clustering: DBSCAN y Clustering Espectral

## Introducción

Los métodos tradicionales de clustering como K-means y clustering jerárquico presentan limitaciones significativas cuando se enfrentan a datos con formas complejas, densidades variables o ruido. Para superar estas limitaciones, se han desarrollado algoritmos más avanzados como DBSCAN (Density-Based Spatial Clustering of Applications with Noise) y Clustering Espectral, que ofrecen enfoques alternativos para la identificación de patrones en los datos.

## DBSCAN: Clustering basado en densidad

### Fundamentos conceptuales

DBSCAN es un algoritmo de clustering basado en la densidad de puntos, introducido por Ester, Kriegel, Sander y Xu en 1996. A diferencia de K-means, que asume clusters esféricos de tamaño similar, DBSCAN puede identificar clusters de formas arbitrarias y manejar eficazmente el ruido.

El algoritmo se basa en dos parámetros clave:
- **Epsilon (ε)**: Define el radio de vecindad alrededor de cada punto.
- **MinPts**: Número mínimo de puntos requeridos dentro del radio ε para formar una región densa.

DBSCAN clasifica los puntos en tres categorías:
1. **Puntos núcleo**: Puntos que tienen al menos MinPts puntos (incluyendo a sí mismos) dentro de su radio ε.
2. **Puntos frontera**: Puntos que tienen menos de MinPts puntos dentro de su radio ε, pero están en la vecindad de un punto núcleo.
3. **Puntos ruido**: Puntos que no son ni núcleo ni frontera.

El proceso de clustering se desarrolla conectando puntos núcleo que son directamente alcanzables por densidad (dentro del radio ε entre sí), y luego asignando puntos frontera a los clusters correspondientes.

### Ventajas y limitaciones

**Ventajas:**
- No requiere especificar el número de clusters a priori.
- Puede identificar clusters de formas arbitrarias.
- Maneja eficazmente el ruido al identificarlo explícitamente.
- Funciona bien con clusters de tamaños y densidades variables (con ciertas limitaciones).

**Limitaciones:**
- Sensible a los parámetros ε y MinPts, cuya selección óptima puede ser desafiante.
- Dificultad para identificar clusters con densidades muy variables.
- Problemas de escalabilidad en conjuntos de datos de alta dimensionalidad debido a la "maldición de la dimensionalidad".
- Mayor complejidad computacional que K-means, aunque existen implementaciones optimizadas.

### Aplicaciones industriales

- **Análisis geoespacial**: Identificación de zonas urbanas, detección de puntos de interés o análisis de patrones de tráfico.
- **Detección de anomalías**: Identificación de transacciones fraudulentas en sistemas financieros.
- **Segmentación de imágenes**: Reconocimiento de objetos y segmentación en visión por computadora.
- **Bioinformática**: Agrupamiento de proteínas o genes con funciones similares.
- **Análisis de redes sociales**: Detección de comunidades en redes complejas.

## Clustering Espectral: Reducción de dimensionalidad y clustering

### Fundamentos conceptuales

El Clustering Espectral combina técnicas de reducción de dimensionalidad con algoritmos de clustering tradicionales. Este método transforma los datos originales a un espacio de menor dimensión donde los clusters son más fácilmente separables, utilizando los eigenvectores (vectores propios) de una matriz de similitud derivada de los datos.

El proceso general del Clustering Espectral incluye:

1. **Construcción de la matriz de similitud**: Generalmente utilizando el kernel RBF (Radial Basis Function) o la similitud del vecino más cercano.
2. **Cálculo del Laplaciano normalizado**: Derivado de la matriz de similitud.
3. **Descomposición espectral**: Extracción de los eigenvectores correspondientes a los k eigenvalores más pequeños (excluyendo el más pequeño).
4. **Embedding en un espacio de menor dimensión**: Utilizando los eigenvectores seleccionados.
5. **Aplicación de K-means**: En el espacio transformado para obtener los clusters finales.

### Ventajas y limitaciones

**Ventajas:**
- Capacidad para identificar clusters de formas complejas y no convexas.
- No asume una forma específica para los clusters.
- Rendimiento superior en muchos casos donde K-means falla.
- Base matemática sólida en teoría espectral de grafos.

**Limitaciones:**
- Requiere especificar el número de clusters a priori.
- Mayor complejidad computacional, especialmente para conjuntos de datos grandes.
- Sensible a la elección de la función de similitud y sus parámetros.
- Puede ser inestable ante pequeñas perturbaciones en los datos.

### Aplicaciones industriales

- **Procesamiento de imágenes**: Segmentación de imágenes médicas y reconocimiento de patrones visuales.
- **Análisis de textos**: Agrupamiento de documentos por similitud temática.
- **Bioinformática**: Análisis de expresión génica y clasificación de proteínas.
- **Recomendación de productos**: Identificación de grupos de usuarios con preferencias similares.
- **Análisis de series temporales**: Detección de patrones en datos financieros o señales.

## Comparación entre DBSCAN y Clustering Espectral

| Característica | DBSCAN | Clustering Espectral |
|----------------|--------|----------------------|
| Número de clusters | Automático | Predefinido |
| Forma de clusters | Arbitraria | Arbitraria |
| Manejo de ruido | Explícito | Limitado |
| Escalabilidad | Media | Baja |
| Densidades variables | Desafío | Maneja bien |
| Dimensionalidad alta | Problemática | Mejor rendimiento |
| Complejidad | O(n²) o O(n log n) con optimizaciones | O(n³) típicamente |

## Implementación en Python

Scikit-learn proporciona implementaciones eficientes de ambos algoritmos:

```python
from sklearn.cluster import DBSCAN, SpectralClustering
import numpy as np

# Datos de ejemplo
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# DBSCAN
dbscan = DBSCAN(eps=3, min_samples=2)
labels_dbscan = dbscan.fit_predict(X)

# Clustering Espectral
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
labels_spectral = spectral.fit_predict(X)
```

## Estrategias para la selección de parámetros

### DBSCAN: Selección de ε y MinPts

1. **Gráfico de distancias k-vecinos**: Ordenar las distancias al k-ésimo vecino más cercano y buscar el "codo" en la curva.
2. **Validación cruzada**: Evaluar diferentes combinaciones de parámetros utilizando métricas internas como silhouette score.
3. **Heurística de MinPts**: Una regla general es establecer MinPts ≥ dimensión + 1, con valores típicos entre 3 y 5 para datos bidimensionales.

### Clustering Espectral: Selección de parámetros

1. **Análisis del espectro de eigenvalores**: Examinar la brecha entre eigenvalores consecutivos para determinar el número de clusters.
2. **Selección de función de afinidad**: Experimentar con diferentes funciones (RBF, vecinos más cercanos) y sus parámetros.
3. **Validación con métricas internas**: Utilizar métricas como silhouette score o índice Davies-Bouldin para evaluar la calidad de los clusters.

## Conclusiones y consideraciones prácticas

La elección entre DBSCAN, Clustering Espectral u otros algoritmos depende de las características específicas de los datos y los objetivos del análisis. Algunas consideraciones prácticas incluyen:

- Para datos con ruido y clusters de formas arbitrarias, DBSCAN suele ser una buena opción.
- Para problemas donde se conoce aproximadamente el número de clusters y estos pueden tener formas complejas, el Clustering Espectral puede ofrecer mejores resultados.
- En conjuntos de datos de alta dimensionalidad, considerar técnicas de reducción de dimensionalidad antes de aplicar clustering.
- La visualización de los resultados es crucial para validar la calidad del clustering, especialmente en aplicaciones exploratorias.
- Considerar enfoques híbridos o ensemble para mejorar la robustez de los resultados.

## Referencias

Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)*, 226-231. https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf

Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). On spectral clustering: Analysis and an algorithm. *Advances in Neural Information Processing Systems*, 14, 849-856. https://proceedings.neurips.cc/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf

Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and Computing*, 17(4), 395-416. https://doi.org/10.1007/s11222-007-9033-z

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. https://scikit-learn.org/stable/modules/clustering.html

Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *Pacific-Asia Conference on Knowledge Discovery and Data Mining*, 160-172. https://doi.org/10.1007/978-3-642-37456-2_14

Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017). DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems (TODS)*, 42(3), 1-21. https://doi.org/10.1145/3068335
