# 4. Métodos básicos de clustering: K-means y clustering jerárquico

## Algoritmo K-means

### Fundamentos y funcionamiento

K-means constituye uno de los algoritmos de clustering más utilizados en la práctica debido a su simplicidad conceptual, eficiencia computacional y efectividad en numerosos escenarios. Este método particiona el espacio de datos en K grupos, donde cada observación pertenece al cluster cuyo centroide presenta la menor distancia euclidiana.

El procedimiento básico del algoritmo K-means comprende los siguientes pasos:

1. **Inicialización**: Selección de K puntos como centroides iniciales.
2. **Asignación**: Cada observación se asigna al cluster cuyo centroide resulta más cercano.
3. **Actualización**: Recálculo de los centroides como el promedio de todas las observaciones asignadas a cada cluster.
4. **Iteración**: Repetición de los pasos 2 y 3 hasta alcanzar convergencia (los centroides no cambian significativamente) o un número máximo de iteraciones.

El objetivo de K-means consiste en minimizar la suma de distancias cuadráticas entre cada punto y el centroide de su cluster, formalmente expresada como:

$$J = \sum_{j=1}^{k} \sum_{i=1}^{n} ||x_i^{(j)} - c_j||^2$$

Donde:
- $x_i^{(j)}$ representa la i-ésima observación perteneciente al cluster j
- $c_j$ denota el centroide del cluster j
- $||x_i^{(j)} - c_j||^2$ corresponde al cuadrado de la distancia euclidiana

### Variantes y optimizaciones

#### K-means++

La inicialización aleatoria de centroides en K-means tradicional puede conducir a resultados subóptimos. K-means++ propone una estrategia de inicialización que selecciona centroides iniciales distantes entre sí, mejorando significativamente la convergencia y calidad de los resultados. El procedimiento es:

1. Seleccionar aleatoriamente el primer centroide entre los puntos de datos.
2. Para cada punto, calcular la distancia al centroide más cercano ya elegido.
3. Seleccionar el siguiente centroide con probabilidad proporcional al cuadrado de esta distancia.
4. Repetir los pasos 2-3 hasta seleccionar K centroides.

#### Mini-batch K-means

Para conjuntos de datos masivos, Mini-batch K-means utiliza subconjuntos aleatorios (mini-batches) de los datos en cada iteración, reduciendo drásticamente los requisitos computacionales mientras mantiene resultados comparables al K-means estándar.

### Consideraciones prácticas

#### Selección del número de clusters (K)

La determinación del número óptimo de clusters representa uno de los principales desafíos al aplicar K-means. Entre las técnicas más utilizadas se encuentran:

- **Método del codo**: Graficar la suma de errores cuadráticos (inercia) para diferentes valores de K y buscar el "codo" donde la reducción de inercia comienza a estabilizarse.
- **Análisis de silueta**: Calcular el coeficiente de silueta promedio para diferentes valores de K y seleccionar aquel que maximice este valor.
- **Gap statistic**: Comparar la inercia del clustering con la esperada bajo una distribución nula de referencia.

#### Limitaciones

K-means presenta ciertas limitaciones que deben considerarse:

- Asume clusters de forma esférica y tamaño similar.
- Sensible a valores atípicos (outliers).
- Requiere especificar el número de clusters a priori.
- Puede converger a mínimos locales, dependiendo de la inicialización.
- No adecuado para clusters de formas complejas o densidades variables.

## Clustering jerárquico

### Fundamentos y funcionamiento

El clustering jerárquico construye una jerarquía de clusters, representada típicamente mediante un dendrograma. A diferencia de K-means, no requiere especificar el número de clusters a priori, permitiendo explorar la estructura de los datos a diferentes niveles de granularidad.

Existen dos enfoques principales:

#### Clustering jerárquico aglomerativo (bottom-up)

1. **Inicialización**: Cada observación constituye un cluster individual.
2. **Fusión**: En cada paso, se fusionan los dos clusters más similares según una métrica de distancia y un criterio de linkage.
3. **Iteración**: El proceso continúa hasta que todos los puntos pertenecen a un único cluster.

#### Clustering jerárquico divisivo (top-down)

1. **Inicialización**: Todos los puntos pertenecen a un único cluster.
2. **División**: En cada paso, se divide el cluster más heterogéneo.
3. **Iteración**: El proceso continúa hasta que cada punto constituye un cluster individual.

En la práctica, el enfoque aglomerativo resulta más común debido a su menor complejidad computacional.

### Criterios de linkage

El criterio de linkage determina cómo se calcula la distancia entre clusters durante el proceso de fusión:

- **Single linkage (vecino más cercano)**: Distancia entre los puntos más cercanos de cada cluster.
  $$d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$$

- **Complete linkage (vecino más lejano)**: Distancia entre los puntos más lejanos de cada cluster.
  $$d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$$

- **Average linkage**: Promedio de distancias entre todos los pares de puntos.
  $$d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$$

- **Ward**: Minimiza el incremento en la suma de errores cuadráticos tras la fusión.
  $$d(C_i, C_j) = \sqrt{\frac{|C_i||C_j|}{|C_i|+|C_j|}} ||c_i - c_j||$$

Cada criterio genera diferentes estructuras de clustering:

- Single linkage tiende a formar clusters alargados (encadenamiento).
- Complete linkage favorece clusters compactos de tamaño similar.
- Average linkage ofrece un compromiso entre single y complete.
- Ward tiende a formar clusters esféricos y de tamaño similar.

### Consideraciones prácticas

#### Determinación del número de clusters

El dendrograma proporciona una representación visual de la jerarquía de clusters, facilitando la identificación del número apropiado de grupos. Se puede determinar:

- Cortando el dendrograma a una altura específica.
- Buscando saltos significativos en las distancias de fusión.
- Utilizando métricas internas como el coeficiente de silueta para diferentes niveles de corte.

#### Ventajas y limitaciones

**Ventajas:**
- No requiere especificar el número de clusters a priori.
- Proporciona una representación jerárquica que puede resultar informativa.
- Flexible en términos de métricas de distancia y criterios de linkage.
- No asume formas específicas de clusters.

**Limitaciones:**
- Mayor complejidad computacional que K-means, especialmente para grandes conjuntos de datos.
- Sensible a ruido y valores atípicos, particularmente con single linkage.
- Una vez fusionados dos clusters, no pueden separarse en niveles superiores (no permite rectificaciones).

## Comparación entre K-means y clustering jerárquico

| Característica | K-means | Clustering jerárquico |
|----------------|---------|----------------------|
| Complejidad temporal | O(n·k·d·i) | O(n²·d) o O(n³) |
| Escalabilidad | Buena | Limitada |
| Forma de clusters | Esférica | Flexible |
| Número de clusters | Predefinido | Determinado post-análisis |
| Interpretabilidad | Centroides | Dendrograma |
| Sensibilidad a outliers | Alta | Variable según linkage |
| Determinismo | Depende de inicialización | Determinista |

Donde:
- n: número de observaciones
- k: número de clusters
- d: dimensionalidad
- i: número de iteraciones

## Aplicaciones industriales

### K-means

- **Segmentación de clientes en retail**: Agrupamiento de consumidores según patrones de compra para personalizar ofertas y estrategias de marketing.
- **Compresión de imágenes**: Reducción del espacio de colores mediante la selección de K colores representativos.
- **Agrupamiento de documentos**: Organización temática de grandes colecciones de textos.
- **Detección de anomalías**: Identificación de puntos distantes de todos los centroides como potenciales anomalías.

### Clustering jerárquico

- **Bioinformática**: Agrupamiento de genes con patrones de expresión similares o proteínas con funciones relacionadas.
- **Análisis de redes sociales**: Identificación de comunidades y subcomunidades en redes complejas.
- **Taxonomía de productos**: Creación de jerarquías de productos basadas en características o patrones de uso.
- **Estratificación de pacientes**: Identificación de subtipos de enfermedades mediante perfiles clínicos o genéticos.

## Implementación en Python

Scikit-learn proporciona implementaciones eficientes de ambos algoritmos:

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

# Datos de ejemplo
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# K-means
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels_kmeans = kmeans.labels_
centroids = kmeans.cluster_centers_

# Clustering jerárquico
hierarchical = AgglomerativeClustering(n_clusters=2).fit(X)
labels_hierarchical = hierarchical.labels_
```

Para visualización de dendrogramas, se utiliza SciPy:

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Cálculo del linkage
Z = linkage(X, method='ward')

# Visualización del dendrograma
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Dendrograma de clustering jerárquico')
plt.xlabel('Índice de muestra')
plt.ylabel('Distancia')
plt.show()
```

## Ejercicios prácticos

1. **Comparación de métodos**: Implemente K-means y clustering jerárquico con diferentes parámetros sobre un conjunto de datos de su elección. Compare los resultados utilizando métricas internas y visualizaciones.

2. **Análisis de sensibilidad**: Estudie el impacto de valores atípicos en ambos algoritmos. Introduzca outliers artificiales y observe cómo afectan a los resultados.

3. **Caso de estudio**: Aplique ambos métodos a un problema de segmentación de clientes utilizando datos de transacciones reales. Interprete los resultados desde una perspectiva de negocio.

4. **Optimización**: Compare el rendimiento computacional de K-means estándar, K-means++ y Mini-batch K-means para conjuntos de datos de diferentes tamaños.

## Referencias

Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms*, 1027-1035. https://doi.org/10.1145/1283383.1283494

Jain, A. K., & Dubes, R. C. (1988). *Algorithms for clustering data*. Prentice-Hall, Inc.

Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical clustering: an overview. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2*(1), 86-97. https://doi.org/10.1002/widm.53

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. https://scikit-learn.org/stable/modules/clustering.html

Xu, R., & Wunsch, D. (2005). Survey of clustering algorithms. *IEEE Transactions on neural networks, 16*(3), 645-678. https://doi.org/10.1109/TNN.2005.845141
