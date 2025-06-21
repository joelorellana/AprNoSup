# 4. UMAP (Uniform Manifold Approximation and Projection)

UMAP (Uniform Manifold Approximation and Projection) es una técnica de reducción de dimensionalidad no lineal desarrollada por Leland McInnes, John Healy y James Melville en 2018. UMAP ha ganado rápidamente popularidad como alternativa a t-SNE, ofreciendo un mejor equilibrio entre la preservación de la estructura local y global de los datos, junto con un rendimiento computacional superior.

## Motivación

UMAP surge como respuesta a las limitaciones de técnicas anteriores:

1. **PCA**: Aunque computacionalmente eficiente, solo captura relaciones lineales.
2. **Kernel PCA**: Captura no linealidades, pero escala pobremente con el tamaño del conjunto de datos.
3. **t-SNE**: Excelente para preservar estructura local, pero pierde estructura global y es computacionalmente costoso.

UMAP busca combinar lo mejor de estas técnicas: la capacidad de capturar relaciones no lineales, preservar tanto estructura local como global, y hacerlo de manera computacionalmente eficiente.

## Fundamentos Matemáticos

UMAP se basa en elementos de la teoría de la geometría diferencial y la topología algebraica. A diferencia de t-SNE, que se fundamenta principalmente en la teoría de probabilidades, UMAP utiliza conceptos de la teoría de manifolds para construir representaciones de baja dimensión.

### Marco Teórico

El fundamento teórico de UMAP se puede resumir en tres etapas principales:

1. **Construcción de un grafo topológico**: Modelar la estructura de los datos de alta dimensionalidad como un grafo ponderado.
2. **Diseño de un grafo de baja dimensión**: Crear una representación en el espacio de baja dimensión.
3. **Optimización**: Minimizar las diferencias entre ambos grafos.

### Formulación Matemática

#### 1. Construcción del Grafo de Alta Dimensión

Para cada punto $x_i$ en el espacio original, UMAP:

a) Encuentra los $k$ vecinos más cercanos.

b) Calcula una métrica de distancia local $\rho_i$ como la distancia al vecino más cercano.

c) Define una función de similitud exponencial para cada vecino $x_j$:

$$v_{ij} = \exp\left(-\frac{d(x_i, x_j) - \rho_i}{\sigma_i}\right)$$

donde $\sigma_i$ se ajusta para satisfacer una restricción de entropía relacionada con el parámetro de "vecindad local" (similar a la perplejidad en t-SNE).

d) Simetriza las conexiones para obtener el grafo ponderado final:

$$w_{ij} = v_{ij} + v_{ji} - v_{ij} \cdot v_{ji}$$

#### 2. Construcción del Grafo de Baja Dimensión

En el espacio de baja dimensión, UMAP define una función de similitud basada en la distancia:

$$w'_{ij} = (1 + a \cdot d(y_i, y_j)^{2b})^{-1}$$

donde $a$ y $b$ son hiperparámetros que controlan la forma de la función, y $y_i$ son las coordenadas en el espacio de baja dimensión.

#### 3. Optimización

UMAP minimiza la divergencia entre los grafos mediante descenso de gradiente estocástico, utilizando una función de pérdida basada en la entropía cruzada:

$$\mathcal{L} = \sum_{i,j} \left[ w_{ij} \log \left(\frac{w_{ij}}{w'_{ij}}\right) + (1-w_{ij}) \log \left(\frac{1-w_{ij}}{1-w'_{ij}}\right) \right]$$

## Hiperparámetros Clave

### Vecinos Más Cercanos (n_neighbors)

Este parámetro controla cuántos vecinos se consideran al construir el grafo topológico. Es análogo a la perplejidad en t-SNE:

- **Valores bajos** (2-15): Enfatizan la estructura local, pueden fragmentar clusters.
- **Valores altos** (30-100): Capturan más estructura global, pueden mezclar clusters cercanos.

### Distancia Mínima (min_dist)

Controla la distancia mínima permitida entre puntos en el espacio de baja dimensión:

- **Valores bajos** (0.0-0.2): Permiten agrupaciones más compactas, útil para visualizar clusters.
- **Valores altos** (0.5-0.99): Distribuyen los puntos más uniformemente, útil para ver relaciones globales.

### Métrica de Distancia (metric)

UMAP permite utilizar diferentes métricas de distancia según el tipo de datos:

- **Euclidiana**: Estándar para datos continuos.
- **Manhattan**: Útil cuando las diferencias en dimensiones individuales son importantes.
- **Coseno**: Apropiada para datos donde la dirección es más importante que la magnitud (ej. texto).
- **Hamming**: Para datos binarios o categóricos.

## Propiedades y Características

### Ventajas

1. **Preservación de estructura global y local**: A diferencia de t-SNE, UMAP mantiene un mejor equilibrio entre relaciones locales y globales.

2. **Escalabilidad**: Con una complejidad de $O(n \log n)$, UMAP es significativamente más rápido que t-SNE para conjuntos de datos grandes.

3. **Modelo generativo**: UMAP puede proyectar nuevos puntos sin reentrenar todo el modelo, lo que no es posible con t-SNE estándar.

4. **Flexibilidad**: Soporta diferentes métricas de distancia y puede aplicarse a diversos tipos de datos.

5. **Preservación teórica de la topología**: El fundamento matemático de UMAP garantiza cierta preservación de la estructura topológica de los datos.

### Desventajas

1. **Sensibilidad a hiperparámetros**: Los resultados pueden variar significativamente según la elección de n_neighbors y min_dist.

2. **Complejidad teórica**: Los fundamentos matemáticos son más complejos que los de PCA o t-SNE, lo que puede dificultar su interpretación.

3. **No determinístico**: Como t-SNE, UMAP puede producir resultados diferentes en distintas ejecuciones debido a la inicialización aleatoria.

4. **Interpretabilidad limitada**: Las distancias en el espacio reducido no tienen una interpretación directa en términos del espacio original.

## Comparación con Otras Técnicas

| Aspecto | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| Tipo de relaciones | Lineales | No lineales | No lineales |
| Preservación | Varianza global | Estructura local | Estructura local y global |
| Complejidad | O(min(n²p, np²)) | O(n²) o O(n log n) | O(n log n) |
| Escalabilidad | Alta | Media | Alta |
| Nuevos puntos | Proyección directa | Requiere reentrenamiento | Proyección directa |
| Sensibilidad a hiperparámetros | Baja | Alta | Media |
| Fundamento teórico | Álgebra lineal | Teoría de probabilidades | Topología algebraica |

### Comparativa Visual

En términos de visualización:

- **PCA**: Tiende a producir proyecciones más dispersas donde los clusters pueden superponerse.
- **t-SNE**: Crea clusters bien definidos y separados, pero puede distorsionar las relaciones globales.
- **UMAP**: Produce clusters bien definidos mientras mantiene mejor las relaciones de distancia global.

## Aplicaciones

UMAP ha demostrado ser particularmente útil en:

1. **Análisis de datos de célula única**: Visualización y análisis de datos de secuenciación de ARN de célula única en bioinformática.

2. **Procesamiento de lenguaje natural**: Visualización de embeddings de palabras y documentos.

3. **Visión por computadora**: Análisis de espacios de características en redes neuronales profundas.

4. **Preprocesamiento para aprendizaje supervisado**: Reducción de dimensionalidad antes de aplicar algoritmos de clasificación o regresión.

5. **Análisis exploratorio de datos**: Descubrimiento de patrones en conjuntos de datos complejos.

## Consideraciones Prácticas

### Selección de Hiperparámetros

La elección de n_neighbors y min_dist debe basarse en el objetivo del análisis:

- Para **exploración de clusters**: n_neighbors bajo (10-15) y min_dist bajo (0.0-0.1).
- Para **estructura global**: n_neighbors alto (30-50) y min_dist alto (0.5-0.8).
- Para **equilibrio**: n_neighbors medio (15-30) y min_dist medio (0.1-0.5).

### Preprocesamiento

- **Normalización**: Como con otras técnicas de reducción de dimensionalidad, la normalización de los datos es importante, especialmente cuando las variables tienen escalas muy diferentes.

- **Reducción previa**: Para conjuntos de datos con miles de dimensiones, puede ser beneficioso aplicar primero PCA para reducir a 50-100 dimensiones antes de aplicar UMAP.

### Interpretación de Resultados

Al interpretar visualizaciones UMAP:

1. **Clusters**: La presencia de clusters suele indicar grupos naturales en los datos.

2. **Distancias relativas**: Las distancias entre clusters tienden a reflejar relaciones en los datos originales mejor que en t-SNE.

3. **Formas**: A diferencia de t-SNE, las formas de los clusters en UMAP pueden ser informativas sobre la estructura de los datos.

4. **Consistencia**: Ejecutar UMAP múltiples veces con diferentes semillas puede ayudar a identificar características estables.

## Extensiones y Variantes

### UMAP Supervisado

UMAP puede incorporar información de etiquetas para crear visualizaciones que enfaticen la separación entre clases, similar a LDA (Análisis Discriminante Lineal).

### UMAP Paramétrico

Implementaciones que utilizan redes neuronales para aprender la transformación UMAP, permitiendo una proyección más eficiente de nuevos datos.

### densMAP

Una extensión que preserva información sobre la densidad local de los datos en el espacio original.

### UMAP Jerárquico

Enfoques que aplican UMAP de manera jerárquica para revelar estructura a diferentes escalas.

## Conclusión

UMAP representa un avance significativo en las técnicas de reducción de dimensionalidad, combinando la capacidad de t-SNE para preservar estructura local con una mejor preservación de la estructura global y una mayor eficiencia computacional. Su fundamento en la teoría de manifolds y la topología algebraica proporciona garantías teóricas sobre la preservación de la estructura de los datos.

Aunque más complejo conceptualmente que PCA o t-SNE, UMAP ofrece un equilibrio atractivo entre rendimiento, calidad de visualización y flexibilidad, lo que explica su rápida adopción en diversos campos. Para científicos de datos que trabajan con conjuntos de datos de alta dimensionalidad, UMAP se ha convertido en una herramienta esencial para la visualización y el análisis exploratorio.

## Conclusiones Generales del Capítulo

A lo largo de este capítulo, hemos explorado un espectro de técnicas de reducción de dimensionalidad, desde enfoques lineales como PCA y sus variantes hasta métodos no lineales avanzados como t-SNE y UMAP. Cada técnica presenta fortalezas y limitaciones particulares, haciendo que sean complementarias entre sí más que competitivas.

El PCA tradicional sigue siendo fundamental por su simplicidad, interpretabilidad y eficiencia computacional, mientras que sus variantes (Sparse, Incremental y Kernel) abordan limitaciones específicas para casos de uso particulares. Por otro lado, t-SNE y UMAP ofrecen capacidades superiores para visualización y descubrimiento de patrones no lineales, con UMAP emergiendo como una solución que equilibra la preservación de estructura local y global con una eficiencia computacional mejorada.

La elección de la técnica más adecuada dependerá del contexto específico, considerando factores como:

1. **Objetivo del análisis**: Visualización, compresión de datos o preprocesamiento para modelos posteriores.
2. **Tamaño y dimensionalidad de los datos**: Conjuntos grandes pueden requerir técnicas escalables como Incremental PCA o UMAP.
3. **Estructura de los datos**: Relaciones lineales o no lineales entre variables.
4. **Necesidad de interpretabilidad**: PCA y Sparse PCA ofrecen mayor interpretabilidad que t-SNE o UMAP.
5. **Requisitos computacionales**: Consideraciones de tiempo y memoria disponible.

En la práctica, es común aplicar múltiples técnicas de forma secuencial o comparativa, aprovechando sus fortalezas complementarias. Por ejemplo, utilizar PCA como paso inicial para reducir la dimensionalidad antes de aplicar t-SNE o UMAP para visualización, o comparar los resultados de diferentes técnicas para obtener una comprensión más completa de la estructura de los datos.

La reducción de dimensionalidad continúa siendo un área activa de investigación, con nuevos métodos y variantes emergiendo regularmente. El dominio de estas técnicas es esencial para cualquier científico de datos que busque extraer conocimiento significativo de conjuntos de datos complejos y de alta dimensionalidad.
