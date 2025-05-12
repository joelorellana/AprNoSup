# 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE (t-Distributed Stochastic Neighbor Embedding) es una técnica de reducción de dimensionalidad no lineal desarrollada por Laurens van der Maaten y Geoffrey Hinton en 2008. A diferencia de PCA, que se enfoca en preservar la varianza global de los datos, t-SNE está diseñada específicamente para preservar la estructura local, lo que la hace particularmente efectiva para la visualización de conjuntos de datos de alta dimensionalidad.

## Motivación

Las técnicas lineales como PCA pueden fallar al capturar estructuras no lineales complejas presentes en muchos conjuntos de datos del mundo real. Aunque Kernel PCA aborda parcialmente esta limitación, sigue enfocándose principalmente en la estructura global. t-SNE surge como respuesta a la necesidad de:

1. Preservar las relaciones de vecindad local en los datos
2. Revelar clusters y patrones que podrían no ser evidentes con métodos lineales
3. Crear visualizaciones más intuitivas de datos de alta dimensionalidad

## Fundamentos Matemáticos

### Idea Central

t-SNE opera en dos etapas principales:

1. Construye una distribución de probabilidad sobre pares de objetos en el espacio de alta dimensionalidad, de modo que puntos similares tengan alta probabilidad de ser seleccionados como vecinos.
2. Define una distribución similar en el espacio de baja dimensionalidad y minimiza la divergencia entre ambas distribuciones.

### Formulación Matemática

#### Espacio de Alta Dimensionalidad

Para cada par de puntos $x_i$ y $x_j$ en el espacio original, t-SNE define una probabilidad condicional $p_{j|i}$ que representa la similitud de $x_j$ a $x_i$:

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

donde $\sigma_i$ es la desviación estándar de la distribución gaussiana centrada en $x_i$. Este parámetro se ajusta automáticamente para cada punto según la densidad local, mediante un hiperparámetro llamado "perplejidad".

La similitud simétrica entre $x_i$ y $x_j$ se define como:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

donde $n$ es el número de puntos.

#### Espacio de Baja Dimensionalidad

En el espacio de baja dimensionalidad, t-SNE define una similitud $q_{ij}$ entre los puntos mapeados $y_i$ y $y_j$ utilizando una distribución t de Student con un grado de libertad:

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

La distribución t de Student tiene colas más pesadas que la distribución gaussiana, lo que ayuda a aliviar el "problema de aglomeración" en el espacio de baja dimensionalidad.

#### Función de Costo

t-SNE minimiza la divergencia de Kullback-Leibler entre las distribuciones $P$ y $Q$:

$$C = KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

Esta minimización se realiza mediante descenso de gradiente.

## Hiperparámetros Clave

### Perplejidad

La perplejidad es el hiperparámetro más importante en t-SNE y puede interpretarse como una estimación del número de vecinos efectivos. Matemáticamente, la perplejidad de la distribución condicional $P_i$ se define como:

$$\text{Perp}(P_i) = 2^{H(P_i)}$$

donde $H(P_i)$ es la entropía de la distribución $P_i$.

Valores típicos de perplejidad están entre 5 y 50. La elección de este parámetro afecta significativamente la visualización resultante:

- **Perplejidad baja**: Enfatiza estructuras locales, puede fragmentar clusters
- **Perplejidad alta**: Preserva más estructura global, puede mezclar clusters cercanos

### Tasa de Aprendizaje

Controla el tamaño de los pasos en el descenso de gradiente. Un valor demasiado alto puede llevar a una visualización subóptima o inestable, mientras que un valor demasiado bajo puede resultar en mínimos locales o convergencia lenta.

### Número de Iteraciones

Determina cuántos pasos de optimización se realizan. t-SNE generalmente requiere miles de iteraciones para converger a una visualización estable.

### Inicialización

Los puntos en el espacio de baja dimensionalidad pueden inicializarse aleatoriamente o utilizando el resultado de otro método como PCA. La inicialización con PCA suele acelerar la convergencia y puede producir resultados más estables.

## Propiedades y Características

### Ventajas

1. **Preservación de estructura local**: Excelente para mantener las relaciones de vecindad.

2. **Visualización de clusters**: Muy efectivo para revelar agrupaciones naturales en los datos.

3. **Robustez**: Funciona bien con una amplia variedad de tipos de datos y estructuras.

4. **Interpretabilidad visual**: Produce visualizaciones intuitivas donde la distancia refleja similitud.

5. **Adaptabilidad**: El parámetro de perplejidad permite ajustar el equilibrio entre estructura local y global.

### Desventajas

1. **Complejidad computacional**: $O(n^2)$ en la implementación estándar, lo que limita su aplicabilidad a conjuntos de datos grandes.

2. **No determinístico**: Diferentes ejecuciones pueden producir resultados diferentes debido a la inicialización aleatoria.

3. **Dificultad para interpretar distancias globales**: t-SNE preserva principalmente relaciones locales, por lo que las distancias a gran escala pueden no ser significativas.

4. **Sensibilidad a hiperparámetros**: Los resultados pueden variar significativamente según la elección de la perplejidad y otros parámetros.

5. **No invertible**: No existe una transformación directa del espacio de baja dimensionalidad al espacio original.

## Variantes y Extensiones

### Barnes-Hut t-SNE

Una aproximación que reduce la complejidad computacional a $O(n \log n)$, permitiendo aplicar t-SNE a conjuntos de datos más grandes.

### Múltiples mapas t-SNE

Genera múltiples mapas t-SNE y los combina para obtener una representación más robusta.

### Parametric t-SNE

Aprende una función paramétrica (por ejemplo, una red neuronal) para mapear puntos del espacio original al espacio reducido, permitiendo proyectar nuevos datos sin reentrenar.

### UMAP (Uniform Manifold Approximation and Projection)

No es estrictamente una variante de t-SNE, pero aborda objetivos similares con un enfoque matemático diferente y generalmente ofrece mejor escalabilidad y preservación de la estructura global.

## Aplicaciones

t-SNE ha demostrado ser particularmente útil en:

1. **Visualización de datos de alta dimensionalidad**: Especialmente efectivo para conjuntos de datos con estructura de cluster.

2. **Análisis exploratorio de datos**: Ayuda a descubrir patrones y relaciones no evidentes con métodos tradicionales.

3. **Procesamiento de imágenes**: Visualización de espacios de características en visión por computadora.

4. **Bioinformática**: Análisis de datos genómicos, proteómicos y de expresión génica.

5. **Procesamiento de lenguaje natural**: Visualización de embeddings de palabras o documentos.

## Consideraciones Prácticas

### Escalabilidad

Para conjuntos de datos grandes (más de 5,000 puntos), se recomienda:
- Usar implementaciones optimizadas como Barnes-Hut t-SNE
- Considerar submuestrear los datos
- Evaluar alternativas como UMAP que escalan mejor

### Preprocesamiento

- **Reducción de dimensionalidad previa**: Para conjuntos de datos con miles de dimensiones, puede ser beneficioso aplicar primero PCA para reducir a 50-100 dimensiones.
- **Normalización**: Estandarizar las variables puede mejorar los resultados, especialmente cuando tienen escalas muy diferentes.

### Interpretación de Resultados

Al interpretar visualizaciones t-SNE, es importante recordar:

1. **Las distancias absolutas no son significativas**: Solo las distancias relativas y la formación de clusters deben interpretarse.
2. **La forma de los clusters no es informativa**: Solo la separación entre clusters es relevante.
3. **Diferentes ejecuciones pueden dar resultados diferentes**: Es recomendable realizar múltiples ejecuciones.
4. **La perplejidad afecta la visualización**: Es útil probar diferentes valores.

## Comparación con Otras Técnicas

| Aspecto | PCA | Kernel PCA | t-SNE | UMAP |
|---------|-----|------------|-------|------|
| Tipo de relaciones | Lineales | No lineales | No lineales | No lineales |
| Preservación | Varianza global | Estructura según kernel | Estructura local | Estructura local y global |
| Complejidad | O(min(n²p, np²)) | O(n³) | O(n²) o O(n log n) | O(n log n) |
| Escalabilidad | Alta | Baja | Media | Alta |
| Determinismo | Sí | Sí | No | Configurable |
| Nuevos puntos | Proyección directa | Proyección directa | Requiere reentrenamiento | Proyección directa |
| Interpretabilidad | Alta | Media | Baja (solo visual) | Baja (solo visual) |

t-SNE representa un avance significativo en las técnicas de visualización de datos de alta dimensionalidad, ofreciendo una capacidad sin precedentes para revelar estructuras no lineales y clusters. Su enfoque en preservar relaciones locales lo hace complementario a técnicas como PCA, que se centran en la estructura global.

A pesar de sus limitaciones en términos de escalabilidad e interpretabilidad de distancias globales, t-SNE sigue siendo una herramienta esencial para análisis exploratorio y visualización. Para conjuntos de datos muy grandes o aplicaciones donde la preservación de la estructura global es crucial, técnicas como UMAP, que se explorará en la siguiente sección, pueden ofrecer alternativas valiosas.
