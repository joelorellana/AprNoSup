# 3. Conceptos básicos y aplicaciones del clustering

## Introducción al clustering

El clustering o agrupamiento constituye una técnica fundamental del aprendizaje no supervisado que permite identificar estructuras intrínsecas en datos no etiquetados. A diferencia de los métodos supervisados, el clustering no requiere datos de entrenamiento con etiquetas predefinidas, lo que lo convierte en una herramienta valiosa para la exploración de datos y la generación de hipótesis en entornos donde las etiquetas no están disponibles o son costosas de obtener.

La premisa fundamental del clustering radica en agrupar objetos de manera que los elementos dentro de un mismo grupo presenten mayor similitud entre sí que con elementos de otros grupos. Esta similitud se cuantifica mediante diversas métricas de distancia o similitud, cuya selección adecuada resulta crítica para el éxito del análisis.

## Fundamentos conceptuales

### Métricas de distancia y similitud

La elección de la métrica de distancia determina cómo se mide la similitud entre observaciones y, consecuentemente, cómo se forman los clusters. Entre las métricas más utilizadas se encuentran:

- **Distancia euclidiana**: Mide la línea recta entre dos puntos en el espacio euclidiano. Es la métrica más intuitiva y ampliamente utilizada, especialmente cuando las variables tienen la misma escala.

- **Distancia de Manhattan**: Suma de las diferencias absolutas entre las coordenadas de dos puntos. Resulta útil en espacios discretos o cuando las diferencias en cada dimensión deben considerarse independientemente.

- **Similitud coseno**: Mide el ángulo entre dos vectores, ignorando su magnitud. Es particularmente útil en análisis de texto y sistemas de recomendación.

- **Distancia de Jaccard**: Apropiada para datos binarios o categóricos, mide la disimilitud entre conjuntos.

### Desafíos fundamentales

El clustering enfrenta diversos desafíos que deben considerarse durante su aplicación:

- **Determinación del número óptimo de clusters**: No existe un método universal para determinar el número "correcto" de clusters. Se emplean heurísticas como el método del codo, análisis de silueta o índice Davies-Bouldin.

- **Maldición de la dimensionalidad**: En espacios de alta dimensionalidad, las distancias entre puntos tienden a homogeneizarse, dificultando la identificación de clusters significativos.

- **Escalabilidad**: Muchos algoritmos de clustering presentan complejidades computacionales que los hacen inviables para conjuntos de datos masivos sin optimizaciones específicas.

- **Interpretabilidad**: Los clusters identificados deben resultar interpretables en el contexto del dominio específico para aportar valor en aplicaciones reales.

## Taxonomía de algoritmos de clustering

Los algoritmos de clustering pueden clasificarse según diversos criterios:

### Métodos basados en centroides

- **K-means**: Particiona el espacio en K grupos minimizando la suma de distancias cuadráticas entre cada punto y el centroide de su cluster.
- **K-medoids**: Similar a K-means, pero utiliza objetos reales como representantes de clusters, resultando más robusto ante valores atípicos.

### Métodos jerárquicos

- **Aglomerativos (bottom-up)**: Comienzan con cada punto como un cluster individual y fusionan progresivamente los más similares.
- **Divisivos (top-down)**: Parten de un único cluster que contiene todos los puntos y lo dividen recursivamente.

### Métodos basados en densidad

- **DBSCAN**: Agrupa puntos basándose en su densidad, identificando automáticamente ruido y clusters de forma arbitraria.
- **OPTICS**: Extensión de DBSCAN que maneja clusters de densidad variable.

### Métodos basados en modelos

- **Gaussian Mixture Models (GMM)**: Asume que los datos provienen de una mezcla de distribuciones gaussianas.
- **Latent Dirichlet Allocation (LDA)**: Para clustering de documentos basado en tópicos.

## Aplicaciones industriales

El clustering encuentra aplicaciones en numerosos sectores industriales:

### Manufactura inteligente

- **Detección de anomalías en procesos industriales**: Agrupamiento de patrones de sensores para identificar desviaciones del comportamiento normal, implementando sistemas de alerta temprana.
- **Mantenimiento predictivo**: Clustering de patrones de vibración, temperatura y otras señales para predecir fallos de equipos y optimizar programas de mantenimiento.

### Retail y marketing

- **Segmentación de clientes**: Identificación de grupos de consumidores con comportamientos similares para personalizar estrategias de marketing y mejorar la experiencia del cliente.
- **Optimización de surtido y layout**: Clustering de productos basado en patrones de compra conjunta para optimizar la disposición en tienda y las estrategias de inventario.

### Finanzas

- **Detección de fraude**: Clustering de transacciones para identificar patrones anómalos que puedan indicar actividades fraudulentas.
- **Gestión de carteras**: Agrupamiento de activos financieros basado en correlaciones y comportamiento histórico para optimizar la diversificación.

### Salud

- **Medicina de precisión**: Clustering de pacientes basado en perfiles genéticos, biomarcadores y respuestas a tratamientos para identificar subtipos de enfermedades y personalizar tratamientos.
- **Descubrimiento de fármacos**: Agrupamiento de compuestos químicos basado en similitud estructural y actividad biológica para acelerar el proceso de descubrimiento.

## Evaluación de clustering

La evaluación de la calidad de un clustering representa un desafío particular debido a la naturaleza no supervisada del problema. Se utilizan principalmente dos tipos de métricas:

### Métricas internas

No requieren información externa (etiquetas verdaderas) y evalúan la calidad del clustering basándose únicamente en los datos:

- **Coeficiente de silueta**: Mide qué tan similar es un objeto a su propio cluster en comparación con otros clusters.
- **Índice Davies-Bouldin**: Evalúa la separación entre clusters en relación con su dispersión interna.
- **Índice Calinski-Harabasz**: Mide la ratio entre la dispersión entre clusters y la dispersión dentro de los clusters.

### Métricas externas

Requieren conocer las etiquetas verdaderas y comparan la asignación de clusters con estas etiquetas:

- **Adjusted Rand Index (ARI)**: Mide la similitud entre dos asignaciones de clusters, ajustada por el azar.
- **Normalized Mutual Information (NMI)**: Cuantifica la información compartida entre dos agrupaciones.

## Implementación práctica

La implementación efectiva del clustering en entornos industriales requiere considerar aspectos como:

- **Preprocesamiento de datos**: Normalización, manejo de valores atípicos y faltantes, reducción de dimensionalidad.
- **Selección de algoritmos**: Considerando la naturaleza de los datos, requisitos computacionales y objetivos específicos.
- **Interpretación de resultados**: Extracción de insights accionables a partir de los clusters identificados.

## Ejercicios prácticos

1. **Análisis exploratorio**: Seleccione un conjunto de datos de su dominio profesional e implemente diferentes algoritmos de clustering. Compare los resultados utilizando métricas internas y visualizaciones.

2. **Caso de negocio**: Desarrolle una estrategia de segmentación de clientes para una empresa de retail utilizando técnicas de clustering. Justifique la selección de variables, algoritmo y número de clusters.

3. **Optimización**: Implemente K-means para un conjunto de datos grande y compare el rendimiento con versiones optimizadas como Mini-batch K-means. Analice el equilibrio entre tiempo de ejecución y calidad de los resultados.

## Referencias

Aggarwal, C. C., & Reddy, C. K. (Eds.). (2014). *Data clustering: algorithms and applications*. CRC Press.

Berkhin, P. (2006). A survey of clustering data mining techniques. In *Grouping multidimensional data* (pp. 25-71). Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-28349-8_2

Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters, 31*(8), 651-666. https://doi.org/10.1016/j.patrec.2009.09.011

Scikit-learn documentation: Clustering. (2023). https://scikit-learn.org/stable/modules/clustering.html

Xu, D., & Tian, Y. (2015). A comprehensive survey of clustering algorithms. *Annals of Data Science, 2*(2), 165-193. https://doi.org/10.1007/s40745-015-0040-1
