# 1. Análisis de Componentes Principales (PCA)

El Análisis de Componentes Principales (PCA, por sus siglas en inglés) es una técnica estadística de reducción de dimensionalidad ampliamente utilizada en ciencia de datos. Esta técnica permite transformar un conjunto de variables posiblemente correlacionadas en un conjunto menor de variables no correlacionadas denominadas componentes principales.

## Fundamentos Matemáticos

### Objetivo del PCA

El objetivo principal del PCA es encontrar una proyección de los datos originales en un espacio de menor dimensión que maximice la varianza. En términos matemáticos, se busca encontrar una transformación lineal que proyecte los datos en un nuevo sistema de coordenadas donde:

1. El primer componente principal captura la mayor varianza posible
2. Cada componente subsiguiente captura la mayor varianza posible bajo la restricción de ser ortogonal a los componentes anteriores

### Formulación Matemática

Dado un conjunto de datos $X$ con $n$ observaciones y $p$ variables, representado como una matriz de $n \times p$:

1. **Centralización de los datos**: Se resta la media de cada variable para obtener datos centrados.
   $$X_{centrado} = X - \bar{X}$$

2. **Cálculo de la matriz de covarianza**: Se calcula la matriz de covarianza $\Sigma$ de los datos centrados.
   $$\Sigma = \frac{1}{n-1} X_{centrado}^T X_{centrado}$$

3. **Descomposición en valores propios**: Se calculan los valores propios $\lambda_i$ y vectores propios $v_i$ de la matriz de covarianza.
   $$\Sigma v_i = \lambda_i v_i$$

4. **Ordenamiento de los componentes**: Los vectores propios se ordenan según sus valores propios correspondientes en orden descendente.

5. **Proyección de los datos**: Los datos originales se proyectan en el espacio de los componentes principales.
   $$Y = X_{centrado} \cdot V$$
   donde $V$ es la matriz cuyas columnas son los vectores propios ordenados.

## Propiedades Importantes

### Varianza Explicada

La proporción de varianza explicada por el $i$-ésimo componente principal es:

$$\frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

Esta medida es crucial para determinar cuántos componentes principales retener en el análisis.

### Ortogonalidad

Los componentes principales son ortogonales entre sí, lo que significa que no existe correlación lineal entre ellos. Esta propiedad facilita la interpretación y el análisis posterior.

### Reducción de Dimensionalidad

Al seleccionar solo los primeros $k$ componentes principales (donde $k < p$), se puede reducir la dimensionalidad de los datos mientras se preserva la mayor cantidad posible de información (varianza).

## Aplicaciones

El PCA tiene numerosas aplicaciones en diversos campos:

- **Visualización de datos**: Reducción a 2 o 3 dimensiones para visualización
- **Preprocesamiento para modelos de aprendizaje automático**: Reducción de dimensionalidad para evitar la maldición de la dimensionalidad
- **Compresión de datos**: Representación eficiente de datos de alta dimensionalidad
- **Eliminación de ruido**: Filtrado de componentes de baja varianza que pueden representar ruido
- **Análisis exploratorio**: Identificación de patrones y estructuras en los datos

## Limitaciones

A pesar de su utilidad, el PCA presenta algunas limitaciones:

1. **Asume relaciones lineales**: PCA solo captura relaciones lineales entre variables
2. **Sensibilidad a la escala**: Las variables con mayor escala tienden a dominar los primeros componentes principales
3. **Interpretabilidad**: Los componentes principales pueden ser difíciles de interpretar en términos de las variables originales
4. **No preserva distancias locales**: Se enfoca en preservar la varianza global, no las relaciones locales entre puntos

Para abordar estas limitaciones, se han desarrollado diversas variantes del PCA, como Sparse PCA, Incremental PCA y Kernel PCA, que se explorarán en las siguientes secciones.
