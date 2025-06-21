# 2.1 Sparse PCA (Análisis de Componentes Principales Disperso)

El Análisis de Componentes Principales Disperso (Sparse PCA) es una variante del PCA tradicional que busca obtener componentes principales con vectores de carga dispersos (sparse), es decir, con muchos coeficientes exactamente iguales a cero. Esta técnica aborda una de las principales limitaciones del PCA estándar: la dificultad en la interpretación de los componentes principales.

## Motivación

En el PCA tradicional, cada componente principal es una combinación lineal de todas las variables originales, lo que dificulta la interpretación cuando se trabaja con conjuntos de datos de alta dimensionalidad. Sparse PCA resuelve este problema al forzar que muchos de los coeficientes en los vectores de carga sean exactamente cero, manteniendo solo las variables más relevantes para cada componente.

## Formulación Matemática

Existen varias formulaciones para Sparse PCA, pero una de las más comunes es la siguiente:

### Formulación como un Problema de Optimización

Dado un conjunto de datos centrados $X \in \mathbb{R}^{n \times p}$ (con $n$ observaciones y $p$ variables), Sparse PCA busca resolver:

$$\max_{V} \text{Tr}(V^T X^T X V) - \lambda \sum_{j=1}^{k} \|v_j\|_1$$

sujeto a $V^T V = I$

Donde:
- $V \in \mathbb{R}^{p \times k}$ es la matriz de vectores de carga para los $k$ componentes principales
- $\text{Tr}(\cdot)$ es la función traza
- $\|v_j\|_1$ es la norma L1 del $j$-ésimo vector de carga
- $\lambda$ es un parámetro de regularización que controla el nivel de dispersión

### Algoritmos de Solución

Existen varios algoritmos para resolver el problema de Sparse PCA:

1. **Método de Descomposición de Valores Singulares Truncada (TSVD) con Regularización L1**
2. **Elastic Net**: Combina regularización L1 y L2
3. **Método de Zouhair Harchaoui**: Basado en optimización convexa
4. **Algoritmo de Zou, Hastie y Tibshirani**: Reformula el problema como una regresión penalizada

## Propiedades y Características

### Ventajas

1. **Interpretabilidad mejorada**: Al tener muchos coeficientes iguales a cero, es más fácil interpretar qué variables originales contribuyen a cada componente principal.

2. **Selección de variables implícita**: Sparse PCA realiza automáticamente una selección de variables, identificando las más relevantes para cada componente.

3. **Estabilidad estadística**: En contextos de alta dimensionalidad donde $p > n$ (más variables que observaciones), Sparse PCA tiende a ser más estable estadísticamente que el PCA tradicional.

4. **Robustez frente al ruido**: Al ignorar variables con contribuciones menores, Sparse PCA puede ser más robusto frente al ruido en los datos.

### Desventajas

1. **Pérdida de varianza explicada**: Al imponer la restricción de dispersión, Sparse PCA generalmente explica menos varianza que el PCA tradicional con el mismo número de componentes.

2. **Mayor complejidad computacional**: Los algoritmos para Sparse PCA suelen ser más costosos computacionalmente que el PCA estándar.

3. **Selección del parámetro de regularización**: La elección del parámetro $\lambda$ que controla la dispersión puede ser difícil y generalmente requiere validación cruzada u otros métodos de selección de hiperparámetros.

## Aplicaciones

Sparse PCA es particularmente útil en contextos donde la interpretabilidad es crucial:

1. **Genómica**: Análisis de datos de expresión génica, donde cada componente puede asociarse con un conjunto específico de genes.

2. **Procesamiento de imágenes**: Extracción de características localizadas espacialmente.

3. **Finanzas**: Identificación de factores de riesgo específicos en portafolios de inversión.

4. **Neurociencia**: Análisis de datos de neuroimagen, donde cada componente puede corresponder a regiones cerebrales específicas.

5. **Quimiometría**: Análisis de espectros químicos, donde cada componente puede asociarse con compuestos específicos.

## Comparación con PCA Tradicional

| Aspecto | PCA Tradicional | Sparse PCA |
|---------|-----------------|------------|
| Vectores de carga | Densos (todos los coeficientes ≠ 0) | Dispersos (muchos coeficientes = 0) |
| Interpretabilidad | Baja cuando hay muchas variables | Alta debido a la selección de variables |
| Varianza explicada | Máxima posible | Subóptima debido a la restricción de dispersión |
| Complejidad computacional | Baja (SVD) | Alta (optimización iterativa) |
| Aplicabilidad en $p > n$ | Problemas de estabilidad | Mayor estabilidad |

## Consideraciones Prácticas

### Selección del Nivel de Dispersión

El nivel de dispersión (controlado por el parámetro $\lambda$) debe seleccionarse cuidadosamente:

- Un valor pequeño de $\lambda$ resulta en componentes menos dispersos, más cercanos al PCA tradicional.
- Un valor grande de $\lambda$ produce componentes muy dispersos, pero puede sacrificar demasiada varianza explicada.

La selección óptima suele realizarse mediante validación cruzada, evaluando el equilibrio entre dispersión y varianza explicada.

### Número de Componentes

Al igual que en el PCA tradicional, la selección del número de componentes a retener es crucial. Sin embargo, en Sparse PCA, esta decisión puede ser más compleja debido al compromiso entre dispersión y varianza explicada.

### Preprocesamiento

El preprocesamiento de los datos (centralización, escalado) es tan importante para Sparse PCA como para el PCA tradicional. En particular, la estandarización de las variables puede ser crucial cuando éstas tienen escalas muy diferentes.

Sparse PCA representa una evolución importante del PCA tradicional, ofreciendo una mayor interpretabilidad a costa de una cierta pérdida en la varianza explicada. Esta técnica es especialmente valiosa en contextos de alta dimensionalidad donde la identificación de las variables más relevantes para cada componente principal es crucial para la interpretación de los resultados.
