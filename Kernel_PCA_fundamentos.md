# 2.3 Kernel PCA (Análisis de Componentes Principales con Kernel)

El Análisis de Componentes Principales con Kernel (Kernel PCA) es una extensión no lineal del PCA tradicional. Mientras que el PCA estándar solo puede capturar relaciones lineales entre variables, Kernel PCA permite identificar componentes principales en espacios de características no lineales, lo que lo hace más potente para conjuntos de datos con estructuras complejas.

## Motivación

Muchos conjuntos de datos del mundo real presentan relaciones no lineales que el PCA tradicional no puede capturar adecuadamente. Por ejemplo:

- Datos que forman estructuras curvas o circulares
- Clusters con formas no convexas
- Variedades de baja dimensión inmersas en espacios de alta dimensión

Kernel PCA aborda estas limitaciones utilizando el "truco del kernel" (kernel trick), que permite trabajar implícitamente en espacios de características de alta dimensión sin calcular explícitamente las transformaciones.

## Fundamentos Matemáticos

### El Truco del Kernel

La idea central de Kernel PCA se basa en el "truco del kernel", que consiste en:

1. Mapear implícitamente los datos originales a un espacio de características de mayor dimensión mediante una función de transformación no lineal $\phi$.
2. Realizar PCA estándar en este espacio de características.

La clave es que no necesitamos calcular explícitamente la transformación $\phi(x)$, sino que trabajamos con funciones kernel $k(x, y) = \langle \phi(x), \phi(y) \rangle$, que computan directamente el producto escalar en el espacio de características.

### Formulación Matemática

Dado un conjunto de datos $X = \{x_1, x_2, ..., x_n\}$ con $x_i \in \mathbb{R}^d$:

1. **Definición de la matriz de kernel**: Se construye la matriz de kernel $K \in \mathbb{R}^{n \times n}$ donde $K_{ij} = k(x_i, x_j)$.

2. **Centrado en el espacio de características**: Se centra la matriz de kernel mediante:
   $$\tilde{K} = K - 1_n K - K 1_n + 1_n K 1_n$$
   donde $1_n$ es una matriz $n \times n$ con todos los elementos iguales a $1/n$.

3. **Descomposición en valores propios**: Se resuelve el problema de valores propios:
   $$\tilde{K} \alpha^{(i)} = \lambda_i \alpha^{(i)}$$
   donde $\alpha^{(i)}$ son los vectores propios y $\lambda_i$ son los valores propios.

4. **Normalización**: Los vectores propios se normalizan para que:
   $$\lambda_i (\alpha^{(i)} \cdot \alpha^{(i)}) = 1$$

5. **Proyección de datos**: Para proyectar un punto $x$ en el $i$-ésimo componente principal en el espacio de características:
   $$y_i(x) = \phi(x) \cdot v^{(i)} = \sum_{j=1}^{n} \alpha_j^{(i)} k(x_j, x)$$
   donde $v^{(i)}$ es el $i$-ésimo vector propio en el espacio de características.

## Funciones Kernel Comunes

La elección de la función kernel determina el tipo de no linealidad que puede capturar Kernel PCA:

1. **Kernel Lineal**: $k(x, y) = x^T y$
   - Equivalente al PCA tradicional.

2. **Kernel Polinomial**: $k(x, y) = (x^T y + c)^d$
   - Captura relaciones polinomiales de grado $d$.
   - El parámetro $c \geq 0$ controla la influencia de los términos de orden inferior.

3. **Kernel Gaussiano (RBF)**: $k(x, y) = \exp(-\gamma \|x - y\|^2)$
   - Captura relaciones de cualquier orden.
   - El parámetro $\gamma > 0$ controla la "anchura" del kernel.

4. **Kernel Sigmoide**: $k(x, y) = \tanh(a x^T y + b)$
   - Similar a las funciones de activación en redes neuronales.

5. **Kernel Laplaciano**: $k(x, y) = \exp(-\gamma \|x - y\|_1)$
   - Variante del kernel RBF que utiliza la norma L1 en lugar de la L2.

## Propiedades y Características

### Ventajas

1. **Captura relaciones no lineales**: Puede identificar estructuras complejas que el PCA tradicional no detecta.

2. **Flexibilidad**: La elección del kernel permite adaptarse a diferentes tipos de no linealidades.

3. **Interpretación geométrica**: Realiza PCA en un espacio de características donde las relaciones no lineales se vuelven lineales.

4. **No requiere cálculo explícito en el espacio de características**: Gracias al truco del kernel, todos los cálculos se realizan utilizando la matriz de kernel.

### Desventajas

1. **Complejidad computacional**: La construcción y descomposición de la matriz de kernel tiene una complejidad de $O(n^3)$, lo que limita su aplicabilidad a conjuntos de datos grandes.

2. **Selección del kernel y sus parámetros**: La elección del kernel y la optimización de sus parámetros pueden ser complejas y generalmente requieren validación cruzada.

3. **Interpretabilidad reducida**: A diferencia del PCA tradicional, los componentes en Kernel PCA son más difíciles de interpretar en términos de las variables originales.

4. **Pre-imagen**: La reconstrucción de datos en el espacio original (problema de pre-imagen) no tiene una solución analítica directa y requiere aproximaciones.

## Comparación con PCA Tradicional y Otras Técnicas

| Aspecto | PCA Tradicional | Kernel PCA | t-SNE | UMAP |
|---------|-----------------|------------|-------|------|
| Tipo de relaciones | Lineales | No lineales | No lineales | No lineales |
| Complejidad computacional | O(min(n²p, np²)) | O(n³) | O(n² log n) | O(n log n) |
| Escalabilidad | Alta | Baja | Media | Alta |
| Preservación de estructura | Global | Global y local (según kernel) | Local | Local y global |
| Interpretabilidad | Alta | Media | Baja | Baja |
| Hiperparámetros | Pocos | Varios | Varios | Varios |

## Aplicaciones

Kernel PCA es particularmente útil en los siguientes escenarios:

1. **Reconocimiento de patrones**: Detección de estructuras no lineales en datos complejos.

2. **Procesamiento de imágenes**: Extracción de características no lineales en imágenes.

3. **Bioinformática**: Análisis de datos genómicos y proteómicos con relaciones complejas.

4. **Preprocesamiento para clasificación**: Como paso previo a algoritmos de clasificación.

5. **Detección de anomalías**: Identificación de puntos que se desvían significativamente de la estructura principal de los datos.

## Consideraciones Prácticas

### Selección del Kernel

La elección del kernel es crucial y depende de la naturaleza de los datos:

- **Kernel Polinomial**: Útil cuando se sospecha que existen relaciones polinomiales.
- **Kernel RBF**: Opción versátil que funciona bien en muchos casos, especialmente cuando la estructura de los datos es desconocida.
- **Kernel Sigmoide**: Puede ser apropiado para datos que presentan patrones similares a los que procesan las redes neuronales.

### Ajuste de Parámetros

Los parámetros del kernel deben ajustarse cuidadosamente:

- Para el kernel RBF, $\gamma$ controla la "anchura" del kernel. Valores pequeños capturan estructuras más globales, mientras que valores grandes se centran en estructuras locales.
- Para el kernel polinomial, el grado $d$ determina la complejidad de las relaciones que se pueden capturar.

### Escalado de Datos

El escalado de los datos es importante, especialmente para kernels sensibles a la magnitud como el RBF:

- La estandarización (media cero, varianza unitaria) suele ser recomendable.
- Alternativamente, la normalización a un rango específico (por ejemplo, [0,1]) puede ser apropiada en algunos casos.

### Problema de Pre-imagen

La reconstrucción de datos en el espacio original (problema de pre-imagen) es un desafío en Kernel PCA:

- No existe una solución analítica directa para la mayoría de los kernels.
- Se utilizan métodos de optimización para encontrar aproximaciones.
- La calidad de la reconstrucción puede variar significativamente según el kernel y los datos.

## Implementación Eficiente

Para conjuntos de datos grandes, se pueden considerar aproximaciones:

1. **Kernel PCA Aproximado**: Utiliza una muestra representativa de los datos para construir una aproximación de la matriz de kernel.

2. **Kernel PCA Incremental**: Actualiza incrementalmente el modelo a medida que llegan nuevos datos.

3. **Kernel PCA Disperso**: Utiliza técnicas de dispersión para reducir el número de vectores de soporte.

Kernel PCA representa una poderosa extensión no lineal del PCA tradicional, capaz de capturar estructuras complejas en los datos que las técnicas lineales no pueden detectar. Su flexibilidad a través de diferentes funciones kernel lo hace adaptable a diversos tipos de datos y problemas.

Sin embargo, su mayor complejidad computacional y la necesidad de seleccionar y ajustar adecuadamente el kernel y sus parámetros requieren un enfoque más cuidadoso en comparación con el PCA tradicional. En conjuntos de datos muy grandes, pueden ser preferibles técnicas más escalables como t-SNE o UMAP, que se explorarán en las siguientes secciones.
