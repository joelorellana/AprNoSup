# Notas para el Instructor - Live Coding Session

## 📋 Resumen de la Sesión (90 minutos)

### Dataset: Mice Protein Expression
- **Fuente:** UCI Machine Learning Repository
- **Características:** 1080 muestras, 77 proteínas, 8 clases
- **Por qué este dataset:**
  - Poco conocido (no es Iris, MNIST, etc.)
  - Datos reales con problemas reales (missing values, outliers)
  - Suficientemente complejo para maestría
  - Biológicamente relevante

---

## ⏱️ Timing Sugerido

### Bloque 1: Exploración y Feature Engineering (15 min)
**Minutos 0-15**

**Puntos clave a enfatizar:**
1. **Imputación estratificada** (KNN vs median)
   - Explicar por qué KNN para <30% missing
   - Discutir trade-offs: precisión vs complejidad

2. **Winsorización vs eliminación de outliers**
   - Mostrar que no perdemos datos
   - Preservamos información pero limitamos extremos

3. **Transformaciones Yeo-Johnson**
   - Por qué no Box-Cox (valores negativos)
   - Importancia de normalidad para PCA

4. **Feature Engineering creativo:**
   - Features estadísticos (mean, std, cv, skew, kurtosis)
   - Ratios y productos de proteínas correlacionadas
   - Explicar el concepto de "domain knowledge"

**💡 Pregunta para estudiantes:**
*"¿Por qué crear features de interacción entre proteínas correlacionadas?"*
- Respuesta: Capturar relaciones no lineales que métodos lineales (PCA) no ven

---

### Bloque 2: PCA (20 min)
**Minutos 15-35**

**Demostración paso a paso:**

1. **Scree Plot** (5 min)
   - Explicar "codo" (elbow method)
   - Discutir trade-off: varianza vs dimensiones
   - Mostrar que 90% varianza ≠ 90% información útil

2. **Visualización 2D** (5 min)
   - Interpretar separación de clases
   - Explicar por qué algunas clases se solapan
   - Limitaciones de proyección lineal

3. **Análisis de Loadings** (10 min)
   - **MUY IMPORTANTE:** Explicar qué son los loadings
   - Mostrar features que más contribuyen a PC1 y PC2
   - Conectar con biología (si es posible)
   
**💡 Pregunta para estudiantes:**
*"Si PC1 explica 40% de varianza, ¿significa que es la dimensión más importante para clasificación?"*
- Respuesta: NO necesariamente. Varianza ≠ discriminación

**⚠️ Advertencias comunes:**
- PCA asume linealidad
- Sensible a escala (por eso escalamos)
- No garantiza separabilidad de clases

---

### Bloque 3: t-SNE (20 min)
**Minutos 35-55**

**Experimentos con perplexity:**

1. **Perplexity = 5** (estructura muy local)
   - Muchos clusters pequeños
   - Puede sobre-fragmentar

2. **Perplexity = 30** (recomendado)
   - Balance típico
   - Buena separación

3. **Perplexity = 50-100** (más global)
   - Clusters más grandes
   - Puede perder detalles

**💡 Pregunta para estudiantes:**
*"¿Por qué t-SNE produce resultados diferentes cada vez?"*
- Respuesta: Optimización estocástica, inicialización aleatoria

**⚠️ Advertencias CRÍTICAS:**
- **NO usar para reducción dimensional en pipelines ML**
- **NO interpretar distancias globales**
- **NO comparar densidades entre clusters**
- Computacionalmente costoso (O(n²))

**Conceptos avanzados a mencionar:**
- t-SNE preserva probabilidades de vecindad
- Usa distribución t de Student (colas pesadas)
- Crowding problem y cómo t-SNE lo resuelve

---

### Bloque 4: UMAP (20 min)
**Minutos 55-75**

**Experimentos con n_neighbors:**

1. **n_neighbors = 5** (muy local)
   - Similar a t-SNE con perplexity baja
   - Estructura granular

2. **n_neighbors = 15** (recomendado)
   - Balance óptimo típico
   - Buena separación

3. **n_neighbors = 30-50** (más global)
   - Estructura más suave
   - Mejor preservación global

**💡 Pregunta para estudiantes:**
*"¿Cuál es la principal ventaja de UMAP sobre t-SNE?"*
- Respuestas:
  1. Más rápido (puede escalar a millones de puntos)
  2. Preserva mejor estructura global
  3. Determinístico (con mismo random_state)
  4. Puede usarse en pipelines ML

**Conceptos avanzados:**
- Teoría de variedades de Riemannian
- Preservación de topología
- min_dist vs n_neighbors

**Comparación directa t-SNE vs UMAP:**
| Aspecto | t-SNE | UMAP |
|---------|-------|------|
| Velocidad | Lento (O(n²)) | Rápido (O(n log n)) |
| Determinismo | No | Sí (con seed) |
| Estructura global | Pobre | Buena |
| Estructura local | Excelente | Excelente |
| Uso en ML | ❌ | ✅ |

---

### Bloque 5: Comparación Cuantitativa (15 min)
**Minutos 75-90**

**Métricas de clustering:**

1. **Silhouette Score** (-1 a 1, mayor mejor)
   - Mide cohesión y separación
   - Fácil de interpretar

2. **Calinski-Harabasz** (mayor mejor)
   - Ratio de dispersión entre/dentro clusters
   - Favorece clusters compactos y separados

3. **Davies-Bouldin** (menor mejor)
   - Promedio de similitud entre clusters
   - Penaliza clusters cercanos

**💡 Pregunta para estudiantes:**
*"¿Por qué UMAP suele tener mejores métricas que t-SNE?"*
- Respuesta: Mejor preservación de distancias globales

**Visualización comparativa:**
- Mostrar las 3 técnicas lado a lado
- Discutir trade-offs
- No hay "ganador" absoluto

---

## 🎯 Puntos Clave de Feature Engineering

### 1. Imputación
```python
# Estrategia híbrida
- KNN: Para missing < 30% (más preciso)
- Median: Para missing > 30% (más robusto)
```

### 2. Outliers
```python
# Winsorización vs Eliminación
- Winsorización: Preserva muestras, limita extremos
- Eliminación: Pierde información
```

### 3. Transformaciones
```python
# Yeo-Johnson vs Box-Cox
- Yeo-Johnson: Funciona con negativos
- Box-Cox: Solo positivos
```

### 4. Features Estadísticos
```python
# Por muestra (axis=1)
- mean, std, cv (coef. variación)
- skewness, kurtosis
- percentiles (q25, q75, IQR)
```

### 5. Features de Interacción
```python
# Entre features correlacionadas
- Ratios: f1 / f2
- Productos: f1 * f2
- Diferencias: f1 - f2
```

---

## 🎓 Preguntas Avanzadas para Discusión

### Nivel 1: Conceptual
1. ¿Por qué PCA no garantiza buena separación de clases?
2. ¿Cuándo preferirías t-SNE sobre UMAP?
3. ¿Cómo afecta el escalado a cada técnica?

### Nivel 2: Práctico
1. ¿Cómo seleccionarías el número de componentes en PCA?
2. ¿Cómo validarías que tu feature engineering mejoró el modelo?
3. ¿Qué harías si t-SNE muestra clusters que PCA no muestra?

### Nivel 3: Avanzado
1. ¿Cómo implementarías UMAP en un pipeline de producción?
2. ¿Qué técnicas usarías para datasets con millones de muestras?
3. ¿Cómo combinarías feature engineering con deep learning?

---

## 🔧 Troubleshooting Común

### Problema 1: "t-SNE tarda mucho"
**Solución:**
- Reducir primero con PCA a ~50 dimensiones
- Usar menos iteraciones (500 en vez de 1000)
- Considerar usar UMAP

### Problema 2: "UMAP da resultados extraños"
**Solución:**
- Ajustar n_neighbors (probar 5, 15, 30, 50)
- Ajustar min_dist (0.0 a 0.99)
- Verificar escalado de datos

### Problema 3: "PCA no separa clases"
**Solución:**
- Verificar que datos estén escalados
- Probar transformaciones (log, sqrt, Box-Cox)
- Considerar PCA no lineal (Kernel PCA)

### Problema 4: "Muchos missing values"
**Solución:**
- Evaluar si eliminar features con >50% missing
- Usar imputación múltiple
- Crear feature binario "was_missing"

---

## 📚 Referencias para Profundizar

### Papers Fundamentales:
1. **PCA:** Pearson (1901) - Original paper
2. **t-SNE:** van der Maaten & Hinton (2008)
3. **UMAP:** McInnes et al. (2018)

### Recursos Online:
- Distill.pub: "How to Use t-SNE Effectively"
- UMAP documentation: https://umap-learn.readthedocs.io/
- Scikit-learn User Guide: Manifold Learning

### Libros:
- "Feature Engineering for Machine Learning" - Alice Zheng
- "Hands-On Machine Learning" - Aurélien Géron (Cap. 8)

---

## 💻 Extensiones Opcionales (si sobra tiempo)

### 1. Kernel PCA
```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca = kpca.fit_transform(X_scaled)
```

### 2. Autoencoder (reducción dimensional con DL)
```python
# Arquitectura simple
# Input -> Dense(64) -> Dense(32) -> Dense(2) -> Dense(32) -> Dense(64) -> Output
```

### 3. Análisis de estabilidad
```python
# Ejecutar t-SNE múltiples veces
# Medir variabilidad de resultados
```

### 4. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif
# Comparar con/sin feature selection
```

---

## ✅ Checklist Pre-Sesión

- [ ] Verificar que todas las librerías estén instaladas
- [ ] Probar descarga del dataset (puede fallar)
- [ ] Tener dataset local como backup
- [ ] Ejecutar notebook completo una vez
- [ ] Preparar respuestas a preguntas comunes
- [ ] Tener ejemplos adicionales listos

---

## 🎯 Objetivos de Aprendizaje

Al final de la sesión, los estudiantes deben poder:

1. ✅ Aplicar feature engineering avanzado a datos reales
2. ✅ Entender diferencias conceptuales entre PCA, t-SNE y UMAP
3. ✅ Seleccionar la técnica apropiada según el contexto
4. ✅ Interpretar visualizaciones de reducción dimensional
5. ✅ Evaluar calidad de embeddings con métricas
6. ✅ Identificar limitaciones de cada técnica

---

## 📝 Notas Finales

**Enfoque pedagógico:**
- Más práctica que teoría
- Enfatizar intuición sobre matemáticas
- Usar visualizaciones constantemente
- Fomentar experimentación

**Mensaje final para estudiantes:**
> "No existe una técnica 'mejor'. La elección depende de:
> - Objetivo (visualización vs ML pipeline)
> - Tamaño de datos
> - Estructura esperada (local vs global)
> - Recursos computacionales
> - Necesidad de reproducibilidad"

¡Buena suerte con la sesión! 🚀
