# Teoría para Live Coding: Reducción Dimensional y Feature Engineering

## 📚 Guía Completa de Conceptos Teóricos para Maestría

---

## PARTE 1: FEATURE ENGINEERING (15 minutos)

### 1.1 Imputación de Missing Values

**Concepto clave:**
> Los datos reales siempre tienen valores faltantes. La forma de manejarlos puede hacer o romper tu modelo.

**Estrategias:**

**KNN Imputation (lo que usamos):**
```python
# Idea: "Dime quién eres y te diré qué valor falta"
# Encuentra K vecinos más similares
# Promedia sus valores (ponderado por distancia)
```

**¿Por qué funciona?**
- Captura relaciones entre features
- Respeta estructura local
- Más preciso que media/mediana

**Nuestra estrategia híbrida:**
```
Missing < 30%  → KNN (vale la pena la precisión)
Missing > 30%  → Median (KNN no confiable)
```

**💡 Pregunta:** *¿Por qué no siempre KNN?*
- Con mucho missing, los "vecinos" no son confiables

---

### 1.2 Tratamiento de Outliers

**Winsorización (lo que usamos):**
```python
# Reemplazar extremos con percentiles
x_new = clip(x, p01, p99)

Ventajas:
✅ No perdemos muestras
✅ Reducimos influencia de extremos
✅ Preservamos orden relativo
```

**Alternativas:**
- **Eliminación:** Pierdes datos
- **Transformación:** Log, sqrt comprime extremos

**💡 Pregunta:** *¿Cuándo eliminar vs winsorizar?*
- Eliminar: Errores obvios, dataset grande
- Winsorizar: Valores extremos legítimos

---

### 1.3 Transformaciones: Yeo-Johnson

**¿Por qué transformar?**
- PCA asume normalidad (Gaussiana)
- Mejora convergencia
- Reduce impacto de outliers

**Yeo-Johnson vs Box-Cox:**
```python
Box-Cox:  Solo valores POSITIVOS
Yeo-Johnson: Funciona con NEGATIVOS ✅

# Datos de proteínas pueden tener negativos
# (log-ratios, z-scores)
```

---

### 1.4 Feature Engineering Creativo

**A. Features Estadísticos:**
```python
mean_expression    # Nivel promedio
std_expression     # Variabilidad
cv_expression      # Variabilidad relativa (std/mean)
skewness          # Asimetría de distribución
kurtosis          # "Peakedness"
```

**¿Por qué funcionan?**
- Capturan patrones globales por muestra
- Reducen ruido
- Pueden ser más discriminativos

**Ejemplo biológico:**
```
Muestra A: mean=10, cv=0.1  → Expresión estable
Muestra B: mean=10, cv=0.8  → Expresión errática
→ Mismo mean, diferente biología
```

**B. Features de Interacción:**
```python
# Ratios
ratio = protein_A / protein_B
# Captura balance entre procesos

# Productos
product = protein_A × protein_B
# Captura co-ocurrencia
```

**¿Cómo seleccionar pares?**
- Correlación alta (>0.7)
- Domain knowledge
- Feature importance

**⚠️ Cuidado:** 77 features → 2,926 pares posibles
→ Usamos solo top 10

---

### 1.5 Escalado: RobustScaler

**¿Por qué escalar?**
```
Feature A: [0, 1]
Feature B: [0, 1000]
→ B domina distancias
→ Algoritmos sesgados
```

**RobustScaler:**
```python
x_new = (x - median) / IQR

Ventajas:
✅ Robusto a outliers (usa percentiles)
✅ Segunda línea de defensa después de winsorización
```

---

## PARTE 2: PCA (20 minutos)

### 2.1 Intuición

**Analogía:**
> Fotografiar un objeto 3D desde el mejor ángulo que captura más información

**Concepto:**
- Encontrar direcciones de máxima varianza
- Proyectar datos en esas direcciones
- Reducir dimensionalidad manteniendo información

### 2.2 Matemáticas (nivel maestría)

**Pasos:**
```python
1. Centrar datos: X_centered = X - mean
2. Matriz covarianza: Σ = X.T @ X / n
3. Eigendecomposición: Σv = λv
4. Ordenar por eigenvalues: λ₁ ≥ λ₂ ≥ ...
```

**Propiedades:**
- **Ortogonalidad:** PCs no correlacionados
- **Ordenamiento:** PC1 captura más varianza
- **Reconstrucción:** Minimiza error de reconstrucción

### 2.3 Interpretación

**Scree Plot:**
```
Buscar "codo" donde curva se aplana
Antes del codo: componentes importantes
Después: ruido
```

**Reglas de selección:**
```
1. Regla del codo (visual)
2. Varianza acumulada (80%, 90%, 95%)
3. Eigenvalue > 1 (Kaiser)
4. Validación cruzada (mejor)
```

**Loadings:**
```python
Loading[i,j] = correlación entre feature i y PC j

|loading| alto → feature importante
loading > 0 → contribución positiva
loading < 0 → contribución negativa
```

**Ejemplo:**
```
PC1 loadings:
  Proteína_A:  0.85  ← Alta contribución +
  Proteína_D: -0.65  ← Alta contribución -

Interpretación:
PC1 = "Eje de A vs D"
Score alto → A alto, D bajo
```

### 2.4 Ventajas y Limitaciones

**✅ Ventajas:**
- Determinístico (reproducible)
- Rápido (escalable)
- Interpretable (loadings)
- Reduce ruido
- Decorrelaciona features

**❌ Limitaciones:**
- Solo lineal
- Sensible a escala (escalar primero)
- Sensible a outliers
- Máxima varianza ≠ máxima discriminación

**💡 Pregunta:** *¿Por qué PCA no siempre separa clases?*
- Varianza y discriminación son diferentes
- Puede haber mucha varianza dentro de clases

---

## PARTE 3: t-SNE (20 minutos)

### 3.1 Motivación

**Problema con PCA:**
```
Datos no-lineales (espiral, Swiss roll)
PCA proyecta linealmente → pierde estructura
```

**Objetivo t-SNE:**
> Preservar estructura LOCAL: puntos cercanos deben permanecer cercanos

### 3.2 Algoritmo (intuición)

**Idea:**
```
1. Calcular similitudes en alta dimensión (Gaussiana)
2. Calcular similitudes en baja dimensión (t-Student)
3. Optimizar para que coincidan (minimizar KL divergence)
```

**¿Por qué distribución t-Student?**
- Colas más pesadas que Gaussiana
- Permite separar puntos moderadamente lejanos
- Resuelve "crowding problem"

### 3.3 Hiperparámetro: Perplexity

**Definición:**
```
Perplexity ≈ "número efectivo de vecinos"
Perplexity = 30 → considera ~30 vecinos
```

**Efectos:**
```
Perplexity BAJO (5-10):
✅ Estructura muy local
❌ Sobre-fragmenta clusters

Perplexity MEDIO (30-50):
✅ Balance local-global
✅ Recomendado

Perplexity ALTO (100+):
✅ Más global
❌ Fusiona clusters pequeños
```

**Regla práctica:**
```
n < 100:      perplexity = 5-15
n = 100-1000: perplexity = 30-50
n > 1000:     perplexity = 50-100
```

### 3.4 Interpretación

**✅ SÍ puedes interpretar:**
- Proximidad local (puntos cercanos son similares)
- Separación de clusters
- Estructura interna de clusters

**❌ NO puedes interpretar:**
- Distancias globales entre clusters
- Tamaño de clusters
- Densidades
- Distancias numéricas

**Ejemplo de mala interpretación:**
```
❌ "Cluster A está 2x más lejos de C que B"
✅ "Clusters A y B están separados de C"
```

### 3.5 Ventajas y Limitaciones

**✅ Ventajas:**
- Excelente visualización
- Captura no-linealidad
- Clusters bien definidos

**❌ Limitaciones:**
- NO para ML pipelines (no determinístico)
- Lento O(n²)
- Sensible a hiperparámetros
- No preserva estructura global
- Puede crear estructura artificial

**💡 Pregunta:** *¿Por qué no usar t-SNE en pipelines ML?*
- No determinístico
- No hay transform() para nuevos datos
- Cada aplicación requiere re-optimización

---

## PARTE 4: UMAP (20 minutos)

### 4.1 Motivación

**Mejoras sobre t-SNE:**
```
✅ Más rápido O(n log n)
✅ Escalable
✅ Preserva estructura global
✅ Determinístico (con seed)
✅ Puede usarse en ML pipelines
```

### 4.2 Algoritmo (intuición)

**Base teórica:**
- Teoría de variedades de Riemannian
- Topología algebraica
- Idea: datos viven en variedad de baja dimensión

**Pasos:**
```
1. Construir grafo fuzzy en alta dimensión
   - Encontrar k vecinos
   - Distancias adaptativas locales
   
2. Optimizar embedding en baja dimensión
   - Minimizar cross-entropy
   - Preservar topología
```

### 4.3 Hiperparámetros

**n_neighbors:**
```
n_neighbors BAJO (5-10):
✅ Estructura muy local
✅ Clusters detallados
❌ Puede fragmentar

n_neighbors MEDIO (15-30):
✅ Balance recomendado
✅ Buena separación

n_neighbors ALTO (50+):
✅ Más global
✅ Estructura suave
❌ Pierde detalles
```

**min_dist:**
```
min_dist = 0.0:  Clusters muy compactos
min_dist = 0.1:  Balance (recomendado)
min_dist = 0.5:  Clusters más dispersos
```

**Regla práctica:**
```
Exploración inicial: n_neighbors=15, min_dist=0.1
Ajustar según necesidad
```

### 4.4 UMAP vs t-SNE

**Comparación:**
```
| Aspecto           | t-SNE      | UMAP        |
|-------------------|------------|-------------|
| Velocidad         | O(n²)      | O(n log n)  |
| Escalabilidad     | <10k       | Millones    |
| Determinismo      | No         | Sí (seed)   |
| Estructura global | Pobre      | Buena       |
| Estructura local  | Excelente  | Excelente   |
| ML pipeline       | ❌         | ✅          |
| Interpretación    | Cualitativa| Semi-cuant  |
```

### 4.5 Ventajas y Limitaciones

**✅ Ventajas:**
- Rápido y escalable
- Balance local-global
- Determinístico
- Transform para nuevos datos
- Puede usarse en ML
- Preserva más estructura que t-SNE

**❌ Limitaciones:**
- Menos maduro que PCA/t-SNE
- Teoría matemática compleja
- Hiperparámetros menos intuitivos
- Puede sobre-optimizar en datos pequeños

---

## PARTE 5: COMPARACIÓN Y MÉTRICAS (15 minutos)

### 5.1 Métricas de Clustering

**Silhouette Score [-1, 1]:**
```python
s = (b - a) / max(a, b)

a = distancia promedio intra-cluster
b = distancia promedio al cluster más cercano

Interpretación:
s ≈ 1:  Bien separado
s ≈ 0:  En frontera
s < 0:  Mal asignado
```

**Calinski-Harabasz (mayor mejor):**
```python
CH = (SSB / SSW) × ((n - k) / (k - 1))

SSB = suma de cuadrados entre clusters
SSW = suma de cuadrados dentro clusters

Interpretación:
Alto → clusters compactos y separados
```

**Davies-Bouldin (menor mejor):**
```python
DB = (1/k) Σ max((σᵢ + σⱼ) / d(cᵢ, cⱼ))

Interpretación:
Bajo → clusters compactos y lejanos
```

### 5.2 Interpretación de Resultados

**Ejemplo de nuestro dataset:**
```
         Method  Silhouette  Calinski-H  Davies-B
            PCA      -0.065        91.4       7.20
t-SNE (perp=30)      -0.007       231.6       5.29
    UMAP (n=15)      -0.005       313.5       6.08
```

**Análisis:**

**Silhouette (todos negativos):**
- Normal en datos biológicos
- Overlap natural entre clases
- UMAP mejor (menos negativo)

**Calinski-Harabasz:**
- UMAP: 313.5 (mejor) ✨
- 3.4x mejor que PCA
- Mejor separación global

**Davies-Bouldin:**
- t-SNE: 5.29 (mejor) ✨
- Clusters más compactos
- Mejor estructura local

**Conclusión:**
- UMAP: Mejor balance general
- t-SNE: Mejor compactness local
- PCA: Limitado por linealidad

### 5.3 Cuándo Usar Cada Técnica

**PCA:**
```
✅ Análisis exploratorio rápido
✅ Reducción para ML pipeline
✅ Interpretabilidad (loadings)
✅ Datasets con relaciones lineales
✅ Cuando necesitas determinismo
```

**t-SNE:**
```
✅ Visualización para papers
✅ Exploración de clusters
✅ Detección de estructura local
✅ Presentaciones
❌ NO para ML pipelines
❌ NO para datos >10k puntos
```

**UMAP:**
```
✅ Balance local-global
✅ ML pipelines (con transform)
✅ Datasets grandes (escalable)
✅ Cuando necesitas velocidad
✅ Preservación de topología
✅ Alternativa moderna a t-SNE
```

### 5.4 Workflow Recomendado

**Exploración inicial:**
```
1. PCA primero (rápido, interpretable)
   - Ver varianza explicada
   - Identificar outliers
   - Entender features importantes

2. t-SNE para visualización
   - Probar múltiples perplexities
   - Identificar clusters
   - Validar con domain knowledge

3. UMAP para confirmación
   - Comparar con t-SNE
   - Verificar estructura global
   - Usar en pipeline si necesario
```

**Para producción:**
```
1. PCA si relaciones lineales
2. UMAP si estructura compleja
3. Nunca t-SNE (no reproducible)
```

---

## CONCEPTOS AVANZADOS (Opcional)

### Kernel PCA
```
Idea: PCA en espacio de features no-lineal
Kernel: rbf, poly, sigmoid
Captura no-linealidad sin manifold learning
```

### Autoencoders
```
Red neuronal para reducción dimensional
Encoder: alta-D → baja-D
Decoder: baja-D → alta-D
Más flexible que PCA
```

### Comparación Teórica

**Preservación:**
```
PCA:   Preserva varianza global
t-SNE: Preserva vecindades locales
UMAP:  Preserva topología (local + global)
```

**Complejidad:**
```
PCA:   O(min(n²p, np²))
t-SNE: O(n² log n) con Barnes-Hut
UMAP:  O(n log n)
```

**Fundamento:**
```
PCA:   Álgebra lineal (eigenvectors)
t-SNE: Probabilidad (KL divergence)
UMAP:  Topología (Riemannian manifolds)
```

---

## PREGUNTAS FRECUENTES

**P: ¿Siempre escalar antes de PCA?**
R: SÍ. PCA es sensible a escala. Features con mayor varianza dominan.

**P: ¿Cuántos componentes retener?**
R: Depende. Regla general: 80-95% varianza. Mejor: validación cruzada.

**P: ¿Por qué t-SNE da resultados diferentes?**
R: Inicialización aleatoria + optimización estocástica. Fijar random_state.

**P: ¿UMAP siempre mejor que t-SNE?**
R: No siempre. t-SNE puede ser mejor para estructura muy local. UMAP mejor para balance.

**P: ¿Puedo usar t-SNE para clasificación?**
R: NO. No hay transform(). Solo visualización.

**P: ¿Reducir con PCA antes de t-SNE?**
R: SÍ, recomendado. Reduce a ~50 dims primero. Más rápido, usualmente sin pérdida.

**P: ¿Cómo validar que reducción es buena?**
R: 
1. Métricas de clustering
2. Clasificación downstream
3. Validación con domain knowledge
4. Múltiples técnicas (triangulación)

---

## MENSAJES CLAVE PARA ESTUDIANTES

1. **Feature Engineering es crucial**
   - Puede mejorar modelo más que algoritmo
   - Requiere domain knowledge
   - Iterativo: experimentar y validar

2. **No hay técnica "mejor"**
   - Depende de objetivo y datos
   - PCA: rápido, interpretable, lineal
   - t-SNE: visualización, local
   - UMAP: balance, escalable

3. **Siempre validar**
   - Métricas cuantitativas
   - Validación cruzada
   - Domain knowledge
   - Múltiples técnicas

4. **Entender limitaciones**
   - PCA: solo lineal
   - t-SNE: no para ML, no global
   - UMAP: menos maduro

5. **Workflow importa**
   - Explorar → Validar → Producción
   - Documentar decisiones
   - Reproducibilidad

---

## RECURSOS ADICIONALES

**Papers:**
- PCA: Pearson (1901)
- t-SNE: van der Maaten & Hinton (2008)
- UMAP: McInnes et al. (2018)

**Tutoriales:**
- Distill.pub: "How to Use t-SNE Effectively"
- UMAP docs: https://umap-learn.readthedocs.io/

**Libros:**
- "Feature Engineering for ML" - Alice Zheng
- "Hands-On ML" - Aurélien Géron

---

**¡Fin de la teoría! Ahora a codear! 🚀**
