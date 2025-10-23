# Live Coding: Reducción Dimensional Avanzada y Feature Engineering

## 🎯 Descripción

Sesión de live coding de 90 minutos para estudiantes de maestría sobre técnicas avanzadas de reducción dimensional (PCA, t-SNE, UMAP) y feature engineering aplicado.

**Dataset:** Mice Protein Expression (UCI ML Repository)
- 1080 muestras de expresión de proteínas
- 77 features originales
- 8 clases experimentales
- Datos reales con missing values y outliers

---

## 📋 Contenido

### Archivos incluidos:
1. **`live_coding_dimensionality_reduction.ipynb`** - Notebook principal para la sesión
2. **`NOTAS_INSTRUCTOR.md`** - Guía detallada para el instructor con timing y puntos clave
3. **`requirements.txt`** - Dependencias necesarias
4. **`README.md`** - Este archivo

---

## 🚀 Instalación

### Opción 1: Usando pip
```bash
pip install -r requirements.txt
```

### Opción 2: Usando conda
```bash
conda create -n dimred python=3.9
conda activate dimred
pip install -r requirements.txt
```

### Verificar instalación:
```python
import pandas as pd
import numpy as np
import sklearn
import umap
import plotly

print("✅ Todas las librerías instaladas correctamente")
```

---

## 📚 Estructura de la Sesión (90 minutos)

### 1. Exploración y Feature Engineering (15 min)
- Carga del dataset
- Análisis exploratorio
- Imputación inteligente (KNN + estadística)
- Detección y tratamiento de outliers
- Transformaciones (Yeo-Johnson)
- Creación de features estadísticos e interacciones

### 2. PCA - Principal Component Analysis (20 min)
- Análisis de varianza explicada
- Scree plot y selección de componentes
- Visualización 2D
- Análisis de loadings
- Interpretación de resultados

### 3. t-SNE - t-Distributed Stochastic Neighbor Embedding (20 min)
- Efecto del parámetro perplexity
- Comparación de diferentes configuraciones
- Visualizaciones interactivas
- Limitaciones y advertencias

### 4. UMAP - Uniform Manifold Approximation and Projection (20 min)
- Efecto del parámetro n_neighbors
- Comparación con t-SNE
- Ventajas computacionales
- Aplicaciones en ML

### 5. Comparación Cuantitativa (15 min)
- Métricas de clustering (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Visualización comparativa lado a lado
- Discusión de trade-offs
- Recomendaciones prácticas

---

## 🎓 Requisitos Previos

Los estudiantes deben tener conocimiento básico de:
- ✅ Python y Jupyter notebooks
- ✅ NumPy y Pandas
- ✅ Conceptos básicos de PCA, t-SNE y UMAP
- ✅ Machine Learning (clasificación, clustering)
- ✅ Álgebra lineal básica

---

## 💡 Conceptos Clave Cubiertos

### Feature Engineering:
- Imputación estratificada (KNN vs median)
- Winsorización de outliers
- Transformaciones para normalidad
- Features estadísticos (mean, std, cv, skewness, kurtosis)
- Features de interacción (ratios, productos)
- Escalado robusto

### Reducción Dimensional:
- **PCA:** Lineal, rápido, interpretable
- **t-SNE:** Preservación de estructura local, visualización
- **UMAP:** Balance local-global, escalable

### Evaluación:
- Métricas de clustering
- Preservación de estructura
- Trade-offs computacionales

---

## 📊 Resultados Esperados

Al ejecutar el notebook completo, obtendrás:

1. **Visualizaciones exploratorias:**
   - Distribución de missing values
   - Distribución de clases
   - Correlaciones entre features

2. **Comparación de transformaciones:**
   - Antes/después de Yeo-Johnson
   - Efecto del feature engineering

3. **Análisis PCA:**
   - Scree plot
   - Varianza acumulada
   - Loadings de componentes principales
   - Visualización 2D con clases

4. **Comparación t-SNE:**
   - 4 configuraciones de perplexity
   - Efecto visual de cada parámetro

5. **Comparación UMAP:**
   - 4 configuraciones de n_neighbors
   - Comparación con t-SNE

6. **Métricas cuantitativas:**
   - Tabla comparativa de las 3 técnicas
   - Visualización lado a lado

---

## 🔧 Troubleshooting

### Problema: Error al descargar el dataset
**Solución:** El dataset se descarga automáticamente desde UCI. Si falla:
```python
# Descargar manualmente desde:
# https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls
# Y cargar localmente:
df = pd.read_excel('Data_Cortex_Nuclear.xls')
```

### Problema: UMAP muy lento
**Solución:** Reducir primero con PCA:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_scaled)
# Luego aplicar UMAP a X_reduced
```

### Problema: Visualizaciones no se muestran
**Solución:** Asegurar que estás en Jupyter:
```python
%matplotlib inline
# Para Plotly en Jupyter Lab:
import plotly.io as pio
pio.renderers.default = 'jupyterlab'
```

---

## 📖 Referencias

### Papers Fundamentales:
1. **PCA:** Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space"
2. **t-SNE:** van der Maaten, L., & Hinton, G. (2008). "Visualizing data using t-SNE"
3. **UMAP:** McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection"

### Recursos Online:
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Scikit-learn: Manifold Learning](https://scikit-learn.org/stable/modules/manifold.html)

### Dataset:
- [UCI ML Repository - Mice Protein Expression](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression)

---

## 🎯 Objetivos de Aprendizaje

Al completar esta sesión, los estudiantes podrán:

1. ✅ Aplicar técnicas avanzadas de feature engineering a datos reales
2. ✅ Entender las diferencias conceptuales entre PCA, t-SNE y UMAP
3. ✅ Seleccionar la técnica apropiada según el contexto y objetivos
4. ✅ Interpretar visualizaciones de reducción dimensional
5. ✅ Evaluar la calidad de embeddings con métricas cuantitativas
6. ✅ Identificar limitaciones y trade-offs de cada técnica
7. ✅ Implementar pipelines completos de preprocesamiento

---

## 🤝 Contribuciones

Este material está diseñado para fines educativos. Siéntete libre de:
- Adaptar el contenido a tu curso
- Agregar ejemplos adicionales
- Experimentar con otros datasets
- Compartir mejoras

---

## 📧 Contacto

Para preguntas sobre el material o sugerencias de mejora, contactar al instructor del curso.

---

## 📄 Licencia

Material educativo de uso libre para fines académicos.

---

## 🌟 Tips para el Instructor

1. **Preparación:** Ejecutar el notebook completo antes de la sesión
2. **Timing:** Usar cronómetro para mantener el ritmo
3. **Interacción:** Hacer preguntas frecuentes a los estudiantes
4. **Flexibilidad:** Ajustar profundidad según respuestas
5. **Práctica:** Animar a experimentar con parámetros

**¡Buena suerte con la sesión!** 🚀
