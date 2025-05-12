#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación práctica del Análisis de Componentes Principales Incremental (Incremental PCA)

Este script demuestra la aplicación del Incremental PCA en conjuntos de datos de gran tamaño,
comparando su rendimiento con el PCA tradicional y mostrando cómo procesar datos por lotes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import fetch_openml, make_blobs
import seaborn as sns
import time
import os
import psutil
from memory_profiler import memory_usage
import warnings

# Suprimir advertencias
warnings.filterwarnings("ignore")

# Configuración para visualizaciones más estéticas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def generar_datos_sinteticos(n_samples=10000, n_features=500, n_informative=50, random_state=42):
    """
    Genera un conjunto de datos sintético de alta dimensionalidad.
    
    Args:
        n_samples (int): Número de muestras.
        n_features (int): Número total de características.
        n_informative (int): Número de características informativas.
        random_state (int): Semilla para reproducibilidad.
    
    Returns:
        numpy.ndarray: Datos generados.
    """
    print("1. Generando datos sintéticos de alta dimensionalidad...")
    
    # Generar datos con estructura de clusters
    X, y = make_blobs(n_samples=n_samples, n_features=n_informative, 
                     centers=5, cluster_std=10.0, random_state=random_state)
    
    # Añadir características no informativas (ruido)
    if n_features > n_informative:
        X_noise = np.random.randn(n_samples, n_features - n_informative)
        X = np.hstack((X, X_noise))
    
    print(f"   - Dimensiones del conjunto de datos: {X.shape}")
    print(f"   - Número de características informativas: {n_informative}")
    print(f"   - Número de características de ruido: {n_features - n_informative}")
    
    return X, y

def cargar_mnist(subsample=None):
    """
    Carga el conjunto de datos MNIST y opcionalmente lo submuestrea.
    
    Args:
        subsample (int, optional): Número de muestras a seleccionar. 
                                  Si es None, se cargan todas las muestras.
    
    Returns:
        tuple: (X, y) donde X son los datos y y son las etiquetas.
    """
    print("1. Cargando el conjunto de datos MNIST...")
    
    # Cargar MNIST
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    
    # Submuestrear si es necesario
    if subsample is not None and subsample < X.shape[0]:
        indices = np.random.choice(X.shape[0], subsample, replace=False)
        X = X[indices]
        y = y[indices]
    
    print(f"   - Dimensiones del conjunto de datos: {X.shape}")
    print(f"   - Rango de valores: [{X.min()}, {X.max()}]")
    
    return X, y

def comparar_rendimiento_memoria(X, n_components=50):
    """
    Compara el rendimiento y el uso de memoria entre PCA tradicional e Incremental PCA.
    
    Args:
        X (numpy.ndarray): Datos de entrada.
        n_components (int): Número de componentes principales a retener.
    
    Returns:
        tuple: (pca, ipca, pca_time, ipca_time, pca_memory, ipca_memory)
    """
    print(f"\n2. Comparando rendimiento y uso de memoria (n_components={n_components})...")
    
    # Función para medir el uso de memoria y tiempo de PCA tradicional
    def run_pca():
        pca = PCA(n_components=n_components)
        start_time = time.time()
        X_pca = pca.fit_transform(X)
        end_time = time.time()
        return pca, X_pca, end_time - start_time
    
    # Función para medir el uso de memoria y tiempo de Incremental PCA
    def run_ipca(batch_size):
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        start_time = time.time()
        X_ipca = ipca.fit_transform(X)
        end_time = time.time()
        return ipca, X_ipca, end_time - start_time
    
    # Medir PCA tradicional
    print("   - Ejecutando PCA tradicional...")
    pca_memory = memory_usage((run_pca, []), max_usage=True)
    pca, X_pca, pca_time = run_pca()
    
    # Determinar un tamaño de lote razonable (aproximadamente 10% de los datos)
    batch_size = max(100, min(1000, X.shape[0] // 10))
    
    # Medir Incremental PCA
    print(f"   - Ejecutando Incremental PCA (batch_size={batch_size})...")
    ipca_memory = memory_usage((run_ipca, [batch_size]), max_usage=True)
    ipca, X_ipca, ipca_time = run_ipca(batch_size)
    
    # Imprimir resultados
    print("\n   - Resultados de rendimiento:")
    print(f"     * PCA tradicional: {pca_time:.4f} segundos, {pca_memory:.2f} MB")
    print(f"     * Incremental PCA: {ipca_time:.4f} segundos, {ipca_memory:.2f} MB")
    print(f"     * Diferencia de tiempo: {(pca_time - ipca_time) / pca_time * 100:.2f}% ({pca_time / ipca_time:.2f}x)")
    print(f"     * Diferencia de memoria: {(pca_memory - ipca_memory) / pca_memory * 100:.2f}% ({pca_memory / ipca_memory:.2f}x)")
    
    return pca, ipca, X_pca, X_ipca, pca_time, ipca_time, pca_memory, ipca_memory

def analizar_varianza_explicada(pca, ipca):
    """
    Compara la varianza explicada entre PCA tradicional e Incremental PCA.
    
    Args:
        pca (sklearn.decomposition.PCA): Modelo PCA tradicional ajustado.
        ipca (sklearn.decomposition.IncrementalPCA): Modelo Incremental PCA ajustado.
    """
    print("\n3. Analizando la varianza explicada...")
    
    # Varianza explicada por componente
    pca_var = pca.explained_variance_ratio_
    ipca_var = ipca.explained_variance_ratio_
    
    # Varianza explicada acumulada
    pca_cumvar = np.cumsum(pca_var)
    ipca_cumvar = np.cumsum(ipca_var)
    
    # Imprimir resultados
    print("   - Varianza explicada (primeros 5 componentes):")
    for i in range(min(5, len(pca_var))):
        print(f"     * Componente {i+1}: PCA = {pca_var[i]:.4f}, IPCA = {ipca_var[i]:.4f}, Diferencia = {abs(pca_var[i] - ipca_var[i]):.4f}")
    
    # Visualizar varianza explicada
    plt.figure(figsize=(12, 5))
    
    # Varianza por componente
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca_var) + 1), pca_var, alpha=0.5, label='PCA Tradicional')
    plt.bar(range(1, len(ipca_var) + 1), ipca_var, alpha=0.5, label='Incremental PCA')
    plt.title('Varianza Explicada por Componente')
    plt.xlabel('Componente Principal')
    plt.ylabel('Proporción de Varianza Explicada')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Varianza acumulada
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca_cumvar) + 1), pca_cumvar, 'o-', label='PCA Tradicional')
    plt.plot(range(1, len(ipca_cumvar) + 1), ipca_cumvar, 's-', label='Incremental PCA')
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Varianza')
    plt.title('Varianza Explicada Acumulada')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Proporción Acumulada')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('incremental_pca_varianza.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calcular el número de componentes necesarios para explicar el 90% de la varianza
    n_components_pca_90 = np.argmax(pca_cumvar >= 0.9) + 1
    n_components_ipca_90 = np.argmax(ipca_cumvar >= 0.9) + 1
    
    print(f"\n   - Componentes necesarios para explicar el 90% de la varianza:")
    print(f"     * PCA tradicional: {n_components_pca_90}")
    print(f"     * Incremental PCA: {n_components_ipca_90}")

def comparar_proyecciones(X_pca, X_ipca, y=None):
    """
    Compara las proyecciones de datos entre PCA tradicional e Incremental PCA.
    
    Args:
        X_pca (numpy.ndarray): Datos transformados por PCA tradicional.
        X_ipca (numpy.ndarray): Datos transformados por Incremental PCA.
        y (numpy.ndarray, optional): Etiquetas de clase para colorear los puntos.
    """
    print("\n4. Comparando proyecciones de datos...")
    
    # Calcular la correlación entre las proyecciones
    corr_pc1 = np.corrcoef(X_pca[:, 0], X_ipca[:, 0])[0, 1]
    corr_pc2 = np.corrcoef(X_pca[:, 1], X_ipca[:, 1])[0, 1]
    
    print(f"   - Correlación entre proyecciones:")
    print(f"     * Primer componente: {corr_pc1:.4f}")
    print(f"     * Segundo componente: {corr_pc2:.4f}")
    
    # Visualizar proyecciones 2D
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Determinar si usar colores por clase
    if y is not None:
        scatter_pca = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=30)
        scatter_ipca = axes[1].scatter(X_ipca[:, 0], X_ipca[:, 1], c=y, cmap='viridis', alpha=0.6, s=30)
        fig.colorbar(scatter_pca, ax=axes[0], label='Clase')
        fig.colorbar(scatter_ipca, ax=axes[1], label='Clase')
    else:
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)
        axes[1].scatter(X_ipca[:, 0], X_ipca[:, 1], alpha=0.6, s=30)
    
    axes[0].set_title('PCA Tradicional')
    axes[0].set_xlabel('Primer Componente Principal')
    axes[0].set_ylabel('Segundo Componente Principal')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Incremental PCA')
    axes[1].set_xlabel('Primer Componente Principal')
    axes[1].set_ylabel('Segundo Componente Principal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('incremental_pca_proyecciones.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualizar la correlación entre componentes
    n_components = min(5, X_pca.shape[1])
    correlations = np.zeros(n_components)
    
    for i in range(n_components):
        correlations[i] = np.corrcoef(X_pca[:, i], X_ipca[:, i])[0, 1]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_components + 1), correlations)
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    plt.title('Correlación entre Componentes de PCA y IPCA')
    plt.xlabel('Componente Principal')
    plt.ylabel('Correlación')
    plt.xticks(range(1, n_components + 1))
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('incremental_pca_correlacion.png', dpi=300, bbox_inches='tight')
    plt.show()

def demostrar_procesamiento_por_lotes(X, n_components=50, batch_sizes=[100, 500, 1000, 5000]):
    """
    Demuestra el procesamiento por lotes en Incremental PCA y su impacto en el rendimiento.
    
    Args:
        X (numpy.ndarray): Datos de entrada.
        n_components (int): Número de componentes principales a retener.
        batch_sizes (list): Lista de tamaños de lote a probar.
    """
    print("\n5. Demostrando procesamiento por lotes...")
    
    times = []
    memories = []
    
    # Probar diferentes tamaños de lote
    for batch_size in batch_sizes:
        print(f"   - Probando batch_size={batch_size}...")
        
        # Función para medir
        def run_ipca_batch():
            ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
            start_time = time.time()
            X_ipca = ipca.fit_transform(X)
            end_time = time.time()
            return ipca, end_time - start_time
        
        # Medir tiempo y memoria
        memory = memory_usage((run_ipca_batch, []), max_usage=True)
        ipca, elapsed_time = run_ipca_batch()
        
        times.append(elapsed_time)
        memories.append(memory)
        
        print(f"     * Tiempo: {elapsed_time:.4f} segundos, Memoria: {memory:.2f} MB")
    
    # Visualizar resultados
    plt.figure(figsize=(12, 5))
    
    # Tiempo vs. Tamaño de lote
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, times, 'o-', linewidth=2, markersize=8)
    plt.title('Tiempo de Ejecución vs. Tamaño de Lote')
    plt.xlabel('Tamaño de Lote')
    plt.ylabel('Tiempo (segundos)')
    plt.grid(True, alpha=0.3)
    
    # Memoria vs. Tamaño de lote
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, memories, 's-', linewidth=2, markersize=8, color='orange')
    plt.title('Uso de Memoria vs. Tamaño de Lote')
    plt.xlabel('Tamaño de Lote')
    plt.ylabel('Memoria (MB)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('incremental_pca_batch_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Imprimir conclusiones
    print("\n   - Conclusiones del procesamiento por lotes:")
    
    min_time_idx = np.argmin(times)
    min_memory_idx = np.argmin(memories)
    
    print(f"     * Tamaño de lote óptimo para tiempo: {batch_sizes[min_time_idx]}")
    print(f"     * Tamaño de lote óptimo para memoria: {batch_sizes[min_memory_idx]}")
    
    if min_time_idx != min_memory_idx:
        print("     * Existe un compromiso entre tiempo y memoria según el tamaño de lote.")
    else:
        print("     * El mismo tamaño de lote optimiza tanto tiempo como memoria.")

def simular_flujo_datos(X, n_components=50, n_batches=5):
    """
    Simula un escenario de flujo de datos donde los datos llegan por lotes.
    
    Args:
        X (numpy.ndarray): Datos de entrada.
        n_components (int): Número de componentes principales a retener.
        n_batches (int): Número de lotes en que dividir los datos.
    """
    print(f"\n6. Simulando flujo de datos ({n_batches} lotes)...")
    
    # Dividir los datos en lotes
    batch_size = X.shape[0] // n_batches
    batches = [X[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    
    # Inicializar Incremental PCA
    ipca = IncrementalPCA(n_components=n_components)
    
    # Inicializar PCA tradicional para comparación
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Métricas para seguimiento
    explained_variances = []
    reconstruction_errors = []
    component_correlations = []
    
    # Procesar lotes incrementalmente
    for i, batch in enumerate(batches):
        print(f"   - Procesando lote {i+1}/{n_batches}...")
        
        # Actualizar el modelo con el nuevo lote
        ipca.partial_fit(batch)
        
        # Calcular varianza explicada acumulada
        explained_variance = np.sum(ipca.explained_variance_ratio_)
        explained_variances.append(explained_variance)
        
        # Calcular error de reconstrucción en todo el conjunto de datos
        X_ipca = ipca.transform(X)
        X_reconstructed = ipca.inverse_transform(X_ipca)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        reconstruction_errors.append(reconstruction_error)
        
        # Calcular correlación con los componentes de PCA tradicional
        # (solo para el primer componente por simplicidad)
        if ipca.components_.shape[0] > 0 and pca.components_.shape[0] > 0:
            corr = np.abs(np.corrcoef(ipca.components_[0], pca.components_[0])[0, 1])
            component_correlations.append(corr)
        else:
            component_correlations.append(0)
        
        print(f"     * Varianza explicada acumulada: {explained_variance:.4f}")
        print(f"     * Error de reconstrucción: {reconstruction_error:.6f}")
        if len(component_correlations) > 0:
            print(f"     * Correlación con PCA tradicional (PC1): {component_correlations[-1]:.4f}")
    
    # Visualizar la evolución de las métricas
    plt.figure(figsize=(15, 5))
    
    # Varianza explicada
    plt.subplot(1, 3, 1)
    plt.plot(range(1, n_batches + 1), explained_variances, 'o-', linewidth=2)
    plt.axhline(y=np.sum(pca.explained_variance_ratio_), color='r', linestyle='--', 
               label=f'PCA Tradicional: {np.sum(pca.explained_variance_ratio_):.4f}')
    plt.title('Varianza Explicada Acumulada')
    plt.xlabel('Número de Lotes Procesados')
    plt.ylabel('Proporción de Varianza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error de reconstrucción
    plt.subplot(1, 3, 2)
    plt.plot(range(1, n_batches + 1), reconstruction_errors, 's-', linewidth=2, color='orange')
    
    # Calcular error de reconstrucción para PCA tradicional
    X_pca = pca.transform(X)
    X_pca_reconstructed = pca.inverse_transform(X_pca)
    pca_reconstruction_error = np.mean((X - X_pca_reconstructed) ** 2)
    
    plt.axhline(y=pca_reconstruction_error, color='r', linestyle='--', 
               label=f'PCA Tradicional: {pca_reconstruction_error:.6f}')
    plt.title('Error de Reconstrucción')
    plt.xlabel('Número de Lotes Procesados')
    plt.ylabel('Error Cuadrático Medio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Correlación de componentes
    plt.subplot(1, 3, 3)
    plt.plot(range(1, n_batches + 1), component_correlations, '^-', linewidth=2, color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Correlación Perfecta')
    plt.title('Correlación con PCA Tradicional (PC1)')
    plt.xlabel('Número de Lotes Procesados')
    plt.ylabel('Correlación Absoluta')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('incremental_pca_flujo_datos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualizar la convergencia de los componentes
    if ipca.components_.shape[0] >= 2 and pca.components_.shape[0] >= 2:
        plt.figure(figsize=(12, 5))
        
        # Primer componente
        plt.subplot(1, 2, 1)
        plt.bar(range(len(pca.components_[0])), pca.components_[0], alpha=0.5, label='PCA Tradicional')
        plt.bar(range(len(ipca.components_[0])), ipca.components_[0], alpha=0.5, label='Incremental PCA')
        plt.title('Primer Componente Principal')
        plt.xlabel('Índice de Característica')
        plt.ylabel('Coeficiente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Segundo componente
        plt.subplot(1, 2, 2)
        plt.bar(range(len(pca.components_[1])), pca.components_[1], alpha=0.5, label='PCA Tradicional')
        plt.bar(range(len(ipca.components_[1])), ipca.components_[1], alpha=0.5, label='Incremental PCA')
        plt.title('Segundo Componente Principal')
        plt.xlabel('Índice de Característica')
        plt.ylabel('Coeficiente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('incremental_pca_convergencia_componentes.png', dpi=300, bbox_inches='tight')
        plt.show()

def visualizar_reconstruccion_mnist(X, pca, ipca, n_samples=5):
    """
    Visualiza la reconstrucción de imágenes MNIST usando PCA tradicional e Incremental PCA.
    
    Args:
        X (numpy.ndarray): Datos MNIST.
        pca (sklearn.decomposition.PCA): Modelo PCA tradicional ajustado.
        ipca (sklearn.decomposition.IncrementalPCA): Modelo Incremental PCA ajustado.
        n_samples (int): Número de muestras a visualizar.
    """
    print("\n7. Visualizando reconstrucción de imágenes MNIST...")
    
    # Seleccionar algunas muestras aleatorias
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_samples = X[indices]
    
    # Transformar y reconstruir con PCA tradicional
    X_pca = pca.transform(X_samples)
    X_pca_reconstructed = pca.inverse_transform(X_pca)
    
    # Transformar y reconstruir con Incremental PCA
    X_ipca = ipca.transform(X_samples)
    X_ipca_reconstructed = ipca.inverse_transform(X_ipca)
    
    # Visualizar resultados
    plt.figure(figsize=(15, 3 * n_samples))
    
    for i in range(n_samples):
        # Original
        plt.subplot(n_samples, 3, i*3 + 1)
        plt.imshow(X_samples[i].reshape(28, 28), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # PCA reconstrucción
        plt.subplot(n_samples, 3, i*3 + 2)
        plt.imshow(X_pca_reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title('PCA Tradicional')
        plt.axis('off')
        
        # IPCA reconstrucción
        plt.subplot(n_samples, 3, i*3 + 3)
        plt.imshow(X_ipca_reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title('Incremental PCA')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('incremental_pca_reconstruccion_mnist.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calcular errores de reconstrucción
    pca_error = np.mean((X_samples - X_pca_reconstructed) ** 2)
    ipca_error = np.mean((X_samples - X_ipca_reconstructed) ** 2)
    
    print(f"   - Error de reconstrucción medio:")
    print(f"     * PCA tradicional: {pca_error:.6f}")
    print(f"     * Incremental PCA: {ipca_error:.6f}")
    print(f"     * Diferencia relativa: {abs(pca_error - ipca_error) / pca_error * 100:.2f}%")

def main_sintetico():
    """Función principal para el análisis con datos sintéticos."""
    print("=== Análisis de Componentes Principales Incremental (Datos Sintéticos) ===\n")
    
    # Generar datos sintéticos
    X, y = generar_datos_sinteticos(n_samples=5000, n_features=200)
    
    # Comparar rendimiento y memoria
    pca, ipca, X_pca, X_ipca, pca_time, ipca_time, pca_memory, ipca_memory = comparar_rendimiento_memoria(X)
    
    # Analizar varianza explicada
    analizar_varianza_explicada(pca, ipca)
    
    # Comparar proyecciones
    comparar_proyecciones(X_pca, X_ipca, y)
    
    # Demostrar procesamiento por lotes
    demostrar_procesamiento_por_lotes(X)
    
    # Simular flujo de datos
    simular_flujo_datos(X)
    
    print("\n=== Análisis con Datos Sintéticos Completado ===")

def main_mnist():
    """Función principal para el análisis con datos MNIST."""
    print("=== Análisis de Componentes Principales Incremental (MNIST) ===\n")
    
    # Cargar datos MNIST (submuestreados para mayor rapidez)
    X, y = cargar_mnist(subsample=5000)
    
    # Comparar rendimiento y memoria
    pca, ipca, X_pca, X_ipca, pca_time, ipca_time, pca_memory, ipca_memory = comparar_rendimiento_memoria(X, n_components=100)
    
    # Analizar varianza explicada
    analizar_varianza_explicada(pca, ipca)
    
    # Comparar proyecciones
    comparar_proyecciones(X_pca, X_ipca, y)
    
    # Visualizar reconstrucción de imágenes
    visualizar_reconstruccion_mnist(X, pca, ipca)
    
    print("\n=== Análisis con MNIST Completado ===")

if __name__ == "__main__":
    # Ejecutar análisis con datos sintéticos
    main_sintetico()
    
    # Ejecutar análisis con MNIST
    # Descomentar la siguiente línea para ejecutar el análisis con MNIST
    # main_mnist()
