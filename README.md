# 🧬 Algoritmos Genéticos Aplicados a Machine Learning

Este repositorio contiene la implementación de **Algoritmos Genéticos (AG)** aplicados a tres problemas fundamentales en el aprendizaje de máquina. El objetivo es demostrar cómo la evolución artificial puede sustituir procesos de búsqueda manuales o exhaustivos.

---

## 📊 Datasets Utilizados

Se utilizan un datasets principales de la librería `scikit-learn`:

* **Breast Cancer Wisconsin**: 569 muestras con 30 características para diagnóstico de tumores (utilizado en Selección y Optimización).

---

## 🔍 1. Feature Selection (Selección de Características)

Se implementa un AG para seleccionar el subconjunto óptimo de variables que maximiza la precisión del modelo, reduciendo la dimensionalidad y eliminando ruido.

* **Cromosoma**: Vector binario de longitud 30 (1 = incluida, 0 = excluida).
* **Fitness**: Accuracy (5-fold CV) con penalización por número de variables.
* **Selección**: Torneo binario.
* **Cruzamiento**: Un punto de corte.
* **Mutación**: Probabilidad de 0.05 por gen.
* **Resultado**: Reducción a **5 características clave** con una precisión de **0.9649**.

---

## ⚙️ 2. Hyperparameter Optimization (Optimización)

Optimización de parámetros críticos de un modelo **Random Forest** para equilibrar sesgo y varianza.

* **Hiperparámetros**: `n_estimators` (10–200) y `max_depth` (1–15).
* **Cromosoma**: Diccionario con los valores de los hiperparámetros.
* **Fitness**: Accuracy promedio mediante validación cruzada (3-fold).
* **Selección**: Torneo (k=3).
* **Mutación**: Variación controlada de los valores (probabilidad 0.2).
* **Resultado**: 
    * Mejor configuración: `n_estimators = 67`, `max_depth = 11`.
    * **Accuracy promedio: 0.9623**.

---

## 🧠 3. NeuroEvolution (Búsqueda de Arquitectura)

Uso de neuroevolución para "auto-diseñar" la estructura de una **Red Neuronal (MLP)**.

* **Cromosoma**: Lista de enteros de longitud variable `(n1, n2, n3)`.
* **Estructura**: De 1 a 3 capas ocultas; entre 4 y 64 neuronas por capa.
* **Fitness**: Accuracy del modelo entrenado.
* **Mutación (Prob. 0.4)**: Operador triple (Cambiar neuronas, Agregar capa o Eliminar capa).
* **Resultado**: El algoritmo convergió en una arquitectura de 3 capas **(30, 6, 44)** con un **Accuracy de 0.9912 (99.12%)**.

---

## 🚀 Ejecución

Los notebooks y scripts pueden ejecutarse en **Google Colab** o de forma local.

### Pasos para iniciar:
1.  **Clonar el repositorio**: `git clone https://github.com/tu-usuario/tu-repo.git`
2.  **Abrir archivos**: Cargar los `.ipynb` en Colab o ejecutar el script `.py`.
3.  **Instalar dependencias**:
    ```bash
    pip install numpy scikit-learn matplotlib
    ```

---

## 🛠️ Requisitos
* Python 3.x
* NumPy
* Scikit-learn
* Matplotlib

---

## 📝 Conclusión

Los algoritmos genéticos demostraron ser una herramienta versátil. Mientras que en **Iris** se alcanzó la perfección rápidamente por su separabilidad, en **Breast Cancer** el límite del **99.12%** representa un estado de generalización óptimo. El proyecto evidencia que los AG pueden automatizar el diseño de modelos de Machine Learning de manera eficiente.
