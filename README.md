Algoritmos Genéticos aplicados a Machine Learning
Este repositorio contiene la implementación de algoritmos genéticos (AG) aplicados a tres problemas fundamentales en el aprendizaje de máquina: selección de características, optimización de hiperparámetros y diseño de arquitecturas neuronales (Neuroevolución). El objetivo es demostrar cómo la evolución artificial puede sustituir procesos de búsqueda manuales o exhaustivos.

Dataset
Se utilizan dos datasets principales de la librería scikit-learn:

Breast Cancer Wisconsin: 569 muestras con 30 características para diagnóstico de tumores (utilizado en Selección y Optimización).

Iris Dataset: 150 muestras y 4 características para clasificación de flores (utilizado en Neuroevolución).

1. Feature Selection (Selección de Características)
Se implementa un AG para seleccionar el subconjunto óptimo de variables que maximiza la precisión del modelo, reduciendo la dimensionalidad y eliminando ruido.

Cromosoma: Vector binario de longitud 30 (1 = característica incluida, 0 = excluida).

Fitness: Accuracy mediante validación cruzada (5-fold) con una penalización por el número de variables seleccionadas.

Selección: Torneo binario.

Cruzamiento: Un punto de corte.

Mutación: Probabilidad de 0.05 por gen.

Resultado: Se redujo el set a 5 características relevantes, alcanzando una precisión de 0.9649.

2. Hyperparameter Optimization (Optimización de Hiperparámetros)
Optimización de los parámetros críticos de un modelo Random Forest para encontrar el equilibrio entre sesgo y varianza.

Hiperparámetros: n_estimators (10–200) y max_depth (1–15).

Cromosoma: Diccionario con los valores de los hiperparámetros.

Fitness: Accuracy promedio mediante validación cruzada (3-fold).

Selección: Torneo (k=3).

Cruzamiento: Combinación de promedio y selección aleatoria.

Mutación: Variación controlada de los valores (probabilidad 0.2).

Resultado:

Mejor configuración: n_estimators = 67, max_depth = 11.

Accuracy promedio: 0.9623.

3. NeuroEvolution (Búsqueda de Arquitectura)
Se utiliza la neuroevolución para "auto-diseñar" la estructura de una Red Neuronal (MLP), determinando el número ideal de capas y neuronas.

Cromosoma: Lista de enteros de longitud variable (n1, n2, n3). Cada gen es el número de neuronas en esa capa.

Estructura: De 1 a 3 capas ocultas; entre 4 y 64 neuronas por capa.

Fitness: Accuracy del modelo entrenado.

Cruzamiento: Punto de corte único para intercambiar bloques de capas.

Mutación (Prob. 0.4): Operador triple: Cambiar neuronas, Agregar capa o Eliminar capa.

Resultado: El algoritmo convergió en una arquitectura de 3 capas (30, 6, 44) con un accuracy de 0.9912 (99.12%).
Ejecución
Los notebooks y scripts pueden ejecutarse en Google Colab o de forma local.

Pasos:
Clonar el repositorio.

Abrir los archivos .ipynb o ejecutar el script .py.

Asegurarse de tener instaladas las dependencias.

Requisitos:
Python 3.x

numpy

scikit-learn

matplotlib

Conclusión
Los algoritmos genéticos demostraron ser una herramienta versátil. Mientras que en Iris se alcanzó la perfección rápidamente por su separabilidad, en Breast Cancer el límite del 99.12% representa un estado de generalización óptimo. El proyecto evidencia que los AG pueden automatizar el diseño de modelos de Machine Learning, encontrando soluciones competitivas sin intervención humana manual.
