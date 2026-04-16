# =====================================
# 1. IMPORTAR LIBRERÍAS
# =====================================
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Reproducibilidad
random.seed(42)
np.random.seed(42)

# =====================================
# 2. CARGAR DATOS
# =====================================
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =====================================
# 3. CREAR POBLACIÓN
# =====================================
def crear_poblacion(n):
    poblacion = []
    for _ in range(n):
        individuo = {
            "n_estimators": random.randint(10, 200),
            "max_depth": random.randint(1, 15)
        }
        poblacion.append(individuo)
    return poblacion

# =====================================
# 4. FUNCIÓN FITNESS
# =====================================
def fitness(individuo):
    modelo = RandomForestClassifier(
        n_estimators=individuo["n_estimators"],
        max_depth=individuo["max_depth"],
        random_state=42,
        n_jobs=-1
    )

    scores = cross_val_score(modelo, X_train, y_train, cv=3)
    return scores.mean()

# =====================================
# 5. SELECCIÓN POR TORNEO
# =====================================
def seleccionar(poblacion, k=3):
    seleccionados = []

    for _ in range(len(poblacion)//2):
        torneo = random.sample(poblacion, k)
        mejor = max(torneo, key=lambda x: fitness(x))
        seleccionados.append(mejor)

    return seleccionados

# =====================================
# 6. CRUZAMIENTO
# =====================================
def cruzar(p1, p2):
    hijo = {
        "n_estimators": int((p1["n_estimators"] + p2["n_estimators"]) / 2),
        "max_depth": random.choice([p1["max_depth"], p2["max_depth"]])
    }
    return hijo

# =====================================
# 7. MUTACIÓN
# =====================================
def mutar(individuo, prob=0.2):
    if random.random() < prob:
        individuo["n_estimators"] += random.randint(-20, 20)
        individuo["n_estimators"] = max(10, min(200, individuo["n_estimators"]))

    if random.random() < prob:
        individuo["max_depth"] += random.randint(-3, 3)
        individuo["max_depth"] = max(1, min(15, individuo["max_depth"]))

    return individuo

# =====================================
# 8. ALGORITMO GENÉTICO
# =====================================
def algoritmo_genetico(generaciones=10, tamaño_poblacion=12):

    poblacion = crear_poblacion(tamaño_poblacion)
    historial = []

    for gen in range(generaciones):
        print(f"\nGeneración {gen}")

        resultados = [(ind, fitness(ind)) for ind in poblacion]
        mejor = max(resultados, key=lambda x: x[1])

        historial.append(mejor[1])

        print("Mejor individuo:", mejor[0])
        print("Fitness (accuracy promedio):", round(mejor[1], 4))

        seleccionados = seleccionar(poblacion)

        nueva = seleccionados.copy()

        while len(nueva) < tamaño_poblacion:
            p1, p2 = random.sample(seleccionados, 2)
            hijo = cruzar(p1, p2)
            hijo = mutar(hijo)
            nueva.append(hijo)

        poblacion = nueva

    mejor_final = max(poblacion, key=lambda x: fitness(x))

    print("\nMejor configuración final:")
    print(mejor_final)
    print("Fitness final:", round(fitness(mejor_final), 4))

    # Gráfica de evolución
    plt.plot(historial)
    plt.title("Evolución del fitness")
    plt.xlabel("Generación")
    plt.ylabel("Accuracy")
    plt.show()

# Ejecutar
algoritmo_genetico()