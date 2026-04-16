"""
╔══════════════════════════════════════════════════════════════════╗
║   NEUROEVOLUCIÓN - EJEMPLO 1                                     ║
║   Hallar la mejor arquitectura de red neuronal con AG            ║
║   Dataset: Breast_cancer (clasificación)                                  ║
╚══════════════════════════════════════════════════════════════════╝

CICLO DEL ALGORITMO GENÉTICO:
  1. Representación (Cromosoma): lista de enteros [n1, n2, n3]
     cada gen = número de neuronas en esa capa oculta
  2. Inicialización: población aleatoria de arquitecturas
  3. Función de aptitud: accuracy del modelo entrenado
  4. Selección: torneo (tournament selection)
  5. Cruzamiento: punto de corte único (single-point crossover)
  6. Mutación: cambio aleatorio de una capa
  7. Terminación: número máximo de generaciones
"""

import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



TAMANIO_POBLACION = 10     
NUM_GENERACIONES  = 8       
PROB_MUTACION     = 0.2     
TAMANIO_TORNEO    = 3        

MIN_CAPAS    = 1
MAX_CAPAS    = 3
MIN_NEURONAS = 4
MAX_NEURONAS = 64


breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
scaler = StandardScaler() #toma todos los datos 
X = scaler.fit_transform(X) #pone todos los datos en la misma escala 
X_train, X_test, y_train, y_test = train_test_split(  #se usa mis datos para dos grupos 
    X, y, test_size=0.2, random_state=42 #asegura que la division siempre sea la misma 
)


def crear_cromosoma():
    
    num_capas = random.randint(MIN_CAPAS, MAX_CAPAS)
    return tuple(random.randint(MIN_NEURONAS, MAX_NEURONAS) for _ in range(num_capas))
#con el tuple empaqueta todos los datos en una cromosoma 


def inicializar_poblacion():
    """Crea la población inicial de cromosomas (arquitecturas)."""
    return [crear_cromosoma() for _ in range(TAMANIO_POBLACION)]


def calcular_aptitud(cromosoma):
    
    modelo = MLPClassifier(
        hidden_layer_sizes=cromosoma, #ejemplo (32, 16)
        max_iter=300,
        random_state=42,
        early_stopping=True, #si la red ya aprende no sigue entrenando 
        n_iter_no_change=20 #si no mejora en 20 vueltas, se detiene el entrenamiento
    )
    modelo.fit(X_train, y_train) #80% de datos
    return modelo.score(X_test, y_test) #20% de datos 


def seleccion_torneo(poblacion, aptitudes):
    
    participantes = random.sample(range(len(poblacion)), TAMANIO_TORNEO)
    ganador = max(participantes, key=lambda i: aptitudes[i])
    return poblacion[ganador]

def cruzamiento(padre1, padre2):
    
    largo = max(len(padre1), len(padre2))
    if largo < 2:
        if random.random() < 0.5: 
            return padre1
        else:
            return padre2
    p1 = list(padre1) + [random.randint(MIN_NEURONAS, MAX_NEURONAS)] * (largo - len(padre1))
    p2 = list(padre2) + [random.randint(MIN_NEURONAS, MAX_NEURONAS)] * (largo - len(padre2))

    punto_corte = random.randint(1, largo - 1)
    hijo = p1[:punto_corte] + p2[punto_corte:]

    hijo = hijo[:MAX_CAPAS]
    return tuple(hijo) if hijo else (random.randint(MIN_NEURONAS, MAX_NEURONAS),)


def mutacion(cromosoma):
    
    cromosoma = list(cromosoma)          

    if random.random() < PROB_MUTACION:
       
        tipo = random.choice(["cambiar", "agregar", "eliminar"])

        print(f"      🧬 MUTACIÓN tipo='{tipo}' | antes: {tuple(cromosoma)}", end="")

        
        if tipo == "cambiar" and cromosoma:
            idx = random.randint(0, len(cromosoma) - 1)
            valor_antes = cromosoma[idx]
            cromosoma[idx] = random.randint(MIN_NEURONAS, MAX_NEURONAS)
            print(f" → cambió capa[{idx}]: {valor_antes} → {cromosoma[idx]}"
                  f" | después: {tuple(cromosoma)}")

    
        elif tipo == "agregar" and len(cromosoma) < MAX_CAPAS:
            pos = random.randint(0, len(cromosoma))   # posición de inserción
            nuevas_neuronas = random.randint(MIN_NEURONAS, MAX_NEURONAS)
            cromosoma.insert(pos, nuevas_neuronas)
            print(f" → insertó {nuevas_neuronas} neuronas en pos[{pos}]"
                  f" | después: {tuple(cromosoma)}")

        elif tipo == "agregar":
            # Condición no cumplida: ya tiene el máximo de capas → sin efecto
            print(f" → BLOQUEADO (ya tiene {len(cromosoma)} capas = MAX)")

        
        elif tipo == "eliminar" and len(cromosoma) > MIN_CAPAS:
            idx = random.randint(0, len(cromosoma) - 1)
            valor_eliminado = cromosoma.pop(idx)
            print(f" → eliminó capa[{idx}] con {valor_eliminado} neuronas"
                  f" | después: {tuple(cromosoma)}")

        elif tipo == "eliminar":
            # Condición no cumplida: ya tiene el mínimo de capas → sin efecto
            print(f" → BLOQUEADO (ya tiene {len(cromosoma)} capas = MIN)")

    else:
        # Sin mutación en este individuo
        print(f"      ⬜ Sin mutación | cromosoma: {tuple(cromosoma)}")

    return tuple(cromosoma)


def algoritmo_genetico():
    print("=" * 60)
    print("  NEUROEVOLUCIÓN — Búsqueda de Arquitectura (Breast_Cancer)")
    print("=" * 60)

    poblacion = inicializar_poblacion()
    mejor_global = None
    mejor_aptitud_global = 0.0

    for generacion in range(1, NUM_GENERACIONES + 1):
        print(f"\n📌 Generación {generacion}/{NUM_GENERACIONES}")

        aptitudes = []
        for i, cromosoma in enumerate(poblacion):
            apt = calcular_aptitud(cromosoma)
            aptitudes.append(apt)
            print(f"   Individuo {i+1}: capas={cromosoma}  → accuracy={apt:.4f}")

        mejor_idx = np.argmax(aptitudes)
        if aptitudes[mejor_idx] > mejor_aptitud_global:
            mejor_aptitud_global = aptitudes[mejor_idx]
            mejor_global = poblacion[mejor_idx]

        print(f"   🏆 Mejor de esta gen: {poblacion[mejor_idx]} "
              f"| accuracy={aptitudes[mejor_idx]:.4f}")

        if mejor_aptitud_global >= 1.0:
            print("\n✅ Terminación anticipada: accuracy = 1.0 alcanzada")
            break

        nueva_poblacion = []
        nueva_poblacion.append(mejor_global)

        print(f"\n   ── Aplicando mutaciones (prob={PROB_MUTACION}) ──")
        while len(nueva_poblacion) < TAMANIO_POBLACION:
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            hijo = cruzamiento(padre1, padre2)
            hijo = mutacion(hijo)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

    print("\n" + "=" * 60)
    print("  RESULTADO FINAL")
    print("=" * 60)
    print(f"  Mejor arquitectura encontrada: {mejor_global}")
    print(f"  Número de capas ocultas:       {len(mejor_global)}")
    print(f"  Neuronas por capa:             {list(mejor_global)}")
    print(f"  Accuracy en prueba:            {mejor_aptitud_global:.4f} "
          f"({mejor_aptitud_global*100:.2f}%)")

    modelo_final = MLPClassifier(
        hidden_layer_sizes=mejor_global,
        max_iter=500,
        random_state=42
    )
    modelo_final.fit(X_train, y_train)
    print(f"\n  Arquitectura completa de la red:")
    print(f"    Entrada  →  {X_train.shape[1]} características")
    for i, n in enumerate(mejor_global, 1):
        print(f"    Capa {i}   →  {n} neuronas (ReLU)")
    print(f"    Salida   →  {len(np.unique(y))} clases (Softmax)")

    return mejor_global, mejor_aptitud_global


if __name__ == "__main__":
    random.seed(7)
    np.random.seed(7)
    mejor_arq, mejor_acc = algoritmo_genetico()
