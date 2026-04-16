"""
╔══════════════════════════════════════════════════════════════════╗
║   ALGORITMO GENÉTICO — FEATURE SELECTION                        ║
║   Objetivo: Seleccionar el subconjunto óptimo de características║
║   Dataset : Wisconsin Breast Cancer (30 features)               ║
╚══════════════════════════════════════════════════════════════════╝

Ciclo del AG implementado:
  1. Representación   → Vector binario (gen=1 incluye la feature)
  2. Inicialización   → Población aleatoria
  3. Función de aptitud → Accuracy (CV 5-fold) penalizada por #features
  4. Selección        → Torneo binario
  5. Cruzamiento      → Un punto
  6. Mutación         → Flip de bit
  7. Terminación      → Máximo de generaciones + convergencia
"""

import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════════════════════════
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
FEATURE_NAMES = dataset.feature_names
N_FEATURES = X.shape[1]          # 30 features originales

# ══════════════════════════════════════════════════════════════════
# PARÁMETROS DEL AG
# ══════════════════════════════════════════════════════════════════
POP_SIZE       = 20    # individuos por generación
GENERATIONS    = 15    # número máximo de generaciones
MUTATION_RATE  = 0.05  # probabilidad de mutar cada bit
CROSSOVER_RATE = 0.8   # probabilidad de cruzamiento
ELITE_SIZE     = 2     # mejores individuos que pasan directamente
PENALTY        = 0.01  # penalización por feature adicional usada


# ══════════════════════════════════════════════════════════════════
# 1. REPRESENTACIÓN
# ══════════════════════════════════════════════════════════════════
def create_individual() -> list:
    """
    Cromosoma: lista binaria de longitud N_FEATURES.
    Gen = 1 → feature INCLUIDA en el modelo.
    Gen = 0 → feature EXCLUIDA del modelo.
    Al menos 1 gen activo para evitar individuos vacíos.
    """
    ind = [random.randint(0, 1) for _ in range(N_FEATURES)]
    if sum(ind) == 0:
        ind[random.randint(0, N_FEATURES - 1)] = 1
    return ind


# ══════════════════════════════════════════════════════════════════
# 2. INICIALIZACIÓN
# ══════════════════════════════════════════════════════════════════
def initialize_population(size: int) -> list:
    """Genera una población inicial de individuos aleatorios."""
    return [create_individual() for _ in range(size)]


# ══════════════════════════════════════════════════════════════════
# 3. FUNCIÓN DE APTITUD
# ══════════════════════════════════════════════════════════════════
def fitness(individual: list) -> float:
    """
    Aptitud = Accuracy (CV 5-fold) − PENALTY × nº_features_usadas.
    La penalización premia subconjuntos pequeños con alta precisión.
    """
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected) == 0:
        return 0.0

    X_sel = X[:, selected]
    clf   = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_sel, y, cv=5, scoring="accuracy")
    return scores.mean() - PENALTY * len(selected)


# ══════════════════════════════════════════════════════════════════
# 4. SELECCIÓN — Torneo binario
# ══════════════════════════════════════════════════════════════════
def tournament_selection(population: list, fitnesses: list) -> list:
    """Selecciona un padre enfrentando 2 individuos; gana el más apto."""
    i, j = random.sample(range(len(population)), 2)
    return population[i] if fitnesses[i] >= fitnesses[j] else population[j]


# ══════════════════════════════════════════════════════════════════
# 5. CRUZAMIENTO — Un punto
# ══════════════════════════════════════════════════════════════════
def crossover(parent1: list, parent2: list) -> tuple:
    """
    Con probabilidad CROSSOVER_RATE divide en un punto aleatorio
    e intercambia segmentos. Si no hay cruce, devuelve copias.
    """
    if random.random() < CROSSOVER_RATE:
        point  = random.randint(1, N_FEATURES - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    else:
        child1, child2 = parent1[:], parent2[:]
    return child1, child2


# ══════════════════════════════════════════════════════════════════
# 6. MUTACIÓN — Flip de bit
# ══════════════════════════════════════════════════════════════════
def mutate(individual: list) -> list:
    """
    Cada bit se invierte (0→1 ó 1→0) con probabilidad MUTATION_RATE.
    Garantiza al menos 1 feature activa.
    """
    mutated = [1 - bit if random.random() < MUTATION_RATE else bit
               for bit in individual]
    if sum(mutated) == 0:
        mutated[random.randint(0, N_FEATURES - 1)] = 1
    return mutated


# ══════════════════════════════════════════════════════════════════
# 7. BUCLE PRINCIPAL + TERMINACIÓN
# ══════════════════════════════════════════════════════════════════
def genetic_algorithm():
    print("╔══════════════════════════════════════════════════════╗")
    print("║       AG — FEATURE SELECTION                        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Dataset : Breast Cancer | Features originales: {N_FEATURES}")
    print(f"  Población: {POP_SIZE} | Generaciones: {GENERATIONS}\n")

    # ── Inicialización ──────────────────────────────────────────
    population   = initialize_population(POP_SIZE)
    best_ind     = None
    best_fit     = -np.inf
    history      = []

    for gen in range(1, GENERATIONS + 1):

        # ── Evaluar aptitud ────────────────────────────────────
        fitnesses = [fitness(ind) for ind in population]

        # ── Registrar mejor global ─────────────────────────────
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = population[gen_best_idx][:]

        n_feat = sum(population[gen_best_idx])
        history.append(gen_best_fit)
        print(f"  Gen {gen:2d} | aptitud: {gen_best_fit:.4f} | "
              f"features usadas: {n_feat:2d} | mejor global: {best_fit:.4f}")

        # ── Elitismo ───────────────────────────────────────────
        sorted_idx  = np.argsort(fitnesses)[::-1]
        new_pop     = [population[i][:] for i in sorted_idx[:ELITE_SIZE]]

        # ── Selección + Cruzamiento + Mutación ─────────────────
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(c2))

        population = new_pop

        # ── Terminación anticipada (convergencia) ──────────────
        if len(history) >= 5 and len(set(history[-5:])) == 1:
            print(f"\n  ✓ Convergencia alcanzada en generación {gen}.")
            break

    # ── Resultado final ────────────────────────────────────────
    selected_features = [FEATURE_NAMES[i] for i, b in enumerate(best_ind) if b == 1]
    raw_acc = best_fit + PENALTY * len(selected_features)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  RESULTADO FINAL                                    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Accuracy obtenida  : {raw_acc:.4f}")
    print(f"  Features usadas    : {len(selected_features)} / {N_FEATURES}")
    print("  Features seleccionadas:")
    for f in selected_features:
        print(f"    • {f}")
    return best_ind, best_fit


if __name__ == "__main__":
    genetic_algorithm()
