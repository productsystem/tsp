import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from scipy import stats
import pandas as pd

NUM_CITIES = 30
WIDTH, HEIGHT = 1000, 1000

T_START, T_MIN, ALPHA_SA, ITERATIONS_PER_T = 1000.0, 1e-4, 0.995, 80
ALPHA_ACO, BETA_ACO, EVAPORATION, NUM_ANTS, NUM_ITERATIONS_ACO = 1.0, 2.0, 0.3, 50, 150
NUM_PARTICLES_PSO, NUM_ITERS_PSO, C1_PSO, C2_PSO = 40, 300, 1.5, 1.5
TABU_MAX_ITERS, TABU_TENURE, TABU_NEIGHBORHOOD = 500, 30, 150
POP_SIZE_GA, NUM_GENERATIONS_GA, MUTATION_RATE_GA = 100, 500, 0.05

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def total_distance(tour, cities):
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += distance(cities[tour[i]], cities[tour[(i + 1) % n]])
    return total

def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

def two_opt_swap(tour):
    new_tour = tour[:]
    n = len(new_tour)
    i, j = sorted(random.sample(range(n), 2))
    new_tour[i:j] = reversed(new_tour[i:j])
    return new_tour

def two_opt_local_search(tour, cities):
    improved = True
    best = tour[:]
    best_cost = total_distance(best, cities)
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                new_tour = best[:i] + best[i:j][::-1] + best[j:]
                new_cost = total_distance(new_tour, cities)
                if new_cost < best_cost:
                    best, best_cost, improved = new_tour, new_cost, True
    return best

def nearest_neighbor_tsp(cities, start=0):
    n = len(cities)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda c: distance(cities[last], cities[c]))
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour

def best_nearest_neighbor(cities):
    best_tour = None
    best_cost = float("inf")
    for start in range(len(cities)):
        tour = nearest_neighbor_tsp(cities, start)
        cost = total_distance(tour, cities)
        if cost < best_cost:
            best_tour, best_cost = tour[:], cost
    return best_tour, best_cost

def simulated_annealing(cities):
    current = random_tour(len(cities))
    current_cost = total_distance(current, cities)
    best, best_cost = current[:], current_cost
    T = T_START
    history, temp_history = [], []
    while T > T_MIN:
        for _ in range(ITERATIONS_PER_T):
            candidate = two_opt_swap(current)
            cost = total_distance(candidate, cities)
            delta = cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current, current_cost = candidate, cost
                if current_cost < best_cost:
                    best, best_cost = current[:], current_cost
        history.append(best_cost)
        temp_history.append(T)
        T *= ALPHA_SA
    return best, best_cost, history, temp_history

def genetic_algorithm(cities):
    n = len(cities)
    pop = [random_tour(n) for _ in range(POP_SIZE_GA)]
    history = []
    
    for _ in range(NUM_GENERATIONS_GA):
        pop.sort(key=lambda t: total_distance(t, cities))
        history.append(total_distance(pop[0], cities))
        
        new_pop = pop[:2]
        while len(new_pop) < POP_SIZE_GA:
            p1 = min(random.sample(pop, 5), key=lambda t: total_distance(t, cities))
            p2 = min(random.sample(pop, 5), key=lambda t: total_distance(t, cities))
            a, b = sorted(random.sample(range(n), 2))
            child = [-1]*n
            child[a:b] = p1[a:b]
            remaining = [item for item in p2 if item not in child]
            r_idx = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = remaining[r_idx]
                    r_idx += 1
            if random.random() < MUTATION_RATE_GA:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]
            child = two_opt_local_search(child, cities)
            new_pop.append(child)
        pop = new_pop
    return pop[0], total_distance(pop[0], cities), history

def get_swap_sequence(source, target):
    res, s = [], source[:]
    pos = {city: i for i, city in enumerate(s)}
    for i in range(len(s)):
        if s[i] != target[i]:
            j = pos[target[i]]
            res.append((i, j))
            pos[s[i]], pos[s[j]] = j, i
            s[i], s[j] = s[j], s[i]
    return res

def apply_swaps(tour, swaps, prob=1.0):
    nt = tour[:]
    for i, j in swaps:
        if random.random() < prob:
            nt[i], nt[j] = nt[j], nt[i]
    return nt

def particle_swarm_optimization(cities):
    n = len(cities)
    particles = [random_tour(n) for _ in range(NUM_PARTICLES_PSO)]
    pbest = [p[:] for p in particles]
    pbest_cost = [total_distance(p, cities) for p in particles]
    gbest = min(pbest, key=lambda t: total_distance(t, cities))
    gbest_cost = total_distance(gbest, cities)
    history = []
    for _ in range(NUM_ITERS_PSO):
        for i in range(NUM_PARTICLES_PSO):
            s1 = get_swap_sequence(particles[i], pbest[i])
            s2 = get_swap_sequence(particles[i], gbest)
            new_p = apply_swaps(particles[i], s1, prob=C1_PSO/(C1_PSO+C2_PSO))
            new_p = apply_swaps(new_p, s2, prob=C2_PSO/(C1_PSO+C2_PSO))
            new_p = two_opt_local_search(new_p, cities)
            cost = total_distance(new_p, cities)
            particles[i] = new_p
            if cost < pbest_cost[i]:
                pbest[i], pbest_cost[i] = new_p[:], cost
                if cost < gbest_cost:
                    gbest, gbest_cost = new_p[:], cost
        history.append(gbest_cost)
    return gbest, gbest_cost, history

def ant_colony_optimization(cities):
    n = len(cities)
    dist = [[distance(cities[i], cities[j]) for j in range(n)] for i in range(n)]
    pheromone = [[1.0 for _ in range(n)] for _ in range(n)]
    best_tour, best_cost, history = None, float("inf"), []
    for _ in range(NUM_ITERATIONS_ACO):
        tours, costs = [], []
        for _ in range(NUM_ANTS):
            curr = random.randint(0, n-1)
            tour, unvisited = [curr], set(range(n))
            unvisited.remove(curr)
            while unvisited:
                u = tour[-1]
                moves = []
                total = 0.0
                for v in unvisited:
                    val = (pheromone[u][v]**ALPHA_ACO) * ((1.0/dist[u][v])**BETA_ACO)
                    moves.append((v, val))
                    total += val
                r, acc = random.random(), 0.0
                for v, val in moves:
                    acc += val/total
                    if acc >= r:
                        tour.append(v)
                        unvisited.remove(v)
                        break
            tour = two_opt_local_search(tour, cities)
            cost = total_distance(tour, cities)
            tours.append(tour); costs.append(cost)
            if cost < best_cost:
                best_tour, best_cost = tour[:], cost
        for i in range(n):
            for j in range(n): pheromone[i][j] *= (1 - EVAPORATION)
        for t, c in zip(tours, costs):
            for i in range(n):
                pheromone[t[i]][t[(i+1)%n]] += 1.0/c
        history.append(best_cost)
    return best_tour, best_cost, history

def tabu_search(cities):
    n = len(cities)
    current, current_cost = best_nearest_neighbor(cities)
    best, best_cost = current[:], current_cost
    tabu_list, history = deque(maxlen=TABU_TENURE), []
    for it in range(TABU_MAX_ITERS):
        bc, bcc, bm = None, float("inf"), None
        for _ in range(TABU_NEIGHBORHOOD):
            i, j = sorted(random.sample(range(n), 2))
            cand = current[:i] + current[i:j][::-1] + current[j:]
            cost = total_distance(cand, cities)
            if (i, j) not in tabu_list or cost < best_cost:
                if cost < bcc: bc, bcc, bm = cand, cost, (i, j)
        if bc:
            current, current_cost = bc, bcc
            tabu_list.append(bm)
            if current_cost < best_cost: best, best_cost = current[:], current_cost
        history.append(best_cost)
    return best, best_cost, history

def plot_tour(tour, cities, title):
    plt.figure(figsize=(6,6))
    x = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    y = [cities[i][1] for i in tour] + [cities[tour[0]][1]]
    plt.plot(x, y, 'ro-')
    plt.title(title)
    plt.show()


def perform_analysis(final_stats):
    print("\n" + "="*50)
    print("      DETAILED STATISTICAL ANALYSIS")
    print("="*50)
    
    print("\n[1] Normality Test (Shapiro-Wilk):")
    for alg, data in final_stats.items():
        _, p = stats.shapiro(data)
        status = "Normal" if p > 0.05 else "Not Normal"
        print(f"  - {alg:<6}: p-value = {p:.4f} ({status})")

    print("\n[2] Kruskal-Wallis H-test (Overall Comparison):")
    h_stat, p_val = stats.kruskal(*[final_stats[alg] for alg in final_stats])
    print(f"  H-statistic: {h_stat:.4f}, p-value: {p_val:.4e}")
    if p_val < 0.05:
        print("  RESULT: Significant difference exists between algorithms.")
    else:
        print("  RESULT: No significant difference found.")

    print("\n[3] Pairwise Comparisons (vs Hybrid GA):")
    target = 'GA'
    for alg in final_stats:
        if alg == target: continue
        u_stat, p_val = stats.mannwhitneyu(final_stats[target], final_stats[alg])
        
        d = (np.mean(final_stats[target]) - np.mean(final_stats[alg])) / \
             np.sqrt((np.std(final_stats[target])**2 + np.std(final_stats[alg])**2) / 2)
             
        significance = "Significant" if p_val < 0.05 else "Insignificant"
        print(f"  - {target} vs {alg:<4}: p={p_val:.4f} | Effect Size(d)={d:.2f} | {significance}")



def run_single_seed(s):
    random.seed(s)
    cities = [(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NUM_CITIES)]
    _, d_nn = best_nearest_neighbor(cities)
    _, d_sa, sa_h, _ = simulated_annealing(cities)
    _, d_aco, aco_h = ant_colony_optimization(cities)
    _, d_pso, pso_h = particle_swarm_optimization(cities)
    _, d_tabu, tabu_h = tabu_search(cities)
    _, d_ga, ga_h = genetic_algorithm(cities)
    
    return {
        'dists': {'NN': d_nn, 'SA': d_sa, 'ACO': d_aco, 'PSO': d_pso, 'Tabu': d_tabu, 'GA': d_ga},
        'histories': {'SA': sa_h, 'ACO': aco_h, 'PSO': pso_h, 'Tabu': tabu_h, 'GA': ga_h, 'NN_val': d_nn}
    }
from concurrent.futures import ProcessPoolExecutor
if __name__ == "__main__":
    seeds = [42, 100, 500,876543,345,543,9382734,35678,2357,68762,36587,1234,13468,1234,777,666,555,444,33,2222]
    final_stats = {alg: [] for alg in ['NN', 'SA', 'ACO', 'PSO', 'Tabu', 'GA']}
    
    print(f"Starting Multiprocessing on {len(seeds)} seeds...")
    
    with ProcessPoolExecutor() as executor:
        results_list = list(executor.map(run_single_seed, seeds))

    for res in results_list:
        for alg in final_stats:
            final_stats[alg].append(res['dists'][alg])

    print(f"\n{'Algorithm':<10} | {'Mean Dist':<12} | {'Std Dev':<10}")
    print("-" * 40)
    for alg, data in final_stats.items():
        print(f"{alg:<10} | {np.mean(data):<12.2f} | {np.std(data):<10.2f}")

    last_h = results_list[-1]['histories']
    plt.figure(figsize=(10,6))
    plt.plot(last_h['SA'], label='SA')
    plt.plot(last_h['ACO'], label='ACO')
    plt.plot(last_h['PSO'], label='PSO')
    plt.plot(last_h['Tabu'], label='Tabu')
    plt.plot(last_h['GA'], label='GA (Hybrid)')
    plt.axhline(y=last_h['NN_val'], color='black', linestyle='--', label='Greedy NN')
    plt.title("Algorithm Convergence Comparison (Parallel Run)")
    plt.xlabel("Iteration / Step")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.boxplot([final_stats[alg] for alg in final_stats], labels=final_stats.keys())
    plt.title("Distribution of Final Distances across 20 Seeds")
    plt.ylabel("Total Distance")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    perform_analysis(final_stats)

