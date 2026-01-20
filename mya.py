import random, math, numpy as np
import matplotlib.pyplot as plt
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import time, sys

def log(msg):
    print(msg)
    sys.stdout.flush()

def timed(label):
    def decorator(func):
        def wrapper(*args, **kwargs):
            log(f"[{label}] Started")
            t0 = time.time()
            res = func(*args, **kwargs)
            log(f"[{label}] Finished in {time.time()-t0:.2f}s\n")
            return res
        return wrapper
    return decorator


NUM_CITIES = 50
WIDTH, HEIGHT = 1000, 1000

T_START, T_MIN, ALPHA_SA, ITERATIONS_PER_T = 1000.0, 1e-4, 0.995, 80
ALPHA_ACO, BETA_ACO, EVAPORATION, NUM_ANTS, NUM_ITERATIONS_ACO = 1.0, 2.0, 0.3, 50, 150
NUM_PARTICLES_PSO, NUM_ITERS_PSO, C1_PSO, C2_PSO = 40, 300, 1.5, 1.5
TABU_MAX_ITERS, TABU_TENURE, TABU_NEIGHBORHOOD = 500, 30, 150
POP_SIZE_GA, NUM_GENERATIONS_GA, MUTATION_RATE_GA = 100, 500, 0.05


def build_distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(cities[i][0]-cities[j][0], cities[i][1]-cities[j][1])
            dist[i,j] = dist[j,i] = d
    return dist

def total_distance(tour, dist):
    s = 0.0
    for i in range(len(tour)):
        s += dist[tour[i]][tour[(i+1)%len(tour)]]
    return s

def random_tour(n):
    t = list(range(n))
    random.shuffle(t)
    return t


def two_opt_local_search(tour, dist):
    n = len(tour)
    best = tour[:]
    best_cost = total_distance(best, dist)
    improved = True

    while improved:
        improved = False
        for i in range(1, n-2):
            a, b = best[i-1], best[i]
            for j in range(i+1, n):
                c, d = best[j-1], best[j % n]
                delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                if delta < -1e-9:
                    best[i:j] = reversed(best[i:j])
                    best_cost += delta
                    improved = True
    return best

def two_opt_swap(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    return tour[:i] + tour[i:j][::-1] + tour[j:]

def best_nearest_neighbor(dist):
    n = len(dist)
    best_cost = float("inf")
    best_tour = None

    for start in range(n):
        unvisited = set(range(n))
        tour = [start]
        unvisited.remove(start)

        while unvisited:
            u = tour[-1]
            v = min(unvisited, key=lambda x: dist[u][x])
            tour.append(v)
            unvisited.remove(v)

        cost = total_distance(tour, dist)
        if cost < best_cost:
            best_cost, best_tour = cost, tour[:]

    return best_tour, best_cost


@timed("SA")
def simulated_annealing(dist):
    n = len(dist)
    current = random_tour(n)
    current_cost = total_distance(current, dist)
    best, best_cost = current[:], current_cost
    T = T_START
    history = []
    step = 0

    while T > T_MIN:
        for _ in range(ITERATIONS_PER_T):
            candidate = two_opt_swap(current)
            cost = total_distance(candidate, dist)
            delta = cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current, current_cost = candidate, cost
                if cost < best_cost:
                    best, best_cost = candidate[:], cost
        history.append(best_cost)
        T *= ALPHA_SA
        step += 1
        if step % 200 == 0:
            log(f"[SA] Temp step {step} | Best = {best_cost:.2f}")

    return best, best_cost, history


@timed("GA")
def genetic_algorithm(dist):
    n = len(dist)
    pop = [random_tour(n) for _ in range(POP_SIZE_GA)]
    fitness = [total_distance(t, dist) for t in pop]
    history = []

    for gen in range(NUM_GENERATIONS_GA):
        pop_fit = sorted(zip(pop, fitness), key=lambda x: x[1])
        pop, fitness = zip(*pop_fit)
        pop, fitness = list(pop), list(fitness)
        history.append(fitness[0])

        if gen % 50 == 0:
            log(f"[GA] Gen {gen}/{NUM_GENERATIONS_GA} | Best = {fitness[0]:.2f}")

        new_pop = pop[:2]
        new_fit = fitness[:2]

        while len(new_pop) < POP_SIZE_GA:
            p1 = min(random.sample(list(zip(pop, fitness)), 5), key=lambda x: x[1])[0]
            p2 = min(random.sample(list(zip(pop, fitness)), 5), key=lambda x: x[1])[0]
            a, b = sorted(random.sample(range(n), 2))
            child = [-1]*n
            child[a:b] = p1[a:b]
            fill = [x for x in p2 if x not in child]
            idx = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = fill[idx]
                    idx += 1
            if random.random() < MUTATION_RATE_GA:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]
            child = two_opt_local_search(child, dist)
            f = total_distance(child, dist)
            new_pop.append(child)
            new_fit.append(f)

        pop, fitness = new_pop, new_fit

    return pop[0], fitness[0], history


def get_swap_sequence(src, tgt):
    s = src[:]
    pos = {c:i for i,c in enumerate(s)}
    swaps = []
    for i in range(len(s)):
        if s[i] != tgt[i]:
            j = pos[tgt[i]]
            swaps.append((i,j))
            pos[s[i]], pos[s[j]] = j,i
            s[i],s[j] = s[j],s[i]
    return swaps

def apply_swaps(tour, swaps, prob):
    t = tour[:]
    for i,j in swaps:
        if random.random() < prob:
            t[i],t[j] = t[j],t[i]
    return t

@timed("PSO")
def particle_swarm_optimization(dist):
    n = len(dist)
    particles = [random_tour(n) for _ in range(NUM_PARTICLES_PSO)]
    pbest = particles[:]
    pbest_cost = [total_distance(p, dist) for p in particles]
    gbest = min(pbest, key=lambda t: total_distance(t, dist))
    gbest_cost = total_distance(gbest, dist)
    history = []

    for it in range(NUM_ITERS_PSO):
        for i in range(NUM_PARTICLES_PSO):
            s1 = get_swap_sequence(particles[i], pbest[i])
            s2 = get_swap_sequence(particles[i], gbest)
            new = apply_swaps(particles[i], s1, C1_PSO/(C1_PSO+C2_PSO))
            new = apply_swaps(new, s2, C2_PSO/(C1_PSO+C2_PSO))
            new = two_opt_local_search(new, dist)
            cost = total_distance(new, dist)
            particles[i] = new
            if cost < pbest_cost[i]:
                pbest[i], pbest_cost[i] = new[:], cost
                if cost < gbest_cost:
                    gbest, gbest_cost = new[:], cost
        history.append(gbest_cost)
        if it % 30 == 0:
            log(f"[PSO] Iter {it}/{NUM_ITERS_PSO} | Best = {gbest_cost:.2f}")

    return gbest, gbest_cost, history

@timed("Tabu")
def tabu_search(dist):
    n = len(dist)
    current, current_cost = best_nearest_neighbor(dist)
    best, best_cost = current[:], current_cost
    tabu = deque(maxlen=TABU_TENURE)
    history = []

    for it in range(TABU_MAX_ITERS):
        best_candidate = None
        best_candidate_cost = float("inf")
        best_move = None

        for _ in range(TABU_NEIGHBORHOOD):
            i, j = sorted(random.sample(range(n), 2))
            cand = current[:i] + current[i:j][::-1] + current[j:]
            cost = total_distance(cand, dist)
            if (i,j) not in tabu or cost < best_cost:
                if cost < best_candidate_cost:
                    best_candidate, best_candidate_cost, best_move = cand, cost, (i,j)

        if best_candidate:
            current, current_cost = best_candidate, best_candidate_cost
            tabu.append(best_move)
            if current_cost < best_cost:
                best, best_cost = current[:], current_cost

        history.append(best_cost)
        if it % 50 == 0:
            log(f"[Tabu] Iter {it}/{TABU_MAX_ITERS} | Best = {best_cost:.2f}")

    return best, best_cost, history

@timed("ACO")
def ant_colony_optimization(dist):
    n = len(dist)
    pher = np.ones((n,n))
    best, best_cost = None, float("inf")
    history = []

    for it in range(NUM_ITERATIONS_ACO):
        tours, costs = [], []
        for _ in range(NUM_ANTS):
            curr = random.randint(0,n-1)
            tour = [curr]
            unvisited = set(range(n))
            unvisited.remove(curr)

            while unvisited:
                u = tour[-1]
                probs = []
                total = 0
                for v in unvisited:
                    val = (pher[u][v]**ALPHA_ACO)*((1/dist[u][v])**BETA_ACO)
                    probs.append((v,val))
                    total += val
                r = random.random()
                acc = 0
                for v,val in probs:
                    acc += val/total
                    if acc >= r:
                        tour.append(v)
                        unvisited.remove(v)
                        break

            tour = two_opt_local_search(tour, dist)
            cost = total_distance(tour, dist)
            tours.append(tour)
            costs.append(cost)
            if cost < best_cost:
                best, best_cost = tour[:], cost

        pher *= (1-EVAPORATION)
        for t,c in zip(tours,costs):
            for i in range(n):
                pher[t[i]][t[(i+1)%n]] += 1/c

        history.append(best_cost)
        if it % 15 == 0:
            log(f"[ACO] Iter {it}/{NUM_ITERATIONS_ACO} | Best = {best_cost:.2f}")

    return best, best_cost, history

def run_single_seed(seed):
    log(f"\n[SEED {seed}] Starting")
    random.seed(seed)
    cities = [(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NUM_CITIES)]
    dist = build_distance_matrix(cities)

    _, d_nn = best_nearest_neighbor(dist)
    log(f"[SEED {seed}] NN done: {d_nn:.2f}")

    _, d_sa, sa_h = simulated_annealing(dist)
    # _, d_aco, aco_h = ant_colony_optimization(dist)
    # _, d_pso, pso_h = particle_swarm_optimization(dist)
    _, d_tabu, tabu_h = tabu_search(dist)
    _, d_ga, ga_h = genetic_algorithm(dist)

    log(f"[SEED {seed}] Completed\n")

    return {
        'dists': {'NN': d_nn, 'SA': d_sa, 'ACO': d_aco, 'PSO': d_pso, 'Tabu': d_tabu, 'GA': d_ga},
        'histories': {'SA': sa_h, 'ACO': aco_h, 'PSO': pso_h, 'Tabu': tabu_h, 'GA': ga_h, 'NN_val': d_nn}
    }

if __name__ == "__main__":
    seeds = [42,100,500,876543,345,543,9382734,35678,2357,68762]
    final_stats = {alg: [] for alg in ['NN','SA','ACO','PSO','Tabu','GA']}

    print("Running fast benchmark with logging...")
    with ProcessPoolExecutor(max_workers=4) as exe:
        results = list(exe.map(run_single_seed, seeds))

    for res in results:
        for alg in final_stats:
            final_stats[alg].append(res['dists'][alg])

    print("\nAlgorithm | Mean Distance | Std Dev")
    print("-"*40)
    for alg,data in final_stats.items():
        print(f"{alg:<8} | {np.mean(data):<13.2f} | {np.std(data):<8.2f}")

    last = results[-1]['histories']
    plt.figure(figsize=(10,6))
    for alg in ['SA','PSO','Tabu','GA']:
        plt.plot(last[alg], label=alg)
    plt.axhline(y=last['NN_val'], linestyle='--', label='NN')
    plt.legend(); plt.grid(); plt.show()
