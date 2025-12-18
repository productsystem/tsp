import random
import math

NUM_CITIES = 30
WIDTH = 1000
HEIGHT = 1000

T_START = 1000.0
T_MIN = 1e-4
ALPHA = 0.995
ITERATIONS_PER_T = 100

NUM_ANTS = 30
NUM_ITERATIONS = 100
ALPHA_ACO = 1.0 
BETA_ACO = 5.0
EVAPORATION = 0.5
Q = 100

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def total_distance(tour, cities):
    total = 0.0
    number_of_cities = len(tour)
    for i in range(number_of_cities):
        current_city_index = tour[i]
        next_city_index = tour[(i + 1) % number_of_cities]
        current_city = cities[current_city_index]
        next_city = cities[next_city_index]
        segment_distance = distance(current_city, next_city)
        total += segment_distance
    return total


def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

def two_opt_swap(tour):
    new_tour = tour[:]
    number_of_cities = len(new_tour)
    index1 = random.randint(0, number_of_cities - 1)
    index2 = random.randint(0, number_of_cities - 1)
    if index1 > index2:
        index1, index2 = index2, index1
    segment_to_reverse = new_tour[index1:index2]
    segment_to_reverse.reverse()
    new_tour[index1:index2] = segment_to_reverse
    return new_tour

def simulated_annealing(cities):
    n = len(cities)
    current = random_tour(n)
    best = current[:]

    current_cost = total_distance(current, cities)
    best_cost = current_cost
    T = T_START

    sa_best_history = []
    sa_temp_history = []

    while T > T_MIN:
        for _ in range(ITERATIONS_PER_T):
            candidate = two_opt_swap(current)
            candidate_cost = total_distance(candidate, cities)
            delta = candidate_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / T):
                current = candidate
                current_cost = candidate_cost
                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

        sa_best_history.append(best_cost)
        sa_temp_history.append(T)
        T *= ALPHA

    return best, best_cost, sa_best_history, sa_temp_history


def ant_colony_optimization(cities):
    n = len(cities)

    dist = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(distance(cities[i], cities[j]))
        dist.append(row)

    pheromone = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(1.0)
        pheromone.append(row)

    best_tour = None
    best_cost = float("inf")
    best_history = []

    for _ in range(NUM_ITERATIONS):
        all_tours = []
        all_costs = []

        for _ in range(NUM_ANTS):
            start = random.randint(0, n - 1)
            tour = [start]

            unvisited = set()
            for i in range(n):
                unvisited.add(i)
            unvisited.remove(start)

            while unvisited:
                u = tour[-1]

                values = []
                total = 0.0

                for v in unvisited:
                    val = (1.0 / dist[u][v]) ** ALPHA_ACO
                    val = val * (pheromone[u][v]) ** BETA_ACO
                    values.append((v, val))
                    total += val

                r = random.random()
                acc = 0.0

                for v, val in values:
                    acc += val / total
                    if acc >= r:
                        tour.append(v)
                        unvisited.remove(v)
                        break

            cost = total_distance(tour, cities)
            all_tours.append(tour)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_tour = tour[:]

        for i in range(n):
            for j in range(n):
                pheromone[i][j] = pheromone[i][j] * (1 - EVAPORATION)

        for tour, cost in zip(all_tours, all_costs):
            add = 1.0 / cost
            for i in range(n):
                a = tour[i]
                b = tour[(i + 1) % n]
                pheromone[a][b] += add
                pheromone[b][a] += add

        best_history.append(best_cost)

    return best_tour, best_cost, best_history


import matplotlib.pyplot as plt

if __name__ == "__main__":
    random.seed(100)

    cities = [(random.randint(0, WIDTH), random.randint(0, HEIGHT))
              for _ in range(NUM_CITIES)]

    sa_tour, sa_dist, sa_best_hist, sa_temp_hist = simulated_annealing(cities)
    aco_tour, aco_dist, aco_best_hist = ant_colony_optimization(cities)

    plt.figure()
    plt.plot(sa_best_hist)
    plt.xlabel("temp step")
    plt.ylabel("best distance")
    plt.title("SA Converge")
    plt.show()

    plt.figure()
    plt.plot(sa_temp_hist)
    plt.xlabel("temp step")
    plt.ylabel("temp")
    plt.title("SA Cooling")
    plt.show()

    plt.figure()
    plt.plot(aco_best_hist)
    plt.xlabel("Iteration")
    plt.ylabel("best distance")
    plt.title("ACO Converge")
    plt.show()

    plt.figure()
    plt.plot(sa_best_hist, label="SA")
    plt.plot(
        [aco_best_hist[min(i, len(aco_best_hist)-1)] for i in range(len(sa_best_hist))],
        label="ACO"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.title("SA vs ACO Performance Comparison")
    plt.legend()
    plt.show()
    print("sa distance:", round(sa_dist, 2))
    print("aco distance:", round(aco_dist, 2))
