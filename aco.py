import random
import math
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NUM_CITIES = 50
NUM_ANTS = 50
ITERATIONS = 1000

ALPHA = 1.0 
BETA = 5.0
EVAPORATION = 0.5
Q = 100

SEED = 42
LOG_FILE = "tsp_aco_log.csv"

random.seed(SEED)
np.random.seed(SEED)


def generate_cities(n):
    return [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(n)]

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def total_distance(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[(i+1)%len(tour)]])
               for i in range(len(tour)))

cities = generate_cities(NUM_CITIES)


dist_matrix = np.zeros((NUM_CITIES, NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        dist_matrix[i][j] = distance(cities[i], cities[j])

heuristic = 1.0 / (dist_matrix + 1e-10)


pheromones = np.ones((NUM_CITIES, NUM_CITIES))


def select_next_city(current, unvisited):
    probs = []
    denom = 0.0

    for j in unvisited:
        tau = pheromones[current][j] ** ALPHA
        eta = heuristic[current][j] ** BETA
        val = tau * eta
        probs.append(val)
        denom += val

    probs = [p/denom for p in probs]
    return random.choices(unvisited, weights=probs, k=1)[0]

def build_tour():
    start = random.randint(0, NUM_CITIES-1)
    tour = [start]
    unvisited = set(range(NUM_CITIES))
    unvisited.remove(start)

    while unvisited:
        next_city = select_next_city(tour[-1], list(unvisited))
        tour.append(next_city)
        unvisited.remove(next_city)

    return tour

def update_pheromones(all_tours, all_lengths):
    global pheromones

    pheromones *= (1 - EVAPORATION)

    for tour, length in zip(all_tours, all_lengths):
        deposit = Q / length
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i+1)%len(tour)]
            pheromones[a][b] += deposit
            pheromones[b][a] += deposit


best_overall = None
best_overall_dist = float("inf")

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Best Distance", "Average Distance", "Iter Time (s)", "Cumulative Time (s)"])

start_time = time.time()

for it in range(ITERATIONS):
    iter_start = time.time()

    all_tours = []
    all_lengths = []

    for _ in range(NUM_ANTS):
        tour = build_tour()
        length = total_distance(tour, cities)

        all_tours.append(tour)
        all_lengths.append(length)

        if length < best_overall_dist:
            best_overall_dist = length
            best_overall = tour[:]

    avg_len = sum(all_lengths)/len(all_lengths)
    best_iter = min(all_lengths)

    update_pheromones(all_tours, all_lengths)

    iter_time = time.time() - iter_start
    cumulative_time = time.time() - start_time

    # Log
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([it, best_iter, avg_len, iter_time, cumulative_time])

    if it % 50 == 0:
        print(f"Iter {it:4d} | Best = {best_iter:.2f} | Time = {cumulative_time:.2f}s")

total_time = time.time() - start_time
print("\nBest tour distance found:", best_overall_dist)
print("Total runtime:", total_time, "seconds")


df = pd.read_csv(LOG_FILE)

plt.figure(figsize=(10,6))
plt.plot(df["Iteration"], df["Best Distance"], label="Best")
plt.plot(df["Iteration"], df["Average Distance"], label="Average")
plt.xlabel("Iteration")
plt.ylabel("Tour Distance")
plt.title("ACO for TSP Convergence")
plt.legend()
plt.grid()
plt.show()


def plot_tour(tour, cities):
    xs = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    ys = [cities[i][1] for i in tour] + [cities[tour[0]][1]]

    plt.figure(figsize=(7,7))
    plt.plot(xs, ys, marker="o")
    plt.title("Best TSP Tour Found (ACO)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

plot_tour(best_overall, cities)
