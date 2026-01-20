import random
import math
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NUM_CITIES = 50
POP_SIZE = 200
GENERATIONS = 1000
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 7
ELITISM = True
SEED = 42

LOG_FILE = "tsp_ga_log.csv"

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


def create_population(size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size)]

def tournament_selection(population, fitnesses):
    selected = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    selected.sort(key=lambda x: x[1])
    return selected[0][0][:]

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    
    child = [-1]*size
    child[a:b] = parent1[a:b]

    ptr = b
    for city in parent2:
        if city not in child:
            if ptr >= size:
                ptr = 0
            child[ptr] = city
            ptr += 1

    return child

def inversion_mutation(tour):
    a, b = sorted(random.sample(range(len(tour)), 2))
    tour[a:b] = reversed(tour[a:b])


population = create_population(POP_SIZE, NUM_CITIES)

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Generation", "Best Distance", "Average Distance", "Gen Time (s)", "Cumulative Time (s)"])

best_overall = None
best_overall_dist = float("inf")

start_time = time.time()

for gen in range(GENERATIONS):
    gen_start = time.time()

    fitnesses = [total_distance(tour, cities) for tour in population]

    best_gen_dist = min(fitnesses)
    avg_gen_dist = sum(fitnesses)/len(fitnesses)

    if best_gen_dist < best_overall_dist:
        best_overall_dist = best_gen_dist
        best_overall = population[fitnesses.index(best_gen_dist)]

    new_population = []
    if ELITISM:
        elite = population[fitnesses.index(best_gen_dist)]
        new_population.append(elite[:])

    while len(new_population) < POP_SIZE:
        p1 = tournament_selection(population, fitnesses)
        p2 = tournament_selection(population, fitnesses)

        child = ordered_crossover(p1, p2)

        if random.random() < MUTATION_RATE:
            inversion_mutation(child)

        new_population.append(child)

    population = new_population

    gen_time = time.time() - gen_start
    cumulative_time = time.time() - start_time

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gen, best_gen_dist, avg_gen_dist, gen_time, cumulative_time])

    if gen % 50 == 0:
        print(f"Gen {gen:4d} | Best = {best_gen_dist:.2f} | Avg = {avg_gen_dist:.2f} | Time = {cumulative_time:.2f}s")

total_time = time.time() - start_time
print("\nBest tour distance found:", best_overall_dist)
print("Total runtime:", total_time, "seconds")


df = pd.read_csv(LOG_FILE)

plt.figure(figsize=(10,6))
plt.plot(df["Generation"], df["Best Distance"], label="Best")
plt.plot(df["Generation"], df["Average Distance"], label="Average")
plt.xlabel("Generation")
plt.ylabel("Tour Distance")
plt.title("GA for TSP Convergence")
plt.legend()
plt.grid()
plt.show()


def plot_tour(tour, cities):
    xs = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    ys = [cities[i][1] for i in tour] + [cities[tour[0]][1]]

    plt.figure(figsize=(7,7))
    plt.plot(xs, ys, marker="o")
    plt.title("Best TSP Tour Found (GA)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

plot_tour(best_overall, cities)
