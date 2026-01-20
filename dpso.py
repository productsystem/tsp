import random
import math
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt


NUM_CITIES = 50
NUM_PARTICLES = 50
ITERATIONS = 1000

W = 0.5
C1 = 1.5
C2 = 1.5

SEED = 42
LOG_FILE = "tsp_dpso_hybrid_fast_log.csv"

random.seed(SEED)


def generate_cities(n):
    return [(random.uniform(0,1000), random.uniform(0,1000)) for _ in range(n)]

def distance(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def total_distance(tour,cities):
    return sum(distance(cities[tour[i]], cities[tour[(i+1)%len(tour)]])
               for i in range(len(tour)))

cities = generate_cities(NUM_CITIES)


def two_opt_fast(tour, cities, max_swaps=30):
    best = tour[:]
    best_dist = total_distance(best, cities)
    n = len(tour)

    for _ in range(max_swaps):
        i, j = sorted(random.sample(range(n), 2))
        if j - i == 1:
            continue
        new = best[:]
        new[i:j] = reversed(new[i:j])
        new_dist = total_distance(new, cities)
        if new_dist < best_dist:
            best = new
            best_dist = new_dist
    return best

def get_swap_sequence(src, target):
    src = src[:]
    swaps = []
    pos = {city:i for i,city in enumerate(src)}

    for i in range(len(src)):
        if src[i] != target[i]:
            j = pos[target[i]]
            swaps.append((i,j))
            src[i], src[j] = src[j], src[i]
            pos[src[j]] = j
            pos[src[i]] = i
    return swaps

def apply_swaps(tour, swaps, prob=1.0):
    tour = tour[:]
    for i,j in swaps:
        if random.random() < prob:
            tour[i], tour[j] = tour[j], tour[i]
    return tour


particles = []
velocities = []
pbest = []
pbest_scores = []

for _ in range(NUM_PARTICLES):
    tour = list(range(NUM_CITIES))
    random.shuffle(tour)
    tour = two_opt_fast(tour, cities)
    particles.append(tour)
    velocities.append([])
    pbest.append(tour[:])
    pbest_scores.append(total_distance(tour, cities))

gbest = pbest[pbest_scores.index(min(pbest_scores))]
gbest_score = min(pbest_scores)

with open(LOG_FILE,"w",newline="") as f:
    csv.writer(f).writerow(["Iteration","Best Distance","Average Distance","Iter Time (s)","Cumulative Time (s)"])

start_time = time.time()

for it in range(ITERATIONS):
    iter_start = time.time()
    scores = []

    for i in range(NUM_PARTICLES):
        tour = particles[i]
        if random.random() < 0.3:
            tour = two_opt_fast(tour, cities)
            particles[i] = tour

        score = total_distance(tour, cities)
        scores.append(score)

        if score < pbest_scores[i]:
            pbest[i] = tour[:]
            pbest_scores[i] = score

            if score < gbest_score:
                gbest = tour[:]
                gbest_score = score

    avg_score = sum(scores)/len(scores)

    gbest = two_opt_fast(gbest, cities, max_swaps=20)
    gbest_score = total_distance(gbest, cities)

    for i in range(NUM_PARTICLES):
        swaps_pbest = get_swap_sequence(particles[i], pbest[i])
        swaps_gbest = get_swap_sequence(particles[i], gbest)

        new_velocity = []

        for swap in velocities[i]:
            if random.random() < W:
                new_velocity.append(swap)

        for swap in swaps_pbest:
            if random.random() < C1 / max(len(swaps_pbest),1):
                new_velocity.append(swap)

        for swap in swaps_gbest:
            if random.random() < C2 / max(len(swaps_gbest),1):
                new_velocity.append(swap)

        velocities[i] = new_velocity
        particles[i] = apply_swaps(particles[i], velocities[i], prob=1.0)

    iter_time = time.time() - iter_start
    cumulative_time = time.time() - start_time

    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([it, gbest_score, avg_score, iter_time, cumulative_time])

    if it % 10 == 0:
        print(f"Iter {it:3d} | Best = {gbest_score:.2f} | Time = {cumulative_time:.2f}s")

total_time = time.time() - start_time
print("\nBest tour distance found:", gbest_score)
print("Total runtime:", total_time, "seconds")


df = pd.read_csv(LOG_FILE)

plt.figure(figsize=(10,6))
plt.plot(df["Iteration"], df["Best Distance"], label="Best")
plt.plot(df["Iteration"], df["Average Distance"], label="Average")
plt.title("Optimized Hybrid DPSO + Fast 2-opt")
plt.xlabel("Iteration")
plt.ylabel("Tour Distance")
plt.legend()
plt.grid()
plt.show()

xs = [cities[i][0] for i in gbest] + [cities[gbest[0]][0]]
ys = [cities[i][1] for i in gbest] + [cities[gbest[0]][1]]

plt.figure(figsize=(7,7))
plt.plot(xs, ys, marker="o")
plt.title("Best TSP Tour (Optimized Hybrid DPSO)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()
