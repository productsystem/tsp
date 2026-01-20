import random
import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NUM_CITIES = 50
NUM_PARTICLES = 100
ITERATIONS = 5000

W = 0.7
C1 = 1.5
C2 = 1.5

SEED = 42
LOG_FILE = "tsp_pso_log.csv"

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


def decode_position(position):
    return list(np.argsort(position))


positions = np.random.rand(NUM_PARTICLES, NUM_CITIES)
velocities = np.random.randn(NUM_PARTICLES, NUM_CITIES) * 0.1

pbest_positions = positions.copy()
pbest_scores = np.array([total_distance(decode_position(p), cities) for p in positions])

gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index].copy()
gbest_score = pbest_scores[gbest_index]

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Best Distance", "Average Distance"])

for it in range(ITERATIONS):
    scores = []

    for i in range(NUM_PARTICLES):
        tour = decode_position(positions[i])
        score = total_distance(tour, cities)
        scores.append(score)

        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = positions[i].copy()

            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i].copy()

    avg_score = sum(scores)/len(scores)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([it, gbest_score, avg_score])

    for i in range(NUM_PARTICLES):
        r1, r2 = np.random.rand(NUM_CITIES), np.random.rand(NUM_CITIES)

        velocities[i] = (W * velocities[i]
                         + C1 * r1 * (pbest_positions[i] - positions[i])
                         + C2 * r2 * (gbest_position - positions[i]))

        positions[i] += velocities[i]

    if it % 50 == 0:
        print(f"Iter {it:4d} | Best = {gbest_score:.2f}")

print("\nBest tour distance found:", gbest_score)


df = pd.read_csv(LOG_FILE)
plt.figure(figsize=(10,6))
plt.plot(df["Iteration"], df["Best Distance"], label="Best")
plt.plot(df["Iteration"], df["Average Distance"], label="Average")
plt.title("PSO for TSP Convergence")
plt.xlabel("Iteration")
plt.ylabel("Tour Distance")
plt.legend()
plt.grid()
plt.show()

best_tour = decode_position(gbest_position)

xs = [cities[i][0] for i in best_tour] + [cities[best_tour[0]][0]]
ys = [cities[i][1] for i in best_tour] + [cities[best_tour[0]][1]]

plt.figure(figsize=(7,7))
plt.plot(xs, ys, marker="o")
plt.title("Best TSP Tour (PSO)")
plt.grid()
plt.show()
