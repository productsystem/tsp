import random
import math
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

NUM_CITIES = 50
ITERATIONS = 1000
TABU_TENURE = 30
NEIGHBORHOOD_SIZE = 100

SEED = 42
LOG_FILE = "tsp_tabu_log.csv"

random.seed(SEED)


def generate_cities(n):
    return [(random.uniform(0,1000), random.uniform(0,1000)) for _ in range(n)]

def distance(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def total_distance(tour,cities):
    return sum(distance(cities[tour[i]], cities[tour[(i+1)%len(tour)]])
               for i in range(len(tour)))

cities = generate_cities(NUM_CITIES)
current = list(range(NUM_CITIES))
random.shuffle(current)

best = current[:]
best_dist = total_distance(best, cities)

tabu_list = []


with open(LOG_FILE, "w", newline="") as f:
    csv.writer(f).writerow(["Iteration","Best Distance","Current Distance","Iter Time (s)","Cumulative Time (s)"])

start_time = time.time()

for it in range(ITERATIONS):
    iter_start = time.time()
    neighbors = []

    for _ in range(NEIGHBORHOOD_SIZE):
        i, j = sorted(random.sample(range(NUM_CITIES),2))
        neighbor = current[:]
        neighbor[i:j] = reversed(neighbor[i:j])
        move = (i,j)
        neighbors.append((neighbor, move))

    best_candidate = None
    best_candidate_dist = float("inf")
    best_move = None

    for cand, move in neighbors:
        dist = total_distance(cand, cities)
        if (move not in tabu_list or dist < best_dist) and dist < best_candidate_dist:
            best_candidate = cand
            best_candidate_dist = dist
            best_move = move

    current = best_candidate

    if best_candidate_dist < best_dist:
        best = best_candidate
        best_dist = best_candidate_dist

    tabu_list.append(best_move)
    if len(tabu_list) > TABU_TENURE:
        tabu_list.pop(0)

    iter_time = time.time() - iter_start
    cumulative_time = time.time() - start_time

    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([it, best_dist, best_candidate_dist, iter_time, cumulative_time])

    if it % 50 == 0:
        print(f"Iter {it:4d} | Best = {best_dist:.2f} | Time = {cumulative_time:.2f}s")

total_time = time.time() - start_time
print("\nBest distance:", best_dist)
print("Total runtime:", total_time, "seconds")


df = pd.read_csv(LOG_FILE)

plt.figure(figsize=(10,6))
plt.plot(df["Iteration"], df["Best Distance"], label="Best")
plt.plot(df["Iteration"], df["Current Distance"], label="Current", alpha=0.6)
plt.title("Tabu Search Convergence")
plt.xlabel("Iteration")
plt.ylabel("Tour Distance")
plt.legend()
plt.grid()
plt.show()
