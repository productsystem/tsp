import random
import math
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

NUM_CITIES = 50
ITERATIONS = 10000

T0 = 1000
ALPHA = 0.999

SEED = 42
LOG_FILE = "tsp_sa_log.csv"

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
current_dist = total_distance(current, cities)

best = current[:]
best_dist = current_dist

T = T0


with open(LOG_FILE,"w",newline="") as f:
    csv.writer(f).writerow(["Iteration","Best Distance","Current Distance","Iter Time (s)","Cumulative Time (s)"])

start_time = time.time()

for it in range(ITERATIONS):
    iter_start = time.time()

    i,j = sorted(random.sample(range(NUM_CITIES),2))
    neighbor = current[:]
    neighbor[i:j] = reversed(neighbor[i:j])
    neighbor_dist = total_distance(neighbor,cities)

    if neighbor_dist < current_dist or random.random() < math.exp((current_dist-neighbor_dist)/T):
        current = neighbor
        current_dist = neighbor_dist

    if current_dist < best_dist:
        best = current[:]
        best_dist = current_dist

    iter_time = time.time() - iter_start
    cumulative_time = time.time() - start_time

    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([it,best_dist,current_dist,iter_time,cumulative_time])

    T *= ALPHA

    if it % 5000 == 0:
        print(f"Iter {it:6d} | Best = {best_dist:.2f} | Time = {cumulative_time:.2f}s")

total_time = time.time() - start_time
print("\nBest distance:", best_dist)
print("Total runtime:", total_time, "seconds")


df = pd.read_csv(LOG_FILE)

plt.figure(figsize=(10,6))
plt.plot(df["Iteration"], df["Best Distance"], label="Best")
plt.plot(df["Iteration"], df["Current Distance"], label="Current", alpha=0.6)
plt.title("Simulated Annealing for TSP Convergence")
plt.xlabel("Iteration")
plt.ylabel("Tour Distance")
plt.legend()
plt.grid()
plt.show()


xs = [cities[i][0] for i in best] + [cities[best[0]][0]]
ys = [cities[i][1] for i in best] + [cities[best[0]][1]]

plt.figure(figsize=(7,7))
plt.plot(xs, ys, marker="o")
plt.title("Best TSP Tour Found (SA)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()
