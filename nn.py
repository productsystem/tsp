import random
import math
import csv
import time
import matplotlib.pyplot as plt

NUM_CITIES = 50
SEED = 42
LOG_FILE = "tsp_nn_log.csv"

random.seed(SEED)

def generate_cities(n):
    return [(random.uniform(0,1000), random.uniform(0,1000)) for _ in range(n)]

def distance(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

cities = generate_cities(NUM_CITIES)

visited = [0]
unvisited = set(range(1,NUM_CITIES))


with open(LOG_FILE,"w",newline="") as f:
    csv.writer(f).writerow(["Step","Current Distance","Step Time (s)","Cumulative Time (s)"])

total_dist = 0
start_time = time.time()

while unvisited:
    step_start = time.time()

    last = visited[-1]
    next_city = min(unvisited, key=lambda c: distance(cities[last],cities[c]))
    step_dist = distance(cities[last], cities[next_city])

    total_dist += step_dist
    visited.append(next_city)
    unvisited.remove(next_city)

    step_time = time.time() - step_start
    cumulative_time = time.time() - start_time

    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([len(visited)-1, total_dist, step_time, cumulative_time])

total_dist += distance(cities[visited[-1]], cities[visited[0]])

final_time = time.time() - start_time
print("NN Distance:", total_dist)
print("Total runtime:", final_time, "seconds")


xs = [cities[i][0] for i in visited] + [cities[visited[0]][0]]
ys = [cities[i][1] for i in visited] + [cities[visited[0]][1]]

plt.figure(figsize=(7,7))
plt.plot(xs,ys,marker='o')
plt.title("Nearest Neighbor TSP Tour")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()
