import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


files = {
    "GA": "tsp_ga_log.csv",
    "ACO": "tsp_aco_log.csv",
    "PSO": "tsp_dpso_hybrid_fast_log.csv",
    "SA": "tsp_sa_log.csv",
    "Tabu": "tsp_tabu_log.csv"
}

EPSILON = 0.05
OUTPUT_DIR = "benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


data = {}
for name, file in files.items():
    if os.path.exists(file):
        df = pd.read_csv(file)
        data[name] = df


plt.figure(figsize=(12,7))
for name, df in data.items():
    plt.plot(df.iloc[:,0], df["Best Distance"], label=name)

plt.xlabel("Iteration")
plt.ylabel("Best Distance")
plt.title("Iteration-based Convergence Comparison")
plt.legend()
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/iteration_convergence.png", dpi=300)
plt.show()


plt.figure(figsize=(12,7))
for name, df in data.items():
    plt.plot(df["Cumulative Time (s)"], df["Best Distance"], label=name)

plt.xlabel("Time (s)")
plt.ylabel("Best Distance")
plt.title("Time-based Convergence Comparison")
plt.legend()
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/time_convergence.png", dpi=300)
plt.show()


times_to_eps = {}

plt.figure(figsize=(12,7))

for name, df in data.items():
    best_i = df["Best Distance"].min()
    threshold_i = best_i * (1 + EPSILON)

    reached = df[df["Best Distance"] <= threshold_i]

    if not reached.empty:
        t_eps = reached["Cumulative Time (s)"].iloc[0]
        times_to_eps[name] = t_eps
        plt.scatter(t_eps, threshold_i, label=f"{name}: {t_eps:.2f}s")
    else:
        times_to_eps[name] = np.nan

plt.title("Time to Reach ε-Convergence (Relative to Own Best)")
plt.xlabel("Time (s)")
plt.ylabel("Distance")
plt.legend()
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/epsilon_convergence_local.png", dpi=300)
plt.show()


plt.figure(figsize=(10,6))
names = list(times_to_eps.keys())
times = [times_to_eps[n] for n in names]

plt.bar(names, times)
plt.title("Time to Reach ε-Convergence (Relative to Own Best)")
plt.ylabel("Time (s)")
plt.grid(axis='y')
plt.savefig(f"{OUTPUT_DIR}/epsilon_bar_local.png", dpi=300)
plt.show()


summary = []
for name, df in data.items():
    summary.append({
        "Algorithm": name,
        "Best Distance": df["Best Distance"].min(),
        "Final Distance": df["Best Distance"].iloc[-1],
        "Total Time (s)": df["Cumulative Time (s)"].iloc[-1],
        "Time to ε-conv (own best) (s)": times_to_eps[name]
    })

summary_df = pd.DataFrame(summary).sort_values("Best Distance")
summary_df.to_csv(f"{OUTPUT_DIR}/summary_table.csv", index=False)

print("\n SUMMARY")
print(summary_df)
